#!/usr/bin/env python3
"""Read-only catalog for the ceph dataset / model layout.

``/mnt/cephfs/xingchao`` scatters one logical asset across several top-level
directories -- SLURP alone occupies ten of them, FSC four, SFC four.  This tool
reads ``configs/ceph-assets-v1.json`` (the declarative family/stage manifest)
and derives three things the raw tree cannot give:

* ``--check``          validate that every declared path still exists, and that
                       no undeclared top-level entry has appeared under
                       ``models/`` or ``datasets/`` (the "不要散落" guard).
* ``--report PATH``    emit a JSON catalog of the tree as it actually is.
* ``--materialize-views``  create ``datasets/_families/<family>/<stage>`` and
                       ``models/_families/<group>/<name>`` symlink views, so
                       there is one stable path per logical asset.

The tool never moves, copies, renames or deletes data.  Existing paths stay
valid on purpose: ~1,595 files in this repository reference them and many are
hash-bound frozen evidence, so a physical relocation would break reproduction
and audits.

Sizes are measured with a *single* ``du -sb`` invocation over all top-level
directories.  That matters here: 13 top-level directories are hardlink mirrors
of ``omni/*`` (same inode, ``nlink=2``), and only one process-wide ``du`` run
dedupes them.  ``getfattr -n ceph.dir.rbytes`` returns nothing on this mount,
so the recursive-server-side numbers are not available -- do not trust them.

Usage::

    python -m tools.ceph_asset_catalog --check
    python -m tools.ceph_asset_catalog --check --report -
    python -m tools.ceph_asset_catalog --check --sizes --report catalog.json
    python -m tools.ceph_asset_catalog --materialize-views --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = "configs/ceph-assets-v1.json"
DEFAULT_ROOT = "/mnt/cephfs/xingchao"
SCHEMA = "ceph-assets-v1"

# Directories the contract lists but that are allowed to be absent: the view
# layer is created on demand by --materialize-views.
OPTIONAL_TOP_LEVEL = {"_families"}


class ManifestError(RuntimeError):
    """The manifest is malformed or internally inconsistent."""


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != SCHEMA:
        raise ManifestError(
            f"{path}: expected schema {SCHEMA!r}, got {manifest.get('schema')!r}"
        )
    for key in ("root", "families", "model_groups", "top_level_contract"):
        if key not in manifest:
            raise ManifestError(f"{path}: missing required key {key!r}")
    return manifest


def iter_declared(manifest: dict):
    """Yield ``(role, family, stage, relpath)`` for every declared asset path."""
    for family in manifest["families"]:
        name = family["family"]
        for stage in family["stages"]:
            yield "dataset", name, stage["stage"], stage["path"]
    for group in manifest["model_groups"]:
        name = group["group"]
        for member in group["members"]:
            yield "model", name, "canonical", member["path"]
            if "derived_view" in member:
                yield "model", name, "derived-view", member["derived_view"]
    for entry in manifest.get("misplaced_models", []):
        yield "model", "misplaced", "training-output", entry["path"]
    for entry in manifest.get("reclaim_register", []):
        yield "model", "reclaim", "candidate", entry["path"]
    for relpath in manifest.get("standalone_datasets", []):
        yield "dataset", "standalone", "root", relpath


def declared_index(manifest: dict) -> dict[str, list[tuple[str, str, str]]]:
    """Map relpath -> [(role, family, stage), ...] to expose double-declaration."""
    index: dict[str, list[tuple[str, str, str]]] = {}
    for role, family, stage, relpath in iter_declared(manifest):
        index.setdefault(relpath, []).append((role, family, stage))
    return index


def _top_level_entries(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(entry.name for entry in os.scandir(directory))


def check(manifest: dict, root: Path) -> dict:
    """Validate the manifest against the tree on disk.  Returns a report dict."""
    problems: list[str] = []
    warnings: list[str] = []

    index = declared_index(manifest)
    missing: list[str] = []
    for relpath in sorted(index):
        if not (root / relpath).exists():
            missing.append(relpath)
            problems.append(f"declared path missing on disk: {relpath}")

    duplicated = {path: hits for path, hits in index.items() if len(hits) > 1}
    for path, hits in sorted(duplicated.items()):
        problems.append(f"path declared {len(hits)} times: {path} ({hits})")

    contract = manifest["top_level_contract"]
    undeclared: dict[str, list[str]] = {}
    absent_contract: dict[str, list[str]] = {}
    for area in ("models", "datasets"):
        allowed = set(contract[area])
        on_disk = _top_level_entries(root / area)
        extra = [name for name in on_disk if name not in allowed]
        if extra:
            undeclared[area] = extra
            for name in extra:
                problems.append(f"undeclared top-level entry: {area}/{name}")
        absent = [
            name
            for name in sorted(allowed)
            if name not in OPTIONAL_TOP_LEVEL
            and not (root / area / name).exists()
        ]
        if absent:
            absent_contract[area] = absent
            for name in absent:
                warnings.append(f"contract lists absent entry: {area}/{name}")

    # Every family stage should live under its family's expected top-level
    # directory; a stage pointing elsewhere is the scattering this manifest is
    # meant to make visible, so surface it rather than silently accept it.
    cross_area: list[dict] = []
    for family in manifest["families"]:
        name = family["family"]
        tops = {stage["path"].split("/")[1] for stage in family["stages"]}
        if len(tops) > 1:
            cross_area.append(
                {
                    "family": name,
                    "top_level_dirs": sorted(tops),
                    "stages": [
                        {"stage": s["stage"], "path": s["path"]}
                        for s in family["stages"]
                    ],
                }
            )

    return {
        "schema": SCHEMA,
        "root": str(root),
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
        "missing_declared_paths": missing,
        "duplicated_declarations": sorted(duplicated),
        "undeclared_top_level": undeclared,
        "contract_absent": absent_contract,
        "scattered_families": cross_area,
        "counts": {
            "declared_assets": len(index),
            "families": len(manifest["families"]),
            "model_groups": len(manifest["model_groups"]),
            "standalone_datasets": len(manifest.get("standalone_datasets", [])),
        },
    }


def _sample_files(directory: Path, limit: int) -> list[str]:
    """Return up to ``limit`` file paths (relative to ``directory``)."""
    found: list[str] = []
    for current, _dirs, files in os.walk(directory):
        for name in files:
            found.append(os.path.relpath(os.path.join(current, name), directory))
            if len(found) >= limit:
                return found
    return found


def hardlink_groups(
    root: Path, areas=("models", "datasets"), sample: int = 40
) -> list[dict]:
    """Detect top-level directories whose files are hardlinks of ``omni/`` twins.

    Mirroring here is per *file* (nlink=2), not per directory, so a directory
    inode comparison finds nothing.  For each top-level directory we sample a
    few files and check whether the same relative path under ``omni/<area>/``
    resolves to the same inode.  A sampled hit rate of 100% is the mirror
    signature; anything else is reported as a partial overlap.
    """
    mirrors: list[dict] = []
    for area in areas:
        directory = root / area
        if not directory.is_dir():
            continue
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            if not entry.is_dir(follow_symlinks=False):
                continue
            paths = _sample_files(Path(entry.path), sample)
            if not paths:
                continue
            matched = 0
            compared = 0
            for relpath in paths:
                twin = root / "omni" / area / entry.name / relpath
                try:
                    twin_info = twin.stat()
                    own_info = (Path(entry.path) / relpath).stat()
                except OSError:
                    continue
                compared += 1
                if (twin_info.st_dev, twin_info.st_ino) == (
                    own_info.st_dev,
                    own_info.st_ino,
                ):
                    matched += 1
            if compared and matched:
                mirrors.append(
                    {
                        "path": f"{area}/{entry.name}",
                        "twin": f"omni/{area}/{entry.name}",
                        "sampled_files": compared,
                        "same_inode": matched,
                        "fully_mirrored": matched == compared,
                    }
                )
    return mirrors


def du_sizes(root: Path, areas=("models", "datasets")) -> dict[str, int]:
    """Run one ``du -sb`` over every top-level directory so hardlinks dedupe."""
    targets: list[str] = []
    for area in areas:
        directory = root / area
        if not directory.is_dir():
            continue
        targets.extend(
            entry.path
            for entry in os.scandir(directory)
            if entry.is_dir(follow_symlinks=False)
        )
    if not targets:
        return {}
    # Single invocation on purpose: du dedupes hardlinks only within one run.
    completed = subprocess.run(
        ["du", "-sb", *sorted(targets)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"du failed ({completed.returncode}): {completed.stderr.strip()}")
    sizes: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        size, path = parts
        sizes[os.path.relpath(path, root)] = int(size)
    return sizes


def build_report(manifest: dict, root: Path, sizes: dict[str, int] | None) -> dict:
    report = check(manifest, root)
    report["hardlink_mirrors"] = hardlink_groups(root)
    report["assets"] = [
        {
            "role": role,
            "family": family,
            "stage": stage,
            "path": relpath,
            "exists": (root / relpath).exists(),
            "bytes": (sizes or {}).get(relpath),
        }
        for role, family, stage, relpath in iter_declared(manifest)
    ]
    if sizes is not None:
        report["top_level_bytes"] = dict(sorted(sizes.items()))
        report["totals"] = {
            area: sum(
                size for path, size in sizes.items() if path.startswith(f"{area}/")
            )
            for area in ("models", "datasets")
        }
    return report


def _relative_link_target(link_path: Path, target: Path) -> str:
    return os.path.relpath(target, link_path.parent)


def materialize_views(manifest: dict, root: Path, dry_run: bool) -> dict:
    """Create idempotent symlink views.  Never overwrites an existing link."""
    created: list[str] = []
    existing: list[str] = []
    conflicts: list[str] = []

    def link(link_rel: str, target_rel: str) -> None:
        link_path = root / link_rel
        target = root / target_rel
        if not target.exists():
            conflicts.append(f"{link_rel}: target {target_rel} does not exist")
            return
        wanted = _relative_link_target(link_path, target)
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_symlink() and os.readlink(link_path) == wanted:
                existing.append(link_rel)
            else:
                conflicts.append(f"{link_rel}: exists and is not the expected link")
            return
        created.append(link_rel)
        if dry_run:
            return
        link_path.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(wanted, link_path)

    for family in manifest["families"]:
        for stage in family["stages"]:
            link(
                f"datasets/_families/{family['family']}/{stage['stage']}",
                stage["path"],
            )
    for group in manifest["model_groups"]:
        for member in group["members"]:
            name = Path(member["path"]).name
            link(f"models/_families/{name}", member["path"])

    return {
        "dry_run": dry_run,
        "created": sorted(created),
        "already_present": sorted(existing),
        "conflicts": sorted(conflicts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="manifest path")
    parser.add_argument("--root", default=None, help=f"ceph root (default: manifest root, else {DEFAULT_ROOT})")
    parser.add_argument("--check", action="store_true", help="validate the manifest (default action)")
    parser.add_argument("--report", default=None, metavar="PATH", help="write catalog JSON ('-' for stdout)")
    parser.add_argument("--sizes", action="store_true", help="measure sizes with one du -sb invocation")
    parser.add_argument("--materialize-views", action="store_true", help="create _families symlink views")
    parser.add_argument("--dry-run", action="store_true", help="with --materialize-views, only report")
    args = parser.parse_args(argv)

    if not any((args.check, args.report, args.materialize_views)):
        args.check = True

    manifest = load_manifest(Path(args.config))
    root = Path(args.root or manifest.get("root") or DEFAULT_ROOT)

    exit_code = 0

    if args.check or args.report:
        sizes = du_sizes(root) if args.sizes else None
        report = build_report(manifest, root, sizes)
        if args.report:
            payload = json.dumps(report, indent=2, ensure_ascii=False)
            if args.report == "-":
                print(payload)
            else:
                Path(args.report).write_text(payload + "\n", encoding="utf-8")
                print(f"catalog written: {args.report}", file=sys.stderr)
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["ok"]:
            exit_code = 1
            for problem in report["problems"]:
                print(f"FAIL {problem}", file=sys.stderr)
        else:
            print(
                f"OK {report['counts']['declared_assets']} declared assets, "
                f"{report['counts']['families']} dataset families, "
                f"{len(report['hardlink_mirrors'])} hardlink mirror groups",
                file=sys.stderr,
            )

    if args.materialize_views:
        views = materialize_views(manifest, root, dry_run=args.dry_run)
        print(json.dumps(views, indent=2, ensure_ascii=False))
        if views["conflicts"]:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
