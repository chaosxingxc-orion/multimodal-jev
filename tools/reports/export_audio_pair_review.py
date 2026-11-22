"""Export offline browser workbooks from a frozen audio pair review packet."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_file, commit_json
from multimodal_jev.local_config import load_config
from tools.reports.build_tpo_actor_pairs import REVIEW, load_packet, read


PAGE = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>音频审核</title>
<style>
body{max-width:56rem;margin:2rem auto;padding:0 1rem;font:1.05rem/1.55 system-ui}
audio,textarea{display:block;width:100%;margin:.5rem 0}
button,select,input,textarea{font:inherit;margin:.2rem}fieldset{margin:1rem 0}
dt{font-weight:bold}dl{margin:.5rem}#notice{color:#8b2614}
</style>
<h1 id="title"></h1><p id="instructions"></p>
<fieldset><legend>答卷</legend>
<label>审核人 <input id="reviewer" autocomplete="off"></label>
<label id="armField">答卷 <select id="arm"><option value="">请选择</option>
<option value="listening-a">盲听 A</option><option value="listening-b">盲听 B</option>
</select></label>
<label id="blindField"><input type="checkbox" id="blind">
我独立听原音并按候选定义作答，未查看台词、配对、参考答案、他人答卷或模型预测</label>
</fieldset>
<label>跳到第 <input id="index" type="number" min="1" value="1"> 条</label>
<button id="prev">上一条</button><button id="next">下一条</button><span id="progress"></span>
<div id="players"></div><p id="condition"></p>
<label>判断 <select id="decision"></select></label>
<label id="labelField">候选 <select id="label"></select></label>
<label id="evidenceField">判断依据 <textarea id="evidence" rows="3"></textarea></label>
<p><button id="save">保存本条并前进</button>
<button id="export">导出答卷</button>
<label>恢复进度 <input id="import" type="file" accept=".json"></label></p>
<p id="notice" role="status"></p>
<details id="definitions"><summary>完整候选定义</summary><dl id="candidateList"></dl></details>
<script id="packet" type="application/json">__PACKET__</script>
<script>
"use strict";
const $ = id => document.getElementById(id);
const data = JSON.parse($("packet").textContent);
const blind = data.mode === "listening";
const statuses = blind
    ? {unreviewed:"未审核", unique:"唯一可判", ambiguous:"多解", unintelligible:"不可听"}
    : {unreviewed:"未审核", confirmed:"只改变指定判别条件", rejected:"不构成该语义对"};
const labelIds = new Set(data.candidates.map(c => c.id));
const empty = () => blind ? {status:"unreviewed", label:null} : {status:"unreviewed", evidence:""};
const rows = data.items;
const ids = rows.map(r => blind ? r.review_id : r.pair_id);
const idSet = new Set(ids);
let index = 0, answers = {}, startedAt = new Date().toISOString(), owner = null, dirty = false;
$("title").textContent = blind ? "独立音频盲听" : "语义对关系复核";
$("instructions").textContent = blind
    ? "每次只听一条录音。只有一个候选符合时选择该候选，多解或不可听请如实记录。"
    : "依次听两条录音，判断是否只改了指定的意图判别条件；写下依据。若你参加了盲听，请先完成并提交盲听答卷。";
for (const id of ["armField", "blindField", "labelField", "definitions"]) $(id).hidden = !blind;
$("evidenceField").hidden = blind;
$("index").max = rows.length;
function option(el, value, description) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = description;
  el.append(item);
}
for (const [key, value] of Object.entries(statuses)) option($("decision"), key, value);
option($("label"), "", "请选择");
for (const candidate of data.candidates) {
  option($("label"), candidate.id, candidate.id);
  const name = document.createElement("dt"), definition = document.createElement("dd");
  name.textContent = candidate.id; definition.textContent = candidate.description;
  $("candidateList").append(name, definition);
}
function count() { return Object.values(answers).filter(a => a.status !== "unreviewed").length; }
function progress() { $("progress").textContent = `已审核 ${count()} / ${rows.length}`; }
function show(position) {
  index = Math.min(rows.length - 1, Math.max(0, position));
  const item = rows[index], answer = answers[ids[index]] || empty();
  $("index").value = index + 1;
  $("players").replaceChildren();
  for (const path of (blind ? [item.audio] : item.audio)) {
    const player = document.createElement("audio");
    player.controls = true; player.preload = "none"; player.src = path;
    $("players").append(player);
  }
  $("condition").textContent = blind ? "" : item.condition;
  $("decision").value = answer.status;
  $("label").value = answer.label || "";
  $("label").disabled = answer.status !== "unique";
  $("evidence").value = answer.evidence || "";
  progress();
  dirty = false;
}
function identity() {
  const reviewer = $("reviewer").value.trim(), arm = blind ? $("arm").value : "relation";
  if (!reviewer || !arm) throw Error("请填写审核人并选择答卷。");
  if (owner && (owner.reviewer !== reviewer || owner.arm !== arm))
    throw Error("当前进度已绑定审核人和答卷；请恢复原身份或在新页面开启另一份答卷。");
  return {reviewer, arm};
}
function save() {
  const who = identity(), status = $("decision").value;
  const answer = blind ? {status, label:status === "unique" ? $("label").value : null}
    : {status, evidence:status === "unreviewed" ? "" : $("evidence").value.trim()};
  if (status === "unique" && !labelIds.has(answer.label)) throw Error("请选择唯一候选。");
  if (!blind && status !== "unreviewed" && !answer.evidence) throw Error("请填写判断依据。");
  owner = who;
  answers[ids[index]] = answer;
  dirty = false;
  progress();
}
function attempt(action) {
  $("notice").textContent = "";
  try { action(); } catch (error) { $("notice").textContent = error.message; }
}
function navigate(position) {
  if (dirty) save();
  show(position);
}
$("decision").onchange = () => { $("label").disabled = $("decision").value !== "unique"; dirty = true; };
$("label").onchange = $("evidence").oninput = () => { dirty = true; };
$("prev").onclick = () => attempt(() => navigate(index - 1));
$("next").onclick = () => attempt(() => navigate(index + 1));
$("index").onchange = () => attempt(() => navigate((Number($("index").value) || 1) - 1));
$("save").onclick = () => attempt(() => { save(); show(index + 1); });
$("export").onclick = () => attempt(() => {
  if (dirty) save();
  const who = identity();
  if (blind && !$("blind").checked) throw Error("请确认盲听声明。");
  if (!count()) throw Error("请至少审核一条。");
  const result = {schema:data.schema, packet_sha256:data.packet_sha256, reviewer_id:who.reviewer,
    started_at:startedAt, finished_at:new Date().toISOString(),
    answers:Object.fromEntries(ids.map(id => [id, answers[id] || empty()]))};
  if (blind) Object.assign(result, {blind_attested:true, audio_only:true});
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type:"application/json"}));
  const link = document.createElement("a");
  link.href = url; link.download = who.arm + ".json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("import").onchange = async () => {
  const [file] = $("import").files;
  if (!file) return;
  try {
    if (dirty || count()) throw Error("请在新页面恢复已导出的进度。");
    const who = identity(), saved = JSON.parse(await file.text());
    if (saved.schema !== data.schema || saved.packet_sha256 !== data.packet_sha256 ||
        saved.reviewer_id !== who.reviewer || !Number.isFinite(Date.parse(saved.started_at)) ||
        !saved.answers || Object.keys(saved.answers).length !== ids.length ||
        Object.entries(saved.answers).some(([id, a]) => !idSet.has(id) || !a ||
          !Object.hasOwn(statuses, a.status) ||
          (blind && (a.status === "unique" ? !labelIds.has(a.label) : a.label !== null)) ||
          (!blind && (typeof a.evidence !== "string" || (a.status !== "unreviewed" && !a.evidence.trim())))))
      throw Error("答卷与录音清单、审核人或审核类型不符。");
    if (blind && file.name.replace(/ \(\d+\)(?=\.json$)/, "") !== who.arm + ".json")
      throw Error("请选择当前 A/B 答卷对应的导出文件。");
    owner = who; startedAt = saved.started_at; answers = saved.answers;
    const next = ids.findIndex(id => answers[id].status === "unreviewed");
    show(next === -1 ? 0 : next);
    $("notice").textContent = "已恢复答卷进度。";
  } catch (error) { $("notice").textContent = error.message; }
};
window.addEventListener("beforeunload", event => {
  if (dirty || count()) { event.preventDefault(); event.returnValue = ""; }
});
show(0);
</script></html>
"""


def render(data):
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return PAGE.replace("__PACKET__", encoded)


def workbook(packet, manifest, destination, mode):
    packet_sha = file_hash(packet / "MANIFEST.json")
    if mode == "listening":
        candidates = [{key: item[key] for key in ("id", "description")}
                      for item in read(packet / "listening/candidates.json")]
        items = [{key: item[key] for key in ("review_id", "audio")}
                 for item in read(packet / "listening/roster.json")]
        audio = [row["audio"] for row in items]
    else:
        candidates = []
        items = read(packet / "relation/roster.json")
        audio = [name.removeprefix("../listening/") for row in items for name in row["audio"]]
        items = [dict(pair_id=row["pair_id"], condition=row["condition"],
                      audio=[name.removeprefix("../listening/") for name in row["audio"]]) for row in items]
    audio = set(audio)
    if (not items or any(not re.fullmatch(r"audio/[a-f0-9]{64}\.flac", name)
                         or name not in manifest["files"]["listening"] for name in audio)):
        raise ValueError("expected only anonymous pinned audio paths")

    def write(stream):
        with ZipFile(stream, "w", compression=ZIP_STORED) as bundle:
            bundle.writestr("review.html", render(dict(
                schema=REVIEW, packet_sha256=packet_sha, mode=mode, candidates=candidates, items=items)))
            for name in sorted(audio):
                bundle.write(packet / "listening" / name, name)

    commit_file(destination, write)
    return dict(sha256=file_hash(destination), entries=len(items), audio=len(audio))


def export(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new workbook output")
    manifest = load_packet(config)
    packet = Path(config["packet"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pair-review-", dir=output.parent) as tmp:
        stage = Path(tmp) / "export"
        stage.mkdir()

        def write(mode):
            name = mode + ".zip"
            return name, workbook(packet, manifest, stage / name, mode)

        with ThreadPoolExecutor(max_workers=min(config["io_workers"], 2)) as pool:
            files = dict(pool.map(write, ("listening", "relation")))
        result = dict(packet_sha256=file_hash(packet / "MANIFEST.json"), files=files)
        commit_json(stage / "MANIFEST.json", result)
        stage.rename(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("local")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config, _ = load_config(args.config, args.local)
    print(json.dumps(export(config, args.output)))
