# 超过索引上限的机器生成证据（fingerprint 索引）

本文件登记仓库中**存在但未索引**的机器生成证据。按工程规约，单个机器生成证据文件超过
1 MB 时不在 GitHub 索引正文，只保留 SHA256 指纹与来源说明；文件本身留在运行主机的
`/home/xingchao/workspace/multimodal-jev` 工作树内，并由 `.gitignore` 排除。

判定与证据边界：本表只证明「该路径在 2026-11-18 的内容身份」，不替代任何 NPU 验收、
完整曲线或独立审核结论。引用这些文件的文档在克隆环境里会出现断链，属于预期行为；
需要正文时按指纹回原主机核对。

| 路径 | 大小 | SHA256 |
| --- | --- | --- |
| `docs/engineering/context-admitted-full-npu-v1.json` | 68.11 MiB | `34defc0ae54c9362e51ce31aa408f85774c5e5ab00b4ddbc27f55869bc0341c5` |
| `docs/engineering/context-checkpoint-shared-npu-v3.json` | 12.18 MiB | `157f11d26419eeb7b901a96a5a1bacc8cfb3afd606bdae07efbe3bdc8b3f4457` |
| `docs/engineering/context-online-shared-npu-v4.json` | 12.18 MiB | `170d427120b84bb92102f29035f0fd7bda85c818df29c751bde39c0b2942fbe5` |
| `docs/engineering/context-online-shared-resume-npu-v4.json` | 6.09 MiB | `c9f589d0c75b5c2d963efc1fe0419b38b1588dab00a2f513f9411a6850e0940c` |
| `docs/research/minds-local-audio-audit-v1.json` | 4.74 MiB | `d022f32e4a7a362495d3911d35fd44fa90933a544a51236f505be138740c2a37` |
| `docs/research/context-stage1-native-structure-diagnostic-v1.json` | 3.22 MiB | `d14d6ca2b8e4894a4a2b5ffa906e057f530fb70646d0086ca585e3634ff15120` |
| `docs/engineering/slurp-batch-updated-components-v1.json` | 1.25 MiB | `9b8360a67e0574271bf701f5b01648c7fd7aa65c396dcd55513c90523f113323` |
| `docs/engineering/context-stage1-selection-step0000-v2.json` | 1.03 MiB | `cdc71f70211a0223041f561629e1f540b782ae723118c8bb389843402af6cdc4` |
| `docs/engineering/context-stage1-selection-step0634-v2.json` | 1.03 MiB | `7e501aaa90177a4e85ff286ba64e9fe5a35c06eac103d4a34b72ddc237ace088` |

其中 `context-stage1-selection-step0000-v2.json`、`context-stage1-selection-step0634-v2.json`、
`context-checkpoint-shared-npu-v3.json`、`context-online-shared-npu-v4.json`、
`context-admitted-full-npu-v1.json`、`minds-local-audio-audit-v1.json` 的完整 SHA256
已由引用它们的证据索引文件（如 `context-probability-endpoint-population-control-v1.json`、
`context-stage1-initial-endpoint-comparison-v1.json`）逐字记录；
`slurp-batch-updated-components-v1.json` 与
`context-stage1-native-structure-diagnostic-v1.json` 以本表为唯一指纹来源。

`.gitignore` 中的对应条目位于「Machine-generated evidence over the 1 MB index limit」段。
新增超限证据时：先写指纹，再加排除项，不把正文提交进仓库。
