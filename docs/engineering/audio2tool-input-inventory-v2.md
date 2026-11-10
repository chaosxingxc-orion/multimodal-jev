# Audio2Tool 无答案输入索引

`prepare_audio2tool_inventory.py` 从固定下载清单和独立 `tools_registry.csv` 生成
可复用输入索引；不打开 query metadata、转写、参考调用或音频波形。
固定 release `f1388da9a3189541ab82adac88824a0661670c43`，核对 registry 的原始
Git blob SHA1，再记录 SHA256；工具定义保留 CSV 顺序、类型签名、描述、默认值及约束。
无基于答案的候选筛选，也不调用官方依赖 gold 的备用 schema builder。

独立工具表有 152 个工具：smart_car 86、smart_home 53、wearables 13；名称/ID 无重复。
文件清单包含 16,843 个 query 目录和 36,419 WAV，与卡片 query 数一致，
WAV 仍比卡片 36,421 少 2 个，原因尚未核定。

| Tier | Query 目录数 | WAV 数 | 索引语义 |
| --- | ---: | ---: | --- |
| 1 | 2,146 | 4,292 | 每条 2 个独立说话人变体 |
| 2 | 3,160 | 6,320 | 每条 2 个独立说话人变体 |
| 3 | 2,146 | 4,292 | 每条 2 个独立说话人变体 |
| 4 | 2,132 | 4,276 | 2,125 条各 2 个、2 条各 3 个、5 条各 4 个变体 |
| 5 | 2,146 | 4,292 | 每条 2 个独立说话人变体 |
| 6 | 2,146 | 4,292 | 每条 2 个独立说话人变体 |
| 7 | 2,146 | 7,834 | 有序多轮对话，不能当作说话人变体 |
| 8 | 821 | 821 | 每条一个混合音频 |

Tier7 按文件名中的数值 turn 排序并保留 user/agent 角色，拒绝重复轮次：
1,445 条 3 段、18 条 4 段、677 条 5 段、6 条 7 段。当前 SFC Omni 运行配置
最多接收 2 段音频，因此不能直接迁移到此 Tier，更不能截断来满足接口。
若其余 tier 每个音频变体独立评测、Tier7 每段完整对话评测一次，则文件清单推导
出 30,731 个请求。这只是未来覆盖计划，不是已执行评测或论文分母认证。

清单还确认 `*_audios/query_*/query_metadata.json` 存在；因此旧 tier loader 有
目录级对应，修正此前“只有平铺 JSON、旧 loader 不兼容”的过强推断。
字段级兼容、成功/失败音频筛选及多轮目标语义仍待单独验证。

产物位于 `/mnt/cephfs/xingchao/experiments/multimodal-jev/audio2tool-input-inventory-v2/`：
`registry.json` 与 `audio-inventory.jsonl`。首次临时脚本生成的 v1 保留；正式工具的
v2 产物哈希与 v1 一致，证明此次改为可复用入口未改变内容。
来源、计数和哈希见 [结果](audio2tool-input-inventory-v2.json)。

音频哈希继承下载清单，没有重新读取或校验所有当前 WAV。
这不是完整 inference request：还需固定 prompt、全部工具的上下文预算、多轮音频
接口和正式评分适配。数据全部保持外部 TEST，不用于训练或教师反馈。
