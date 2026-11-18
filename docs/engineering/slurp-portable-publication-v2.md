# SLURP 发布目录的审核依赖修复

审核证据是正确的数据溯源依赖。将开发机的审核文件、源码和原始发布目录绝对路径作为
每次训练必须读取的依赖，是 v1 发布器/加载器的可移植性 bug。
原先 `publish()` 把输入配置原样写入 publication-config.json，`load()` 再读取其中
所有绝对路径；因此数据集本身完整、训练容器正常，也可能在加载阶段 FileNotFoundError。

源端已增加 `src/multimodal_jev/slurp_portable_release.py`，默认发布命令
`tools/jev/publish_slurp_official_intent.py` 现在输出 v2。审核包、审核结论、原生请求和
标注、任务定义、原始代码证据全部以独立文件随发布目录保存，manifest覆盖完整文件集。
publication-config.json 的可执行引用及pins均为相对路径；禁止越界、符号链接和漏文件。
原始绝对路径仅在证据内容中保留溯源语义，不作为加载路径。音频仍由共享音频/本地镜像
提供，逻辑音频标识未改；本修复不声称把音频资源也复制进发布目录。

`slurp_official_training.py` 的原始 v1 实现保持字节不变，因为旧发布已经pin该源码。
新加载入口兼容读取v1，但不会把v1悄悄解释成便携版本。旧实验及冻结包继续可复现；
新训练包明确使用v2。不能直接改旧MANIFEST或抹掉审核校验以规避失败。

实际新发布：
`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/official-clean-intent-v2`

manifest SHA256：`2418e2a974b529fe95a4fd41c0c99673e10dc0b698e91e1136eef127a7dd72f2`。
来源v1 manifest：`38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482`。
40,513 learning、9,226 selection（TRAIN内部选模）、647排除项；六个原有数据/任务/排除
payload文件逐字节保持原哈希。源请求、原生标签与完整分母由独立verify脚本再次复算通过。
迁移不重新评估标签、不改变gold、不加入selection训练，不重新下载或复制主模型。

迁移命令通过本地宏解析：

```bash
PYTHONPATH=src:tools/jev python -m tools.jev.migrate_slurp_portable_release \
  --config configs/portable/operations/slurp-portable-publication.json \
  --local .local/config/slurp-portable-publication-v2.json
```

证据 `/srv/multimodal-jev-xingchao/operations/slurp-portable-publication-v2/migration-audit.json`。
测试包括删除原始审核/source目录、迁移发布目录后加载和独立复核、审核文件破坏、重写
manifest后错标/缺样本拒绝、路径逃逸和符号链接拒绝、selection不能训练、旧版兼容。
数据发布相关16tests及连训练/恢复包43相关tests通过。该结论不是NPU训练收益。

新DP2包macro-training-v4使用新加载入口，删除临时publication文件挂载方案；旧全量日程
和audio mirror原有release_id仅通过显式schedule_release_sha256关联。准备阶段验证迁移
来源及六项payload原哈希一致，日程内容和全量采样顺序不变。无NPU preflight实际加载
全量数据及634日程后才创建NPU服务。v3只是临时封包，未派发，不得继续使用该补挂载方案。

固定镜像v4无NPU预检发现Python版本的sum舍入差导致audio_hours精确比较失败。
加载器和独立verify的时长统计改math.fsum，绝对容差仅1e-9小时（3.6微秒）；
计数/标签/文件哈希仍精确，17发布相关测试通过，包括拒绝真实时长偏差。
v5封包纳入此修复，仍使用同一不可变v2数据manifest。v4未创建NPU服务。

发布权限也在源端统一：临时目录原子重命名前，目录设为0755、文件设为0644，
避免mkdtemp的0700权限使其他训练或收集用户无法读取。实际v2根目录已修复为0755，
manifest及数据内容哈希未变。

最新复核：`tests/test_slurp_portable_release.py`与
`tests/test_slurp_official_training.py`共18项测试通过，包括搬迁发布目录并删除原始
审核/source目录后仍可加载及独立校验。实际v2再次通过全量加载和独立人口/标签复算；
40,513/9,226、647排除项及六个payload哈希均保持不变。

评测入口也已切换到兼容v1/v2的便携加载器：evaluate_context_event_records.py、
evaluate_context_selection_vllm.py、audit_context_rl_selection_v2.py。
搬迁v2发布并删除原审核/source目录后，评分结果与v1一致；评分、曲线、评测入口、
审核和收集共70项回归测试通过（162.72秒）。实际v2清单的26个文件哈希再次逐项核验通过。
这些测试验证发布和评测兼容性，不代表新模型已取得任务收益。
