# SLURP原生评分的输入条件与运行归属

2026-11-12 UTC。检查发现`score_slurp_native.py`已有完整请求/预测覆盖、
发布文件哈希和官方metric源码检查，但评分结果未保留音频条件、模型和提示词身份。
尤其TRAIN零波形诊断可产生没有显式条件字段的分数，容易在后续汇总时混用。

评分入口现在要求config/result一致声明audio-direct、冻结参数、无adapter、
模型文件不变；audio condition必须在配置、结果和每条预测之间一致。
零波形只允许TRAIN诊断，并输出`audio_task_result=false`；真实音频为true。
该字段仅表明输入条件，不表示正式基准或学术复现认证。
结果还记录identity哈希、模型身份、提示词哈希和推理runner哈希。
结果中的请求/预测计数必须与实际文件一致，Boolean计数不接受为整数。

验证：`tests/test_slurp_scoring_entry.py`、`tests/test_slurp_score.py`、
`tests/test_slurp_native.py`共32项通过。新增入口测试使用实际临时发布和运行文件，
保留真实预测归属检查，官方metric数值用替身隔离；它不代替已有上游metric fixture，
也不是新的模型推理成绩。没有改动冻结运行快照或重写历史分数。

同轮07:23 UTC运行观察：本机D控制器PID3552952及后继七个控制器均实际存活，
命令匹配已登记配置。第8步训练容器
`7766f5b9b87d45f01dff1db8467636d4722161c4525ae65ac172252f9f468ec1`
实际退出0且镜像匹配固定镜像；验收容器
`160e8350dc003367daf5664ffd2f0c68e436ac7d9494a3d6b258baf8c423aa38`
观察时运行中。因此已接受曲线仍截至第7步，不能将第8步计为已验收。
下一次普通状态观察不早于07:33 UTC。
