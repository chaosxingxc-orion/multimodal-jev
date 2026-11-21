# Stage1 rc7 两端评测和 Stage2 数据准备

Stage1 rc7 已通过独立训练审核：318 步，40,513 条 clean TRAIN，两轮 81,026 次有效曝光。此次只比较预登记的 step 0 与 step 318；中间检查点和源域完整曲线另行保留。所有待评测音频由同一个语音多模态策略直接消费，使用稳定前缀布局及固定阈值 0.5。

| 主机 | 物理 NPU | 工作 | 每个端点录音数 |
| --- | --- | --- | ---: |
| 69 | 0–1 | FSC native31，step 0 / 318 | 3,793 |
| 69 | 2–3 | MINDS native14，step 0 / 318 | 8,168 |
| 69 | 4–7 | step 318 的 TRAIN 数值验收及完整 reference；reference 服务实际使用 6–7 | 40,513 |
| 175 | 4–7 | SLURP clean TEST，step 0 / 318 | 12,991 |

三个目标集分别走独立链：所在主机本地编码并重读校验输入，256 条无标签音频验收两端 HF 与 vLLM 的完整决策宽度，校验通过后全量推理，最后重建推理日志并独立读取封存真值评分。不同数据集之间无启动依赖；正式推理两端使用同一套数据、同一主机、同一组卡。目标标签不参与运行时验收、选模、阈值或 Stage2 样本构建。

操作记录根为 `/srv/multimodal-jev-xingchao/operations/stage1-endpoints-and-stage2-inputs-v0.0.131/`。其中 `native-endpoints-v0.0.134/{fsc,minds,slurp}/run` 为缓存及独立 HF 容器记录，`native-endpoints-v0.0.135/{fsc,minds,slurp}/run` 为对照、全量预测及评分链记录。v133 在 CPU 启动阶段发现缺失源模块，保留失败终态；未进入 NPU。冻结镜像和源音频走已有共享只读资产；新缓存、预测和运行状态写在服务器本地盘。

TRAIN reference 与目标预测隔离。TRAIN 的独立审核通过后，后台采样程序接续构建并复算 Stage2 训练视图，记录在 `stage2-difficulty-successor/`。这里不启动 Stage2 正式训练。
