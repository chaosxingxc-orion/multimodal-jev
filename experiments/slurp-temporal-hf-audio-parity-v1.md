# vLLM 实时 HF 音频编码诊断

依据[投影回放结果](slurp-temporal-projection-replay-v1.md)，测试同一 checkpoint 的 HF eager 音频编码与 vLLM 语言模型组合是否降低与 HF actor 的概率差异。此版本仅限诊断，不修改模型文件，不使用保存的激活或转写，不训练。

vLLM engine 初始化后，通过本地可信 callable 在其 TP1 worker 内安装同一 checkpoint 的 HF audio tower + projector。实时输入仍是 actor 的 mel 特征；适配器将 packed 特征还原为 HF 的按窗补零张量及 mask，输出长度必须与 vLLM 音频 token 数相同。加载时显式采用 CPU/float32 默认构造环境、BF16 参数、eager attention，再迁移到 NPU，以避免 worker 的默认环境改变非持久位置缓冲区。完整 HF 模型只在安装时临时加载，其语言模型不加入采样路径。当前实现未支持权重热更新、TP>1 或分布式训练，不能当作正式部署适配。

与原诊断相同的两条 learning 录音、六个提示、12 条响应，选择不依赖模型正确性；BF16、TP1、max_num_seqs=1、temperature=1、top_p=1、top_k=-1、seed=17。新后端重新真实采样，原 logprob 不替换。采样后比较 merged full、merged cached、base+LoRA full 三臂，最大响应 token logprob 差阈值仍为 0.02。外层诊断完成不代表三个内层通过。

采样 identity 显式保存 `diagnostic_only=true` 和后端名，worker 回执记录 checkpoint identity。正式 `train` 拒绝该标记。后续主线另把共享 rollout loader 改为默认拒绝，仅显式只读审计允许读取；正在运行的只读代码快照早于后一个 loader 加固，保持其原始哈希，不原地更新。

测试：宿主 torch 下 packed 音频还原及 rollout 回归共 13 passed；项目环境 rollout/准入相关回归 31 passed（其中有重叠，不合并为 44 个独立用例）。覆盖短窗、整窗、多窗、多录音顺序、非法长度、诊断数据拒绝训练和显式只读加载。

沿用固定 NGA/verl/vLLM 镜像，只挂载物理 4 号卡，8 CPU/64 GiB、无网络、900 秒超时。只在此隔离容器允许可信安装 callable 序列化，不改变镜像默认配置。

运行根目录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-temporal-hf-audio-parity-v1`。
容器：`c2cade56115d1a7a24eada43133c951875072ded57beaf2282fb6f736be4bf8b`。
详见[启动凭证](../docs/engineering/slurp-temporal-hf-audio-parity-v1-launch.json)。已建立单个完成等待句柄；结果待核实，未准入 RL。
