# vLLM 原生多 Span pooling 诊断

启动前协议：复用 HF smoke-v2 同一条学习录音、同一基座、模板和候选映射；无权重更新，无生成。物理卡 4，固定镜像 `52ba94e1...756c`，TP1 BF16 eager，prefix cache/chunked prefill 关闭，最长输入 1024。

独立适配 `Qwen3ASRForSpanDecisions` 继承 vLLM 原生 Qwen3-ASR，保留 LM head，使用 pooling runner。自定义 pooler 在完整前向后按显式位置取得候选 logits，逐请求核验 expanded input IDs。选择该方式是因为通用 pooling 转换会移除 LM head，而 STEP pooling 默认读 marker 同位，不满足 p−1 契约。代码独立编写，参考固定镜像 vLLM Apache-2.0 接口，未修改上游文件。

音频使用 HF 提取的同一特征，交给 vLLM 原生音频塔；不使用转写或预存音频投影。输入压缩 audio_pad 后由引擎重新展开，最终必须与 HF 输入 token 完全相同。成功条件为引擎返回有限的 3×2 候选 logits、无生成 token；概率一致性另以最大候选 logprob 误差 ≤0.02 判定，超过则不准入 RL。单录音通过不代表多样本/适配器一致性已通过。

根目录 `/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-vllm-spans-pooling-v1`。硬截止 900 秒，单个 docker wait 完成句柄；保留原始失败证据。

v1 终态 1：parent 中覆盖原架构注册没有传到 spawn engine worker，worker 加载回原始模型，内存预检报告无 pooling task。没有产生有效音频 pooling 输出。

v2 改为独立架构名，并在入口模块导入时注册以覆盖 spawn；同时对 vLLM 内部全零 token 的 profiling 请求使用真实 LM head 预热。真实请求仍必须提供位置、候选和完整 token identity。运行根为 `slurp-vllm-spans-pooling-v2`，原协议不变。

v2 已退出 0：真实 pooling 返回 3×2 有限候选 logits，600 个展开输入 token 与 HF 一致，无 token 生成。最大候选 logprob 差 0.06325805，超过预设 0.02，**概率准入失败**。这证明新接口能运行，不代表 HF/vLLM 已一致；未执行 RL 更新。旧音频塔差异是线索，尚未在新接口确定误差来源。
