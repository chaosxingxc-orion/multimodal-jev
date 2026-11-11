# Omni 完整 CPU 加载与 meta 准备

本步骤为后续两卡 FSDP 概率验证准备显式 Thinker 加载路径，保持 BF16 参数
和原始 FP32 位置编码 buffer。没有初始化 FSDP/HCCL、NPU、前向或优化器。
不把 CPU 加载成功解释为完整分布式 actor 已可用。

## 实现约束

`src/multimodal_jev/omni_fsdp_loading.py` 提供三个边界：

- `build_meta_thinker(full_config)`：使用原配置和 SDPA，显式保留真实 buffer，
  仅对 meta 参数转换 BF16，不调用整模型 dtype cast。冻结模型并校验参数
  仍全在 meta、buffer 的值和 dtype 未变。
- `load_cpu_thinker(path)`：显式 HF Thinker `from_pretrained`，本地 BF16
  参数、CPU 设备、SDPA，无 remote code；拒绝 missing/mismatched/error
  keys，只允许完整 Omni checkpoint 中的 talker/code2wav 额外键。
- `require_buffers(model, expected)`：全部四个非持久化位置编码 buffer
  必须是有限 FP32，其名称、形状、dtype 和原始连续字节 SHA 全部匹配。
  仅 `.float()` 恢复 dtype 无法通过发生过舍入的数值校验。

这些辅助函数供已认证的运行器使用，不能凭路径字符串或参考字典证明模型
真实性；运行器负责固定源码、config、checkpoint 和参考记录。当前函数为
冻结诊断准备，不创建优化器，也没有 rank 分片初始化。

## 真实预检范围

`tools/jev/probe_omni_fsdp_loading.py` 在固定镜像中执行：

1. 按已认证历史 prefix identity 校验原模型全部 25 份文件的 SHA。
2. 构造完整 meta Thinker，校验 31,719,205,488 个参数、1,407 个参数张量，
   四个 FP32 buffer 与[已认证参考](omni-buffer-precision-v1.md)完全一致。
3. 实际 CPU 加载完整 checkpoint；比较 CPU/meta 全部参数名称、形状和
   dtype，并验证冻结状态与四个非持久化 buffer。
4. 遍历所有原始 Thinker checkpoint 张量，将 HF 合并的 expert gate/up/down
   参数按原索引切片，逐一核对 shape、dtype 和每个元素。保留全部原始键的
   检查记录，不以参数总数或只抽查第一层代替完整覆盖。
5. 再次检查 buffer、参数结构和模型全部文件 SHA，保存终态、耗时与峰值 RSS。

无权重复制或新 checkpoint 写入；只保存小型 buffer 张量和检查元数据。
HF 从完整模型目录读取权重，运行环境无网络、无 NPU 设备映射，模型目录只读。

## 验证边界

固定镜像完整预检已完成，源码 `87aa0b3c4c791c9f0ea445556a84c63c7690bb89`，
容器 `d0253dcc9e9235e235a507334b073312e19a1f1c57c83088363a4b9e136f20d3`，
8 CPU / 128 GiB、无网络/设备，正常退出 0、无 OOM。耗时 246.361 秒，进程
峰值 RSS 118.326 GiB。19,743 个原始 Thinker 张量、31,719,205,488 个
参数全部逐元素匹配，合并后的 1,407 个参数张量与 meta 清单完全一致，
四个 FP32 buffer 与参考精确相同。

外部按实际 Docker 状态、源码 Git 对象、所有产物 SHA 核验；独立比较保存
buffer 与已认证参考，并由原 safetensors header 重建全部合并参数形状。
checkpoint 元素比较由固定源码探针在容器内执行，外部复核其算法与完整
覆盖，没有为核验再次加载完整模型。见[外部核验记录](omni-fsdp-loading-v1-verified.json)。

118.326 GiB 是整个进程的峰值 RSS，不是 NPU 显存或加载完成后的常驻内存。
它包含 checkpoint 读取/转换等阶段，不能仅按约 59.082 GiB 的 BF16 权重
大小配置 CPU 加载容量。后续两 rank 初始化须重新规划主机内存，并采集
真实分片初始化期间的资源峰值。

宿主 `tests/test_omni_fsdp_loading.py` 的 9 项测试通过，覆盖 buffer 值在
BF16 往返后即使 dtype 恢复仍拒绝、完整 buffer 清单/持久化属性、参数冻结
与 meta/BF16 约束，以及 expert gate/up 的位置和索引。宿主无 Transformers，
真正的完整模型构造与加载由固定镜像预检执行。

后续仍需验证 rank0 CPU 实权重与其他 rank meta 参数如何由 FSDP 初始化、
完整参数分片与通信、buffer 同步后的身份、真实峰值 NPU 内存及完整响应
概率。不能直接将 meta 模型 `to_empty` 后的未初始化参数或 buffer 当作权重。
原 NPU 单层探针消费已保存位置编码，其通过也不能替代完整加载验收。

后续[两卡初始化探针](omni-fsdp-initialization-plan-v1.md)已准备但未派发。
非递归meta参数物化回调保留FP32 buffer，固定镜像19项契约测试通过；
完整探针将逐一聚合108个单元校验全部参数，不做前向或训练。
