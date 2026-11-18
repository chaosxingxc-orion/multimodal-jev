# MINDS声学数值定位：卷积与投影有限诊断

状态：2026-11-17 UTC。诊断已执行、收集并核验容器和传输身份；不是全模型一致性准入。正式训练尚不能据此放行。

## 问题与研究边界

同一MINDS259条有限面板和同一Stage1策略下，原service32配置有10条logit差异，声学KV16候选有196条差异及4次0.5阈值翻转。两个运行同时改变声学实现和KV调度容量，不能把全部误差归因于其中一项。目标标签不用于诊断、调参或难例采样。

本次检验两个局部假设：相同BF16卷积输入因chunk batch形状改变而变化；HF缓存尾部特征与推理侧补零不一致。继而检查GELU、固定形状conv_out和投影tile槽位。没有修改既有actor、推理worker、冻结包或数值容差。

## 方法

固定镜像`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，175物理7单卡。模型只读、真实缓存只读、输出与所有热缓存落本地盘，复用原有限任务生命周期和排他租约。

从同一冻结MINDS输入回执按padded mel帧数选择300、1100、2400、2500、2600、2700、2800、2900、3100各一条；每长度取字典序第一ID，未读取gold或预测。对应有效帧为290、1100、2321、2416、2552、2688、2764、2884、3068。运行前核对每个缓存文件hash和全部输入tensor身份。

读取现有模型的conv2d1/2/3与conv_out真实权重，记录权重tensor身份；只建立这些层，不加载文本模型。前向使用BF16、推理模式。

- 原录音按100帧拆块，与补齐到32块的同一输入对照。
- 每层同时比较相同上一层输入的局部batch差异和累积差异。
- 比较真实缓存padding与按有效mask清零后的输入及卷积输出。
- v4增加每层GELU后的比较、项目固定tile_rows128/tile_batch8投影、前置1和7个邻接块后的槽位比较。
- 每项记录exact、非一致元素数、最大绝对误差与首批坐标；有差异时额外保存首个差异块tensor。非有限数直接失败。

## 结果与证据

| 验证 | 9条有限样本结果 |
| --- | --- |
| 缓存padding与清零后的输入 | 完全一致；padding本来就是0 |
| 三层卷积局部与累积batch比较 | 全部exact，最大误差0 |
| 各层GELU后的batch比较 | 全部exact，最大误差0 |
| 固定conv_out投影batch比较 | 全部exact，最大误差0 |
| conv_out投影槽位偏移1/7 | 全部exact，最大误差0 |

operations根由本地宏绑定；当前为`/srv/multimodal-jev-xingchao/operations`。

- v3：`minds-acoustic-numeric-v3/jev-minds-acoustic-numeric-v3-collected/evaluation/report.json`。
  manifest `df7bd8fa94b4e5b06eb700baa6193a7f4224a95d543dd748be8872e9fef78a3a`；report SHA256 `3e6131acea5cbca7c8d2a2c26abee86e0db077d94d1994338ebd4a20a3725f4e`。
- v4：`minds-acoustic-numeric-v4/jev-minds-acoustic-numeric-v4-collected/evaluation/report.json`。
  manifest `6328a76145324f918f8bdbc63609c343624025fca5576d055ff3f587b431db34`；report SHA256 `ad1099e212d670650267daee678f060dad7b0463468db657a5ec1c77040582dc`。
- 两次驱动exit0，独立回执为`numeric_diagnostic_collected_not_full_model_admission`；真实容器terminal、镜像/挂载/设备、transport文件hash均已核验。
- v1在导入阶段失败：最小封包漏带包入口，意外被镜像安装包遮蔽。未进入NPU计算；原容器、失败输出和导入日志保留。v2仅封包未派发；v3补齐包入口的传递依赖后执行成功。
- `tests/test_context_acoustic_numeric_probe.py`三项CPU测试通过：布局可逆、单个BF16扰动及坐标检测、shape/dtype/NaN/Inf拒绝。这些测试不代替NPU数值结果。

## 解释与下一步

在已测输入和批形状下，不支持卷积、GELU或conv_out批形状/槽位是差异来源的假设。不能据9条局部测试排除所有声学形状、真实多录音邻居组合或设备执行上下文；也不能把局部exact写成全模型exact。

下一诊断应在真实HF与vLLM路径捕获同一输入的声学位置编码、每个encoder block、ln_post、最终projector及语言层入口。比较单录音与真实多录音批次，固定策略与输入，找到首个差异层；若声学全程exact，再定位语言调度及head。不得先放宽概率容差、删除MINDS样本或跳过14候选宽度验收。

源全量训练包已生成并重验完整文件清单：`acoustic-kv-formal-v1-prepared/PREPARED.json`状态`macro_formal_prepared_only`，manifest `33c2567247debabc17658d93d835077296d7d2838a172f492eb1849770c1db17`；training及finite_npu dispatch仍为false。数值、NPU稀疏恢复与launch集成仍是正式训练前置条件。

## 可复用入口

诊断：`tools/reports/context_acoustic_numeric_probe.py`。
封包/校验/派发：`tools/reports/context_acoustic_numeric_operations.py`。
共享模板：`configs/portable/operations/context-acoustic-numeric-probe.json`，宏对象`ACOUSTIC_NUMERIC_PROBE`。
本机绑定：`.local/config/minds-acoustic-numeric-v4.json`，仅用于追溯已执行版本，不能再次launch。

新实验需新run/tag/root/输出与封包；保持原缓存回执及选择规则，运行prepare→validate→launch。所有主机路径来自本地宏。源码后续修订不应覆盖已封包版本；历史比较依据冻结源码和原始报告。
