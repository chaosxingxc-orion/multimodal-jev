# Audio2Tool-Jev独立决策推理准备

评测总体保持30733条发布记录×457决策，不以缺失参数标注跳过调用。当前完成
分片账本和专用引擎容量入口，尚未完成全量Jev runner、独立actor验收或NPU运行。
它们不能用来声明任务性能、Jev收益或RL训练可用。

## 专用引擎边界

`audio2tool_jev_engine.py`从已验收的完整context结果中读取12个真实processor案例：
三种输入模式×三种决策，以及七段音频×三种决策。输入边界选择只依赖上下文预算，
不读取targets或缺失监督列表。`prepare_request`先验证完整输入、bank和决策，再从
同一字节缓冲区认证并解码音频；保持原始多轮/前景输入与全部工具定义。

`probe_audio2tool_jev_engine.py --preflight-only`在固定镜像只构造引擎输入，核对
prompt哈希、完整音频收据/波形哈希、帧数与旧真实HF/vLLM证据逐项一致。它不加载
模型，也不重新执行音频feature processor，不能代替旧processor验收或NPU容量。
预登记CPU输入检查预算2 CPU/16 GiB/1200秒、无网络设备，首次派发身份另存。

非preflight路径保留相同32768上下文、TP2、BF16、max_num_seqs=1、7音频、关闭
prefix cache的固定引擎配置；全模型资产前后验哈希，12次完整prefill各生成一token，
要求实际展开token哈希等于旧Jev processor结果（canonical JSON摘要）。它不沿用
native结果使用的另一种JSON空白编码摘要。输出EOS或length也必须恰好一token。
实际NPU派发仍需资源协调、固定预算/源码/精确容器与独立验收，当前没有启动。

## 有界文件账本

`audio2tool_jev_ledger.DecisionLedger`固定完整请求及bank顺序，默认4096决策一个
JSONL分片，全量约3429分片，另加manifest和complete两份文件。运行时只保持一个
分片写句柄；目录fsync和最终manifest读写有短暂的独立句柄，不宣称进程总共只有
一个FD。同一请求的音频收据只写一次context，各决策attempt引用context摘要。

生成调用前先持久写attempt，生成后持久写terminal，均flush+fsync。所有事件带
递增序号和前序摘要；各分片落盘后保存文件摘要，完成记录只有全部实际调用成功
结束才能写出。receipt/输入/原始输出不修复，空解码文本和非法JSON仍保留。
完整性哈希不是签名，不能抵御把所有文件一致重写，更不能证明引擎真的执行了；
仍须未来独立actor验收绑定精确容器、源码、音频及实际输入/输出token解码。

跨分片继续同一请求时不得改变音频收据。一个writer只允许一个待完成attempt，
必须按原始请求及决策顺序执行。磁盘写入或fsync异常会使writer失效，即使数据可能
已经落盘也不自动重写。真实推理异常只记录已尝试项并停止，不伪造后续失败记录。
旧目录不能重开、resume或重放。每事件fsync的真实Ceph开销尚未测量，不能凭文件
数量减少声称吞吐已提升。

只读`records`以调用者独立固定的完整请求/bank核验manifest、分片集合、链条、
context/attempt/terminal顺序、输入绑定、输出合同与最终计数。必须耗尽迭代器才
算完成，不能用部分yield的记录宣布验收。`export_predictions`在账本外排他创建
逐行原始scorer接口；中途失败保留部分文件且不返回成功，不能将其当完整评测。
导出声明`actor_execution_verified=false`，未来必须接入完整生成与独立来源验收。

相关76项测试通过（ledger、engine、renderer、scorer）：包含跨请求/跨分片轮转、
同长度token错序、实际/probe输出边界、非法决策先于音频读取拒绝、缺失/额外/篡改
账本、重哈希但改变决策顺序、上下文绑定、错误计数、未结束不能完成、原始文本和
真实/probe错误分类。`/proc/self/fd`检查确认每次轮转后仅一个分片写句柄；终态fsync
故障反例确认不允许重试同一结果。测试数据明确为fixture，不是完整模型推理证据。

## 首次真实CPU引擎输入预检查已验收

容器`27e52b3cdd4d24920e7ddb9e2007f5ab9d1335e587a20c67f06587242bc14ef1`
使用源码`22e0157da6ae629cdef832b54391134ef89e5d19`、固定镜像及上述CPU预算，退出0。
主体7.96秒，12项请求的prompt与完整音频收据均匹配此前实际processor边界。
独立验收重新inspect精确容器，核对命令、挂载/环境、资源、源码快照与Git一致性、
输入依赖、identity、12份独立结果与应用终态。见
[输入预检查验收](audio2tool-jev-engine-preflight-v1-verified.json)。
模型未加载、NPU未使用；全量生成runner及独立actor验收仍待完成。

## 完整生成入口已实现，尚未派发

`infer_audio2tool_jev.py`配合`audio2tool_jev_inference.py`固定全部30733×457独立
调用。`population`重新绑定每条native/Jev请求、全部音频预算与2742个真实tokenizer
profile，并拒绝缺失布局/决策、改变输入或超出32768窗口。`--preflight-only`仅执行
完整清单检查，不读取音频或加载模型；该宿主检查已通过，结果另存。

实际执行保持greedy、seed17、1024输出、TP2/BF16/max_num_seqs=1及关闭prefix cache。
每条音频只从源字节认证加载一次并在457次独立调用间保留同一波形对象；每次仍独立
执行HF完整feature处理及vLLM生成，无KV/feature缓存新机制。逐项检查完整帧数、
feature有限性、profile输入长度、实际vLLM token IDs与本次HF展开逐位一致。
未知监督对actor不可见；重复发布输入独立执行。原始非法JSON、空解码文本和length
截断全部落盘，实际引擎失败只保存该attempt后立即停止。模型文件实际前后验SHA，
完整成功才耗尽分片验收迭代器并导出scorer接口，不写伪造的剩余预测。

`audio2tool_jev_capacity.require_capacity`只接受协调者实际捕获并预先固定SHA的
精确NPU6/7成功容量容器：固定镜像、16 CPU/128 GiB/8 GiB shm、一小时加30秒
宽限、完整命令、只读源码挂载、环境/设备、不可变source、版本/模型/processor
依赖、12份真实一token边界结果及成功应用终态。probe源码必须匹配此前已验收
CPU preflight的原始快照；不同源码须另登记验证，不能悄悄替换。该验收不把CPU
preflight当作NPU证据，调用者提供的inspection本身也不替代实际Docker观察。

新增生成循环与容量检查连同账本/引擎共56项测试通过。生成测试使用模拟HF/engine，
验证每决策单独调用、音频对象复用、帧数/非有限值/token错序拒绝、原文与截断保留、
实际异常后不生成假剩余行。容量测试使用明确的合成容器及mock边界证据，覆盖
运行中/失败/OOM/错设备/错命令/错资源/preflight/缺案例/错token/错模型/改源码/改SHA。
它们不是NPU或完整actor执行证据。全量Jev正式launch plan、真实吞吐预算、独立
actor来源验收尚未完成；当前只有输入清单preflight，不能给出模型性能或RL收益。
