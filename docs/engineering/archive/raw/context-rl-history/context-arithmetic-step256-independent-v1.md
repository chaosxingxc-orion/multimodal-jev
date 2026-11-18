# 正式GRPO第256步保存产物独立审核

原容器e564df04f946f853、固定镜像52ba94e1，未停止/恢复/重建任务。
从live-v42已确认提交的checkpoint-0256读取有限文件，未额外查询最新进度。
沿用第64步的只读收集和CPU审核逻辑，独立输出step256产物，原证据不覆盖。

结果通过：

- 提交记录中的四rank checkpoint和执行文件hash匹配，契约及更新次数均为256。
- 四卡FP32 master/Adam状态满足原checkpoint协议。
- 全部128参数、6,878,209元素，从四卡master按本固定FSDP布局重组并转换BF16，
  与assembled execution逐元素相等；execution与published policy也相等。
- 原生adapter与联合策略一致，decision marker保持Stage1固定值。
- 四rank更新记录与已发布策略c411c683639f匹配，激活成功；每rank记录的16项行为一致性
  检查均通过且threshold flips=0。这里是审核已保存记录，不是本轮重新推理。

依据：context-arithmetic-step256-independent-v1.json、
context-arithmetic-step256-master-export-v1.json，包含收集文件和脚本hash。
独立副本位于.local/decision-objective-research/context-arithmetic-step256-independent-v1/collected。
本轮364项原基线身份再次核验通过；没有改运行包、算法、日程或队列。

第256步rank0 total104.49秒，preparation75.87秒、actor24.76秒；128/128输入cache命中。
四rank allocator peak约38.04–40.72GB（十进制），不能当作设备全部HBM使用量或瞬时占用。
详见context-arithmetic-step256-runtime-summary-v1.json。本步含checkpoint，不作稳态吞吐统计。
上述记录与前缀分析一致：单纯提高actor microbatch不能消除rollout准备的大头成本。

本审核仅证明该真实保存点的产物完整性和master/执行/发布一致性。
它不证明新进程NPU恢复、完整634步训练通过、selection/TEST成绩或RL提升。
完整终态审核和原11点选择集队列继续，不能用中途checkpoint审核替代。
