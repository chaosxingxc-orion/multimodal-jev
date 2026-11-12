# FSC单次Span配置迁移契约

2026-11-12 UTC。`configs/tasks/fsc-span-transfer-v1.json`固定FSC原生31个合法action/object/location联合命令，组件支持为6/14/4。候选集合只从本地官方TRAIN提取，校验其CSV哈希；配置没有参考语句、类别频率或模型反馈。本项目历史上已有FSC实验，不能宣称从未接触目标域；此次没有利用其评测结果挑选模板。

配置提供两个模板：direct只输出Final；structured一次输出action、object、location、Final。两者Final问题及完整候选定义相同，所有问题预先给定，输入只有音频、任务定义及thinking requirements。`none`保留为合法对象/位置类别，未增加unknown或缺候选任务。输出ID通过当前模板绑定回原生语义，不沿用SLURP分类ID。

`fsc_span_transfer.py`支持native、完整候选permutation以及rebind。rebind同时移动语义在候选列表中的位置并使用新ID，避免仅改ID但原生A/B决策符号仍对应原语义的弱检查。所有变换只由配置和固定seed决定，不依赖gold；目标命令不会被用于筛选候选。评分单独消费gold，区分Final正确、Final三个组件正确、过程组件正确及过程/Final兼容，不用组件组合修复模型Final。

5项本地测试通过，覆盖全部31命令的三种绑定、合法none、非法输出、组件正确但Final错误等情况。固定镜像真实CPU processor运行18请求：1/10/30秒合成静音×direct/structured×三个变体，最长3352tokens，候选均为独立单token符号，D/Q声学特征一致。实际容器退出0及来源哈希已核对，见`fsc-span-transfer-processor-cpu-v1-verified.json`。这不是实际FSC音频/NPU概率验收，也不是跨域收益证据。

主实验仍为SLURP D/Q/P RL。FSC只用于冻结源模型检查点的配置迁移：固定模板比较起点与预定后期检查点，分开报告模型更新收益和模板结构收益。不使用FSC训练/验证/测试标签选模板或checkpoint；后续真实FSC评测须独立核验官方划分、音频身份和两后端完整概率。真实音频超出上下文时不得静默截断。模板版本或解释若改变，应记录新版本并披露原因，不能根据目标测试结果不断修词。
