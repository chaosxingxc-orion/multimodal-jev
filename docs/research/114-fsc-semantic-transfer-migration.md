# FSC语义迁移协议收口

当前迁移入口为`configs/tasks/fsc/transfer-protocol-v3.json`。保留原生31种
action/object/location组合为Final分类目标，另定义31个I_present语义谓词。
二者不宣称等价，不能将原生单标签投影为全部语义问题的正负真值。
旧v2成员重编码协议保留历史身份。

本轮读取TRAIN全部248种原始参考表达，绑定23,132录音、77位说话者及31原生类；
同文本未发现冲突原生组合。全部248×31=7,688个语义判决仍为未评估，准入0。
没有打开TEST文件；项目历史FSC接触仍须披露，不能称未接触过的全新任务。

必须保留以下语义边界：location=none要求匹配请求未指定房间，不是任意房间的
通配符或缺失标注；多请求分别匹配，不强制整句话one-hot。仅有Play/Resume等
表达时不能借原生标签补音乐领域；最大音量不等于相对调高，静音不等于保留声音
的调低；打开语言设置不等于执行语言切换。lamp与一般lights采用显式配置类别，
不暗加分类层级闭包。17个边界例仅作待审核问题，不是gold。

审核输入编译器`tools/jev/build_fsc_semantic_review_packet.py`为每个参考表达固定
8个面板，前7组4问、末组3问，共1,984请求。语义字段显式保留对象、位置、局部
请求约束，原生标签、路径和provenance不进入教师wire。该包仅为离线审核输入，
未调用API，不是actor输入编译完成。

迁移比较冻结源域Stage1及RL权重。Direct与Q_input每次共享完整native31菜单、
当前面板规约；每条音频都跑固定8面板，分别报告及等权平均，不挑最佳面板或投票。
Q_credit是源域训练条件，不在FSC上做奖励学习。目标测试结果不选择模板、检查点
或算法。原生Final保持固定完整人口；语义未知仅mask语义指标，不缩原生分母。
FSC action/object/location属性不代替S_present，因此本协议尚不提供槽位语义迁移证据。

产物位于共享目录：

- `/mnt/cephfs/xingchao/datasets/jev/fsc-semantic-transfer-train-inventory-v1/`
- `/mnt/cephfs/xingchao/datasets/jev/fsc-semantic-review-packet-v1/`

[独立结构检查](fsc-semantic-transfer-design-check-v1.json)重新读取原始TRAIN CSV，
核对全部录音/说话者/行身份、248参考、定义字段投影、完整日程及协议hash；
3项编译器测试通过。它不是独立语义裁决或音频复听。

主task与supervision plan已接v3。七类依赖设计均有当前候选，但语义准入、生产renderer、
上下文和后端验收仍未完成；training/evaluation dispatch均为false。本轮没有训练。

2026-11-13计分补充：v3 metrics引用SLURP evaluation-v2的共享语义算术，原生macro
F1仍固定31类；I_present按正例支持定义等权，不对4问和3问面板macro作等权替代。
负例非法输出另报失败率，零准入F1为null；没有S_present，不报告槽位迁移指标。
这是计分协议补充，不修改原始TRAIN审核包或历史独立结构检查hash，也不构成scorer
验收。新协议hash见[算术复核](semantic-metric-contract-check-v2.json)。

## 原生与语义读出绑定（2026-11-13）

上述算术复核绑定的是当时协议，后续协议修订不回写历史hash。当前v3新增
`readout_design`指向[evaluation-readout-layout-candidate-v1.json](../../configs/tasks/fsc/evaluation-readout-layout-candidate-v1.json)，
生成器为`tools/jev/build_fsc_semantic_readout.py`。新设计绑定当前协议、原生类别身份
文件及语义定义文件hash，不读取TRAIN/TEST样本或产生标签。

原生31个合法三元组使用C00–C30及A–Z/a–e输出符号；语义谓词独立使用I00–I30，
每个布尔Span内A=false/B=true。相同字符在不同Span具有各自绑定，不能沿用SLURP
原生类别语义，也不能从FSC原生Final补出语义真值。旧native定义文件仅提供ID与三元组，
其I_class文字没有复制成独立语义充分必要条件；完整native描述仍须renderer明确供给。

前七面板各4问、末面板3问。Direct的Final位置0；Q_input的过程位置0–3（末面板0–2），
Final位置4（末面板3）。两种目标输入均共享完整native菜单和面板语义知识，Q_credit
仅标识源训练条件，不增加目标训练或第三种目标输入。规范评测每音频/检查点16请求。

同时声明两个独立诊断布局：候选展示顺序变化时符号绑定不变；符号重绑时展示位置不变。
这两个布局尚未加入主评测预算，不能自动扩充调用。它们只检查绑定等变性，不证明
定义理解或迁移收益。native31与boolean符号映射需用实际processor验证token身份。

25项相关测试通过；其中目标映射穷举8面板×3布局、Direct及Q_input所有布尔向量、
31个Final符号共11,904种合法输出，另拒绝重复谓词、重复原生命令、缺面板、额外目标臂
及native语义投影。检查的是设计映射与现有参考解码函数，不是模型前向、真实分布式
推理或生产scorer验收。当前语义准入仍0，冻结测试人口/标签、renderer与后端仍待完成。
