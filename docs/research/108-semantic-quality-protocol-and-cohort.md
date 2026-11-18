# 全量语义监督的质量审核协议与固定队列

## 人工盲审交付（2026-11-13）

完整材料位于`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/semantic-quality-human-review-v1`。
128来源、16面板、2048请求、15360空白判决；manifest SHA256为
`1cba1dc5971f4349d9ee458be1fced31850ed1793cf7e3b47ef07db03dadf2d8`。
包含完整展开规则JSON/Markdown、sources.json、schedule.json、answers.csv、
reviewer.json及中文README，不包含native标签、作者参考、教师答案、source_id或录音绑定。
全部规则展开、36固定文件pin、来源/日程与空白坐标独立核验，见
[核验记录](semantic-quality-human-review-check-v1.json)。

审核者填写答案表与身份表后，通过prepare_human_semantic_review.py的collect子命令
传入--packet、--answers、--identity、--output导入新目录。空白/缺行/重复/未知坐标、
缺理由及改动规则会被拒绝。身份表记录方法、起止时间、先前暴露及盲审声明；
声明本身不能认证身份或独立性。未审阅请留空，不能用unavailable代替未完成。

导入格式jev-human-semantic-review-pass-v1与模型传输分开，保留原CSV及身份hash，
不伪造Chat Completions响应。现有质量scorer可消费为blind_reviewer臂，并逐次
重建原始答案与完整输入；仍需proposer、裁决记录及独立性判断。人工完整率表示
判决覆盖，不称API成功率。所有工具仍输出semantic_admission=0。

19项相关测试通过，含全128组/15360坐标合成CSV→导入→评分，验证缺行、篡改与
空白不能被当作已完成。实际材料答案全空、审核者未定；已询问用户是否有人参与，
未向第三方发送。材料准备不解决教师语义准入、全量标注或NPU验收；主目标仍未完成。
冻结包保留生成时实现hash，后续导入验证增强不回写冻结包。

本轮将全量监督前的审核范围、身份与通过标准具体化，未修改本体或训练实现。
入口为[质量协议](../../configs/reviews/slurp-semantic-quality-protocol-v1.json)。
其状态是预登记设计，审核者身份、API预算及执行器仍未验收，不触发批量派发。

2026-11-13评分实现补充：`tools/jev/score_semantic_quality_audit.py`按固定全128组
和15,360坐标接收两份完整终态审核，失败保分母，缺审核/裁决不计正确；一致判决也须
裁决才能计入分子。每族正负分别按原协议做10,000次来源组bootstrap，零分母显式
不可估计，报告零错误退化及逐类型覆盖。10项评分/队列测试通过，包括独立标量复算、
失败/截断保分母、原始响应与规范化结果不符拒绝、输入文件hash及重复身份拒绝。

pass使用`jev-semantic-quality-pass-v1`；input_manifest_path及sha绑定
`jev-semantic-quality-input-manifest-v1`，其中requests逐条提供request_id/path/sha256。
成功请求evidence必须指向带唯一stop choice的原始Chat Completions响应，解析内容
必须与response字段一致。失败请求仅提供失败说明及原始证据，不消费部分答案。
输入hash证明文件身份，不证明盲审语义或传输真实性；不同reviewer_id不证明独立性。
该工具永远输出semantic_admission=0及training_dispatch_allowed=false，尚无真实
双审质量结果，也不替代正式监督发布或执行器验收。

2026-11-13完整输入与收集链路补充：prepare_semantic_quality_audit.py新增
prepare_inputs及对应CLI，按冻结队列重建全部source×panel覆盖、template/request身份，
逐条只输出system/user两条消息；user字段与原规则/问题/参考文本精确一致。
score_semantic_quality_audit.py现在重新核对实际内容，不能通过给泄漏标签或改写文本
重新计算hash绕过检查。system提示固定hash，但其语义及审核者独立性仍须另审。

候选请求包位于
`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/semantic-quality-inputs-envelope-v1`，
128来源、16面板、2048请求、15360判决；manifest SHA256
`32d72acc36d082863f5bf7052cfdadb8491deec70b5ed24f0c530593abc42e0f`。
所有user字段、系统提示、source×panel日程及文件hash均独立重建，见
[核验记录](semantic-quality-inputs-envelope-check-v1.json)。API_sent=false；角色登记
blind_reviewer仅表示输出格式，不代表已选定独立审核者。候选系统提示含通用JSON外层
结构说明，六例通路对照与全128组请求包严格分开，不因前者成功自动派发后者。

新增collect_semantic_quality_pass.py读取冻结input manifest和完整执行index，核对
每条terminal/request/result/response及其hash，实际发送messages必须等于冻结输入。
index使用jev-semantic-quality-execution-index-v1，记录input_manifest_sha256、
reviewer_id、model_identity、requested_model及requests；每条request_id绑定上述
四种证据的path/sha256。HTTP失败可不提供response；成功传输必须提供完整原始响应。
截断、裸数组、格式错误、进程/API失败均保留失败记录；未完成或缺失日程拒绝发布。
只有完整核验后原子发布pass文件，仍固定semantic_admission=0，不发送/重试API。

16项质量队列/评分/收集测试通过：包括完整输入投影、重算hash后的标签泄漏、缺面板、
不完整执行index及多类失败保分母。另以完整128组/15360坐标的合成临时夹具贯通文件输入→双审评分，验证缺失审核保分母；这些不是实际双审质量结果。
冻结数据包保留生成时实现hash；后来CLI扩展不回写已冻结包，输入内容验证独立于CLI。

## 固定代表性队列

使用shared-v3全部9192来源的显式source_group绑定，排除当前意图/槽位评估及历史
定义参考涉及的1205来源，整组排除后7976组可选。按固定seed与组ID的hash排序取
128组，实际128来源；保留原16面板及120问题，2048请求/15360判决每轮。
选择不使用类别、模型输出或参考文本内容。未声称这些来源从未被任何人阅读。
全部参考文本仍只用于离线标注，actor继续直接输入音频。

包位于`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/semantic-quality-audit-cohort-v1/`，
report SHA256 `1716c2c5a610bddbf2f7330d5b8b78cca9f2271fdd381e6df06e7e3bb6cf0f4f`。
工具为[prepare_semantic_quality_audit.py](../../tools/jev/prepare_semantic_quality_audit.py)。
三项测试验证组闭包、顺序不变、无法满足规模/身份歧义时拒绝；另外独立从全量原始包
重建排除集合、hash排序、来源绑定及全部日程，模板字节未变，见
[核验记录](semantic-quality-audit-cohort-check-v1.json)。

## 审核与发布

先验通过标准：每轮完整请求比例至少98%；不能用总体比例掩盖某面板系统失败。
语义质量按I/S及预测正/负分层，对最终审核判决的正确率报告来源组bootstrap下界，
候选门槛95%；每层至少30来源组。随机队列不能保证稀有类型覆盖，缺证据时单独冻结
正负覆盖补充集，不把定向补样混入总体准确率或比例估计。bootstrap零观测错误时可能
退化，不能单凭这个数字认证标签；逐类型证据、审核方法与系统错误排查同为准入条件。

审核者先独立看相同规约与输入，不看教师输出、理由、native标签或作者参考。
所有分歧/未决保留原始判决并裁决；质量分子中的一致项也必须检查，不能将一致即正确。
同一作者重审或同模型重跑不叫独立审核；不同模型意见也不自动成为人工gold。
审核身份与方法尚未选定，当前0语义准入。真正输入歧义保留mask；定义矛盾阻塞对应
定义发布。记录参考文本证据与逐录音听审区别，不自动声称所有录音被听过。

全量主人口仍为40513录音/9192来源，128组只做质量控制，不能替代训练。
单轮全量147072请求，在每请求4096输出候选上限下最多602406912输出token；
128组审核单轮上限8388608。输入费用须基于实际面板usage另估，双审核与裁决另计，
这些数字不是费用承诺或批量运行授权确认。首先收尾既有89请求槽位校准，定位失败与
分歧，再选择可执行审核方案并冻结总资源预算；不自动扩大请求或重试。

任务仍未发布；需要全量监督、固定actor菜单与CE/reward/评测协议共同收口后，才进入
工程验收与算法发布。基线复现、RL持续提升和FSC冻结迁移仍是未完成的主目标。

槽位校准本轮在距上次606秒后观察：PID3831451实际cmdline仍匹配原runner，
71/89请求终态、总终态未写，未评分或重启。观察时间1794525361.849376，
记录`.local/research/slot-teacher-calibration-v1/observation-v2.json`；
下一普通检查至少1794525961.849376。这是观察时状态，不推断后续完成。

候选系统提示六例已终态：格式6/6，但7参考仅2一致、5将明确缺席判未决；见doc115。该输入包的结构核验仍有效，prompt/model语义准入失败，不能直接执行2048请求。未改原始语义规则或将未决重映射为负例。
