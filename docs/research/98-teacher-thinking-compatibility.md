# 教师接口兼容性对照与首个标注响应

doc97的长输入请求HTTP400不能证明上下文超限。本轮做三个新的有限请求，未重发
任何相同请求：最小disabled、最小enabled、原长输入enabled。两组对照分别核验
实际request.json，唯一差异均为thinking字段；模型、stream、输出4096上限、消息
和其他服务端字段在每对中不变。

| 请求 | thinking | HTTP | 输入token | 输出token | 秒 |
|---|---|---:|---:|---:|---:|
| 最小JSON响应控制 | disabled | 400 | 未报告 | 未报告 | 1.88 |
| 相同最小控制 | enabled | 200 | 53 | 284 | 4.50 |
| doc97最长JSON标注输入 | disabled | 400 | 未报告 | 未报告 | 3.26 |
| 相同长输入 | enabled | 200 | 24,593 | 1,856 | 34.84 |

两个disabled请求返回同一类内部MaaS 400001错误。最小enabled准确返回
`{"ok":true}`；长输入enabled结束原因stop，8个谓词均显式返回status/rationale，
通过既有响应验证器。输出token中1,439为reasoning token，不能将4096全部视作最终
JSON预算。这是离线教师的推理模式，不改变actor一次Span输出、无文本CoT的契约。

长输入参考为出租车接机请求；7个音乐/播客/电台/关系/体育谓词判negative，time判
positive。逐条阅读理由没有发现明显偏离当前定义；时间短语的具体数值解释不属于
本任务目标，也不随教师理由作为槽位值标签准入。该单条不是正负均衡校准、不证明
共享规则与字面规则对模型等价，更不是独立语义复核或全量监督。

两组单因素观察支持当前接口使用enabled模式；不能据此推断所有提供商永远不支持
disabled，也不能完全排除时变服务因素。完整响应、请求与原错误均保留；公共证据
见[对照凭证](teacher-thinking-compatibility-v1.json)。本轮没有读取或输出凭据值。

另纠正文档错误：当前executor SHA为`c9f9bbd3…`，与历史固定版本相同。上轮文字说
“不同”与实际记录矛盾，本轮按源码及两次intent hash核实，已修订doc97与AGENTS。
这不影响已记录请求体、HTTP结果或共享规则测量。

传输候选登记于`configs/reviews/slurp-semantic-teacher-profile-candidate-v1.json`：
glm-5.3-flash、thinking enabled、stream、输出4096、总截止180秒、无自动重试。
不复用Omni研究长提示或其131072输出配置。该候选绑定当前executor、工作包和
实际对照证据，bulk_dispatch_allowed=false，semantic_admission=0。

下一步检查全16面板的实际token/输出额度、按极性及歧义校准和字面/共享表示对照，
明确全量预算、独立审核与分歧裁决，才派发规模化标注。当前仅证明最长JSON样本
能够被接受；最长JSON不一定等于最高token输入，不能据单条把全量上下文标绿。
