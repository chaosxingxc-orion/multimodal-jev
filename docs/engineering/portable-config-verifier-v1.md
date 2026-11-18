# 固定配置等价校验入口

`tools.verify_portable_configs`将共享宏模板与已固定SHA256的原始JSON配置作严格值比较。
复用统一宏解析器，不运行driver、评分器、模型或SSH。它不会自动接受差异或从当前文件
生成一个新的“期望hash”以绕过来源变化。

将[配置示例](../../configs/config-verification-macros.example.json)的CONFIG_VERIFICATION_ENTRIES
合并到本地宏文件，并保留各被测模板需要的本地宏。每项显式指定唯一id、template、
source及预先验证的source_sha256。运行：

```bash
PYTHONPATH=src:tools/jev:. python -m tools.verify_portable_configs \
  --config configs/portable/operations/verify-configs.json \
  --local .local/config/macros.json
```

工具只在stdout输出JSON报告，不写文件。调用方如需归档，应使用现有本地持久存储
发布流程。报告只包含来源路径/hash与宏名，不展开本地宏值。示例单列GRPO，列表可
登记多个任务。空列表、重复ID、来源hash不符、验证过程中输入变化均拒绝。

比较忽略JSON对象键顺序和空白，保留列表顺序及值类型；false与0、整数1与浮点1.0
视为不同。NaN/Infinity拒绝。该通用入口不允许例外字段；actor派发false的历史特殊
对比仍使用原`tools.verify_portable_training`，不修改其已经发布的规则。

22项相关测试通过，覆盖类型/列表差异、来源变化、绑定变化、只读重复执行及原宏解析。
固定脚本已对GRPO、RLOO、FSC迁移、SLURP clean TEST、MINDS和FSC参考比较六份真实
冻结配置全部验证通过。证据见[版本记录](../../configs/releases/portable-config-verifier-v1/validation.json)。

本检查只证明配置值与原件一致，不验证配置引用的模型/音频/预测，不证明运行许可、
跨主机兼容或任务收益。此次没有修改冻结源配置、替换队列或派发任务。
