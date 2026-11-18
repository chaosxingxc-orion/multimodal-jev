# GRPO/RLOO实验入口宏配置

共享模板补齐原`tools.reports.run_context_arithmetic_experiment`的两份入口配置。
它们绑定完整训练发布包、共同Stage1源、实验协议及selection模板；算法与超参数仍由
已发布包决定，不因改文件名或宏而切换算法。

使用[本地宏示例](../../configs/rl-experiment-macros.example.json)填写本机资产路径及
真实PACKAGE清单SHA256，然后运行：

```bash
PYTHONPATH=src:tools/jev:. python -m tools.run_operations \
  --config configs/portable/operations/prepare-rl-experiments.json \
  --local .local/config/macros.json
```

生成目录为RL_EXPERIMENT_CONFIG_OUTPUT，含grpo.json、rloo.json及各自provenance。
生成工具不调用原driver、不派发训练、不检查未来远程执行的可行性。PACKAGE清单hash
与正式RELEASE发布ID是不同身份，不能互换。当前队列继续消费原冻结spec，不切换入口。

实际两份arithmetic-v2配置解析值与原件完全相等，共享源绑定也一致；源字节不变，
真实本地持久盘检查及重入通过。验收记录见
[validation.json](../../configs/releases/rl-experiment-configs-v1/validation.json)。
最初临时验证命令的宏名正则漏了数字字符，解析RL_STAGE1字段时报错；修正验收命令后
通过，未修改生产宏解析器或原训练配置。

此项只证明配置值等价，未重验整个训练包，不替代目标主机有限验收。原driver的设备、
镜像和remote prefix约束仍适用，跨团队部署须以新版本资产及运行证明准入。
已存在的actor/service宏模板继续负责底层执行配置，本入口负责绑定已发布资产。

相关入口：[迁移/TEST](transfer-portable-configs-v1.md)、
[FSC参考比较](fsc-comparison-portable-config-v1.md)、[算法登记](../../configs/releases/rl-baselines-v2/README.md)。
