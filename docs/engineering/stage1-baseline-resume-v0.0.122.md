# Stage1 基础配置恢复

用户授权回退已验证基础配置并继续实验。使用rc6原始冻结source/run.py字节不变，每卡有效batch64，micro8，累积8，DP4物理4–7，global256，BF16，纯InfoNCE。完整40513条clean TRAIN两轮，终点318。新根目录隔离输出，恢复点step32；停止时未提交的33–40重算，不计为额外训练轮次。

通过原始发布链校验、第32步commit及四rank分片hash、FP32参数/Adam独立恢复、BF16 export从master分片重构后启动。恢复入口只改变根路径、start_update、resume_root和resume pins；objective/schedule/LR/模型契约不变。4个恢复契约CPU测试通过。

现场四rank均exact_masters=true、exact_adam=true、completed_updates=32，四rank完成update33，容器仍running。恢复后的运行继续到318；不是完整训练完成或效果验收。

共享入口tools/reports/context_infonce_formal_resume.py，配置模板configs/portable/operations/context-infonce-formal-resume.json，本地绑定.local/config/context-infonce-prefix-resume-v0.0.122.json。训练完成后同一后台控制器collect并审核恢复段；完整曲线应将原运行1–32与续训33–318拼接，旧运行33–40仅保留重放诊断，不能重复计入。

证据：/srv/multimodal-jev-xingchao/operations/context-infonce-prefix-resume-v0.0.122/startup-summary.json
