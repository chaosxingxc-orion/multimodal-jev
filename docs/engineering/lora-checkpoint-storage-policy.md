# LoRA训练的checkpoint范围

用户约束：LoRA训练不得反复保存主模型权重。冻结基座保存在共享models目录，实验通过
配置宏提供路径，并固定base_id/文件hash；每个step不得复制或合并导出整份基座。

允许保存的内容：

- LoRA A/B参数及配置。
- Jev可训练Transformer head；它是决策能力的一部分，不能只保存LoRA而遗漏head。
- 当前模型新增的必要特殊token embedding行；不保存整个embedding矩阵。
- 上述可训练参数的FP32恢复值、Adam状态、step及恢复所需配置/随机状态。
- 数据、源模型、代码与执行配置的身份引用和评测结果。

当前实际导出execution.pt检查为13,803,531字节：112个LoRA张量、16个head张量、1行
marker，无冻结backbone权重。原trainer仅收集requires_grad参数传入checkpoint capture，
export_checkpoint_policy只替换LoRA/head并保留单行marker，native adapter只导出LoRA。
完整四rank step634实际checkpoint检查见`lora-checkpoint-scope-observed-v1.json`。

新增`context_lora_checkpoint_scope.validate_trainable_scope`在新版dual trainer启动时
拒绝误解冻的基座层、完整embedding或其他不在LoRA/head白名单内的保存参数，允许FSDP
无本地所有权的空分片。每rank生成checkpoint-scope.json，不能仅凭requires_grad声称
保存范围正确。当前token行冻结，训练checkpoint不保存它的优化器状态。

FP32恢复分片、BF16执行导出及vLLM原生adapter是不同用途的适配器/head产物，不是完整
主模型。它们仍可能有冗余；持久保留策略应区分恢复断点、预定评测点和临时激活文件，
在消费者完成后通过固定脚本回收。当前冻结运行按原协议收尾，不原位删除评测/恢复
仍依赖的产物。该规定不取消本地checkpoint的fsync。
