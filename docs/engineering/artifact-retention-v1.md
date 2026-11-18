# 已封闭实验的产物保留清理

tools.retain_experiment_artifacts通过固定JSON清单处理已封闭实验。复用统一宏解析、
本地存储检查、持久提交和cache_cleanup.delete_one的无符号链接逐级打开/删除实现。
不扫描目录推测文件用途，不调用模型判断结论，不递归删除目录。

从configs/artifact-retention-macros.example.json取所需宏加入本地绑定，在仓库根执行：

```bash
PYTHONPATH=src:. python -m tools.retain_experiment_artifacts \
  --config configs/portable/operations/artifact-retention.json \
  --local .local/config/macros.json
```

默认生成清单；在独立输出版本配置下加--apply执行。apply也先持久化plan，再核验依赖、
归档及文件身份后逐文件删除，记录intent和deleted回执。无自动重试；部分失败保留现场。
重复成功执行只核验记录，若被删除路径重新出现则拒绝再次删除。

上游必须提供SHA256绑定的inventory，结构如下（示意，digest需真实填写）：

```json
{
  "schema": "closed-experiment-inventory-v1",
  "closed": true,
  "evaluation_complete": true,
  "terminal": "terminal.json",
  "terminal_sha256": "<sha256>",
  "selected_checkpoint_steps": [512],
  "files": [
    {"path": "config.json", "sha256": "<sha256>", "role": "config", "step": null},
    {"path": "result.json", "sha256": "<sha256>", "role": "final", "step": null},
    {"path": "conclusion.md", "sha256": "<sha256>", "role": "conclusion", "step": null},
    {"path": "checkpoint-0512.pt", "sha256": "<sha256>", "role": "checkpoint", "step": 512},
    {"path": "checkpoint-0634.pt", "sha256": "<sha256>", "role": "checkpoint", "step": 634},
    {"path": "intermediate-analysis.json", "sha256": "<sha256>", "role": "intermediate", "step": null}
  ],
  "conclusion_archives": {"conclusion.md": "/absolute/path/outside/experiment/conclusion.md"}
}
```

这是上游产物清单协议，实际生成器应从同一宏文件读取归档根，不能把示意路径复制为
共享部署配置。terminal.json须有exit_code=0。清单同时声明closed和evaluation_complete；
清理器核验文件身份及这些声明，但无法证明上游声明真实、结论科学正确或发现未登记的
后台任务。因此，必须由已验收的实验收尾流程发布清单，且所有写入者/清理者协作使用同一锁。
此处尚未把主实验自动收尾链迁入，不能手工将活跃实验标成closed绕过保护。

保留规则：最终结果、配置、结论原件、终态收据与inventory永远保留；结论须在实验根
之外有逐字节一致的归档；checkpoint保留清单中的最大step、选中step，以及任一文件被
依赖pin/保护路径引用的整个step。四rank分片、adapter和head需按同一step完整登记。
未登记文件一律不动。其余明确登记的intermediate和旧checkpoint才成为候选；候选为
普通单硬链接文件，拒绝符号链接、共享models/datasets类别及超预算清单。

pin_manifests支持路径→SHA256映射或其pins字段。工具核验清单身份并保护被引用文件，
不能自动发现所有运行/排队依赖。示例空列表不是“没有依赖”的证明。
当前GRPO完整11点评测仍需要全部对应checkpoint，不满足本入口的封闭条件，禁止删除。

20项保留清理/缓存测试通过，涵盖实际删除、整个checkpoint分片保护、结论归档错误、
非封闭声明、路径替换和中途失败不重试。实际存储检查下仅对新建合成实验执行删除验收，
不对现有研究目录执行删除。最新checkpoint指显式完整inventory中的最新项，不是全盘搜索结果。

该入口已可供固定脚本调用；统一DAG的自动收尾清单生产和保留清理步骤尚未接入，
不宣称全存储已清理或所有日常流程已经闭合。锁文件见configs/releases/artifact-retention-v1/。
