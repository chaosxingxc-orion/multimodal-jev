# 训练归档的配置化收集入口

固定入口tools.collect_training_artifacts复用原collect_context_rl_training，不启动模型
或训练、不轮询、不调用模型分析。共享模板为configs/portable/operations/training-collection.json；
从configs/training-collection-macros.example.json取宏加入自己的本地绑定文件。

```bash
PYTHONPATH=src:. python -m tools.collect_training_artifacts \
  --config configs/portable/operations/training-collection.json \
  --local .local/config/macros.json
```

adopt模式只核验已完成归档，不访问远程。collect模式调用原收集器，先检查准确容器
与镜像且已exited，再通过只读继承挂载的临时reader容器导出；reader无NPU设备。
主机使用显式SSH别名；launch文件hash绑定container ID、镜像、远程根和原配置身份。
已有collection/transport/archive时拒绝再次收集，不接管原owner。

入口校验transport状态、launch绑定、所有声明文件及archive SHA256；符号链接路径
和路径越界拒绝。默认保留失败训练的完整输出，不将ExitCode0或收集成功视为训练成功。
adopt校验已有archive字节hash和声明文件；归档的实际解包结构检查由原收集器负责，
该入口不再次解包。collect的网络与挂载权限仍沿用目标机配置，不认证任意新主机。

output_directory必须与collection分离。按任务统一lock_path，并对发布目录另加锁；
所有协作调用者必须使用同一任务锁，工具不会自动发现不协作的旧收集器。
热写执行现有本地存储检查；失败保存status，不自动重试。完成清单记录sources和
artifacts，重入重新校验后返回already_complete。大归档使用流式SHA256，避免一次读入内存。

本版本是配置入口，不改旧collector的持久化细节或失败恢复策略；不宣称断点续传，
也未接入run_operations的自动collect→trajectory动态绑定。轨迹导出仍使用既有
tools.export_training_trajectory，通过其配置显式绑定本次transport及SHA256。

16项收集入口/轨迹测试通过，覆盖离线接入、输入篡改、未完成状态、路径越界及失败
不重试。实际旧两步prefix归档已用于离线接入验收；没有新增SSH收集或碰当前训练评测。
研究中间产物清理与完整流程串接仍需完成，不能以本入口代替全量日常归并验收。
