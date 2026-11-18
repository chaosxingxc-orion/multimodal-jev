# 当前GRPO正式运行的源评测计划

状态：离线计划生产器已实现并完成合成测试，尚无实际634步正式运行、完整源曲线
或新版迁移成绩。旧`context_rl_evaluation_plan.py`保留其旧算法/四卡契约，不重标为
当前双卡双信用GRPO。

入口`tools/reports/context_macro_source_evaluation_plan.py`读取portable配置
`configs/portable/evaluation/context-macro-source-plan.json`，机器路径由本地宏解析。
调用formal operations的`audit_collected`，重新核对本地完整归档hash、逐文件收集
身份、准确容器命令/设备/镜像/终态及正式训练输出；不再次访问远程或复制归档。
只有完整634步formal产物可生成计划，100步效率数据不能冒充正式训练。

计划绑定0、64、128、192、256、320、384、448、512、576、634共11点。step0使用
正式包的共同Stage1策略，其余点使用同一次运行提交的检查点；逐个读取BF16执行
bundle、验证原生LoRA adapter和共享head，核对算法、起点、发布人口和commit身份。
它继承正式包的初始策略来源约束，不新增“Stage1训练过程已审核”的证明。

每点要求完整9,226条SLURP selection，固定60候选和0.5阈值。主指标set-F1，补充
top1、macro-F1、binary NLL和Brier。计划不读取目标数据、不产生评分、不选择模型。
下一步须将这些策略接入统一macro评测生命周期，独立重算每点指标，再生成
`jev-macro-transfer-source-evaluation-v1`并按set-F1→NLL→较早step确定源选模型。
目标推理只消费已封存角色；不能直接把本计划当作已有完整源成绩。

验证：源计划、训练输出审核、formal operations共40项测试22.15秒通过；共享
收集复核的6项容器/命令/字节/归档/driver故障测试1.93秒通过。前者模型tensor及
完整训练输入使用替身，后者执行真实容器inspect字段校验但观测数据为合成。
实际v6两步既有收集产物用修改后的默认`audit_collected`重审通过，manifest保持
`6794abab2d758339007fa152ff3d4e578df1c2954948422ebe4e40fbba0a1d85`。

尚缺正式launcher与NPU尾批/稀疏恢复验收、实际正式训练、更新态11点评测封包与
执行、完整源曲线生产及FSC/MINDS目标宽度后端一致性与真实推理。上述验证不证明
源表现或迁移收益。SFC等复杂任务按最新范围全部延后，不作为当前阶段门槛。

## 更新态检查点的评测封包与生命周期

`prepare_context_source_point_macro.py`通过portable模板
`configs/portable/operations/prepare-source-point-macro.json`指定源计划、hash和step，
重新审核实际formal产物并重建完整计划后，复用已有selection封包流程。只替换新
副本内的LoRA/head及adapter，两个服务与客户端都绑定同一个新policy；原formal
包和已准备的Stage1评测包保持原样。检查processor身份与原始Stage1完全相同，
因此音频编码缓存可复用；不重新拟合任何标签或阈值。

`context_source_point_macro_operations.py`提供validate/run/collect/audit，复用
`configs/portable/operations/selection-macro-operations.json`本地宏和原双实例
生命周期。默认不准入派发，运行时仍要求显式dispatch配置、设备空闲和独占lease。
旧Stage1入口拒绝新的source_curve_point角色，避免将更新模型误写为初始基线。
全量输入、完整macro收集、评分独立复算沿用现有代码。

源point验证除全包hash外，还核对11点顺序、算法、源manifest、原审核文件、所选
step与policy manifest，目标调参必须关闭；重新封存文件hash不能绕过这些语义
检查。完整formal来源由公开prepare入口重新审核，不把SOURCE-PLAN文件hash本身
当作容器/权重真实运行的证明。

封包、错配拒绝、旧入口、容器收集及源计划回归共40项通过（2.77秒）。这些封包
测试的formal审计/模型/缓存为替身，真实文件复制、pins和服务策略绑定执行。
实际已封存Stage1包用新默认validator复核通过，manifest保持
`d38ab9ffb0b2645cb0d729578ad7ff6d56a06d8e0905d8c19480f61e2b0c80a0`。
尚未生成实际formal检查点评测包或派发NPU评测；完整曲线汇总生产器仍待接入。
另以真实v6第二步更新策略`1281238cbdeeb83a5c1f1844f2c32136d2955b3362db6a1cd223101a1be7e2e6`
重读执行bundle并验证native adapter，processor与共同Stage1保持相同
`41f8edc45719bb2d02b385f57520b862f605df05d38fea60788eed8115ee96fe`。
这是实际更新权重可复用输入processor的证据，不能代替全量selection推理或收益。

## 完整曲线汇总与迁移角色输出

`collect_context_macro_source_curve.py`读取portable配置
`configs/portable/evaluation/context-macro-source-curve.json`：源计划及其hash、按11点
顺序列出的评测operations配置/本地宏、输出目录。每次重新审核正式训练并重建
源计划，再逐点评估归档、容器、完整macro journal和独立评分。完整验证通过后，
写出每点score、完整curve、审核证据、source-evaluation与三角色绑定，使用本地
持久盘的原子提交入口，不覆盖既有目录，不使用目标数据。

不同点评测必须保持镜像、源码runtime、processor、输入收据、缓存证据、任务
契约、macro大小和两实例服务容量一致；只允许策略本身变化。source-point validator
现进一步读取实际policy bundle并核对native adapter，避免只验封包hash而未核对
模型张量。全部11点齐全才按set-F1→NLL→较早step选择；允许step0胜出，保留固定
终点退化。输出`jev-macro-transfer-source-evaluation-v1`由现有迁移入口重算选择并
验证共同Stage1、源选GRPO和固定终点三个角色。

61项曲线生产、原曲线/迁移客户端和源点封包回归通过（3.15秒），包括缺点/重复点/
错序、错policy、镜像/runtime/服务batch不一致、输入变化、错误阈值或人口、审核
失败、源成绩篡改、step0胜出和终点退化。模型及容器审计为显式测试替身，实际
曲线计算、hash绑定、原子写入和下游角色读取执行；测试临时tmpfs的存储检查单独
替换，不修改生产存储约束。实际Stage1冻结包默认validate仍通过，manifest未变。

这补齐了固定脚本的源曲线生产路径，但没有新的真实训练或评分。正式634步训练、
11点实际NPU推理、目标宽度parity及迁移部署/收集仍待完成；复杂任务已延后。

## 独立clean TEST接入同一macro接口

9,226条selection是TRAIN内部分割，不是官方TEST。当前另有冻结的12,991条
clean TEST（原13,078条排除87条），作为源域独立测试，不能记作跨数据集迁移。
新增`context_slurp_test_macro_inputs.py`复用既有完整TEST processor证据，并重新
读取当前portable v2的40,513 learning及9,226 selection，核对任务定义/requirements
不变、TEST录音ID及音频文件hash与两源集合均无交集。该检查不证明语义重复或
解码后波形重复不存在，也不打开TEST标注。

`context_transfer_macro_inputs.py`新增显式`SLURP_clean_TEST60`适配；复用同一
macro推理、独立journal审核和三策略角色矩阵。配置模板为
`configs/portable/operations/prepare-slurp-test-macro-inputs.json`及
`configs/portable/evaluation/context-slurp-test-macro.json`，本地路径仅在宏文件。
源角色绑定必须匹配实际source-evaluation中的release hash，不能用另一份训练
发布定义替换本次源曲线。全60候选、0.5阈值保持冻结。

真实输入准备已成功：本地宏`.local/config/slurp-test-macro-inputs-v1.json`，产物
`/srv/multimodal-jev-xingchao/operations/slurp-test-macro-inputs-v1`。完整12,991条
请求/渲染hash、60个marker位置及既有tensor身份通过；共同processor为
`41f8edc45719bb2d02b385f57520b862f605df05d38fea60788eed8115ee96fe`。
population hash为`10076c4ae5ce561e6e6c7d41d34fb4f5c077c9e19328eb63da3d276c4c2daa4a`，
独立输入收据hash为`76a355d9ccfc6494b6be8d2545dceeff27d4346c80a73951c9c428accd4d54c3`。
本次复核未读取远端cache字节、未启动推理、未打开TEST标签或生成分数。

`context_slurp_test_macro_truth.py`和共用`score_context_transfer_macro.py`连接原生
标注重建与三角色评分。全部独立推理输出通过审核后才读取预先冻结的标注契约；
核对实际执行输入、原发布标签/排除清单hash与TRAIN候选映射。旧单请求评分默认
协议保持不变，macro分支显式验证实际logits及compact request hash，不改写请求ID。
结果标注`evaluation_kind=source_domain_held_out`，与FSC/MINDS的迁移结果分开。
评分组件仍明确`runtime_provenance_verified=false`；上游部署和collector必须补足
实际容器/策略/后端对齐证据，不能把局部输出审核当成完整运行验收。

37项输入及共用推理/审核回归通过2.96秒；32项三角色全人口合成评分、旧TEST评分
和FSC/MINDS来源回归通过73.91秒；7项TEST冻结来源/人口错误拒绝通过2.33秒。
输入角色隔离使用完整人口夹具，实际模型/容器或标注重建部分为显式替身；另有上述
真实全TEST输入证据。70个效率序列pins均未变。实际macro部署/推理/收集尚未执行，
当前没有新的源TEST或迁移收益结论。
