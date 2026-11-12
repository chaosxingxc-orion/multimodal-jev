# 四卡第二步后的独立验收自动衔接

2026-11-12 UTC。独立队列PID3510242已启动，实际进程命令行核验一致；首轮处于waiting_exact_resume。每600秒检查一次，不占NPU，不改现有四卡恢复准备包或前驱队列。

当前链路：seed29两项固定评估结束 → 原DP4队列确认4–7空闲并派发四卡恢复 → 恢复容器退出0且四rank第二步记录完整 → 本队列冻结导出adapter和证据 → 物理卡4独立HF重载/48音频vLLM TP1验收 → 逐rank恢复审计。已有seed29两项训练均完成，评估运行中；这是01:37 UTC的实际Docker观察，不是完成估计。

后继使用独立目录`/mnt/cephfs/xingchao/experiments/multimodal-jev/span-fsdp-audio-resume-suite-v1`与固定镜像；在实际adapter存在且四rank成功前不生成运行config。冻结config包含adapter文件、四rank master/Adam、恢复前张量、selection和导出后概率的hash。创建前和启动前检查物理卡占用，创建超时不重复派发。

`run_span_fsdp_resume_acceptance.py`调用既有HF/vLLM完整套件，再调用`verify_span_fsdp_resume_audio.verify_resume`：核验原断点master/Adam精确恢复、原生step=2、下一全局平衡批次、导出后独立HF误差0、48输入完整候选及联合动作概率gate。通过后才写`ACCEPTED.json`。不减小验收人口或放宽概率阈值。

终态含义：队列terminal=0仅表示后继已派发；套件run/READY只证明HF/vLLM步骤；最终还须controller-terminal=0及ACCEPTED，才能说明四卡第二步恢复验收完成。任何失败保留原容器和输出，不重启训练。

21项CPU测试通过：精确前驱身份、训练仍运行、前驱失败、镜像不符、创建超时、回执先于start、四rank缺陷、step/adapter不符、依赖文件改变和重复发布均有拒绝检查。真实NPU恢复和后继验收尚未执行，此结果不代表完整verl连续trainer已完成。
