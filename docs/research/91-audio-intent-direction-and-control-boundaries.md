# 音频意图：相对方向、绝对目标与控制动作

完整读取四个原生audio_volume类的227个学习来源：down44、mute89、other11、up83。
本轮集中完成六张音频语义卡的边界修订，发布
[目录v2](../../configs/tasks/slurp/semantic-intent-catalog-candidate-v2.json)和
[共同边界规则](../../configs/tasks/slurp/audio-intent-boundary-resolution-v1.json)。
总数仍65；其余55张本地卡和4张引用卡逐对象比较不变。未改原生Final或原生标签。

## 关键真值区别

相对方向问的是请求是否表达了更响/更轻，物理变化问的是执行后音量是否提高/降低。
两者不是同一任务。`set volume to 65%`明确请求设值，但没有比较方向；在完整语义
审核下，adjust=true、louder=false、quieter=false。未知当前音量不会将明确的语义缺失
变成unknown。`decrease to 10%`则明确带有相对减小要求，quieter和adjust均可true。

mute/unmute是明确的音频控制操作，都属于adjust；unmute本身不是相对升高请求，也
不是静音请求。`do not speak`或暂停明确的音乐播放表达了抑制该声源，但不要求修改
音量/静音控制，所以silence=true、adjust=false。silence不是adjust的无条件子类。
这说明不能按谓词名称或目录层级自动补标签。

`stop`没有目标、`down`没有领域时保留相关谓词未决；`where is the fire`则明确是其他
信息请求，六项音频谓词都为false。`volume settings`可能是当前值或设置界面；明确
打开/显示界面与明确询问当前值应分别判断。所有判断不使用native类别补缺失语义。

## 对照来源与建议

从完整阅读队列中选27个对照来源，对全部六项谓词给出162建议：25positive、120negative、
17unavailable，关联217录音、1,302条来源—录音—定义绑定。该子集用于边界验证，
不替代完整227来源标注或40,513录音主训练。

| 来源 | 新规约下的区别 |
|---|---|
| 1708 设置音量65%、2342设置60、5066设置19 | adjust=true，未表达相对方向 |
| 1716 speak more softly、2628 lower by50% | quieter=true，即使原生标签为up |
| 5284 please unmute yourself | adjust=true、silence=false，即使原生标签为mute |
| 3111/4242 do not speak | silence=true、adjust=false |
| 2828 pause the music | 抑制明确声源，不自动当音量控制 |
| 3583 set volume to zero | silence与adjust均true，不是保留声音的quieter |
| 2488 volume louder softer | 调整目的可辨，方向冲突未决，不按共存规则强行两者true |
| 4072 turn that down | 对象领域不明，不能从native down推出音频调整 |

[逐项参考](../../configs/reviews/slurp-audio-intent-boundary-reference-v1.json)包含理由、
原metadata、定义和边界hash，发布在共享`datasets/slurp-jev-benchmark/audio-intent-boundary-reference-v1`。
这些是当前规约下作者建议，不能把和native标签的差异计为官方错标，也未听审或准入训练。

## 核验和剩余工作

[独立核验](audio-intent-boundary-check-v1.json)重建原始metadata/targets、完整阅读队列、
27×6判决身份、全部绑定及统计，确认目录只改六卡。目录v1及历史资料保留，当前监督
计划指向v2；旧slot工作包不变，没有自动迁移任何正式训练标签。

本轮没有模型或训练代码变更。结构核验不能证明语义判决已由独立审稿人认可。
“事实属性信息”与领域信息、集合操作、非请求和媒体领域边界仍按doc90范围待审，
不因音量六卡完成就称整个本体冻结。下一优先处理广义事实信息的重叠和冗余监督问题，
然后统一标签覆盖及菜单/奖励配置；不能提前用65替换旧loss分母。
