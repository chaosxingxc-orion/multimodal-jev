# 集合与音乐行为边界收口

当前目录更新为`semantic-intent-catalog-candidate-v4.json`，只改六卡：
list_create_add、list_information、list_remove、music_positive_preference、
music_feedback_record、music_content_save。55张其他本地卡与4张引用卡不变，
共享request/nonrequest政策不变；新增共同边界政策
`collection-music-boundary-resolution-v1.json`。所有定义仍为候选。

完整阅读学习集原生lists三类424来源、music喜欢/不喜欢/设置149来源，另读其他类别
含list/playlist/favorites/favourites的193来源，共766来源。此规则只用于离线阅读
取样，不用于运行时选Q或生成语义标签。对40来源明确记录88项边界建议：34正、41负、
13未决，绑定180录音、376判决×录音关系。没有把全部766来源宣称为已标注。

| 共同规则 | 真实参考的区分 |
|---|---|
| list作为答案格式不等于一般清单对象 | 12676“list the available train trips”不是一般清单查询；10482“make a list of clothing”可请求构造一般集合 |
| 内容主题不决定容器功能 | 11208把会议加入project list可为一般清单；11037从contact list删除人属于专用容器 |
| 原生类别不补语义 | 7657虽属calendar_set，实际“give me my to do list”支持一般清单查询 |
| 保存内容与记录评价分开 | 932保存歌曲仅支持内容保存；1585保存对歌曲的意见支持反馈记录，但极性未明 |
| 明确favorite指定可以重叠 | 222加入my favorite list支持保存、反馈和正偏好；1342“this song is my favorite”支持偏好，不自动要求持久化 |
| 音乐集合不因list词面变一般清单 | 11135从歌曲创建playlist支持音乐保存，不支持一般list_create_add |
| 歧义保留 | 10477“make sure bread is on…”可为检查或确保添加；11028“edit list”不确定具体操作；不按native强判 |

favorite规则区分评价性类别与字面集合名，引用已有favorites文件夹不自动构成新评价。
数字评分继续使用原nonrequest政策，不从五颗星推断未给出的量表。一般清单可以包含
人物、会议、电影名；只有实际表达专用容器操作才按专用行为判断。此处没有新增
本体类别，也没有强迫每条样本都落入某个语义谓词。

统一清单已发布`intent-presence-inventory-v2`并接监督计划，保留v1及原包：
全40,513录音/9,192来源/65卡，1,146建议覆盖530来源，439正609负98未决，
596,334坐标未评。15卡具有正负参考，2卡缺正例，48卡无参考。历史1,058项没有
因本次修订失效；这不意味着它们经过独立语义认证。新语义训练准入仍为0。

原metadata重建阅读集合、目录精确六卡delta、参考多重集以及完整录音绑定和65卡统计
均通过核验，见[边界检查](collection-music-boundary-check-v1.json)与
[全量清单检查](intent-presence-inventory-check-v2.json)。结构核验与规则作者的参考
判决分开；没有独立语义审核或音频复听，不把hash一致当作语义正确。

下一步按同一全本体系统性覆盖剩余谓词与规模化监督，集中处理共有逻辑缺陷；
不继续把每个新词例都变为新政策。当前仍需有效监督覆盖、独立分歧处理与固定菜单/
目标/评测/FSC迁移统一后才能冻结任务，不能用这40来源作主训练集。
