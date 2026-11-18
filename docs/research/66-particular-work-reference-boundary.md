# 作品名称、特定引用与开放内容选择

2026-11-13。任务规则收敛，本轮没有模型forward、训练或下载。

## 共用判决规则

七类作品/集合谓词共用“特定引用+作品类型”条件：audiobook_name、song_name、
podcast_name、music_album、playlist_name、movie_name、game_name。
名称/标题可以建立引用；指代、所有格、序号、播放历史也可以。无需恢复真实作品ID。
仅给题材、风格、评分、提供者等开放筛选条件，不自动形成特定作品引用。

| 学习来源表达 | 类型 | 新规则参考建议 |
|---|---|---|
| play last song from my favourite playlist | song_name | true：上一首是历史引用 |
| i would like to listen to my audiobook | audiobook_name | true：明确的所有格作品引用 |
| play a podcast about latest yoga videos | podcast_name | false：开放主题筛选 |
| what's the best podcast | podcast_name | false：开放质量推荐 |
| what is my most played podcast | podcast_name | true：个人历史识别某个节目 |
| play podcast named ringcast | podcast_name | true：类型与名称明确 |
| play imagine dragons | song_name | unavailable：句子未消除艺人/歌曲等类型歧义 |

冠词、所有格和最高级不是机械关键词规则，必须结合整句的特定引用关系。提供者/
创作者名字本身不保证已识别作品；“NPR podcast”“Jocko's podcast”在本次对照中
保留未决，不用外部目录知识决定它是单一节目还是任意该提供者内容。

游戏包含在游戏语境下明确命名的规则系统，例如chess和rock paper scissors；不能
把所有运动或开放游戏类别当具体游戏作品。播放列表的名称、序号、所属关系可识别
集合；只要求按条件生成任意新列表则不同。其他类型必须各自满足角色证据。

## 证据与版本

读完七类原生阳性并集353来源/1592录音/12意图的完整参考文本。各类来源计数为：
有声书60、歌曲102、播客48、专辑1、播放列表53、电影13、游戏77；类型有交集，
不得直接把计数相加当独立来源。上游指南138/150/182/191段提供作品与游戏名称示例，
但不证明本项目加入的指代/描述性引用与原生标注等价。

规则：`configs/tasks/slurp/work-reference-boundary-resolution-v1.json`。
统一入口：`slot-consolidated-design-v4.json`，类型v6/政策v6/边界v7。
七类语义及21条相邻边界同步更新，48张其他类型卡逐字段不变。保留此前news与event
修订，旧版本不覆写。原生字段名保留作映射键，新语义范围必须独立发布监督与指标。

## 对照建议与核验

43来源45判决坐标：23 true、13 false、9 unavailable；222录音判决绑定、216独立
录音。13项与native投影不同，不代表13个官方错标；9项未决不补false。
首批固定16来源中3个作品引用未决项已在新规则下重判：11334的song/podcast为false，
9877的audiobook为true。只处理这3个受影响未决坐标，不宣称完成新版全55类型重审。

作者记录：`configs/reviews/slurp-work-reference-boundary-v1.json`。
共享产物：`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/work-reference-boundary-evidence-v1`。
全部来源/metadata、原始learning/target及222绑定已另行重建，类型/政策hash和
55/55/89依赖关系核对通过，记录见`docs/research/work-reference-boundary-check-v1.json`。

这些是同一agent参考文本建议，不是独立标注者复核或音频听审。正式准入0，不将
边界对照替代完整学习人口，不以原生正类选择作为在线Q路由。没有生产代码变更，
本轮不运行无关测试；结构核验不验证语义建议正确性或模型效果。

剩余公共边界是显式属性名/未赋值问句，以及个人属性、关系与一般事实/定义查询。
这些收敛后统一迁移监督，按来源继续全量审核。当前文本尚无processor/NPU验收，
任务和工程未冻结，训练派发仍为false。
