# 一般列表、日历和联系人集合的共同边界

完整阅读151个原生list_name学习来源、387个calendar/schedule词法来源和48个
联系人集合词法来源，去重582来源；对应各组582、1634、153录音。参考上游指南
139–141、146、148、174–176的事件、联系人和列表例子；没有听审。

新版规则以“明确集合角色”为依据：

- 一般事项/任务列表、作为事件集合的calendar/schedule、用于记录查询或修改的
  contacts/address book可以满足list_name，不要求列表标题或恢复所有条目。
- schedule作为安排动作、on schedule作为准时状态、单个事件或泛指待办活动
  不自动建立集合身份。新建的有名称/组织描述的列表可以算。
- 音乐播放列表、音乐库、邮件文件夹和普通存储容器不因“含有元素”自动算一般列表；
  明确表达一般事项清单用途时独立判断。list of books to order和DVR recording list
  有清单证据，my music和wallet alone没有。
- 裸calendar/contacts可能指应用、集合或片段，继续保留未决；明确absence不是
  “未标注”，也不应因可能想象出别的上下文就一律转unknown。

这是项目新语义定义选择，不能称作官方原生标签的唯一还原。固定Q算子不变，不按
样本类别选题。原triage的集合组5项（7242、8505、7414、15896、17089）均在
新规则下给list_name=true；11017的待办活动查询未表达集合，给false。

39项参考建议为22真、11假、6未决，绑定165录音，0准入。18项与原生投影不同
不等于18处官方错标。新包[slot-consolidated-design-v9](../../configs/tasks/slurp/slot-consolidated-design-v9.json)
使用类型v11/政策v11/边界v12，仅改变list_name一张类型卡及两条相邻边界，
其余54卡逐字段不变。原email_folder/playlist_name卡未改，但相邻语义契约变化
使其旧参考需要明确复审，不自动继承。

合并为`annotation-workpack-current-rules-v8`，40513录音/9192来源及全部历史
证据保留。5712评估覆盖4749坐标：3480当前明确、60未决、1209需重审，500811未评。
明确参考218正3262负；31类正负都有、22仅负，email_folder/playlist_name两类
当前暂无有效明确参考。这是相邻规则迁移影响，不是数据类型消失。

证据：

- [共同规则与原始证据组](../../configs/tasks/slurp/collection-role-boundary-resolution-v1.json)。
- [参考建议](../../configs/reviews/slurp-collection-role-boundary-v1.json)。
- [规则/人口/绑定独立核验](collection-role-check-v1.json)。
- [全工作包重建核验](current-rule-migration-check-v8.json)。
- [55类覆盖复算](current-semantic-coverage-check-v4.json)。

原metadata和完整learning/targets重建三组人口及全部绑定，核对卡片/相邻规则
变更范围、工作包全评估多重集、版本适用性及55类覆盖，通过。生产实现未改，
未重复运行未受影响测试；结构核验不认证语义、不替代听审。
工作包报告SHA为`9f37e3af595ae2dfd28684dadff411ced4ff586aa65740c3653f16aac51a0f59`。

任务/监督/上下文入口已指向v9；历史triage明确标为旧工作包计数，不原位改写。
固定审核offset仍80。下一处理属性激活/省略事件及受影响参考复审；全局范围候选
仍未应用，正式训练和稳定工程验收尚未开展。
