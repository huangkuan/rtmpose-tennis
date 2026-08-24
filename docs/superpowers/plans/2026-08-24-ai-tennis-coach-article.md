# AI Tennis Coach Article Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `docs/` 下完成《如果我有个 AI 网球教练》，以典型目标用户的第一人称表达对 AI 教练产品功能蓝图的期待，并准确衔接当前原型能力。

**Architecture:** 正文是一篇单文件 Markdown 叙事文章，沿“学习痛点 → 当前原型起点 → 训练前/中/后/长期功能期待 → 真人教练协作 → 产品边界 → 用户价值”推进。源码与 README 只作为事实校准依据；正文将工程实现翻译为非技术用户语言，并持续用时态和措辞区分现有能力与未来愿景。

**Tech Stack:** Markdown；事实来源为 `rtmpose_tennis/app.py`、`README.md` 与已确认的设计稿；验证使用 `rg`、`wc`、`git diff --check` 和人工内容核对。

---

## File Structure

- Create: `docs/if-i-had-an-ai-tennis-coach.md` — 唯一正式文章，承载第一人称叙事、功能蓝图、边界与结论。
- Reference: `docs/superpowers/specs/2026-08-24-ai-tennis-coach-article-design.md` — 已确认的人物、结构、事实映射与验收标准，不复制进正文。
- Reference: `rtmpose_tennis/app.py` — 校准当前姿态检测、主要球员选择、裁剪跟随、最新帧和指标能力。
- Reference: `README.md` — 校准项目定位、运行模式、性能取舍以及动作识别、反馈、球拍/球检测等后续方向。

### Task 1: 建立人物声音与文章骨架

**Files:**
- Create: `docs/if-i-had-an-ai-tennis-coach.md`
- Reference: `docs/superpowers/specs/2026-08-24-ai-tennis-coach-article-design.md`

- [ ] **Step 1: 写入标题、导语与完整章节骨架**

创建下列章节，正文标题保持中文：

```markdown
# 如果我有个 AI 网球教练

## 三十多节私教课之后，我缺的仍然是课后的那双眼睛
## 现在的它，先学会了看见我
## 训练开始前：先理解今天的我
## 训练过程中：少说一点，但说到关键处
## 训练结束后：别只给我一个分数
## 几个月以后：请告诉我是否真的进步了
## 它不应该取代我的真人教练
## 我也不希望它假装无所不知
## 我期待的，是一个真正陪我练球的伙伴
```

- [ ] **Step 2: 完成第一人称导语**

导语一次性交代“30 多节一对一私教课”，通过转肩、引拍、降低重心、随挥等课堂细节表现人物经验。写清核心矛盾：作者认真、热爱且认可真人教练，但独自练球时缺乏及时、可信、可持续的反馈。不要把人物写成技术人员或刚上体验课的新手。

- [ ] **Step 3: 检查人物设定是否自然出现**

Run:

```bash
rg -n "三十多节|30 多节|一对一|私教|转肩|引拍|重心|随挥" docs/if-i-had-an-ai-tennis-coach.md
```

Expected: 私教次数只在开篇明确出现一次；其余命中来自自然的训练细节，不是重复炫耀经验。

- [ ] **Step 4: 提交人物与骨架**

```bash
git add docs/if-i-had-an-ai-tennis-coach.md
git commit -m "docs: frame AI tennis coach user story"
```

### Task 2: 用非技术语言写清当前原型起点

**Files:**
- Modify: `docs/if-i-had-an-ai-tennis-coach.md`
- Reference: `rtmpose_tennis/app.py`
- Reference: `README.md`

- [ ] **Step 1: 写“现在的它，先学会了看见我”**

用 3–5 个自然段说明作者研究项目后确认的事实：它能读取摄像头或录像；多人出现时优先关注画面中较大、较居中的主要球员；估计 17 个身体位置及可信程度；利用身体位置让观察区域平滑跟随；看不清、可信位置太少，或人物接近尚可向源画面内部扩展的裁剪边缘时会重新寻找。如果命中的裁剪边缘已经贴住源画面边界，重新检测无法看到画面外的内容，系统会记录但抑制这类无效请求。缩小预览不改变原始动作坐标。

还要通俗、准确地区分诊断信息的位置：画面覆盖层显示处理延迟、画面年龄、序列滞后与丢帧比例等即时指标；终端的会话总结进一步报告检测刷新的触发原因和边缘诊断。由此引出用户判断：即时建议不仅要算得准，还必须基于足够新的动作画面。正文不必逐项复述指标名，但不能错误声称画面覆盖层会直接展示重检测原因。

涉及实时性的措辞必须带有限定：只有启用异步摄像头或实时录像模式时，系统才持续保留最新画面并主动替换来不及处理的旧画面；后台检测也需要显式启用。可以写这种设计“避免旧画面或检测任务持续排队”，不能承诺它“永远不会变迟”“零延迟”或适用于所有运行模式。

- [ ] **Step 2: 在本节结尾划清当前边界**

明确把当前阶段比作“眼睛和注意力”，并写明它还不能判断正手、反手或发球，不能理解一整段动作，也尚未真正给出网球教学建议。

- [ ] **Step 3: 排除工程师口吻**

Run:

```bash
if rg -n "LatestFrameCapture|LatestDetectorWorker|select_player|DetectorRequest|线程|MPS|RTMDet|--[a-z]" docs/if-i-had-an-ai-tennis-coach.md; then exit 1; else echo "No implementation jargon found"; fi
```

Expected: `No implementation jargon found`。

- [ ] **Step 4: 对照源码和 README 逐项核对事实**

Run:

```bash
rg -n "COCO_KEYPOINT_NAMES|select_player\(|update_crop_from_pose|assess_pose_redetection|LatestFrameCapture|LatestDetectorWorker|preview_scale" rtmpose_tennis/app.py
rg -n "normalize|1–3 seconds|classify|racket/ball|persistent person tracking|feedback" README.md
```

Expected: 当前能力与未来方向均能在事实来源中找到对应位置；正文没有把 README 的 next phase 写成已实现。

- [ ] **Step 5: 提交当前能力章节**

```bash
git add docs/if-i-had-an-ai-tennis-coach.md
git commit -m "docs: explain current pose prototype to users"
```

### Task 3: 完成训练全旅程的产品功能期待

**Files:**
- Modify: `docs/if-i-had-an-ai-tennis-coach.md`
- Reference: `docs/superpowers/specs/2026-08-24-ai-tennis-coach-article-design.md`

- [ ] **Step 1: 写训练前功能期待**

从“我曾经拍下一堆不能用的视频”的痛点出发，覆盖智能机位引导、拍摄完整性、不同机位用途、身高与身体比例、惯用手/水平/伤病限制/今日目标、个人基线以及视频隐私与留存选择。每个功能至少回答“它解决我的什么问题”。

- [ ] **Step 2: 写训练中功能期待**

覆盖动作阶段划分、可解释的身体与时序反馈、每回合只说一个优先问题、反馈冷却、耳机短提示，以及置信不足时承认“没有看清”。重点体现上过大量私教课的用户知道口令过多会破坏击球节奏。

- [ ] **Step 3: 写训练后功能期待**

覆盖自动切分有效击球、最佳/典型错误/无法判断片段、阶段定格、与过去的自己比较、把问题转化为下一次练习，以及简洁的训练总结。明确反对脱离水平与身体条件的职业动作照抄和孤立总分。

- [ ] **Step 4: 写长期成长功能期待**

覆盖能力地图、稳定性和趋势、区分偶然好球与习惯形成、训练负荷提醒、关键片段与结构化报告分享给真人教练。保持用户语言，不提出后台架构或模型实现方案。

- [ ] **Step 5: 检查每阶段都由痛点驱动**

Run:

```bash
rg -n "训练开始前|训练过程中|训练结束后|几个月以后" docs/if-i-had-an-ai-tennis-coach.md
rg -n "我希望|我期待|如果它能|对我来说|我需要" docs/if-i-had-an-ai-tennis-coach.md
```

Expected: 四个阶段均存在；功能期待持续与第一人称问题或价值相连，而不是孤立的产品清单。

- [ ] **Step 6: 提交产品旅程章节**

```bash
git add docs/if-i-had-an-ai-tennis-coach.md
git commit -m "docs: describe AI coaching product journey"
```

### Task 4: 完成人机协作、产品边界与结尾

**Files:**
- Modify: `docs/if-i-had-an-ai-tennis-coach.md`
- Reference: `README.md`
- Reference: `docs/superpowers/specs/2026-08-24-ai-tennis-coach-article-design.md`

- [ ] **Step 1: 写真人教练与 AI 的互补关系**

用私教课堂中的示范、手感、战术、心理、安全与即时调整说明真人教练的不可替代性；让 AI 承担高频重复观察、客观记录、课后陪练和长期趋势回顾。用“真人定方向，AI 陪我练，证据再回到真人教练手里”收束关系。

- [ ] **Step 2: 诚实写出产品边界**

明确当前没有球拍和网球检测，不能可靠理解触球点、拍面和落点；单帧二维身体位置会受遮挡、肢体交叉、运动模糊、出画和机位影响；当前轻量跟随不是比赛级持久身份跟踪；AI 应表达不确定性，避免用虚假的小数精度制造权威感，也不能做医疗诊断。

- [ ] **Step 3: 写回用户最终价值**

结尾避免技术路线图，落在典型目标用户愿意长期使用和付费的理由：系统记得个人目标、把大量重复连接成成长轨迹、给出少而可执行的提示，并知道什么时候应该闭嘴。

- [ ] **Step 4: 提交完整初稿**

```bash
git add docs/if-i-had-an-ai-tennis-coach.md
git commit -m "docs: complete AI tennis coach article"
```

### Task 5: 完成准确性、人物声音与可读性核验

**Files:**
- Modify if needed: `docs/if-i-had-an-ai-tennis-coach.md`
- Reference: `docs/superpowers/specs/2026-08-24-ai-tennis-coach-article-design.md`
- Reference: `rtmpose_tennis/app.py`
- Reference: `README.md`

- [ ] **Step 1: 核对文章长度与结构**

Run:

```bash
wc -m docs/if-i-had-an-ai-tennis-coach.md
rg -n '^#' docs/if-i-had-an-ai-tennis-coach.md
```

Expected: 正文约 3,000–4,500 个中文字符，章节顺序完整，无空章节。

- [ ] **Step 2: 核对“已有能力”和“未来愿景”的措辞**

人工逐段检查所有“现在、已经、能够”附近的陈述都能由 `app.py` 或 README 的现状部分支持；所有动作分类、反馈、球拍/网球、个人档案、训练计划和长期成长能力都使用“希望、期待、如果”等未来措辞。确认文章没有虚构运行结果、用户数量、识别准确率或尚不存在的模型。

- [ ] **Step 3: 做去术语复读测试**

暂时忽略或删读模型名、算法名和实现词，再通读文章。Expected: 非技术读者仍能准确理解当前原型与未来产品的分界；作者全程像认真研究过产品的网球爱好者，而不是工程师。

- [ ] **Step 4: 做目标市场典型性检查**

确认人物既不是完全新手，也不是职业运动员；其痛点能代表愿意持续投入课程、器材和练习时间，却缺少课后高频反馈的进阶爱好者。避免把个人需求写成所有网球用户都完全相同。第一人称是经过明确设计的目标用户模拟，不得写成 AI 或实际作者拥有这些真实生活经历。

- [ ] **Step 5: 执行仓库级文本检查**

Run:

```bash
git log --check --oneline -- docs/if-i-had-an-ai-tennis-coach.md
if rg -n '[[:blank:]]+$' docs/if-i-had-an-ai-tennis-coach.md; then exit 1; else echo "No trailing whitespace found"; fi
if rg -n "TODO|TBD|待补充|待确认" docs/if-i-had-an-ai-tennis-coach.md; then exit 1; else echo "No placeholders found"; fi
git status --short
```

Expected: 整篇文章在所有提交中均无空白错误，当前文件无行尾空格和占位符；状态只包含本任务预期改动。

- [ ] **Step 6: 提交最终润色（仅在有修改时）**

```bash
git add docs/if-i-had-an-ai-tennis-coach.md
git commit -m "docs: polish AI tennis coach article"
```
