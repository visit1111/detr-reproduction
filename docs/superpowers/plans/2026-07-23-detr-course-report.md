# DETR Course Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份约 2000 字、数据可追溯、图片可正常显示的 DETR 论文阅读与复现实验 Markdown 报告。

**Architecture:** 先从论文 PDF、本地评估产物、服务器训练日志和 Git 仓库中收集证据，再将可复用图片整理到 `reports/assets/`。根据完整训练日志生成曲线，最后撰写正文并用自动检查验证字数、图片路径、敏感信息和 Git 状态。

**Tech Stack:** Markdown、MinerU flash-extract、PowerShell、SSH、Python 3.12、Matplotlib、Git。

## Global Constraints

- 正文约 2000 字，采用正式课程实验报告语体。
- 所有实验数字必须来自实际日志。
- 区分官方权重复现、全量数据 20 轮训练和 10% 数据 100 轮训练。
- 训练曲线必须同时呈现 train loss、val loss 和 COCO AP。
- Markdown 图片使用 `assets/...` 相对路径。
- 不写入密码、私钥、服务器地址或 SSH 配置。
- 最终只提交报告、必要图片和实施文档，不提交原始 COCO 数据、模型权重或完整服务器日志。

---

### Task 1: 提取论文与实验事实

**Files:**
- Read: `C:/Users/liuhuayong/Zotero/storage/W2GRMJ7L/Carion 等 - 2020 - End-to-End Object Detection with Transformers.pdf`
- Read: `eval-output/detr_r50_coco_val2017/stdout.log`
- Read: `eval-output/detr_r50_coco_val2017/visualizations/selected-samples.json`
- Read: `configs/detection_10pct_100ep.json`
- Create (ignored working file): `artifacts/report-work/detr-paper.md`
- Create (ignored working file): `artifacts/report-work/train10-log.jsonl`
- Create (ignored working file): `artifacts/report-work/train-full20-log.jsonl`

**Interfaces:**
- Consumes: 用户提供的论文、本地评估输出、服务器 `/data/lhy` 范围内的训练日志。
- Produces: 可核对的论文文本、两组训练 JSONL 日志和实验事实清单。

- [ ] **Step 1: 建立忽略的工作目录**

Run:

```powershell
New-Item -ItemType Directory -Force artifacts/report-work | Out-Null
git check-ignore artifacts/report-work
```

Expected: 输出 `artifacts/report-work`，证明临时材料不会进入 Git。

- [ ] **Step 2: 用 MinerU 提取论文**

Run:

```powershell
npx --yes mineru-open-api flash-extract `
  "C:\Users\liuhuayong\Zotero\storage\W2GRMJ7L\Carion 等 - 2020 - End-to-End Object Detection with Transformers.pdf" `
  -o artifacts/report-work/detr-paper.md `
  --language en
```

Expected: 生成非空 Markdown，包含 `DETR`、`bipartite matching`、`object queries` 和 `panoptic segmentation`。

- [ ] **Step 3: 核对本地官方权重评估结果**

Run:

```powershell
rg -n "Average Precision|AP50|AP75" `
  ..\eval-output\detr_r50_coco_val2017\stdout.log `
  results\reference\official_detr_r50_coco_val2017.json
```

Expected: 可核对 AP 42.0、AP50 62.4、AP75 44.2。

- [ ] **Step 4: 从服务器读取两组训练日志**

Run read-only SSH commands to:

1. locate output directories under `/data/lhy/output`;
2. copy only `log.txt` for the full-data 20-epoch run and 10%-data 100-epoch run into `artifacts/report-work`;
3. verify line counts are 20 and 100;
4. verify every line parses as JSON.

Expected: 两个 JSONL 文件分别包含 20 和 100 个完整 epoch，且不包含凭据。

- [ ] **Step 5: 形成事实清单**

Use Python to print:

- epoch 0、10、20、40、60、80、99 的 train loss、val loss 和 AP；
- train/val loss 的最小值及 epoch；
- 最佳 AP 及 epoch；
- 最近 10/20 轮趋势；
- 全量 20 轮首末 train loss。

Expected: 输出足以支持收敛和过拟合判断，不依赖单轮波动。

### Task 2: 整理报告图片

**Files:**
- Create: `reports/assets/detr-architecture.png`
- Create: `reports/assets/official-coco-metrics.png`
- Create: `reports/assets/ground-truth-vs-detr.jpg`
- Create: `reports/assets/training-curves-10pct-100ep.png`

**Interfaces:**
- Consumes: 官方仓库架构图、本地评估图和 Task 1 的 100 轮 JSONL。
- Produces: 四张可由 Markdown 相对引用的最终图片。

- [ ] **Step 1: 复制已有图片**

Copy:

```text
.github/DETR.png
../eval-output/detr_r50_coco_val2017/visualizations/coco-metrics.png
../eval-output/detr_r50_coco_val2017/visualizations/ground-truth-vs-detr.jpg
```

to their exact names under `reports/assets/`.

Expected: 三个目标文件存在且大小大于 10 KB。

- [ ] **Step 2: 生成训练曲线**

Use Matplotlib with two vertically aligned panels:

- upper panel: train loss and val loss against epoch;
- lower panel: COCO box AP in percentage points against epoch;
- mark the learning-rate drop at epoch 80;
- use Chinese-independent English labels so the image renders without extra fonts;
- save at 180 dpi with tight bounding box.

Expected: `training-curves-10pct-100ep.png` exists, covers epochs 0–99, and visually matches the JSONL values.

- [ ] **Step 3: 逐张检查图片**

Open all four assets and check:

- no cropping or unreadable labels;
- metric numbers match logs;
- prediction comparison remains legible at report width;
- architecture image attribution is stated in the report.

Expected: 四张图片均可用于提交。

### Task 3: 撰写 Markdown 报告

**Files:**
- Create: `reports/DETR论文阅读与复现实验报告.md`

**Interfaces:**
- Consumes: 设计文档、Task 1 事实清单、Task 2 图片。
- Produces: 可直接提交给老师的完整报告。

- [ ] **Step 1: 写入报告骨架**

Use these exact top-level sections:

```markdown
# DETR论文阅读与复现实验报告

## 摘要
## 1. 论文方法
## 2. 对论文的理解
## 3. 实验设计与结果
### 3.1 官方预训练权重评估
### 3.2 全量数据20轮训练
### 3.3 10%数据100轮训练
## 4. 工程化复现
## 5. 总结
## 参考文献
```

Expected: 章节覆盖设计文档全部内容。

- [ ] **Step 2: 写方法与个人理解**

Accurately explain:

- CNN backbone and flattened feature sequence;
- positional encoding and Transformer encoder-decoder;
- 100 object queries as learnable prediction slots;
- one-to-one Hungarian matching;
- class, L1 box and GIoU losses;
- segmentation extension;
- removal of anchor design and NMS;
- slow convergence and weak small-object performance.

Expected: 方法来自论文文本，个人理解与事实描述明确区分。

- [ ] **Step 3: 写实验部分**

Include:

- data split, hardware and key settings;
- AP/AP50/AP75 and size-specific AP for official evaluation;
- first/last loss for the 20-epoch full-data run;
- fixed seed 42, 11829 images and four RTX 4090 GPUs for the 100-epoch run;
- complete trend-based overfitting conclusion;
- a compact result table plus four figures with captions.

Expected: 每个数字都能追溯到 Task 1。

- [ ] **Step 4: 写工程化复现与总结**

Mention:

- preserved official Git history;
- public repository `https://github.com/visit1111/detr-reproduction`;
- `reproduce.ps1` and `reproduce.sh`;
- checksum, dataset, metric and run-manifest checks;
- what was learned and what could be improved.

Expected: 不包含连接服务器所需的任何信息。

### Task 4: 验证、提交与发布

**Files:**
- Verify: `reports/DETR论文阅读与复现实验报告.md`
- Verify: `reports/assets/*`

**Interfaces:**
- Consumes: Task 2 and Task 3 outputs.
- Produces: 通过检查的 Git 提交和更新后的 GitHub `main`。

- [ ] **Step 1: 验证报告结构和图片路径**

Run a Python check that:

- finds exactly four Markdown image references;
- resolves every referenced image relative to the report;
- confirms every image is nonempty;
- confirms all required headings are present.

Expected: 输出 `report links and headings: OK`。

- [ ] **Step 2: 验证内容与安全性**

Run:

```powershell
rg -n "tlf123456|115\.156|IdentityFile|BEGIN OPENSSH PRIVATE KEY" reports
git diff --check
```

Expected: `rg` 无匹配，`git diff --check` 退出码为 0。

- [ ] **Step 3: 检查篇幅和图片**

Count Chinese characters and Latin words, targeting approximately 1800–2400 Chinese characters excluding image paths and references. Open the generated loss curve and one composed detection image for final visual QA.

Expected: 正文接近 2000 字，图片清晰。

- [ ] **Step 4: 提交并推送**

Run:

```powershell
git add reports
git commit -m "docs: add DETR reproduction report"
git push student HEAD:main
```

Expected: remote `main` points to the new commit and the GitHub report renders with all images.
