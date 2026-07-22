# DETR 实验复现说明

本分支在官方 DETR 提交 `29901c51` 的完整 Git 历史上增加了两个可验证入口：Windows 单卡评估官方 DETR-R50，以及 Linux 四卡训练 COCO 10% 子集。当前复现范围是目标检测；实例/全景分割仍保留官方代码和说明，但不属于这两个入口的验收结果。

## 1. 从 Git 获取代码

将下面地址替换为本项目发布后的学生仓库地址：

```bash
git clone <你的仓库地址> detr-repro
cd detr-repro
git checkout feat/reproducible-workflows
```

数据、权重、虚拟环境和训练输出均被 `.gitignore` 排除。COCO 数据不能通过 Git 一起下载，需要单独准备。

数据目录应为：

```text
coco2017/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/                # 训练时需要，118287 张
└── val2017/                  # 评估时需要，5000 张
```

## 2. Windows：复现官方 DETR-R50 评估

环境要求：Windows 11、PowerShell、Python 3.10–3.12 和 NVIDIA 显卡。脚本默认创建 `.venv-repro`，安装 PyTorch 2.6.0、torchvision 0.21.0 和 CUDA 12.4 版本依赖，并校验官方权重的 SHA-256。

先做快速检查，不运行 5000 张图片的完整评估：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\reproduce.ps1 -CocoPath D:\datasets\coco2017 -QuickCheck
```

完整评估并生成可视化图片：

```powershell
.\reproduce.ps1 `
  -CocoPath D:\datasets\coco2017 `
  -OutputDir D:\detr-results\official-r50-val2017 `
  -Visualize
```

已有经过验证的环境时，可用 `-PythonExecutable` 指定 Python，并配合 `-SkipInstall` 跳过安装。脚本若发现已有 `eval_metrics.json` 会拒绝覆盖，请换一个输出目录。

参考结果允许绝对误差 ±0.2 AP：

- AP = 42.0
- AP50 = 62.4
- AP75 = 44.2

RTX 4060 Laptop GPU 上完整评估通常约 10–30 分钟，首次运行还需要下载约 159 MB 权重并安装依赖。结果目录包含 `eval_metrics.json`、`run_manifest.json`、COCO evaluator 文件以及可选的 `visualizations/`。

## 3. Linux：四卡训练 10% COCO

训练脚本从 `configs/detection_10pct_100ep.json` 读取唯一配置。它按图像 ID 排序后使用 seed 42 确定性抽样，得到 11829 张训练图像；原始图片通过符号链接复用，不会复制或修改 COCO 源数据。

先在服务器快速验证。该命令检查完整数据数量、Git 状态、依赖和单元测试，不启动 GPU 训练：

```bash
bash reproduce.sh \
  --coco-path /data/your-name/data/coco2017 \
  --python /path/to/python \
  --skip-install \
  --quick-check
```

进行一次一轮 smoke test：

```bash
bash reproduce.sh \
  --coco-path /data/your-name/data/coco2017 \
  --output-dir /data/your-name/output/detr-smoke \
  --smoke
```

正式运行四卡、10% 数据、100 轮实验：

```bash
bash reproduce.sh \
  --coco-path /data/your-name/data/coco2017 \
  --subset-root /data/your-name/data/coco2017_train10_seed42 \
  --output-dir /data/your-name/output/detr-r50-10pct-100ep
```

默认每卡 batch size 4、每进程 8 个数据加载线程、4 个进程，学习率在第 80 轮下降。四张 RTX 4090 的实际耗时受机械盘吞吐和共享负载影响，通常可先按约 6–12 小时估计；一轮 smoke test 一般为数分钟。10% 数据实验的主要目标是验证 loss 持续下降并观察能否过拟合，不应期待达到官方全量 COCO 500 轮的 AP。

脚本在前台运行，退出码会直接反映训练成功或失败。输出包含 `checkpoint.pth`、各轮 checkpoint、`log.txt` 和 `run_manifest.json`。已有输出不会被静默覆盖。

中断后续训：

```bash
bash reproduce.sh \
  --coco-path /data/your-name/data/coco2017 \
  --subset-root /data/your-name/data/coco2017_train10_seed42 \
  --output-dir /data/your-name/output/detr-r50-10pct-100ep \
  --resume /data/your-name/output/detr-r50-10pct-100ep/checkpoint.pth \
  --skip-install
```

## 4. 怎样判断“可复现”

一次结果不能只看终端没有报错，还应同时核对：

1. `run_manifest.json` 中的 Git commit、命令、Python 和依赖版本是否与报告一致。
2. Windows 的 `eval_metrics.json` 是否在参考容差内。
3. Linux 的 `log.txt` 中 loss 是否下降，是否出现 `NaN`、OOM、NCCL 错误或 Traceback。
4. 使用的子集是否为 fraction 0.1、seed 42、11829 张图片。
5. `git status --short` 是否没有把数据、权重或输出误提交进仓库。

## 5. 常见问题

- `expected 5000 images`：检查是否把 COCO `val2017` 和 `instances_val2017.json` 放在同一数据根目录下。
- `checksum mismatch`：权重下载不完整或文件并非官方 DETR-R50，脚本会隔离错误文件并停止。
- `Need 4 CUDA GPUs`：检查 `nvidia-smi` 和 `CUDA_VISIBLE_DEVICES`，或等其他任务释放 GPU。
- `Refusing to overwrite`：使用新输出目录；只有确实续训时才传 `--resume`。
- 读取机械盘较慢：10% 子集已避免图片复制；若 GPU 利用率长期偏低，可把 COCO 放到本人的 SSD 目录后重新指定 `--coco-path`。
- Windows 首次安装慢：后续会复用 `.venv-repro`，不需要重复安装。

仓库的快速测试可独立执行：

```bash
python -m unittest discover -s tests -v
```
