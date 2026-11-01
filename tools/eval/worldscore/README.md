# WorldScore 子集评测

用 WorldScore（ICCV 2025）评一个 minWM 系列的 checkpoint：原版 minWM、①、②、③ 都走这一套。对应 proposal 7.1 的"WorldScore 子集，按相机运动类型分开报"。

## 评什么

全部用写实风格。每个样例生成 77 个像素帧（20 个 latent 帧，5 个 chunk），832×480，16 fps。子集大小由 `WS_PREPARE_ARGS`（传给 `prepare.py`）决定：

| 规模 | 内容 | 个数 | `WS_PREPARE_ARGS` |
|---|---|---|---|
| 最小 | 4 种运动 × 室内/室外 × 1 | 8 | `--moves push_in,move_left,orbit_left,pan_left --static-per-group 1 --dynamic-per-group 0` |
| **小 benchmark**（`run_small.sh`，Colab 默认） | 5 种运动 × 室内/室外 × 4 | 40 | 由 `run_small.sh` 固定：`--moves push_in,pull_out,move_left,orbit_left,pan_left --static-per-group 4 --dynamic-per-group 0` |
| 完整（proposal 7.1） | 静态 8 种运动 × 室内/室外 × 6，加动态 5 类 × 5 | 121 | 空（默认） |

- **小 benchmark 怎么选的**：move、orbit、pan 的左右两个方向在相机轨迹上是镜像，只留左边。前后两个方向不是镜像：push_in 留在原场景里，pull_out 往后退，画面边缘会露出没见过的内容，WorldScore 把它和 move、pan 一样算作"走进新场景"，所以两个都留。push_in、pull_out、move_left、orbit_left 有视差，pan_left 只转不移、没有视差，用来做"有视差 / 无视差"的对比。动态子集（相机固定）用来检查各方法的动态有没有掉，只评原版 minWM 时用不上，比较方法时再加。
- **小 benchmark 的样本量**：每种运动 8 个，"有视差"一行 32 个、"无视差"一行 8 个，10 个场景类别都覆盖到。够看各种运动的趋势、看视频；单种运动之间差几分仍说明不了问题，正式比较方法用完整规模。
- **子集只增不减**：同一个种子下，小的子集包含在大的里面。同一个 run 名先跑小的，再把 `WS_PREPARE_ARGS` 改大重跑，只会生成和评新增的样例，已有的直接沿用。改小、换风格或换种子会报错，要换 run 名。抽到的样例记在 `$OUT_DIR/subset/`。
- **生成顺序**：每组先出一个，再每组出第二个，以此类推。中途停掉，已完成的部分也均匀覆盖各种运动。

## 怎么接的

- **prompt**：原样用 WorldScore 给视频模型的 prompt，例如 `Camera move left. <场景描述>. photorealistic. Everything in the scene remains completely motionless.`
- **相机**：不靠 prompt 里的文字。WorldScore 为每个样例生成真值轨迹（`camera_data.json`，每个像素帧一个相机），打分时拿它和 DROID-SLAM 从视频里估出的轨迹比。这里把同一条真值轨迹换算成 minWM 的输入：每个 latent 帧一个 w2c 位姿（OpenCV 约定，第一个为单位阵），第 k 个 latent 帧取第 4k 个像素帧的相机，通过 benchmark JSON 的 `poses` 字段交给 PRoPE。
  - **平移**乘 1.52，使 push_in / pull_out 恰好等于 minWM 的 `w*19` / `s*19`，也就是训练时的步长（每个 latent 帧 0.08）。move 的距离是 push 的一半，orbit 的半径等于 push 的距离，比例与 WorldScore 一致。WorldScore 的平移误差会先拟合尺度再算，所以这个系数不影响打分的公平性，只决定让模型走多远。
  - **旋转**不缩放（旋转误差按角度算）：pan 和 orbit 都转 30°，pan_left 恰好等于 minWM 的 `j*10`。
  - `prepare.py` 会逐个检查换算后的相机是否朝名字所说的方向运动，约定弄反了会直接报错，不会悄悄生成反向的视频。
- **T2V**：和 WorldScore 论文对 T2V 模型的处理一样，不用输入图，第 0 帧由 prompt 生成。
- **帧**：minWM 以 `inference.save_frames=True` 把每帧存成无损 PNG，避免打分时把 H.264 压缩的瑕疵也算进去。
- **打分**：每个样例的指标直接调 WorldScore 的 `process_batch`，与官方评测同一份代码。样例目录（`image_data.json`、`camera_data.json`、`input_image.png`）由 WorldScore 自己的 dataloader 写出。模型在进程内注册，WorldScore 仓库一个文件都不改。

## Colab

不用 Delta 时，用同目录的 `colab.ipynb`（默认调用 `run_small.sh`）：在 Colab 里打开（「文件 → 从 GitHub 打开」选这个分支的这个文件，或上传），运行时选 **A100**，从上往下运行全部单元格。输出存在 Google Drive，会话断了重跑全部单元格即可接着做。

- **卡**：A100 40 GB。推理时 T5 编码器（11 GB）、DiT、CFG 的两份 KV cache（约 11.5 GB）和 VAE 解码都在卡上，估计峰值 30 GB 以上，L4（24 GB）和 T4（16 GB，且不支持 bf16）不行。H100 可以，笔记本会给 DROID-SLAM 补编 sm_90。
- **省时间**：WorldScore 环境的编译（约 30–40 分钟，主要是 DROID-SLAM）只吃 CPU，笔记本让它在后台跑，同时在前台装 minWM、下载权重、生成视频（`WS_STAGE=gen`），生成完再等编译结束、打分（`WS_STAGE=score`）。`prepare.py` 直接从 `WS_ROOT` 导入 WorldScore 的纯 Python 部分，所以生成不用等编译。
- **耗时估计**（A100，未实测）：小 benchmark 约 2.5–4 小时（装环境和下载 10 分钟、生成 40 × 2–3 分钟、打分 40 × 1.5–3 分钟，编译藏在生成后面），完整子集约 9–11 小时。

## 一次性准备（Delta）

`setup_delta.sh` 在 `$WORK` 下一次装好全部东西：minWM 环境（`venvs/WaveFront`）、WorldScore 环境（`venvs/worldscore`）、WorldScore 代码和指标权重、数据集、Wan 2.1 base 和 Stage 1 权重（`ckpts/`，仓库里的 `ckpts` 链接过去）。在登录节点后台跑，不需要 GPU，约 1 小时，大半是下载和编译 DROID-SLAM：

```bash
export WORK=/work/hdd/<项目>/$USER          # 也写进 ~/.bashrc
git clone -b feature/benchmark https://github.com/legallchaperone/WaveFront.git $WORK/WaveFront
cd $WORK/WaveFront
nohup bash tools/eval/worldscore/setup_delta.sh > $WORK/setup_worldscore.log 2>&1 &
```

- **为什么在登录节点**：NCSA 的说明把装软件和传文件放在登录节点；计算节点能不能上网，文档里没有确切说法。
- **CUDA**：WorldScore 环境用 torch 2.5.1+cu121，编译扩展要 CUDA 12 的 nvcc。Delta 默认加载的 `cudatoolkit` 是 CUDA 13.2，脚本会把它换成 `module load cuda/12.9`。
- **之后只需要 `WORK`**：`run.sh` 的 `WS_ROOT`、`WS_DATASET`、`WS_PYTHON` 在设了 `WORK` 时默认指向这套布局。
- 中途失败就重跑同一条命令，做完的步骤会跳过。WorldScore 环境单独重装用 `install_worldscore.sh`（Colab 也用它）。
- 没装 flash-attn：minWM 的 attention 在没有它时自动退回 PyTorch 的 SDPA。

## Delta 的使用规则

按 NCSA 的 Delta 用户手册（[Data Management](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html)、[Login](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html)、[Running Jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html)、[Software](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/software.html)）核对过：

- **放哪里**：
  - 代码、两个 Python 环境、权重、数据集、缓存和输出都在 `$WORK`（`/work/hdd/<项目>/$USER`）。手册说这里是"where I/O from jobs should occur"。
  - `$HOME`（100 GB、75 万个文件）"not intended as a source/destination for I/O during jobs"。所以安装和作业里的 uv、pip、Hugging Face、torch 缓存都由 `delta_env.sh` 指到 `$WORK/.cache`，Triton 的编译缓存放在计算节点本地的 `/tmp`（作业结束即清空）。
  - `/work` 没有快照和备份，Delta 也不提供归档：结果（`summary.md`、`evaluation.json`、视频）要拉回本地保存。
- **登录节点**：每人最多用 16 个核、37 GB 内存（硬上限 62 GB，没有 swap）。`setup_delta.sh` 在这里下载和编译，编译用 8 个并行任务。生成和打分只在作业里跑；不要在登录节点跑 Jupyter。
- **计算节点能不能上网**：手册没写。所以 `setup_delta.sh` 在登录节点就把作业要用的全部权重下好，包括各指标第一次运行时才会下载的那些（`prefetch_metric_weights.py`），作业里设 `HF_HUB_OFFLINE=1`。
- **Slurm**：
  - 可抢占分区手册"only recommended for jobs that include checkpointing"。这里按样例续跑：被抢占最多丢掉正在做的那一个样例；`--requeue` 让作业自动重新排队，重排后脚本从头开始，做完的会跳过。
  - 每张卡配 16 个核、60 GB 内存。A40 节点有 64 个核、约 243 GB 可分配内存，4 张卡（240 GB）刚好放得下。
  - 单个作业最长 48 小时。interactive 分区只用于 1 小时内的调试，每人同时只能有一个作业。

## 运行

在 Delta 的仓库根目录用 `submit_delta.sh` 提交，它生成作业脚本、交给 sbatch，所有输出放进 `$WORK/outputs/WaveFront/<run 名>/`（`summary.md`、样例目录和视频、`slurm-<作业号>.out`、提交的作业脚本 `job-<作业号>.sbatch`）。默认用可抢占的 `gpuA40x4-preempt`（扣费系数 0.25），被抢占会自动重新排队；作业失败或超时就用同一个 run 名重交，每一步都会跳过已经做完的部分。

**小 benchmark**：`run_small.sh` 固定那 40 个样例，默认评原版 minWM（Stage 1 权重，50 步 UniPC，CFG 3.0）。耗时估计（未实测）一张 A40 5–8 小时，两张大约减半；默认给 10 小时。

先冒烟，只生成并评 2 个，顺便量出每个样例的实际耗时：

```bash
TIME=01:00:00 bash tools/eval/worldscore/submit_delta.sh 2026-11-01-ws-minwm-s1 run_small.sh inference.limit=2
```

没问题就用同一个 run 名交完整的小 benchmark，冒烟做完的 2 个直接沿用：

```bash
bash tools/eval/worldscore/submit_delta.sh 2026-11-01-ws-minwm-s1
```

换模型就在 run 名后面写 `run_small.sh`、推理 config 和覆盖项，例如 `... 2026-11-02-ws-x run_small.sh configs/wan21/action2v/infer/stage1_ar_tf.py inference.checkpoint=<路径>`。

**扩到完整子集**（121 个），用同一个 run 名改用 `run.sh`，已有的 40 个直接沿用：

```bash
NGPU=4 TIME=12:00:00 bash tools/eval/worldscore/submit_delta.sh 2026-11-01-ws-minwm-s1 run.sh \
  configs/wan21/action2v/infer/stage1_ar_tf.py
```

- Slurm 参数用环境变量改：`ACCOUNT`（默认 `bjki-delta-gpu`）、`PARTITION`、`NGPU`（每卡配 16 个 CPU、60 GB 内存）、`TIME`。
- 看进度：`squeue -u $USER`，`tail -f $WORK/outputs/WaveFront/<run 名>/slurm-<作业号>.out`。
- `run.sh` 第一个参数是 minWM 的推理 config，后面的参数原样传给 `tools/infer_mwm.py`，例如换权重：`inference.checkpoint=<路径>`。
- 换成 4 步的 DMD 权重只需换 config：`configs/wan21/action2v/infer/stage3_ar_dmd.py`，快得多，适合先看一眼流程。
- `WS_PREPARE_ARGS` 改抽样，见上面"评什么"的表；只能改大，改小要换 run 名。
- **一个 run 名只评一个模型**：第一次生成时把推理 config 和覆盖项记进 `gen_args.txt`，之后换了 config 或权重再接着跑会报错，免得一张表里混进两个模型（`inference.limit` 不算）。
- `WS_STAGE=gen` 只准备和生成，`WS_STAGE=score` 只打分和汇总（不需要 config 参数），默认 `all` 两者都做。
- 生成和打分都是每张卡一个进程。耗时还没实测：粗估 Stage 1 在 A40 上每个视频几分钟，WorldScore 每个样例也是几分钟（DROID-SLAM 要跑两遍）。冒烟之后用实测值重新估 walltime。

## 输出（`$OUT_DIR`）

| 路径 | 内容 |
|---|---|
| `summary.md` / `summary.json` | 按相机运动类型汇总的表 |
| `worldscore_output/static/<风格>/<室内外>/<类别>/<名字>/` | 一个静态样例：`videos/output.mp4`（看视频）、`frames/`、`camera_data.json`（真值相机）、`evaluation.json`（各指标） |
| `worldscore_output/dynamic/<风格>/<运动类型>/<名字>/` | 一个动态样例，同上，没有 `camera_data.json` |
| `gen/manifest.json` | minWM 推理的完整 config 和每条的 seed |
| `bench.json` | 最近一次交给 minWM 的条目（只含当时还没生成的） |
| `eval-shard*.log` | 打分日志，某个样例报错时看这里 |
| `subset/`、`worldscore_commit.txt` | 抽到的样例、WorldScore 的 commit |
| `gen_args.txt` | 第一次生成时的推理 config 和覆盖项 |
| `slurm-<作业号>.out`、`job-<作业号>.sbatch` | Delta 作业日志和提交的作业脚本（`submit_delta.sh` 写） |

把视频拉回 Mac 看：

```bash
rsync -a --include='*/' --include='output.mp4' --exclude='*' \
  delta:/work/hdd/bjki/wchen38/outputs/WaveFront/<run 名>/worldscore_output/ ./local/ws_videos/
```

## 读 summary

- 每列 aspect（camera_control、3d_consistency 等）是 WorldScore 归一化后的分数 ×100，越高越好，汇总方式与 WorldScore 相同。content_alignment 只对 pull_out、move、pan 这类"走进新场景"的运动计算。
- 原始指标列越低越好：`rot err (deg)` / `trans err` 是相机误差（平移已拟合尺度），`reproj err` 是 DROID-SLAM 的重投影误差（3D 一致性），`flow AEPE` 是前向-后向光流误差（光度一致性）。
- **有视差 / 无视差**两行对应 proposal 7.1 的机制检验：只有平移产生视差，pan 只转不移。先近后远的收益应该集中在"有视差"一行。
- `sentinels` 是 WorldScore 某个指标内部出错、记了占位分（±100）的次数。归一化分数照 WorldScore 的做法把它算成最差，原始指标列不计入。不为 0 时去 `eval-shard*.log` 找原因。`scored < total` 表示有样例还没生成或没打完分。
- 顶上的 WorldScore-Static / -Dynamic 用的是官方公式，但只在这个子集上算，又是 T2V、77 个像素帧，不能和榜单直接比。各行之间（原版 minWM、①、②、③）可以比。

## 已知的偏差

这些对所有方法一样，影响绝对数值，不影响方法之间的比较：

- WorldScore 的 DROID-SLAM 对所有模型都用固定内参（`[500, 500, 256, 256]`），minWM 生成时的内参是归一化焦距 0.5（水平视场 90°）。
- WorldScore 的 pan 和 orbit 在 77 个像素帧里转 30°，每个 latent 帧约 1.6°，比 minWM 训练时的 3° 慢。minWM Stage 1 训练时每个 latent 帧只做一种动作，边移边转的 orbit 和完全不动的相机（动态子集）都没见过。
- prompt 里的 "Camera move left." 这类文字为了与 WorldScore 的协议一致而保留，minWM 的相机实际由 PRoPE 控制。
