# WorldScore 子集评测

用 WorldScore（ICCV 2025）评一个 minWM 系列的 checkpoint：原版 minWM、①、②、③ 都走这一套。对应 proposal 7.1 的"WorldScore 子集，按相机运动类型分开报"。

## 评什么

- **静态**：写实风格、只有一段相机运动的样例，8 种运动（push_in、pull_out、move_left/right、orbit_left/right、pan_left/right）× 室内/室外 × 每组 6 个，共 96 个。
- **动态**：写实风格，5 种运动类型（articulated、deformable、fluid、multi_motion、rigid）× 每类 5 个，共 25 个，相机固定。
- 每个样例生成 77 个像素帧（20 个 latent 帧，5 个 chunk），832×480，16 fps。
- 子集按固定种子抽，同样的参数抽到同样的样例，写在 `$OUT_DIR/subset/`。改了抽样参数就换一个 run 名。

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

不用 Delta 时，用同目录的 `colab.ipynb`：在 Colab 里打开（「文件 → 从 GitHub 打开」选这个分支的这个文件，或上传），运行时选 **A100**，从上往下运行。它装好两套环境、下载权重和数据，再调用同一个 `run.sh`。输出存在 Google Drive，会话断了重跑全部单元格即可接着做。

- **卡**：A100 40 GB。推理时 T5 编码器（11 GB）、DiT、CFG 的两份 KV cache（约 11.5 GB）和 VAE 解码都在卡上，估计峰值 30 GB 以上，L4（24 GB）和 T4（16 GB，且不支持 bf16）不行。H100 可以，笔记本会给 DROID-SLAM 补编 sm_90。
- **每个新会话**重装环境约 30–40 分钟（编译 DROID-SLAM 最久）。
- 冒烟和完整子集用同一个 RUN_NAME，冒烟生成的 2 个样例会直接沿用。

## 一次性准备（Delta）

以下没有在 Delta 上实际跑过，第一次照做时请留意报错。

1. **WorldScore 代码**，固定在接入时读过的 commit：

   ```bash
   git clone https://github.com/haoyi-duan/WorldScore.git $WORK/WorldScore
   git -C $WORK/WorldScore checkout 096c75fc4ada9c7d92e03140c3c1f0c8f383b61f
   ```

2. **WorldScore 的评测环境**：按它 README 的 "Evaluation → Environment Setup" 一节装（Python 3.10、torch 2.5.1 + cu121、编译 DROID-SLAM、Grounded-SAM、SAM2、VFIMamba，最后 `pip install .`）。DROID-SLAM 要编 CUDA 扩展，在 GPU 节点上（`srun`）装。环境建议放在 `$WORK/venvs/worldscore`，例如 `conda create -p $WORK/venvs/worldscore python=3.10`。它和 minWM 的环境（`$WORK/venvs/WaveFront`）分开，因为 torch 版本不同。

3. **指标用的 checkpoint**：按它 README 的 "Download Evaluation Checkpoints" 一节下载，放到 `$WORK/WorldScore/worldscore/benchmark/metrics/checkpoints/`。

4. **数据集**（1.2 GB）：

   ```bash
   mkdir -p $WORK/data && cd $WORK/data
   wget https://huggingface.co/datasets/Howieeeee/WorldScore/resolve/main/WorldScore-Dataset.zip
   unzip -q WorldScore-Dataset.zip && rm WorldScore-Dataset.zip
   ```

5. **minWM 的权重**：Wan 2.1 T2V 1.3B 的 base 和 Stage 1 权重，见仓库根目录 README 的 "Download the demo checkpoints"，Stage 1 的权重是 `Wan21/Action2V/ar_diffusion_tf`，链接成 `ckpts/Wan21/Action2V/stage1_ar_tf`。

6. **环境变量**，写进 `~/.bashrc`：

   ```bash
   export WS_ROOT=$WORK/WorldScore
   export WS_DATASET=$WORK/data/WorldScore-Dataset
   export WS_PYTHON=$WORK/venvs/worldscore/bin/python
   ```

## 运行

在仓库根目录，用 run-submit 提交。评测放可抢占的 `gpuA40x4-preempt`；被抢占就用同一个 run 名重交，每一步都会跳过已经做完的部分。

先冒烟，只生成并评 2 个样例：

```bash
run-submit 2026-11-01-ws-smoke 1 01:00:00 --partition=gpuA40x4-preempt -- \
  bash tools/eval/worldscore/run.sh configs/wan21/action2v/infer/stage1_ar_tf.py inference.limit=2
```

原版 minWM（Stage 1 权重，50 步 UniPC，CFG 3.0）的完整子集：

```bash
run-submit 2026-11-01-ws-minwm-s1 4 08:00:00 --partition=gpuA40x4-preempt -- \
  bash tools/eval/worldscore/run.sh configs/wan21/action2v/infer/stage1_ar_tf.py
```

- `run.sh` 第一个参数是 minWM 的推理 config，后面的参数原样传给 `tools/infer_mwm.py`，例如换权重：`inference.checkpoint=<路径>`。
- 换成 4 步的 DMD 权重只需换 config：`configs/wan21/action2v/infer/stage3_ar_dmd.py`，快得多，适合先看一眼流程。
- `WS_PREPARE_ARGS` 可以改抽样，例如 `env WS_PREPARE_ARGS="--static-per-group 2" bash tools/eval/worldscore/run.sh ...`。改了就换 run 名。
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

把视频拉回 Mac 看：

```bash
rsync -a --include='*/' --include='output.mp4' --exclude='*' \
  delta:<OUT_DIR>/worldscore_output/ ./local/ws_videos/
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
