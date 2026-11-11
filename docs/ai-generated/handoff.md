# 协调交接状态

接手的 agent 先读这份，再读 `decisions.md`。这里只记"现在做到哪、下一步是什么"；已定的模型与实验决定在 `decisions.md`。每次状态变化后更新本文件并提交。

最后更新：2026-11-10 CST

## 角色与规矩

- 协调 session 负责盯数据、模型、评测三条线，自己起 subagent，不让用户传话。回复用户用中文。
- 集群任务：每个 sbatch / scontrol 改动都要用户逐个批准后才提交。队列长就等，不在别的集群开备份任务。
- 删除任何数据前先看清目标，再要用户明确批准。
- 训练和它的开发用评测放进同一个 sbatch（训完 → 导出 → 生成 → 打分），不要训完再单独排评测任务：fairshare 已被训练用掉，评测会再等好几天（② 评测 11-08 提交，到 11-10 仍无开始估计）。前提是训练所在集群的评测环境已验证。
- 不准用 a100、g3、g4；cwx 账号非万不得已不用，且每次要批准；Great Lakes 低调用。
- 代码改动一次一小步：先给用户看 diff，批准后再提交。代码风格见仓库根目录 `CLAUDE.md`。
- HF / W&B 密钥只由用户输入。ssh 连接（Duo）过期时请用户在自己终端重连：`! ssh deltaai true`、`! ssh delta true`。

## 机器与路径

- DeltaAI（`ssh deltaai`，GH200，aarch64），账户 `bjki-dtai-gh`，分区 `ghx4`。`W=/work/hdd/bjki/wchen38`（与 Delta 共享文件系统）。
  - 训练 venv：`/work/nvme/bjki/wchen38/venvs/WaveFront-gh`。不要把 venv 放 /work/hdd：import 会卡十几分钟。
  - 计费按 TRES 取最大：每张卡 ≤ 72 CPU、≤ 约 110 GB 内存。
- Delta（`ssh delta`，A40），账户 `bjki-delta-gpu`，评测用 `gpuA40x4`。
- 数据：`$W/data/lmdb/spatialvid/{train,val,train_disparity,val_disparity}`。
- 代码副本：`$W/WaveFront`（① 用）、`$W/WaveFront-data`（② 和数据脚本用，现为 `ee58432`）、`$W/WaveFront-eval`（评测用，现为 `c396005`）。均跟 `feature/model`。
- 任务脚本：`$W/logs/*.sbatch`、`*.sh`；本地副本 `/tmp/wfmon/gh/`。
- checkpoint：HF 仓库 `cwx-umich/wavefront-checkpoints`。流程：导出全部 DCP → 上传 HF → 核对大小 → 只留最后一个 DCP。

## 当前状态

| 项 | 状态 |
|---|---|
| 数据：train / val 主 LMDB | 完成（12,128 / 8 条） |
| 数据：val disparity | 完成（8/8） |
| 数据：train disparity | 完成（2026-11-05）：写入 12,017，跳过 111（全部 inverted_depth），失败 0；std(2d−1)=0.6445，a 仍为 1.5 |
| ① `b1-continue` | 训完（3000 步，val/loss_rgb 0.2009 → 0.1880，W&B `cwxlab/wavefront/0ockapbw`）。`model_{1000,2000,3000}.safetensors` 已上传 HF；DCP 只留 `checkpoint_3000`。导出文件在 `$W/outputs/WaveFront/b1-continue/ckpts/exported/` |
| ② `b2-joint-depth`，DeltaAI 任务 3529078 | **训完**（11-08 10:51–19:28，8 h 37 min，约 34.5 SU，exit 0）。val/loss_rgb 0.2009 → 0.2322（250 步）→ 0.2015（3000 步），① 同步数为 0.1880；分噪声水平 @0.1/0.5/0.9 为 0.513/0.119/0.084（① 0.482/0.110/0.079）。val/disp_err 0.1919 → 0.1328。三个都已导出到 `ckpts/exported/`。HF 有 `model_{1000,2000}`（大小已核对）；HF 私有存储已满，`model_3000` 存在用户 Google Drive `gdrive:wavefront-checkpoints/b2-joint-depth/`（rclone，md5 已核对，见 memory `gdrive-rclone`）。DCP 只留 `checkpoint_3000`（1000/2000 经用户批准于 11-08 删除） |
| ② 的 disparity 分支没训好（2026-11-10 发现） | `val/loss_disp` 停在 1.0（噪声方差），各噪声水平相同；`val/disp_err` = 0.266 × 噪声水平。`model_3000` 里 `disp_embed`、`head.disp_head` 都是秩 1（token 均值进、token 常数出），rms 6.6e-4 / 1.2e-3，RGB 对应层是 3.3e-2 / 2.1e-2。只学到粗深度，噪声没去掉。原因：零初始化 + lr 2e-6，3000 步最多走 6e-3。② 的评测结果要标"深度分支没训好" |
| ② 短跑 `b2-lr-probe`，DeltaAI 任务 3574932 | **排队中**（`ghx4-interactive`，1 卡，1.5 h，约 2 SU；11-10 提交）。`disp_embed.`、`head.disp_head.` 用 50 倍 lr（1e-4），250 步，每 50 步验证。配置 `b2_joint_depth_lr_probe.py`，脚本 `$W/logs/b2-lr-probe.sbatch`，输出 `$W/outputs/WaveFront/b2-lr-probe/`。达标线：`val/loss_disp` 明显低于 1.0，且随噪声水平变化。不达标就试 `disp_embed` 非零初始化 |
| 开发用评测：原版 minWM，8 样例（试跑），K 选项 B | 完成。输出 `$W/outputs/WaveFront/2026-11-04-ws-minwm-k601/`。WorldScore-Static 68.19；VBench 四维均值 subject 0.978、background 0.956、motion 0.994、imaging 0.760（每条在各自 `vbench.json`，`summary.md` 不含 VBench） |
| DeltaAI 额度 | 2026-11-10 查：存入 1042、余 965（② 用约 35）。DeltaAI 上没有我们的任务 |
| 开发用评测：① 40 样例，K 选项 B | **完成**（Delta 任务 23648178，11-08 03:59 开始，10:49 出 summary，40/40 打分无失败）。输出 `$W/outputs/WaveFront/2026-11-07-ws-b1-k601/` |
| 开发用评测：② 40 样例 | **排队中**，Delta 任务 23673998（gpuA40x4，1 卡，10 h），11-10 仍无开始估计，priority 1108（age 766 + fairshare 333），前面约 1,026 个待排任务。输出 `$W/outputs/WaveFront/2026-11-08-ws-b2-k601/`，checkpoint 为 ② 的 `model_3000.safetensors` |
| 开发用评测：原版 minWM 40 样例 | **完成**（Delta 任务 23648597，11-09 12:29–18:50）。WorldScore-Static 64.50（8 样例试跑时 68.19）。输出 `$W/outputs/WaveFront/2026-11-04-ws-minwm-k601/` |

## DeltaAI 评测环境（WorldScore 部分已装好，未上 GPU 验证）

- 用户 2026-11-08 选 (a)：DeltaAI 上搭 WorldScore 环境，搭好后用它给 ① 已生成的 40 个视频重新打分（`WS_STAGE=score`），逐例对比 A40 结果，量出两套环境的差异，再决定能否混用。
- aarch64 上最早的 PyTorch CUDA 轮子是 2.6.0+cu126，所以环境与 Delta 的 2.5.1+cu121 不同；`install_worldscore.sh` 已改（`9376e93`、`90dacde`：torch 2.6、不装 xformers、源码编 torch-scatter、DROID-SLAM 的 `.type()` 补丁）。
- 布局：评测代码 `$W/WaveFront-eval-gh`（`WaveFront-eval` 的 git worktree，detached 在 origin/feature/model），WorldScore `$W/WorldScore-gh`（指标权重软链到 `$W/WorldScore` 的），venv `/work/nvme/bjki/wchen38/venvs/worldscore-gh`。脚本 `$W/logs/setup_worldscore_gh.sh`，日志同名 .log。
- 2026-11-09：`causal_conv1d` / `mamba_ssm` 的 PyPI 安装在 aarch64 上装进 x86 `.so`（setup.py 下载写死 linux_x86_64 的预编译轮子；causal-conv1d sdist 缺 csrc/）。改为从 GitHub tag 源码编（`$W/logs/fix_mamba_gh2.sh`，源码在 `$W/src-gh`），triton 用 3.5.1（PyPI 首个 aarch64 轮子；与 torch 2.6 非官方配套）。全部指标模块 import 通过，权重已预取。同样改法写进 `install_worldscore.sh`，**待用户批准提交**。登录节点无 GPU，指标能否真跑要等 GPU 任务。
- 用户 2026-11-09 决定：Delta 上两个评测不挪（DeltaAI 新 1 卡任务估计 11-19 才开始，fairshare 0.013），也不改 preempt 分区。
- DeltaAI 账户上还有 swift-odeinit（另一个项目，共用 `bjki-dtai-gh` 的 fairshare）。11-09 经用户批准：取消 16 个 p00 冒烟任务，改在登录节点上用 CPU 跑（`$W/logs/p00_local_cpu.sh`）；修好 `cf-smoke.sbatch`（`torchrun` 用容器 python，缺 omegaconf → 改 `python -m torch.distributed.run`），提交到 `ghx4-interactive`（任务 3566977，1.5 h）。同日用户要求取消全部 swift 任务：3566977 未开始即取消；本地 p00 在 data、evalset、kd[0]、kd[1] 完成后被停（launcher 幂等，可续跑）。DeltaAI 上现在没有我们的任务。新的 `ghx4` 任务 11-09 估计要等到 11-19；短任务用 `ghx4-interactive`（2 SU/卡时，限 2 h）排队短得多。
- VBench 还没做：它依赖的 decord 在 aarch64 上没有轮子也没有 sdist，要从源码编（需 FFmpeg 开发库，DeltaAI 上有没有未查）。

## ② 排队慢的原因

DeltaAI 用 Fair Tree。我们账户份额只占全机 0.0021%，① 用了约 41 SU 就超额约 11 倍，fairshare 排名掉到约 1%。用量半衰期 1 天，要几天才恢复；排队时长分 2 天后封顶。份额看起来按额度给（各账户份额多为 100/200/500/1000），所以换到 1000 小时、同步后份额应升到约 1000。细节在 memory `compute-inventory`。

## 下一步

1. **① 对原版 minWM 的配对报告**：11-09 21:45 生成完（`$W/outputs/WaveFront/reports/2026-11-09-minwm-vs-b1/index.html` + `report.json`）。WorldScore-Static：minWM 64.50，① 59.58。40 例配对差（① − minWM，95% CI 不含 0 的）：reproj err +0.105、flow AEPE +0.275（pan 一组 +1.07 拉高）、VBench subject −0.015、background −0.015、motion −0.006；rot err、trans err、imaging 无显著差。orbit_left 的 rot err ① 更好（−4.8°）。② 评测跑完后再加进来。
2. **② 开跑后**：查 step 0 的 `val/loss_rgb` 是否等于 ① 的 0.2009（同噪声），再看 250/500 步的 disp_err、每步时间、显存。日志在 `$W/outputs/WaveFront/b2-joint-depth/slurm-<id>.out`。注意 ② 用 `$W/WaveFront-data`（`1d3513a`），比提交时的 `ac6ce4b` 多了评测脚本和 disparity 构建修复，训练代码没变。
3. **② 训完**：按 checkpoint 流程导出、上传、核对、清理（删除前要用户批准）。
4. **开发用评测（每个任务要批准）**：原版 minWM 的同一 run 扩到 40 样例；① 用 K 选项 B 跑 40 样例（`inference.checkpoint` 指向 `model_3000.safetensors`）；之后 ②。三者齐了跑 `report.py` 出配对 HTML 报告给用户看。

## 开训前检查（2026-11-10 用户确认）

往预训练模型里加新层或新 loss 的 run，提交前逐条过：

1. 每个新指标写下三个参考值（没训、走捷径、学会）和达标线。
2. 新层算 lr × 步数，和预训练对应层的权重量级比；差一个量级就单独设 lr 或换初始化。
3. 正式跑前做一次"能不能学会"的短跑：新分支的 loss 要低于捷径值。
4. 正式跑在 250 或 500 步设关卡，不达标就停。

## 等用户决定

- 评测任务把 venv 换到 nvme（改 `submit_delta.sh`）：最近两次评测光启动就花了近一小时。要先给用户看 diff。
- DeltaAI 上整篇论文要用多少 SU（③ 和放大实验）：还没估。

## 以后再做

- 把线程数限制写进 `build_spatialvid_disparity_lmdb.py`（现在靠调用脚本设 `OMP_NUM_THREADS=1`）。
- ③ 实现；验证集按噪声水平分更细的组报 loss；算力抽象层；拿到 SpatialVID-RAW 权限后做源视频过滤。
