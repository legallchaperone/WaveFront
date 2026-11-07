# 协调交接状态

接手的 agent 先读这份，再读 `decisions.md`。这里只记"现在做到哪、下一步是什么"；已定的模型与实验决定在 `decisions.md`。每次状态变化后更新本文件并提交。

最后更新：2026-11-07 12:48 CST

## 角色与规矩

- 协调 session 负责盯数据、模型、评测三条线，自己起 subagent，不让用户传话。回复用户用中文。
- 集群任务：每个 sbatch / scontrol 改动都要用户逐个批准后才提交。队列长就等，不在别的集群开备份任务。
- 删除任何数据前先看清目标，再要用户明确批准。
- 不准用 a100、g3、g4；cwx 账号非万不得已不用，且每次要批准；Great Lakes 低调用。
- 代码改动一次一小步：先给用户看 diff，批准后再提交。代码风格见仓库根目录 `CLAUDE.md`。
- HF / W&B 密钥只由用户输入。ssh 连接（Duo）过期时请用户在自己终端重连：`! ssh deltaai true`、`! ssh delta true`。

## 机器与路径

- DeltaAI（`ssh deltaai`，GH200，aarch64），账户 `bjki-dtai-gh`，分区 `ghx4`。`W=/work/hdd/bjki/wchen38`（与 Delta 共享文件系统）。
  - 训练 venv：`/work/nvme/bjki/wchen38/venvs/WaveFront-gh`。不要把 venv 放 /work/hdd：import 会卡十几分钟。
  - 计费按 TRES 取最大：每张卡 ≤ 72 CPU、≤ 约 110 GB 内存。
- Delta（`ssh delta`，A40），账户 `bjki-delta-gpu`，评测用 `gpuA40x4`。
- 数据：`$W/data/lmdb/spatialvid/{train,val,train_disparity,val_disparity}`。
- 代码副本：`$W/WaveFront`（① 用）、`$W/WaveFront-data`（② 和数据脚本用，现为 `1d3513a`）、`$W/WaveFront-eval`（评测用，现为 `c396005`）。均跟 `feature/model`。
- 任务脚本：`$W/logs/*.sbatch`、`*.sh`；本地副本 `/tmp/wfmon/gh/`。
- checkpoint：HF 仓库 `cwx-umich/wavefront-checkpoints`。流程：导出全部 DCP → 上传 HF → 核对大小 → 只留最后一个 DCP。

## 当前状态

| 项 | 状态 |
|---|---|
| 数据：train / val 主 LMDB | 完成（12,128 / 8 条） |
| 数据：val disparity | 完成（8/8） |
| 数据：train disparity | 完成（2026-11-05）：写入 12,017，跳过 111（全部 inverted_depth），失败 0；std(2d−1)=0.6445，a 仍为 1.5 |
| ① `b1-continue` | 训完（3000 步，val/loss_rgb 0.2009 → 0.1880，W&B `cwxlab/wavefront/0ockapbw`）。`model_{1000,2000,3000}.safetensors` 已上传 HF；DCP 只留 `checkpoint_3000`。导出文件在 `$W/outputs/WaveFront/b1-continue/ckpts/exported/` |
| ② `b2-joint-depth`，DeltaAI 任务 3529078 | **排队中**（2026-11-04 13:58 提交，时间上限已改为 11 h）。优先级 1131 = fairshare 122 + 排队时长 1000（已满）+ 分区 10 |
| 开发用评测：原版 minWM，8 样例（试跑），K 选项 B | 完成。输出 `$W/outputs/WaveFront/2026-11-04-ws-minwm-k601/`。WorldScore-Static 68.19；VBench 四维均值 subject 0.978、background 0.956、motion 0.994、imaging 0.760（每条在各自 `vbench.json`，`summary.md` 不含 VBench） |
| DeltaAI 额度 | 用户已在 ACCESS 换到 1000 小时（2026-11-07），集群上**尚未同步**：`accounts` 仍显示存入 200、余 158，Slurm 份额仍为 200 |

## ② 排队慢的原因

DeltaAI 用 Fair Tree。我们账户份额只占全机 0.0021%，① 用了约 41 SU 就超额约 11 倍，fairshare 排名掉到约 1%。用量半衰期 1 天，要几天才恢复；排队时长分 2 天后封顶。份额看起来按额度给（各账户份额多为 100/200/500/1000），所以换到 1000 小时、同步后份额应升到约 1000。细节在 memory `compute-inventory`。

## 下一步

1. **额度同步**：查 `accounts` 存入额是否变 1000、`sshare -A bjki-dtai-gh` 的 RawShares 是否随之变化。一两天不变就建议用户给 NCSA help 发邮件。
2. **② 开跑后**：查 step 0 的 `val/loss_rgb` 是否等于 ① 的 0.2009（同噪声），再看 250/500 步的 disp_err、每步时间、显存。日志在 `$W/outputs/WaveFront/b2-joint-depth/slurm-<id>.out`。注意 ② 用 `$W/WaveFront-data`（`1d3513a`），比提交时的 `ac6ce4b` 多了评测脚本和 disparity 构建修复，训练代码没变。
3. **② 训完**：按 checkpoint 流程导出、上传、核对、清理（删除前要用户批准）。
4. **开发用评测（每个任务要批准）**：原版 minWM 的同一 run 扩到 40 样例；① 用 K 选项 B 跑 40 样例（`inference.checkpoint` 指向 `model_3000.safetensors`）；之后 ②。三者齐了跑 `report.py` 出配对 HTML 报告给用户看。

## 等用户决定

- 评测任务把 venv 换到 nvme（改 `submit_delta.sh`）：最近两次评测光启动就花了近一小时。要先给用户看 diff。
- 原版 minWM 扩到 40 样例、① 的 40 样例评测：等批准后提交。
- DeltaAI 上整篇论文要用多少 SU（③ 和放大实验）：还没估。

## 以后再做

- 把线程数限制写进 `build_spatialvid_disparity_lmdb.py`（现在靠调用脚本设 `OMP_NUM_THREADS=1`）。
- ③ 实现；验证集按噪声水平分更细的组报 loss；算力抽象层；拿到 SpatialVID-RAW 权限后做源视频过滤。
