# 设计决定

## 2026-10-31：基线 ②（联合深度）

基线 ② 见提案 7.2：在 Stage 1 模型上加一个 disparity 通道，与 RGB 用同一个噪声水平一起去噪，训练时 chunk 内噪声统一（s=0）。

1. **disparity 作为第 17 个 latent 通道。** 放在 latent 网格上（60×104，每个值对应 8×8 像素），和 16 个 RGB 通道拼成一个张量。加噪、采样器、CFG、KV refresh 都不用改，只改模型，以及 VAE 解码前切掉这个通道。
   - 与提案 4.3 不同：新增 `disp_embed = Conv3d(1→1536, 1×2×2)` 和 `disp_head = Linear(1536→4)`，都零初始化。③ 需要每个 token 一个深度值时，取 2×2 的最大值。
2. **loss = RGB 项 + λ·disparity 项，λ = 1。** 两项各自在自己的元素上取 MSE 均值，所以每个 token 的 RGB（64 个值）和 disparity（4 个值）权重相同。两项分开记日志；按噪声水平分档的统计以后再加。
3. **disparity 缩放 a = 1（提案默认 2）。** 网络输入为 a·(2d−1)，d ∈ [0,1]。每个 token 有 4 个独立加噪的 disparity 值，深度在 2×2 内变化小，模型平均后有效幅度约翻倍，所以这里的 a=1 约等于提案的 a=2。

## 2026-11-01：SpatialVID 数据格式（数据、模型、评测三条线共用）

数据线负责写 LMDB，模型线负责读，评测线生成视频时输入的相机位姿也按这里的规则来。这一节一改，三条线要一起改。标【待定】的项由括号里写的那条线定下来再填；全部填完之前，不写正式的 LMDB。

**片段**

1. 训练片段长 45 个像素帧，即 12 个 latent 帧、3 个 chunk；16 fps，480×832（提案 6.2）。原始帧率怎么重采样、宽高比怎么裁剪【待定·数据线】。
2. 评测线现在按 20 个 latent 帧（77 个像素帧）生成，见 `tools/eval/worldscore/_common.py` 里的 `NUM_LATENT_FRAMES`，和训练长度不同。提案 6.2 允许评测长度和训练长度分开定，但所有方法必须一致；评测长度在提案 7.1 里定，不在这一节定。
3. 训练集和验证集分成两个 LMDB：`spatialvid/train` 和 `spatialvid/val`。验证集 8 段，两个集合不能出自同一个源视频。

**LMDB 键**：沿用 `tools/data/wan21/build_worldplaygen_lmdb.py` 的写法，每个键有一个 `<key>_shape`，每个样本有一个 `<key>_<i>_data`。

| 键 | 每个样本的形状 | dtype | 内容 |
|---|---|---|---|
| `latents` | (12, 16, 60, 104) | float16 | Wan VAE latent |
| `prompts` | 标量 | str | caption |
| `intrinsics` | (4,) | float32 | [fx/W, fy/H, cx/W, cy/H]，W、H 是裁剪和缩放之后的 832、480 |
| `poses` | (12, 7) | float32 | w2c，OpenCV 坐标系，[tx,ty,tz,qx,qy,qz,qw] |
| `disparity` | (12, 1, 60, 104) | float16 | d ∈ [0,1]，1 最近，0 无穷远 |
| T5 嵌入 | 【待定·模型线】 | | |

**相机位姿**

1. 第 k 个 latent 帧用第 4k 个像素帧的位姿，并对齐到第 0 个像素帧，所以第 0 个位姿是单位阵。评测线 `worldscore_to_minwm_poses` 的取法（`c2w[4·arange(F)]`，对齐到第一帧）与此相同。
2. 内参每段只存一个，数据集类会把它铺到所有帧。SpatialVID 的内参如果逐帧不同，取哪一帧的【待定·数据线】。
3. 平移缩放：所有平移乘同一个缩放系数，让量级接近基座（约每个 latent 帧 0.08 单位，提案 6.2）。用全局常数还是逐段系数（× 第一个 chunk 的 P₉₀ disparity）【待定·数据线，正在 group_0001 上检验】。
   - 评测线的 `translation_scale` 必须用同一条规则。如果选逐段系数，评测时也得知道第一个 chunk 的 P₉₀，可 WorldScore 并不提供深度。
   - 这个系数不要叫 c，术语表里的 c 是"进度参数"。
4. 转换完用 `tools/data/wan21/check_dataset_bev.py` 把轨迹和视频画在一起检查。

**disparity**

1. 深度 z 换成 1/z，天空和无效深度记为 0。
2. 因果归一化：d = clip(0.9·(1/z)/P₉₀, 0, 1)。P₉₀ 是第一个 chunk（前 13 个像素帧）内有效像素 disparity 的 90% 分位数，后续 chunk 沿用同一个 P₉₀（提案 6.2）。
3. 空间：在每个 latent 格覆盖的 8×8 像素上做高分位聚合（90%），得到 60×104。提案 6.2 原本是在每个 token 的 16×16 像素上聚合，得到 30×52；为什么改见 2026-10-31 第 1 条。③ 需要每个 token 一个值时，取 2×2 的最大值。
4. 时间：第 0 个 latent 帧对应第 0 个像素帧；第 k 个 latent 帧（k ≥ 1）对应第 4k−3 到 4k 个像素帧。SpatialVID 不是每个像素帧都有深度（第一个 chunk 只有 5 个深度帧），没有深度的像素帧怎么处理【待定·数据线】。
5. LMDB 里存 γ=1 时的 d，γ 次幂在训练端做。

**T5 嵌入**【待定·模型线】：Wan 的 `text_len=512`、`text_dim=4096`（`minwm/modeling/adapter.py:53`）。键名、存不存补齐到 512 的部分、dtype 由模型线定，数据线照着写。定下来之前，数据线不算 T5 嵌入。

**代码位置**：预处理脚本放在 `tools/data/wan21/`。读 disparity 和 T5 嵌入的数据集类也在 `feature/data` 分支上写，这个分支从 `feature/model` 拉出，写完合回 `feature/model`。

## 2026-11-01：先只训 ②，① 以后补

1. **② 在 Colab 训。** 用 1 张 A100 40GB，约 400 个计算单元，按估算刚好够一条 run。单卡放不下在线运行的 T5，所以 T5 嵌入必须离线算好（见上一节）。GBS 用梯度累积凑到 4（`training.grad_accum_steps=4`）。
2. **① 以后回 Delta 补跑。** 估计约 50 卡时。它和 ② 共用 `_stage1_continue.py`，GBS 也必须是 4：Delta 上 4 张卡、每卡 batch 1，不开累积。
3. **代价**：① 跑完之前，② 相对原版 Stage 1 的提升里，「新数据加上多训几千步」和「深度监督」这两个因素分不开。
