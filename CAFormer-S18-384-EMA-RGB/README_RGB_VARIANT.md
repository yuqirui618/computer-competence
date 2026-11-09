# CAFormer-S18-384 RGB Loss 本地实验副本

这是从仓库 `feat/model01-mambaout` 的 `CAFormer-S18-384-EMA/` **复制出来的本地实验副本**。
**仓库内容和原 `CAFormer-S18-384-EMA/` 目录都没有被修改。**

## 改了什么

- 新增 `training/losses.py`：RGB Loss（Reference-Guided Gradient Balancing / ClassAsyDiff）。
- `training/ema_core.py`：新增 `RGB_EMA` 分支，训练损失从 BCE 换成 RGB Loss。
- `training/ema_continue.py`：新增 `--arm RGB_EMA` 参数。
- `training/ema_common.py`：`ARMS` 增加 `RGB_EMA`。
- `training/ema_recipe.json`：增加 `RGB` 参数（`tn_gamma_neg=3.0, fp_gamma_neg=3.0, ratio=1.0, eps=1e-4`）。

## 损失是怎么算的

原版是 Study 级「逐图 sigmoid 后取 Study 平均概率」再算 BCE。
RGB 版本保持同样的 Study 聚合：
1. 每张图过完整分类头，得到 logits；
2. 同一 Study 内逐图 `sigmoid` 后取平均；
3. 平均概率 clamp 后转 logit；
4. 对 Study 级 logits 计算 RGB Loss；
5. 按 Study 数归一化，保持和原 BCE 相同的 loss 量级。

## 怎么跑

需要先有原实验的原生 CAFormer-S18 第 5 轮 checkpoint 及其验证预测 NPZ：

```powershell
$py = 'C:\Users\czy\Desktop\计算机能力赛\repo\.venv\Scripts\python.exe'
$root = 'C:\Users\czy\Desktop\计算机能力赛\CAFormer-S18-384-EMA-RGB'
$data = 'D:\cxr\raw\cxr_first'
$manifest = 'D:\cxr\manifests\manifest.csv'
$initial = 'D:\cxr_runs\native384\checkpoints\epoch_05.pth'
$initVal = 'D:\cxr_runs\native384\predictions\epoch_05.npz'
$out = 'D:\cxr_runs\rgb_ema'

# 先算 SHA（脚本要求显式传入）
$initSHA = (Get-FileHash -Algorithm SHA256 -LiteralPath $initial).Hash.ToLower()
$valSHA = (Get-FileHash -Algorithm SHA256 -LiteralPath $initVal).Hash.ToLower()

$args = @('--data-root',$data,'--manifest',$manifest,
          '--initial-checkpoint',$initial,'--initial-predictions',$initVal,
          '--initial-sha256',$initSHA,'--initial-predictions-sha256',$valSHA,
          '--output-dir',$out,'--arm','RGB_EMA')

# 1. CPU 预检（会同时验证 BCE 和 RGB 的数值/梯度）
& $py -B "$root\training\ema_continue.py" @args --check-only

# 2. GPU 快速预检 + 显存检查
& $py -B "$root\training\ema_continue.py" @args --smoke

# 3. 正式 5 轮 RGB+EMA 续训
& $py -B "$root\training\ema_continue.py" @args --run
```

## 重要注意事项

1. **输出目录必须新建**，不能放在 `CAFormer-S18-384-EMA-RGB/` 里面（脚本会拒绝）。
2. `ema_common.py` 里 `MANIFEST_SHA` 是原实验的固定值 `b7c280...`；如果你的 manifest 不同，需要把这个常量改成你本地 manifest 的 SHA（只改这个本地副本）。
3. `--initial-sha256` / `--initial-predictions-sha256` 必须传你本地文件的实际 SHA。
4. 这是**第二阶段（EMA 续训）**的 RGB 变体；第一阶段原生 384 训练仍保持 BCE（未改）。
5. RGB Loss 官方实现是 GPL-3.0，论文是 Lin et al., MICCAI 2026；本副本只用于本地实验，提交前要按规则申报来源。
6. 小 batch 下 RGB 可能不稳定；建议先看 `--check-only` 和 `--smoke` 是否通过，再跑正式 5 轮。
