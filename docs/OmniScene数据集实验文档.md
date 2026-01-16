# OmniScene 数据集实验说明（latentSplat）

> 本文档用于指导在 `comp_svfgs` 分支中接入 OmniScene（魔改 nuScenes）并完成对比实验的实现方案。实现风格与代码细节对齐 depthsplat。参数配置以本项目（latentSplat）的模型与损失为准，数据集与训练节奏参照 depthsplat。

## 配置概览
- **数据集配置**：新增 `config/dataset/omniscene.yaml`，结构参考 `config/dataset/re10k.yaml`，并对齐 depthsplat：
  - `name: omniscene`
  - `roots: [datasets/omniscene]`
  - `defaults.view_sampler: all`（固定输入/输出视图，view_sampler 不参与采样）
  - `image_shape: [224, 400]`（默认分辨率；另在实验配置中覆盖为 `112x200`）
  - `background_color: [0.0, 0.0, 0.0]`
  - `cameras_are_circular: false`
  - 额外字段与 depthsplat 对齐：`max_fov`、`baseline_epsilon`、`augment`、`near`、`far`、`baseline_scale_bounds`、`test_len`、`train_times_per_scene` 等（具体字段取决于 OmniScene 数据集类实现）
- **实验配置**：新增 `config/experiment/omniscene_112x200.yaml` 与 `config/experiment/omniscene_224x400.yaml`，覆盖基础配置：
  - `defaults: override /dataset: omniscene`（并使用本项目 re10k 的模型/损失配置）
  - `dataset.image_shape: [112, 200]` 或 `[224, 400]`
  - `data_loader.train/val/test.batch_size: 1`
  - `trainer.max_steps: 100_001`
  - `trainer.val_check_interval: 0.01`
  - `train.use_dynamic_mask: true`
  - `wandb.name/tags` 按分辨率区分
- **模型/损失配置**：沿用本项目 `config/experiment/re10k.yaml` 作为基线（autoencoder/encoder/decoder/discriminator、loss 权重与调度保持一致），仅在 OmniScene 实验中覆盖数据集与训练节奏相关参数。
- **训练/验证/测试节奏**：
  - 本项目使用 Lightning 的 `trainer.val_check_interval` 控制验证频率，OmniScene 设为 `0.01`（每 1% epoch 验证一次）。
  - 本项目暂无 `train.eval_model_every_n_val` 机制；若需对齐 depthsplat 的“每 N 次验证触发完整测试”，需要新增该字段并在训练回调或 `ModelWrapper` 内实现（可作为可选扩展）。

## 数据加载流程
1. **注册入口**：在 `src/dataset/__init__.py` 增加 `omniscene -> DatasetOmniScene` 映射，并扩展 `DatasetCfg` 联合类型。
2. **数据类实现**：新增 `src/dataset/dataset_omniscene.py` 与 `src/dataset/utils_omniscene.py`，实现方式对齐 depthsplat：
   - 读取 `bins_*_3.2m.json`：
     - `train`：`bins_train_3.2m.json` 全量。
     - `val`：`bins_val_3.2m.json` 前 30000 个每 3000 抽 1，再取前 10。
     - `test`：`bins_val_3.2m.json` 每 14 抽 1（默认 mini-test）；如需完整测试，放开抽样。
   - `__getitem__` 读取 `bin_infos_3.2m/{token}.pkl`，对 6 个环视相机取 key-frame 作为输入视图；输出视图来自同 bin 的非关键帧（通常 index `[1,2]`），并把输入视图拼回输出（确保渲染监督覆盖输入帧）。
   - 相机姿态：使用 `sensor2lidar_transform` 作为 `c2w`；内参读取 `samples_param_small/sweeps_param_small`，按分辨率 resize 并归一化（`K[0]/W`, `K[1]/H`），满足本项目“归一化内参”的约定。
3. **图像与掩码加载**：在 `utils_omniscene.load_conditions` 中完成：
   - 图像从 `samples_small/sweeps_small` 读取并 resize 到目标分辨率。
   - 输出视图加载 `samples_mask_small/sweeps_mask_small` 的动态掩码；输入视图使用全 1 掩码。
   - 仅在 `stage=test` 时按需加载相对深度（`samples_dpt_small/sweeps_dpt_small` 的 `.npy`），并转为相对深度用于 PCC。
4. **返回结构**：与当前数据集对齐，返回 `context/target`，包含 `extrinsics/intrinsics/image/near/far/index`。
   - 为支持动态掩码与 PCC，`target` 增加 `masks` 与 `rel_depth` 字段。
   - 本项目当前 `src/dataset/types.py` 未包含 `masks/rel_depth`，需扩展类型定义并在 `apply_patch_shim` 中同步裁剪这些字段。

## 主程序调用方式
- **训练/测试命令**：延续本项目 README 的入口，新增 OmniScene 实验后可使用：
  - `python3 -m src.main +experiment=omniscene_112x200`
  - `python3 -m src.main +experiment=omniscene_224x400`
- **调用链差异**：
  - 本项目 `src/main.py` 在 Hydra 配置完成后实例化 `DataModule`，而后在 `ModelWrapper` 中执行训练/验证/测试；数据集只要返回标准字段即可被现有流程处理。
  - 与 depthsplat 不同：本项目没有 `eval_model_every_n_val` 的周期性全量测试；如需对齐，需要新增回调或在 `ModelWrapper` 中增加额外评测入口。
  - 本项目的 `encoder` 在 `get_data_shim` 中会做 patch 裁剪（`apply_patch_shim`），因此 OmniScene 掩码与相对深度要在 shim 中同步裁剪以保证像素对齐。
- **动态掩码接入**：
  - 当前训练损失不支持掩码，需要在 `loss` 或 `ModelWrapper.training_step` 中加入 mask 逻辑（建议仅对 `target_render_image` / `target_combined` 的图像类损失应用掩码）。
  - 若保持与 depthsplat 一致，可新增 `train.use_dynamic_mask` 开关，并以 `target.masks` 过滤损失计算区域。

## PCC 指标补充方案
1. **相对深度加载（仅 test）**：
   - 在 `utils_omniscene.load_conditions` 中按 depthsplat 方案读取 DepthAnything-v2 的 disparity（`.npy`），resize 后转为相对深度并做 min-max 归一化。
   - 仅在 `stage=test` 下返回 `rel_depth`，并写入 `target`。
2. **测试时保存预测深度**：
   - `Decoder.forward` 默认返回深度图，`ModelWrapper.test_step` 仅负责将 `output.depth` 保存为 `depth/{index:0>6}.npy`，不在测试阶段计算指标。
   - 通过 `test.save_depth=true` 控制是否保存深度，避免默认测试开销过大。
3. **PCC 计算位置**：
   - 在 `src/evaluation/metrics.py` 增加 `compute_pcc`（参考 depthsplat，使用 `torchmetrics.PearsonCorrCoef`）。
   - 在 `src/evaluation/metric_computer.py` 中读取预测深度（来自 `depth/*.npy`）与数据集 `rel_depth`，与 PSNR/SSIM/LPIPS 并列计算。
4. **PCC 统计与汇总**：
   - 复用 `MetricComputer` 的 `scores` 汇总逻辑，新增 `pcc` key 输出到 `per_scene_metrics.json` 与 `evaluation_metrics.json`。
   - 若缺少预测深度文件，则该方法跳过 PCC 统计但不影响其它指标输出。

## 与 depthsplat 的差异与可复用性
1. **数据结构**：已扩展 `BatchedViews` 支持 `masks/rel_depth`，并在 patch shim 中同步裁剪；与 depthsplat 对齐。
2. **损失掩码**：已在图像类损失中引入 `train.use_dynamic_mask` 过滤逻辑，作用范围较 depthsplat 更偏向 loss 内部。
3. **测试/评测机制**：depthsplat 支持 `eval_model_every_n_val` 与完整测试评估；latentSplat 仍保持 `test` 与 `compute_metrics` 分离，PCC 通过保存深度后在 `compute_metrics` 中计算。
4. **可复用部分**：OmniScene 的数据解析与相对深度/掩码处理基本可直接复用 depthsplat 的 `dataset_omniscene.py` 与 `utils_omniscene.py`，但需按 latentSplat 的数据规范（归一化内参、Patch shim、Loss 结构）做适配。

## 小结
OmniScene 接入在 latentSplat 侧主要是“补齐数据字段 + 训练/评测流程适配”。数据加载逻辑可以高度复用 depthsplat；核心改动集中在：配置新增、类型扩展、掩码损失接入、测试深度渲染与 PCC 统计。待文档审阅通过后，将按此方案进行代码实现。
