# 技术规范:OpenWAM-AutoDrive — 数据链路(P1 契约 + P2 数据)

> 本文档是 `spec_zh.md` 的 §3(契约)、§5(数据层)、§8.1(部署预处理对等)、§11(兼容矩阵)
> 的数据侧细化,覆盖 **P1 契约重落地** 与 **P2 数据链路**。模型侧契约(§3.2 的
> `BackboneOutput`/`ConnectorOutput`/`ActionHeadOutput`/`SystemOutput`)、模型层(§6)、
> 训练引擎(§7、§4.2 `config_freezer.py`)不在本文档范围;归一化统计的冻结写入(§4.2 的
> `dataset_statistics.json`)按 §3 此处定义。章节号引用 `spec_zh.md`。

## 0. 基座裁定(必须遵守)

1. **在 `main` 上重落地 contracts**,不采用分支 `refactor/p1-contracts` 的代码(其
   `dataloader/builder.py`、`model/builder.py` 已被 `spec_zh.md` Rev 3 §1.3 规则 1 否决
   —— 注册表保留、builder 删除)。分支上的 `data_contract.py`/`model_contract.py`/
   `collate.py`/`config_schema.py` **仅供参考**,落地时与本文档逐条对齐。
2. **注册表是权威实例化机制**(§1.3 规则 1)。`dataloader/registry.py` 的装饰器注册、
   `build_dataset(config, split)`、`MixtureDataset.from_config` 均保留;Track B 读取器
   在 `dataloader/registry.py` 注册,预处理器在 `dataloader/preprocessors/registry.py`
   建同模式注册表(`register_preprocessor`)。**不新增 builder。**
3. **纯张量模型图 + 高复用**。`PipelineContext` 是 worker 内可变单元,`ModelInputBatch`
   是进入模型的冻结单元;VAE latent / 文本嵌入 / VACE 条件化绝不进 dataloader。

---

## 1. 契约层(`openwam/contracts/`)

字段命名沿用 `spec_zh.md` §3.2 约定:state 按来源命名(`last_hidden_state`、
`hidden_states`),mask 按所掩蔽的序列命名(`attention_mask`、`context_mask`、
`condition_mask`、`video_mask`、`trajectory_mask`)。

### 1.1 `data_contract.py`

`spec_zh.md` §3.1 的 `PipelineContext` / `ModelInputBatch` 是权威字段表,此处仅补两处:

- **`video_mask` 入 `PipelineContext`。** §3.1 只在 `ModelInputBatch` 列了 `video_mask`
  (`[B, T_obs] bool`,WAM 帧有效性),但 `PipelineContext` 无对应来源,collate 无法产出。
  裁定:`PipelineContext` 增补一个通道 ——

  ```python
  # [T_obs] bool — 观测窗口的帧有效性(WAM 损失掩码)。Track A 读取器今日即产出
  # (canonical sample 的 "video_mask");驾驶窗口恒满 ⇒ None。
  video_mask: Optional[np.ndarray] = None
  ```

  位置在块 4(原始解码物理量)末尾,与 `raw_trajectory_valid_mask` 并列。collate 将其
  堆叠为 `ModelInputBatch.video_mask`。

- **`ego_state_mask` 入两契约。** Track A 读取器的 `proprio_mask`(逐维状态有效性,2-D
  `(1, D)`)被 `prepare_inputs` 桥接进 `inputs["proprio_mask"]`,但两个契约均无槽位。按
  §3.2 命名规约(mask 按所掩蔽序列命名,状态序列 = `ego_state_tensor`)落位为
  `ego_state_mask`:`PipelineContext` 块 5(`ego_state_tensor` 旁,`[D]`/`[T_hist, D]`),
  `ModelInputBatch`(`[B, T_hist, D]`);collate 复用既有 1-D/2-D 调和后堆叠。驾驶状态恒
  有效 ⇒ None。

- **`frame_timestamps_ns` 由解码器填充。** §5.2 的 Lance 索引未列该列,而 §3.1 块 3 要求
  它存在于每个 Track B 运行。裁定:该字段是**解码期**产物 —— `MultiCamVideoDecoder`
  从 MP4 的 PTS 提取每相机曝光时刻填充;不进 Lance、不在 ingest 预计算。

其余字段、块分类、`Track A 映射(规范性)` 与 `ModelInputBatch` 三布局语义
(`wam_canvas`/`standard_tensor`/`qwen_vl_patch`)均原文照 `spec_zh.md` §3.1,不再重复。

### 1.2 `collate.py`(组批,唯一权威)

`collate_pipeline_contexts(ctxs, *, camera_order) -> ModelInputBatch`。规则(规范):

- **视觉三布局互斥**:每个批恰好其一。`pixel_values`(wam_canvas 整批堆叠)、
  `pixel_values_by_cam`(按相机逐相机堆叠)、`pixel_values`+`image_grid_thw`(qwen_vl_patch
  沿 dim0 拼接三者 `image_grid_thw` 拼接)。三者同时非空或混合出现是硬错误;全空仅允许
  合成/纯状态批。
- **按相机堆叠**:`pixel_values_by_cam` 相机迭代序 = `camera_order` 参数(缺省取首样本
  插入序)。任一相机在任一样本缺失是硬错误,不是零填充。批内同相机形状必须一致。
- **`image_grid_thw` 拼接**:`[N_img, 3]` 沿 dim0 跨帧/相机/批拼接,无 batch 维、无填充
  (HF Qwen-VL 约定,与 `qwen3_vl_backbone.batch_vlm_inputs` 字节兼容);图像边界经
  `cumsum(t*h*w)` 恢复。拼接顺序 = prompt 处理器占位符展开顺序。
- **语言路径互斥**:WAM 原始 `prompt: List[str]` XOR VLA `input_ids`。`input_ids`/
  `attention_mask`/`vlm_labels` 右填充到批内最长序列(pad 0 / mask 0 / labels -100)。
  `attention_mask`、`vlm_labels` 各自 all-or-none。
- **可选张量字段 all-or-none**:任一字段在某样本有值、另一样本缺省即硬错误。
- **`trajectory_mask` 秩调和**(复用 `BaseArchitecture.prepare_inputs` 的既有调和规则):
  全 1-D → `[B, T]`;全 2-D → `[B, T, D]`;混秩 → 硬错误。驱动用 1-D,机器人 80-D 统一
  空间用 2-D。
- **`ego_state_tensor` 提升**:`[D]` → `[1, D]` 后再堆叠(接受遗留 `[B, D]`)。
- **`sample_weights` 恒出**:`[B] float32`,取自 `ctx.sample_weight`(默认 1.0),损失侧
  全局归一化的稳定输入。
- 任何架构不得自行组批(§3.3)。

### 1.3 `config_schema.py`(数据相关块)

范围控制:`全量 DictConfig→dataclass 不是目标`;仅新块被类型化,构建时
`OmegaConf.structured(...)` 校验,失败在构建期而非 forward。数据侧类型化两块:

```python
@dataclass
class ProcessorSpec:
    name: str                            # 预处理器注册键(video_decoder / time_sync / …)
    kwargs: Dict[str, Any] = field(default_factory=dict)  # 处理器私有 kwarg,构建时由处理器自校验

@dataclass
class PreprocessorChainConfig:
    chain: List[ProcessorSpec] = field(default_factory=list)  # 顺序即执行序;可空(纯适配器/Track A)
```

每个 processor 的 `from_config(kwargs)` 在构建时校验自身 kwarg(快失败);`ProcessorSpec`
不做刚性 schema,因各 processor kwarg 异构。`model_contract.py` 的 `ConnectorConfig` /
`ActionHeadConfig`(模型侧)不在本文档。

`契约源码基线`:落地时直接复用分支草稿 `data_contract.py`/`collate.py` **的逻辑**,但
(a) 删其 `video_mask` 上位的既有写法改为块 4 增补字段、(b) 去掉 `dense docstring` 里
任何对 builder 的引用、(c) 对齐本文档 §1.2 的 collate 规则。分支 DO NOT COPY:
`dataloader/builder.py`、`model/builder.py`、`config_schema.py` 里的模型块。

---

## 2. 归一化统计契约

- **新运行**:`tools/ingest_lance.py` 离线计算或首次加载时 rank-0 计算(其他 rank 等待,
  同今日 auto-stats 协议)→ `dataset_statistics.json`,随 checkpoint 冻结(§4.2)。
- **加载器双读**:`dataset_statistics.json`(新)与遗留 `normalization_stats.npy`(嵌套
  `{joint, eef}` schema)并存;旧机器人 checkpoint 部署不变。
- **schema**(`dataset_statistics.json`):

  ```json
  {
    "mode": "q99",                     // q99 | min_max | mean_std（复用 Normalizer 模式）
    "ego_state":   {"min": [...], "max": [...], "q01": [...], "q99": [...], "mean": [...], "std": [...]},
    "trajectory":  {"min": [...], "max": [...], "q01": [...], "q99": [...], "mean": [...], "std": [...]},
    "T_hist": 1, "D_state": 6, "T_pred": 40, "D_action": 3
  }
  ```

  归一化公式复用 `dataloader/utils/normalization.py::apply_normalization` /
  `dataloader/transforms/normalize.py::Normalizer`(同一套 train/deploy 位等价保证);
  子表(如 `_state` 方向性)沿用 `Normalizer` 的 `_NORMALIZE_STATS_KEY` 机制。

---

## 3. Track A — 适配器(机器人域,薄封装)

`spec_zh.md` §5.1:四个预训练读取器 **保持原样**;一个薄适配器把其 canonical sample dict
包进 `PipelineContext`。

**数据流**:`LeRobotV3Reader.__getitem__` → canonical dict(`video`/`action`/`action_mask`
/`video_mask`/`proprio`/`proprio_mask`/`prompt`) → `robot_sample_to_context(dict) →
PipelineContext` → collate。**Track A 的预处理器链为空** —— 读取器已在内部完成
解码/多视图合成/归一化,适配器只做字段重命名(见 §7 待确认)。

规范映射:

| canonical dict | PipelineContext | 语义 |
|---|---|---|
| `video`(canvas/PIL 列表,已合成) | `pixel_values`(wam_canvas) | 已解码;不重走解码器 |
| `video_mask` | `video_mask` | `[T_obs] bool` |
| `action`(已归一化) | `gt_trajectory_norm` | 读取器已归一化,不重走 `EgoStateProcessor`(非 `raw_*`) |
| `action_mask` | `trajectory_mask` | 2-D `[T_pred, D]`,80-D 保留维在损失外 |
| `proprio` | `ego_state_tensor` | `[1, D]` |
| `proprio_mask` | `ego_state_mask` | 逐维状态有效性;按 §3.2 命名规约落位(见 §1.1) |
| `prompt` | `prompt`(WAM) 或 `input_ids`(VLA 经 tokenize) | VLA Track A 走 `VLAPromptTokenizeProcessor` |
| 簿记 | `sample_id`/`clip_uri`/`source_name`/`nav_instruction` | `clip_uri`=dataset_dir;`anchor_timestamp_us`=0 合成;`cam_frame_offsets`={} |

未映射:`calib`/`frame_timestamps_ns`/`raw_images`/`raw_ego_history`/`raw_gt_trajectory`
/`raw_trajectory_valid_mask` 全为 None(Track A 无标定/时序、原始槽位被读取器内联消化)。
`unify_action_map`、颜色抖动状态是**读取器内部**配置,不进上下文(§3.1 Track A 映射规范)。

对等闸口(契约层,P2):固定种子下,adapters→空链→collate 产出的 `ModelInputBatch`
与今日 transforms 路径在纯字段重命名上**逐比特一致**,且 `pixel_values` 精确恢复 uint8
(`fl(fl(u8/255)*255)==u8`,模型侧按 `(x*255)*(2/255)-1` 转换即与 wan preprocess 位等价),
由 `test_robot_adapter_parity.py` 钉死。损失级位等价闸口随 P3 模型侧 `ModelInputBatch→forward`
适配执行,通过后删除 `transforms/` 包(独立 PR)。`video` 的 PIL→tensor 转换发生在图像
处理器,而 Track A 已由读取器产出 canvas,故适配器直接用读取器内部同一
`assemble_multiview_layout` 的结果 —— 唯一新增负担是 PIL→tensor + `permute` 的正确性由
对等闸口验证。

---

## 4. Track B — 驾驶数据存储与读取器

### 4.1 存储布局

```text
<storage_root>/raw_clips/<clip_id>/
    cams/<cam_name>.mp4            # ~7 环视;每相机分辨率层级(front > side > rear)
                                   # H.264/H.265;PTS 即曝光时间戳
    topics/ego_motion.parquet      # 高频车辆总线(读时插值历史)
    topics/localization.parquet    # 世界系 ego 位姿(仅 ingest 用,推导 gt_future_traj_ego)
    topics/navigation.parquet      # 指令(仅 ingest 用,折入 nav_instruction/cot_text)
    calib.json                     # 每相机内参/外参(相机->ego-rig)
<storage_root>/lance_indices/<dataset_name>.lance
```

**读时需读的文件**仅 `cams/*.mp4` + `topics/ego_motion.parquet` + `calib.json`(其余在
ingest 期消化)。

### 4.2 topics / calib schema

```text
topics/ego_motion.parquet
    timestamp_ns: int64                       # 时间戳
    <state cols>: float32 × D_state           # 速度/加速度/横摆角速度/转向…(数据集特定)
topics/localization.parquet
    timestamp_ns: int64
    <pose cols>: float32/float64              # x,y,yaw 或 SE3/quat(数据集特定,ingest 用)
topics/navigation.parquet
    timestamp_ns: int64
    instruction: string
    cot_text: string                          # 可选
calib.json
{
  "cameras": {
    "<cam>": {
      "intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]],   # [3,3]
      "extrinsics": [[…],[…],[…],[…]]                 # [4,4] 相机 -> ego-rig
    }
  }
}
```

### 4.3 Lance schema

```text
列（schema 固定，tools/ingest_lance.py 写入）:
    sample_id:            string                       # 连接键
    clip_uri:             string                       # raw_clips/<clip_id>/
    anchor_timestamp_us:  int64                        # 锚点相机曝光时刻（topic 插值原点）
    cam_frame_offsets:    map<string, list<int64>>     # 每相机待读帧索引（ingest 预计算；读取器零索引计算）
    nav_instruction:      string                       # 纯世界模型样本 = ""
    cot_text:             string                       # 可空
    scene_tags:           list<string>                 # 过滤/课程/分层（SQL 下推）
    sample_weight:        float32                      # 默认 1.0
    gt_future_traj_ego:   fixed_size_list<float32>     # [T_pred, D_action] 展平，ego-rig 系
    valid_mask:           fixed_size_list<int8>        # [T_pred] 1-D（驾驶逐步有效性）
```

`T_pred`/`D_action`/`D_state` 是数据集级常量,写入 `dataset_statistics.json`,不在逐行存储。
`cam_frame_offsets` 的 `map` 若 SQL 下推受限,退化为固定相机集的 `struct`(per-cam
`list<int64>` 字段),由 ingest 工具按 calib 相机名生成 —— 读取器两者皆容。

### 4.4 读取器

- `autodriving_dataset.py::AutoDrivingDataset`:打开 `<dataset_name>.lance`,SQL 过滤下推
  (`lance.dataset(uri).to_table(filter=...)`)、列投影、`scene_tags` 子集/课程选择。
  `__getitem__` 产出一个 `PipelineContext` 的**索引块 + 原始块的一部分**:
  - 索引块全量(`sample_id`/`clip_uri`/`anchor_timestamp_us`/`cam_frame_offsets`/`source_name`),
  - `nav_instruction`/`cot_text`/`scene_tags`/`sample_weight`,
  - `raw_gt_trajectory` ← `gt_future_traj_ego`(reshape `[T_pred, D_action]`),
  - `raw_trajectory_valid_mask` ← `valid_mask`(`[T_pred]`),
  - `calib`(从 `calib.json` 读,进程缓存)。
  `frame_timestamps_ns`/`raw_images`/`raw_ego_history` 留给预处理器链。
- `clip_reader.py::ClipReader`(被 `MultiCamVideoDecoder`/`TopicTimeSynchronizer` 复用):
  给定 `clip_uri`,按需读 `cams/*.mp4` 帧 + `topics/ego_motion.parquet` + `calib.json`,
  每 worker 单例句柄 + 有界 LRU(§5.3 逐句)。
- 统计:`tools/ingest_lance.py` 离线算或首次加载 rank-0 算,写 `dataset_statistics.json`(§2)。
- 数据治理:ingest 负责去标识化与 `clip_uri` 访问控制;本仓库假定只读、已脱敏存储(§12 O5)。

---

## 5. 可组合预处理器链(`dataloader/preprocessors/`)

`BasePreprocessor.process(ctx: PipelineContext) -> PipelineContext`,属性
`is_training_only: bool`;`PreProcessorPipeline` 依 `PreprocessorChainConfig.chain` 顺序
执行。注册:`preprocessors/registry.py::register_preprocessor` + `build_preprocessor_chain(cfg)`。

| 处理器 | 仅训练 | 输入 → 输出 |
|---|---|---|
| `MultiCamVideoDecoder` | 是 | `clip_uri`+`cam_frame_offsets` → `raw_images: {cam:[T_obs,H,W,3]uint8}` + `frame_timestamps_ns`(MP4 PTS)。decord/PyAV,每 worker 单例句柄 + 有界 LRU;颜色抖动(仅训练、时间一致)从 `transforms/video.py` 移入。 |
| `TopicTimeSynchronizer` | 是 | `clip_uri`+`anchor_timestamp_us`+`frame_timestamps_ns`+`calib` → `raw_ego_history: [T_hist,D_state]`(在锚点插值 `ego_motion.parquet`,经 `calib` 变换到 ego-rig 系)。插值类型/容差闸口/超闸策略是配置字段(§12 O1 默认值未决)。旋转工具复用 `utils/poses.py`(6D/四元数)。**注:`raw_gt_trajectory` 不在此时产生 —— 它已在 Lance 预计算(§4.4),本处理器只做历史插值(见 §7)。** |
| `EgoStateProcessor` | 否 | `raw_ego_history`+`raw_gt_trajectory`+`raw_trajectory_valid_mask` → `ego_state_tensor`/`gt_trajectory_norm`/`trajectory_mask`。用 `dataset_statistics.json` 归一化,可选 Fourier 嵌入;提供 `denormalize_trajectory` + π0.5 风格 `state2str`(供 prompt 处理器)。 |
| `MultiViewImageProcessor` | 否 | `raw_images` → 三布局之一(配置 `per_cam_resolution: {cam:[H,W]}` + `mode`):`wam_canvas`(视差合成到单画布,槽位随层级)、`standard_tensor`(`{cam:[T_obs,C,H,W]}` 不上采样)、`qwen_vl_patch`(动态分辨率分块 + `image_grid_thw`)。缩放/归一化参数 checkpoint 冻结。合并进单 token 流在模型侧(§6.2),不在本处理器。 |
| `VLAPromptTokenizeProcessor` | 否 | `nav_instruction`+相机 token(占位符展开顺序)+可选 `state2str`+`cot_text` → `input_ids`/`attention_mask`/`vlm_labels`(`cot_text` 存在时 prompt token 置 -100)。 |

双模式:

- **训练**:DataLoader worker 内跑配置链 → `contracts/collate.py` 组批。
- **推理**:`InferenceProcessorPipeline.from_checkpoint(ckpt_dir)` 从冻结 `config.yaml`
  重建链,过滤 `is_training_only=True`(训练只用 `EgoStateProcessor`+`MultiViewImageProcessor`
  +`VLAPromptTokenizeProcessor`),把实时输入(`{cam: frame}` 字典、最新 ego 读数、指令)
  映射进 `PipelineContext`,再走同一链 —— 训练/部署对等(starVLA `PolicyNormProcessor` 教训)。

---

## 6. 混合、加权与采样(Track A/B 跨格式)

`spec_zh.md` §5.4:条目可指向任意 Track A 读取器**或** Track B 表。存在两种训练范式,采样
机制不同:**epoch-based**(跑满 epoch,现有 `MixtureDataset`)与 **iter-based**(固定总步数
`max_steps`,大概率跑不满一个 epoch,自驾量产主路径)。

### 6.1 epoch-based(现有 `MixtureDataset`)

`weight_strategy`(`proportional/manual/uniform/inverse_size`)语义保留,新增 `annealed`
(epoch 驱动权重退火,见下);逐 epoch 确定性重洗牌保留(带种子索引映射 — 现状行为)。

实现要点:

- `MixtureDataset.from_config` 已在 `build_dataset` 上多态;Track B 条目由注册表分派到
  `AutoDrivingDataset`(经 `build_dataset`)。**无需改 index-map 逻辑** —— 只需让
  `build_dataset` 能构造 Track B 类,以及 `shape_fields` 同源校验
  (`num_frames/video_stride/height/width`,见 `mixture.py`)对 `AutoDrivingDataset` 解耦
  (Track B 的 `T_obs`/层级分辨率由 `MultiViewImageProcessor` 配置,不属读取器)。
- `sample_weight`:数据集侧由权重器产出(先 `constant`),损失侧消费并做跨 rank 与累积步
  全局归一化(FluxVLA `reduce_action_bc_loss` 模式)。
- `source_name`:Track B 的 `source_name` 用 `<dataset_name>/<clip_id>` 或仅 `<dataset_name>`
  (由配置决定),供 mixture 日志与按源指标。

- **权重退火(epoch 驱动,`weight_strategy: annealed`)**:每个子数据集给定 `weight_start`
  + `weight_end`(相对权重,语义同 `weight`,构建期仍归一化到和=1),缺省回退到 `weight`
  (静态)。退火进度随 epoch:线性 `alpha = e/(E-1)`(默认)或 cosine
  `alpha = (1 - cos(e·π/(E-1)))/2`,`E` = `num_epochs`(训练引擎注入)。`set_epoch(e)` 先
  算 `alpha(e)` → 更新 `_weights` → 再 `_build_index_map()`(复用现有重建);权重变化只在
  各源间重分配虚拟样本(总数 ≈ `total_real`)。**step 级退火不在本路径**(归 iter-based,
  §6.2):epoch-based 下改采样比例受 worker fork 约束,只能在 epoch 粒度生效。

  ```yaml
  type: mixture
  weight_strategy: annealed
  weight_anneal_schedule: cosine        # linear(默认) | cosine
  datasets:
    robocoin:   {enabled: true, weight_start: 0.1, weight_end: 1.0}
    oxe_droid:  {enabled: true, weight_start: 1.0, weight_end: 0.5}
  ```

### 6.2 iter-based 数据管线(在线/离线采样序列)

自驾数据量大、内容与量频繁变动 → **iter-based trainer**:固定总步数 `max_steps`,大概率跑
不满一个 epoch(类比 LLM 固定 token 预算)。采样按"全局 step"驱动,不再按
"epoch × len(dataset)"。**trainer 侧需改成 iter-based**(`spec_zh.md` §7 范围;数据层假设
其已就位:按 `max_steps` 迭代,无 epoch 概念)。

**两层 dataset 包装**:

- 底层:常规 map-style `AutoDrivingDataset`(以 Lance 表为核心,§4.4)。
- 上层:轻量分发器 `IterMixtureDataset`(区别于 epoch-based 的 `MixtureDataset`),持有多个底层
  dataset 实例;给定 `(dataset_index, sample_index)` 调对应 dataset 取样本。不做加权
  index-map、不做 `set_epoch`。

**`WeightAnnealingSampler`(custom sampler)**,持有 `IterMixtureDataset`,按两类预生成序列取样本:

1. **`dataset_index`**(采样哪个 dataset):每样本一个,长度 = `batch_size × total_steps`
   (逐 rank)。由 **init/final weight** 迭代生成(退火进度,线性/cosine)。
2. **`sample_index`**(该 dataset 内第几个样本):**每个 dataset 一条序列**,所有 rank 共享。

**协同采样(rank/world_size 跳跃)**:某 dataset 的共享 `sample_index` 序列 `S_D`(同一
种子、全 rank 一致),rank `r` 取位置 `r, r+w, r+2w, …`(stride = `world_size`);各 rank 拼
起来恰好顺序覆盖 `S_D`,无重叠、无遗漏 —— "所有 rank 在同一套序列上协同采样"。

**online 模式**(早期快速 baseline):

- `dataset_index`:每个 rank 独立种子(与 `rank_index` 绑定),启动前按 `batch_size ×
  total_steps` + init/final weight 预生成自己那份 `dataset_index`(不落盘)。
- `sample_index`:每个 dataset 共享同一种子,生成该 dataset 的随机采样序列生成器;实际取
  用按 rank/world_size 跳跃。
- 特点:零额外存储、改配置即刻生效。

**offline 模式**(后期迭代稳定,数据顺序成为资产):

- 序列提前生成落盘,**保证所有训练任务数据一致**;数据顺序是迭代开发的一部分。
- `dataset_index`:离线生成**一条全局 curriculum 序列**落盘(只存一份,复用 online 生成器
  + 固定种子;生成时已按课程顺序提前排布好整条序列)。分布式训练**整体从前往后**消费 ——
  每全局 step 消耗序列上一段长 `world_size × batch_size` 的**连续窗口**,rank `r` 取窗口
  内第 `r` 个 `batch_size` 连续切片(**step 级交错,非逐 rank 连续块**);保证所有 rank 同刻
  处于相近的 curriculum 相位。
- `sample_index`:每个 dataset 各自一条序列、所有 rank 共享;加载时按跳跃逻辑,**每 rank
  只保留自己要采的 index**(strided 切片,降内存)。
- **比例调整只改槽位**:数据比例变化时,基于前一份 `dataset_index` 序列做**特定槽位的局部
  改写**(而非整体重生成),避免"加一个/调一个数据集数量"导致全序列大抖动。要求序列存储
  可版本化 + 可按槽位更新。
- **world_size 不变性(后期,预留接口即可)**:后期存在不同卡数配置,需保证改 `world_size` 时
  数据分布不变 —— 跳跃 stride 由**固定上限(最大步数/最大卡数)** 控制,而非绑定当前
  `world_size`:全局序列一经生成,其 curriculum 顺序与各位置内容即固定,`world_size` 只决定
  每全局 step 并发消化的位置数。生成器预留 `stride` 参数(当前默认 = `world_size`),后续参数
  化不重排已有序列。

**存储方式**:

- 两类序列都是**数据资产**,都要**版本化**("比例调整只改槽位" = 在上一版序列上做差量)。
- `dataset_index`:`[world_size × batch_size × total_steps]` int(数据集 id,量小)。→
  **Lance 表**(单列,int;版本化 + 槽位级更新)。
- `sample_index`:每个 dataset 一条记录序列,每条 = **样本身份**(`clip_uuid` + `frame_uuid`
  + 相机/时间戳等),可达**亿级**。→ **Lance 表**(列式 + 版本化 + 随机/步进读 +
  memory-mapped scan)。**排序键 = 显式 `order_id` 列**(int64 代用键,离线生成时写死,稳定
  可 join):每 rank 按 `order_id % world_size == rank` 只加载自己 strided 切片;跨版本
  "只改槽位"的差量也以 `order_id` 定位。用**身份而非位置索引**是为稳恒:数据表重排/插入
  行时,身份仍指向同一物理样本。加载侧经 `clip_uuid` 直达 `raw_clips/<clip_uuid>/` 定位。
- **memmap(`.npy`)不再适用**:`sample_index` 已非单 int64 数组,而是结构化身份字段 + 需版本
  化;memmap 既表达不了多字段、也无可版本语义。
- **pkl 排除**:eager 全量载入、无 memmap、对象开销大、无版本化/随机访问,亿级不可行。

### 6.3 断点续训(数据侧)

训练中途中断后,依据最近 ckpt + 完整可恢复状态恢复到断点瞬间;恢复后数据流与"不中断连续
训练"从断点起**逐字节一致**——这是"完整恢复"的唯一验收标准。**trainer 侧**的模型权重/
optimizer/scheduler/RNG 恢复属 `spec_zh.md` §7(不在本文档);此处约定**数据侧需暴露的可恢复
状态**及其确定性前提。

**数据侧可恢复状态**:采样组件(`WeightAnnealingSampler`、`IterMixtureDataset`)实现
`state_dict() -> dict` / `load_state_dict(dict)`,**命名与 torch 原生 ckpt 保存一致**
(`nn.Module.state_dict` / `optimizer.state_dict`);trainer 在 `save_full_states_for_resume=true`
时与 model/optimizer 一样聚合进整份 ckpt(`ckpt["sampler"] = sampler.state_dict()`)。状态含:

- **主游标**:iter-based 用 `global_step`;epoch-based 用 `epoch`(整数,`set_epoch` 已可确定性
  重建 index map)。
- **样本游标**:`cursor[rank][D]` = 该 rank 从 dataset D 采过的次数。可由 `dataset_index
  [0..step-1]` 重推,但**显式持久化**省去 O(step) 重放(便宜,推荐)。
- **序列版本指针(offline)**:本次所用 `dataset_index` / `sample_index` 的 **Lance 版本 id**。
  resume 必须 checkout 同一版本,否则数据顺序的迭代改动会破坏一致性。online 只需存 `seed`
  + `world_size`。

**确定性 + 可寻址**(resume 正确性的前提):

- `dataset_index[step] = f(seed, step, weights(alpha(step)))` —— 逐 step **O(1) 可算**,而非流
  式递推。用 hash 化 RNG(`np.random.SeedSequence([seed, step])` / splitmix / philox)实现任意
  step 直达:resume 无需回放 step 0..K-1,也无需存 RNG 内部状态。
- `sample_index` 每 dataset 一条**不可变**序列(共享种子生成 / offline 落盘);游标 `c_D` 由
  `dataset_index` 计数导出,rank r 第 j 次采 D 取 `S_D[r + j·world_size]`。

**恢复流程**:

1. `resume_ckpt_path` 载入模型权重 + optimizer/scheduler,并对采样组件调 `load_state_dict`
   (`sampler.load_state_dict(ckpt["sampler"])`)恢复数据侧状态。
2. 重建 sampler 并 seek 到断点:iter-based 由 `global_step` 直达 `dataset_index[step]`、由游标
   直达各 `sample_index` 偏移;offline 按版本 id checkout 序列、按 `order_id` 偏移继续;
   epoch-based 调 `set_epoch(epoch)`(种子确定性重建 index map)。
3. 与 `finetune_ckpt_path` 区分:finetune 是 warm-start 权重,从 **step/epoch 0** 起,不恢复
   数据游标、不 checkout 序列版本。

**DDP**:各 rank 恢复各自游标;`global_step`/`epoch` 全 rank 一致(同读一份 ckpt)。world_size
变化时的一致性归 O6(后期),当前 assume 固定 world_size。

**验证(对等闸口)**:固定种子下,断点恢复后的前 N 个批与连续训练同一步区间**逐字节一致**
(回归测试);直接复用 §6.2 的确定性生成,是"完整恢复"的硬门。

---

## 7. 未决 / 待确认

- **`待确认 A — Track A "经新链" 口径`。** `spec_zh.md` §10 P2 写"Track A 经新链运行",而
  §5.1 又说读取器"保持原样"+薄适配器。本文档裁定:Track A 走适配器(填已处理槽位)且链为空;
  "经新链"即"经新数据管线(adapters→collate)不失真"。若你的意图是让 Track A 读取器改吐
  原始数据、由预处理器链**重新**解码/归一化(重实现对等),则需扩改四个读取器,工作量与
  §5.1"保持原样"冲突 —— 请确认。
- **`待确认 B — 驾驶 GT 轨迹来源`。** 本文档依 §5.2 裁定 GT 轨迹 `gt_future_traj_ego` 在
  ingest 预计算进 Lance,`TopicTimeSynchronizer` 只插值历史;§5.3 表格措辞把
  `raw_gt_trajectory` 也归给同步器。若你需要**读时**重推导(不打 Lance 轨迹列),同步器须
  额外读 `localization.parquet` 并做 ego 系投影 —— 请确认走预计算还是读时推导。
- **O1 — 时间同步参数**:插值类型(线性 vs 样条)、容差闸口(提议 ±5 ms)、超闸策略(无效
  掩码 vs 丢弃)。`TopicTimeSynchronizer` 的字段默认值待定。
- **O2 — 离线 VAE latent 预编码**:采纳则以 `LatentCacheReader` 处理器短路解码+编码,
  架构接受预计算 latent;决策须在 P2 结束前(影响 Lance schema 与 ingest 工具)。
- **O5 — 数据治理**:去标识化与访问控制在 ingest 侧,所有权在本仓库之外确认。
- **O6 — world_size 不变性(后期,预留接口即可)**:`dataset_index`(及后续 `sample_index`)
  的跳跃 stride 由固定上限(最大步数/最大卡数)控制,而非绑定当前 `world_size`,改卡数不改
  变数据分布;前期只**预留 `stride` 参数接口**(当前默认 = `world_size`),不实现其参数化扩展。

---

## 8. 交付与验收(数据部分)

| 阶段 | 范围 | 验收 |
|---|---|---|
| P1 | `contracts/` 数据侧(`data_contract.py` 含 `video_mask` 增补、`config_schema.py` 数据块;**不含 `collate.py`**)+ 遗留兼容(注册表不动、旧 checkpoint `config.yaml` 经 `resolve_architecture_config` 原样加载) | 现有 pytest 全绿;契约构建期校验快失败;遗留黄金 checkpoint 可加载 |
| P2 | `contracts/collate.py` + Track A 适配器(`robot_data_adapter.py`,§3)+ 预处理器链(`preprocessors/`)+ 归一化统计双读(§2)+ Track B(`autodriving_dataset.py`+`clip_reader.py`)+ `MixtureDataset` 跨格式与权重退火(§6.1)+ iter-based 管线(§6.2,`IterMixtureDataset`+`WeightAnnealingSampler` 在线/离线)+ `configs/dataloader/` 重组(`robot/`+`driving/`+`preprocessors/`,Hydra 包重映射,cfg 树回归)+ `tools/ingest_lance.py` | 固定种子下 Track A 经新管线(适配器→空链→collate)**契约层位等价**:纯重命名字段(action/mask/proprio/prompt)与 transforms 路径逐比特一致,`pixel_values` 满足 `fl(fl(u8/255)*255)==u8` 精确恢复(模型侧按 `(x*255)*(2/255)-1` 转换即与 wan preprocess 位等价,见 `test_robot_adapter_parity.py`);配置移动不改变组合出的 cfg 树;退火权重各 epoch 归一化且重分配守恒单测;iter-based 在线采样序列/断点续训冒烟;**离线序列加载与 lance 写入随 lance 环境落地**。模型侧 `ModelInputBatch→forward` 适配与损失位等价闸口在 P3(随 connectors 一起,届时 wan 张量入口 + 变体条件接受张量首帧) |

`transforms/` 删除推迟到 P3 模型侧闸口(损失逐比特一致)通过后(独立 PR);五个注册表保留(§1.3 规则 1),不列入删除。