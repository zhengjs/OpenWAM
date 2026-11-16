# 技术规范:OpenWAM-AutoDrive — 重构与扩展

## 1. 概述

### 1.1 目标

将 **OpenWAM**(一个机器人操作 WAM/VLA 框架)改造为面向生产的自动驾驶 VLA/WAM 套件,复用:

- **OpenWAM(基础仓库)**:`dual_system` / `tri_system` / `single_system` WAM 架构、
  video backbone(Wan2.1/2.2、Cosmos-Predict2.5、Cosmos3-Edge)、视觉编码器(DINOv3、
  V-JEPA 2.1、Wan22-VAE / SVAE / Flux2-VAE)、VLM backbone(Qwen3-VL)、Accelerate+DeepSpeed
  训练运行时,以及部署栈(WebSocket 策略服务器 + 自包含 checkpoint)。
- **starVLA(吸收)**:VLA 变体。QwenPI / QwenOFT / QwenGR00T **已移植完成**
  (`openwam/model/architectures/vla/`、`openwam/model/action_backbone/vla_heads/`、`configs/model/vla.yaml`)。
  仍待移植:**词表内嵌的 FAST 头**、**RTC(Real-Time Chunking)训练与推理**、Qwen 特殊 token 词表扩展工具。
- **FluxVLA(吸收,仅工程部分)**:FSDP 集群工程 — 混合分片、按前缀的逐模块学习率、
  聚合式 checkpoint + 优化器重分片恢复、确定性多源均衡采样,以及样本权重器(RA-BC / AW-BC)。

### 1.2 已确认决策

| ID | 决策 |
|----|------|
| D1 | **包名冻结。** `openwam/model`、`openwam/dataloader`、`openwam/train`、`openwam/deploy` 保留名称;重构只**新增**子包/文件(`contracts/`、`dataloader/preprocessors/`、`interfaces/`、`train/` 下新文件)。不做全仓库 import 变更,不做 `model → models` 重命名。 |
| D2 | **双训练后端。** 保留现有 Accelerate+DeepSpeed ZeRO 路径并作为默认;新增 torchrun+FSDP runner,通过 `training.backend: deepspeed \| fsdp` 选择。两者共享同一架构、配置与 checkpoint 契约。 |
| D3 | **双轨数据层。** 保留 LeRobot v3 读取器(机器人域,四个预训练源);新增 Lance 索引 + 原始片段存储(驾驶域)。两条轨道输出相同的 `PipelineContext`;`MixtureDataset` 升级为跨格式混合。 |
| D4 | **FAST 头采用词表内嵌方式**(starVLA 原始设计:FAST BPE → VLM 词表内的 `<robot_action_xx>` 特殊 token → next-token CE 训练 → `model.generate` 推理 + FAST 解码)。**不**建模为独立的 `ActionHead`。 |
| D5 | **移除 `presets/fast.env` / "RBA" 引用** — 本仓库不存在这些内容。加速规范(§7.5)引用真实机制:逐层梯度检查点、冻结编码器 CPU offload(现有 `wan/shared/core/vram` 管理)、冻结 `no_grad` 前向包装。 |

### 1.3 强制工程规则

1. **基于注册表的实例化(保留)。** 组件实例化继续走仓库现有的装饰器注册表
   (`model/architectures/registry.py`、`model/video_backbone/registry.py`,
   `model/video_backbone/encoder/registry.py`、`model/vlm_backbone/registry.py`,
   `dataloader/registry.py`)。评审结论:现有注册表**不**存在别处禁用 MMEngine 风格
   所针对的调试/导航痛点 — 注册是静态的(每文件一个类、import 时完成),
   配置错误在构建时即失败,调用点可 grep — 因此它们保持权威机制,
   **不**列入删除目标。新组件类(connectors、preprocessors、Track B 读取器)
   按同一模式注册 — 注册到其组件类别对应的现有注册表,或在其旁新建同模式注册表文件。
   `model/architectures/registry.py::resolve_architecture_config` 的
   `framework/variant → class` 解析逻辑**原地保留**,
   因为旧 checkpoint 配置依赖它(§4.3)。
2. **纯张量模型图。** `nn.Module` 架构(`WAMDualSystem`、`WAMTriSystem`、
   `VLASystem`、…)只接受和返回张量(以及冻结的 dataclass 契约)。
   **边界澄清(重要):** PreProcessor 链负责视频解码、坐标变换、状态归一化、
   图像缩放/分块、prompt 渲染与*分词* — 全部是 CPU worker 侧操作。GPU 侧编码器
   **留在模型内部**:VAE/视觉编码器 latent 编码、冻结文本编码器嵌入(WAM)、
   VLM 视觉前向(VLA)。因此 `ModelInputBatch` 携带**原始尺度归一化后的像素张量**,
   且对 WAM 携带 **prompt 原始字符串列表**(如今天一样由冻结文本编码器在 GPU 上编码);
   对 VLA 携带 `input_ids` / `attention_mask`(在 dataloader worker 中分词)。
   VAE latent、文本嵌入与 VACE 条件化绝不放在 dataloader 侧。
3. **高复用 / 最小修改半径(DRY)。** 共享逻辑(归一化器、注意力掩码构建、
   损失聚合、Flow-Matching 组件、调度器)放在共享基类或 `*_components` 模块中
   (现有 `vla_heads/fm_components.py`、`action_backbone/components.py` 即为范例)。
   WAM 与 VLA 路径之间不得有胶水代码重复。
4. **可组合 Hydra 配置 + 自包含 checkpoint。** 开发期保留 Hydra 组合。
   启动时 rank 0 解析所有插值,并将完全解析的配置写入**每个 checkpoint 目录**的
   **`config.yaml`**(文件名不变 — `deploy/model_loader.py` 与 pytest 套件依赖它),
   外加 `git_meta.json` 与归一化统计(§5.2)。
5. **与 OpenWAM 核心向后兼容。** `dual_system`(3 变体)、`tri_system`、
   `single_system`(vanilla/moe)、`vla`(oftp/pi/groot)、所有 video backbone、
   所有视觉编码器、Qwen3-VL backbone、部署自包含 checkpoint 契约
   (`config.yaml` + `checkpoint_step_*.safetensors` + `normalization_stats.npy` + 部署资产),
   以及现有 pytest 套件,必须始终可用。

---

## 2. 目标目录结构(增量式)

```text
OpenWAM/
├── configs/
│   ├── train.yaml                            # [已有] + training.backend: deepspeed|fsdp
│   ├── model/
│   │   ├── dual_system.yaml                  # [已有]
│   │   ├── tri_system.yaml                   # [已有]
│   │   ├── single_system.yaml                # [已有]
│   │   ├── vla.yaml                          # [已有;扩展 connector/cotraining 键]
│   │   ├── video_backbone/                   # [已有,不动]
│   │   ├── vlm_backbone/                     # [已有]
│   │   ├── action_backbone/                  # [一个分组 — WAM ActionDiT 与 VLA 头是同一
│   │   │                                     #   概念;镜像代码布局 model/action_backbone/,
│   │   │                                     #   含 vla_heads/]
│   │   │   ├── separate_action_dit.yaml      # [已有] dual/tri ActionDiT
│   │   │   ├── shared_action_backbone.yaml   # [已有]
│   │   │   ├── mlp_l1.yaml                   # [VLA OFT 头]
│   │   │   ├── layerwise_fm.yaml             # [VLA PI 头]
│   │   │   └── dit_fm.yaml                   # [VLA GR00T 头]
│   │   └── connector/                        # [新增] layerwise_hidden.yaml、last_hidden_tokens.yaml、
│   │                                         #       last_hidden_cross_attn.yaml
│   ├── dataloader/
│   │   ├── mixture.yaml                      # [已有;升级:跨格式条目]
│   │   ├── robot/                            # [移动] 全部 11 个现有 LeRobot v3 读取器配置
│   │   │   └── robotwin.yaml, agibotworld.yaml, …
│   │   ├── driving/                          # [新增] Lance 数据集配置 — 预期扩展到
│   │   │   ├── _base.yaml                    #       数十/上百个;共享模板,每个数据集
│   │   │   └── <dataset>.yaml                #       文件保持精简(存储根/过滤/标签)
│   │   └── preprocessors/                    # [新增] wam_canvas_chain.yaml、vla_qwen_chain.yaml
│   ├── runner/
│   │   └── fsdp.yaml                         # [新增] FSDP 分片/包装/混合精度
│   └── deploy.yaml                           # [已有]
├── openwam/
│   ├── contracts/                            # [新增]
│   │   ├── config_schema.py                  # 校验新 YAML 块(connector:、preprocessor 链)的类型化 dataclass
│   │   ├── data_contract.py                  # PipelineContext、ModelInputBatch
│   │   ├── model_contract.py                 # BackboneOutput、ConnectorOutput、ActionHeadOutput、SystemOutput
│   │   └── collate.py                        # PipelineContext -> ModelInputBatch 组批
│   ├── dataloader/
│   │   ├── bases/                            # [已有] LeRobot v3 读取器(Track A,保留)
│   │   ├── *.py                              # [已有] 各数据集读取器
│   │   ├── transforms/                       # [已有] 保留至 preprocessors 达到对等(§10 P2)
│   │   ├── lance_dataset.py                  # [新增] Lance 切片/标签/过滤下推读取器(Track B)
│   │   ├── clip_reader.py                    # [新增] 原始多相机 MP4 + topics parquet 读取器
│   │   ├── mixture.py                        # [已有;升级为跨格式]
│   │   ├── registry.py                       # [已有,保留] Track B 读取器在此注册(§1.3 规则 1)
│   │   └── preprocessors/                    # [新增]
│   │       ├── base.py                       # BasePreprocessor、PreProcessorPipeline、InferenceProcessorPipeline
│   │       ├── video_decoder.py              # MultiCamVideoDecoder
│   │       ├── time_sync.py                  # TopicTimeSynchronizer
│   │       ├── ego_processor.py              # EgoStateProcessor
│   │       ├── image_processor.py            # MultiViewImageProcessor
│   │       └── prompt_processor.py           # VLAPromptTokenizeProcessor
│   ├── model/
│   │   ├── architectures/
│   │   │   ├── base.py                       # [已有] BaseArchitecture / BaseWAMArchitecture
│   │   │   ├── dual_system/ tri_system/ single_system/   # [已���]
│   │   │   ├── vla/                          # [已有] base_vla.py、qwen_pi/oft/groot(并入 §6.3 设计)
│   │   │   └── connectors/                   # [新增] layerwise_hidden.py、last_hidden.py(joint_self_attn 留在 WAM)
│   │   ├── action_backbone/
│   │   │   ├── vla_heads/                    # [已有] MLP / FlowMatching / LayerwiseFM / DiT 头
│   │   │   └── scheduler.py components.py    # [已有]
│   │   ├── video_backbone/                   # [已有,不动]
│   │   └── vlm_backbone/
│   │       ├── qwen3_vl_backbone.py          # [已有]
│   │       └── fast_vocab/                   # [新增] 移植 starVLA add_qwen_special_tokens + FAST BPE 包装
│   ├── train/
│   │   ├── openwam_trainer.py                # [已有] Accelerate+DeepSpeed 路径(默认后端)
│   │   ├── fsdp_runner.py                    # [新增] torchrun FSDP runner
│   │   ├── checkpoint_manager.py             # [新增] 双后端统一保存/恢复
│   │   ├── config_freezer.py                 # [新增] 完全解析的 config.yaml + git_meta.json 写入器
│   │   └── utils/                            # [已有]
│   ├── deploy/                               # [已有] 契约不变(§8.1)
│   └── interfaces/                           # [新增桩]
│       ├── rl_stub.py
│       ├── eval_server_stub.py
│       └── deploy_trt_stub.py
├── scripts/
│   ├── train.py / train.sh                   # [已有] DeepSpeed 路径
│   ├── train_fsdp.sh                         # [新增]
│   └── smoke_test.py                         # [新增] 组件契约与前向/反向测试器
└── tools/                                    # [新增] 离线数据工具
    ├── ingest_lance.py                       # 从原始片段构建 Lance 表(含 cam_frame_offsets 预计算)
    └── fit_fast_tokenizer.py                 # 按动作空间拟合/保存 FAST BPE
```

未列出的现有文件一律不动,仅有两处例外,各自在阶段闸口(§10)落地,
绝不在同一 PR 内既增又删:(a) `configs/dataloader/robot/*.yaml` 是**移动** —
`train.yaml` 默认值与 `mixture.yaml` 必须重映射 Hydra 包
(`- dataloader/robot@dataloader: robotwin`、`- robot/agibotworld@datasets.agibotworld`,…),
使 `cfg.dataloader.*` — 进而 `build_dataset(cfg.dataloader)` 与每个读取器 — 保持源码兼容;
由回归测试断言移动前后组合出的配置树逐字节一致。(b) 计划末尾的删除目标:
preprocessors 达到对等后的 `dataloader/transforms/`。五个注册表文件保留
(§1.3 规则 1),**不**列入删除目标。

---

## 3. 契约(`openwam/contracts/`)

所有跨模块通信使用这些类型。`PipelineContext` 是 DataLoader worker 内部的可变单元;
`ModelInputBatch` 是进入模型的冻结单元。

### 3.1 `data_contract.py`

**设计意图。** 恰好一个上下文类同时服务两条轨道。不存在机器人专用或驾驶专用的
字段*槽位* — 域差异由"谁填充某个槽位、如何填充"表达(Track A 适配器 vs
Track B 预处理器),绝不通过额外的类型表达。下文"必需"按每次运行评估:
运行配置声明它使用哪些能力(状态输入?动作损失?协同训练?),
数据集层随后必须保证相应字段被填充。

```python
@dataclass
class PipelineContext:
    """可变、链式传递的上下文。恰好对应一个训练切片(一个片段的一个窗口)。"""

    # ── 1. 索引元数据 — 共享,始终必需 ──────────────────────────────────────
    #    由数据集层填充(Lance 行 / LeRobot 窗口适配器)。
    #    目的:身份 + 存储寻址 + 读取计划。这里没有任何内容
    #    在读取时派生(在 ingest 时预计算,§5.2)。

    # 用于日志 / 评估 / 恢复审计的连接键。
    sample_id: str

    # 源片段的存储定位符(驾驶:raw_clips/<id>/;机器人:
    # LeRobot 仓库根目录)。只有解码器/读取阶段解引用它。
    clip_uri: str

    # 同步原点。驾驶:锚定相机的曝光时刻;机器人:窗口起点。
    # 所有 topic 插值以它为锚。
    anchor_timestamp_us: int

    # 每相机待读帧索引 — 完整读取计划。驾驶:在 ingest 时
    # 预计算进 Lance。机器人:由适配器从窗口边界合成(LeRobot
    # 按相机存储 MP4 分块,因此同样的读取计划存在)。
    # 解码器执行零索引计算。
    cam_frame_offsets: Dict[str, List[int]]

    # 本切片来自哪个数据集/表(mixture 日志、按源指标、统计解析)。
    source_name: str

    # 任务指令。驾驶:导航文本;机器人:其 prompt(见下方
    # Track A 映射)。纯世界模型样本(无动作损失、无语言条件化)时为 ""。
    nav_instruction: str = ""

    # ── 2. 可选元数据 — 共享 ─────────────────────────────────────────────────

    # 来自权重器(constant / success_rate / progress_delta);由损失
    # 消费并做全局归一化(§5.4)。1.0 = 普通 BC。
    sample_weight: float = 1.0

    # 过滤 / 课程学习 / mixture 分层(Lance SQL 下推)。
    scene_tags: Optional[List[str]] = None

    # 思维链(CoT)目标。存在 ⇒ prompt 处理器构建 vlm_labels
    # (协同训练 §6.5);不存在 ⇒ vlm_labels 保持 None。
    cot_text: Optional[str] = None

    # ── 3. 标定与时序 — 仅驾驶 ────────────────────────────────────────────────
    #    机器人 rig 固定且隐含在画布布局中,因此 Track A 将两个槽位留 None。

    # 每个 Track B 运行必需。每相机:{"extrinsics": [4,4]
    # 相机->ego-rig,"intrinsics": [3,3]}。从 calib.json 读取(§5.2);
    # 由 TopicTimeSynchronizer 消费(坐标变换)。
    calib: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    # 每个 Track B 运行必需。真实的每相机曝光时间戳;同步
    # 容差闸口(§12 O1)读取它们。Track A:None(均匀 fps 是
    # LeRobot 的保证,无需闸口)。
    frame_timestamps_ns: Optional[Dict[str, List[int]]] = None

    # ── 4. 原始解码物理量 — 共享槽位,分轨填充 ────────────────────────────
    #    由张量化预处理器消费;绝不离开 worker。

    # {cam: [T_obs, H_cam, W_cam, 3] uint8},每相机**原生**分辨率。
    # 分辨率层级(front > side > rear,§3.1 批注)在此保留;
    # 重缩放唯一归图像处理器所有。
    raw_images: Optional[Dict[str, np.ndarray]] = None

    # [T_hist, D_state] float32,ego-rig 坐标系。驾驶:在锚点插值的
    # ego 运动学历史。机器人:T_hist=1 的本体感知(当前状态)。
    # None = 无状态运行。
    raw_ego_history: Optional[np.ndarray] = None

    # [T_pred, D_action] float32,ego-rig 坐标系。驾驶:未来 ego 规划;
    # 机器人:动作窗口。运行训练动作头时必需。
    raw_gt_trajectory: Optional[np.ndarray] = None

    # [T_pred] 或 [T_pred, D_action] bool。只要设置了
    # raw_gt_trajectory 就必需。2-D 形式保留**逐维**有效性,
    # 机器人 80-D 统一空间强制使用(保留/未映射维度
    # 必须留在损失之外);1-D 形式足以表达驾驶规划。
    raw_trajectory_valid_mask: Optional[np.ndarray] = None

    # ── 5. 处理后张量 — 共享,由预处理器链产出 ─────────────────────────────

    # wam_canvas / qwen_vl_patch 布局(批注见 §3.1)。
    pixel_values: Optional[Tensor] = None

    # standard_tensor 模式,每样本:{cam: [T_obs, C, H_cam, W_cam]},
    # 该相机的层级分辨率。Collate 按相机逐个堆叠
    # → [B, T_obs, C, H_cam, W_cam]。
    pixel_values_by_cam: Optional[Dict[str, Tensor]] = None

    # 仅 qwen_vl_patch,[N_img, 3]。
    image_grid_thw: Optional[Tensor] = None

    # WAM 路径:原始字符串;模型侧文本编码。
    prompt: Optional[str] = None

    # VLA 路径(worker 内分词)。
    input_ids: Optional[Tensor] = None

    # 同上(每样本 [L_seq];collate 时填充)。
    attention_mask: Optional[Tensor] = None

    # 存在 cot_text 时,prompt token 置 -100。
    vlm_labels: Optional[Tensor] = None

    # [T_hist, D_enc](机器人本体感知 T_hist=1)。
    ego_state_tensor: Optional[Tensor] = None

    # [T_pred, D_action]。
    gt_trajectory_norm: Optional[Tensor] = None

    # [T_pred] 或 [T_pred, D_action] bool。
    trajectory_mask: Optional[Tensor] = None
```

**字段分类总表**

| 块 | 字段 | 域 | 何时必需 |
|---|---|---|---|
| 1 索引 | `sample_id, clip_uri, anchor_timestamp_us, cam_frame_offsets, source_name` | 共享 | 始终 |
| 1 索引 | `nav_instruction` | 共享 | 语言条件化或动作损失运行 |
| 2 可选 | `sample_weight, scene_tags, cot_text` | 共享 | 分别对应:权重器 / 过滤 / 协同训练 |
| 3 标定 | `calib, frame_timestamps_ns` | **仅驾驶** | 所有 Track B 运行(Track A 为 None) |
| 4 原始 | `raw_images` | 共享 | 任何视觉模型 |
| 4 原始 | `raw_ego_history` | 共享(驾驶 `T_hist>1`;机器人 `T_hist=1` 本体感知) | 状态条件化运行 |
| 4 原始 | `raw_gt_trajectory, raw_trajectory_valid_mask` | 共享 | 动作损失运行(机器人用 2-D 逐维掩码) |
| 5 处理后 | `pixel_values … trajectory_mask` | 共享 | 由所配置的链产出 |

**Track A 映射(规范性)。** 机器人读取器填充**相同**的槽位 — 本体感知 →
`raw_ego_history`(T_hist=1),动作窗口 → `raw_gt_trajectory`,动作/本体逐维掩码 →
`raw_trajectory_valid_mask`(2-D 形式),prompt → `nav_instruction`。仅机器人侧的
簿记信息**不**成为上下文数据:80-D `unify_action_map`(读取器 yaml 中的静态每源配置)、
颜色抖动状态(处理器内部)。Track A 不得派生第二个上下文类。

```python
@dataclass(frozen=True)
class ModelInputBatch:
    """纯张量批。只有这些进入 BaseArchitecture.forward()。

    多相机分辨率层级(驾驶):一帧携带约 7 个环视相机,算力通过
    分辨率分层控制 — 前相机最大,侧相机较小,后相机最小。每相机
    目标分辨率是静态配置(MultiViewImageProcessor.per_cam_resolution,
    冻结进 config.yaml — 绝非逐样本批数据)。

    规范性:分辨率异构性**贯穿**视觉编码器。每个相机按自身层级
    编码;各相机特征网格空间尺寸不同;各流仅在逐相机展平为 token
    之后合并 — 沿配置相机顺序沿序列维拼接,统一注意力掩码,
    合并前附加每相机 RoPE / 位置嵌入。合并点在**模型侧**
    (视觉编码器包装 / video-backbone 适配器,§6.2)— 绝不在
    预处理器链。三种布局,每个批恰好其一:

      wam_canvas       pixel_values [B, T_obs, C, H_canvas, W_canvas]
                       视图合成到一个画布;槽位尺寸跟随相机层级
                       (front 大,rear 小)— 泛化机器人 L 形;
                       一个规则网格,零填充浪费。Tier-1 路径:
                       与今天的 backbone 不变即用。
      standard_tensor  pixel_values_by_cam {cam: [B, T_obs, C, H_cam, W_cam]}
                       各相机张量保持自身层级分辨率 — 不上采样
                       小相机;backbone 适配器逐相机分块并在
                       token 级合并(§6.2)。
      qwen_vl_patch    pixel_values 按图像分块打包(数量可变);
                       原生可变分辨率;image_grid_thw 记录每张图
                       的 [T, H/patch, W/patch] — 零填充浪费。
                       格式 = HF Qwen-VL 约定,与
                       qwen3_vl_backbone.batch_vlm_inputs 字节兼容
                       (§3.1 注)。
    """

    # 视觉(按预处理器链,以上三种布局恰好其一):

    # wam_canvas 或 qwen_vl_patch。
    pixel_values: Optional[Tensor] = None

    # 仅 standard_tensor。字典插入顺序 = 配置相机顺序;
    # H/W 跨相机按层级不同,批内同相机内一致。
    pixel_values_by_cam: Optional[Dict[str, Tensor]] = None

    # 仅 qwen_vl_patch:[N_img_total, 3] long,每张图 14-px patch
    # 单位的 (t, h, w),跨帧/相机/批**拼接** — 无 batch 维、
    # 无填充(HF 约定;qwen3_vl_backbone.batch_vlm_inputs 沿
    # dim=0 拼接)。图像边界通过 cumsum(t*h*w) 恢复;
    # 相机身份/顺序由 prompt 处理器的占位符展开顺序确定。
    image_grid_thw: Optional[Tensor] = None

    # [B, T_obs] bool,帧有效性(WAM)。
    video_mask: Optional[Tensor] = None

    # 语言(恰好一条路径):

    # WAM:模型侧文本编码(GPU,冻结)。
    prompt: Optional[List[str]] = None

    # VLA:[B, L]。
    input_ids: Optional[Tensor] = None

    # [B, L]。
    attention_mask: Optional[Tensor] = None

    # [B, L],-100 掩码。
    vlm_labels: Optional[Tensor] = None

    # 状态 / 动作监督:

    # [B, T_hist, D_enc];接受遗留 [B, D]。
    ego_state_tensor: Optional[Tensor] = None

    # [B, T_pred, D_action]。
    gt_trajectory_norm: Optional[Tensor] = None

    # [B, T_pred] bool(涵盖 action_is_pad)。
    trajectory_mask: Optional[Tensor] = None

    # [B] float32。
    sample_weights: Optional[Tensor] = None
```

> 刻意**缺席**(模型侧,绝不在 dataloader 侧):VAE latent
> (`input_latents` / `first_frame_latents`)、文本嵌入(`context`/`context_mask`)、
> VACE 条件化、`condition_mask`、`num_clean_prefix_frames`。WAM 架构现有的
> `prepare_inputs`/`preprocess`(GPU,no_grad)消费 `pixel_values`/`prompt`
> 并按今天的方式产出它们 — 机制不变。

### 3.2 `model_contract.py`

**命名约定(规范性)。** 两个正交轴:*状态*字段按来源命名
(`last_hidden_state`、`hidden_states` — 哪一层);*掩码*字段按其掩蔽的
序列命名(`attention_mask`、`context_mask`、`condition_mask`、`video_mask`、
`trajectory_mask`)。掩码是 token **序列**的属性,从不是单层的属性 —
一个掩码服务所有层的状态,因此掩码命名从不跟随层命名。
(HF 没有定义输出侧掩码约定;此处为内部一致性规则。)

```python
@dataclass
class BackboneOutput:
    """VLMBackbone / VideoBackbone 的标准化输出。

    命名遵循人人皆知的 HF 约定:last_hidden_state 是最后一层;
    hidden_states 是逐层堆栈,hidden_states[-1] **就是**
    last_hidden_state(别名,不拷贝)。
    """

    # [B, L_ctx, D_model] — 最终层隐藏状态;默认条件化来源,
    # 以及 encode_scene_tokens 消费的低频编译单元。
    last_hidden_state: Tensor

    # [B, L_ctx] — token 序列上的有效性(所有层共享)。
    context_mask: Tensor

    # 可选:transformer 各层的隐藏状态,HF output_hidden_states
    # 约定 — embeddings 在前,每层一个 [B, L_ctx, D] 条目,
    # hidden_states[-1] **就是** last_hidden_state。len() 自描述
    # 层数 — 下游无需查配置。仅在运行配置要求时填充(连接器
    # 声明其层需求,例如 connector=layerwise_hidden 且
    # num_layers=N — backbone 则恰好返回那些层)。消费者:
    #   - QwenPI 逐层交叉注意力条件化(§6.3)
    #   - DeepStack 风格多层路由(Qwen3-VL 已有)
    #   - 辅助深监督损失(未来)
    # 默认 None:完整堆栈代价约 N_layers × L_ctx × D × 2 字节
    # (4B VLM 每样本数百 MB)— 绝不投机性物化。
    hidden_states: Optional[List[Tensor]] = None

    # 视频流匹配损失(WAM 世界流)。
    world_pred_loss: Optional[Tensor] = None

    # 语言 CE(VLA 协同训练,§6.5)。
    vlm_loss: Optional[Tensor] = None

    # 解码阶段缓存(仅推理)。
    kv_cache: Optional[Dict[str, Tensor]] = None

@dataclass
class ConnectorOutput:
    """连接器的标准化输出,由动作头消费。

    每条被消费的 VLM 层一条条件流;列表长度自描述层数
    (镜像 BackboneOutput.hidden_states)。
    """

    # 逐层条件流,每个条目 [B, L_src, D_head] — 所有条目在
    # **同一** token 序列上(VLM 的),因此长度相同、
    # 逐条目掩码相同:
    #   - 标准路由(last_hidden_tokens、last_hidden_cross_attn):
    #     len == 1;单流头取 condition_tokens[0]。
    #   - 逐层路由(layerwise_hidden / QwenPI):len == N_layers;
    #     DiT 第 i 层交叉注意力 condition_tokens[i]。层数错误
    #     会在头的 len()/shape 断言处大声失败 — 没有任何东西
    #     被打包成需要外部配置解码的形状。
    # 层→DiT 层映射("最后 N 层按序" vs "均匀桥接")是
    # 连接器配置;列表本身零打包歧义。
    condition_tokens: List[Tensor]

    # [B, L_src] — 一个掩码服务所有流(逐条目相同)。
    condition_mask: Tensor

@dataclass
class ActionHeadOutput:
    """动作头的标准化输出。"""

    # 监督损失(FM 速度 MSE / L1 / FAST token CE)。
    action_loss: Optional[Tensor] = None

    # [B, T_pred, D_action] — 连续头(FM / MLP)。
    pred_trajectory_norm: Optional[Tensor] = None

    # [B, L_tok] — FAST 路径 token id(§6.4);此时 trajectory 为 None。
    pred_action_tokens: Optional[Tensor] = None

    # 为 RL 后训练保留(§8.2)。
    action_logprobs: Optional[Tensor] = None

@dataclass
class SystemOutput:
    """BaseArchitecture.forward() 的顶层返回。"""

    total_loss: Optional[Tensor] = None

    action_loss: Optional[Tensor] = None

    world_pred_loss: Optional[Tensor] = None

    vlm_loss: Optional[Tensor] = None

    pred_trajectory_norm: Optional[Tensor] = None

    pred_action_tokens: Optional[Tensor] = None

    # 标量日志指标(已 detach)。
    metrics: Optional[Dict[str, float]] = None
```

### 3.3 组批与可逆性

- `contracts/collate.py` 负责 `PipelineContext -> ModelInputBatch`:逐字段堆叠、
  填充(`attention_mask`、`image_grid_thw` 拼接 + 填充到 `max(N_img)`)、
  掩码和解(1-D/2-D 掩码调和 — 复用 `BaseArchitecture.prepare_inputs` 中
  已有的调和规则)。`pixel_values_by_cam` 按相机逐个跨批堆叠
  (相机迭代顺序 = 配置相机顺序;缺相机是硬错误,不是零填充)。
  任何架构不得自行组批。
- 可逆性是一等性质:`EgoStateProcessor.denormalize_trajectory(norm) -> ndarray`,
  图像操作保留反演所需参数;部署路径通过**同一**对象反归一化
  (用一个归一化器实例替代 starVLA 独立的 "unapply" 变换,
  从 checkpoint 本地统计加载)。

---

## 4. 组件装配与配置冻结

### 4.1 组件装配(基于注册表)

- 实例化继续走**现有装饰器注册表**(§1.3 规则 1):架构走
  `model/architectures/registry.py`,video/vlm backbone 走各自注册表,
  数据集走 `dataloader/registry.py`。新组件类按同一模式接入 — connectors
  与架构并列注册,preprocessors 在 `dataloader/preprocessors/` 建注册表
  (或在 `dataloader/registry.py` 中登记),沿用同一装饰器模式。
- 动作配置组保持**单一** — WAM ActionDiT 与 VLA 头(MLP / FlowMatching / LayerwiseFM)
  是一个概念,共享一个配置组(`configs/model/action_backbone/`,
  镜像代码布局 `model/action_backbone/` 含 `vla_heads/`),
  按现有 `action_model_type` 判别器分派;文件内块键(`vla.yaml` 中的 `action_head:`、
  `dual_system.yaml` 中的 `action_backbone:`)保持名称以兼容配置。
  新 YAML 块(`connector:`、动作配置、preprocessor 链)在 `contracts/config_schema.py`
  中定义 dataclass;构建时用 `OmegaConf.structured(...)` 校验 —
  schema 错误快速失败,而非在前向时。
  **范围控制**:全量 DictConfig→dataclass 转换**不是**目标;
  只有新块和架构顶层参数被类型化。

### 4.2 配置冻结(`train/config_freezer.py`)

- 启动时,Hydra 组合之后、**第一个分布式集合通信之前**(使所有 rank 看到
  同一真相,rank 0 的写入不可能发散),rank 0 写入 `<work_dir>` 与每个
  未来 checkpoint 目录:
  - `config.yaml` — `OmegaConf.to_yaml(cfg, resolve=True)`
    (所有插值与环境变量已解析)。文件名与今天的 `save_config` 写一次行为一致;
    本次将其内容升级为完全解析。
  - `git_meta.json` — `{commit, dirty, diff}`(diff 是强制的:开发期配置
    在"干净"提交之间也会漂移)。
  - 归一化统计:新运行写 `dataset_statistics.json`;加载器**双读**遗留
    `normalization_stats.npy`(嵌套 `{joint, eef}` schema),
    旧 checkpoint 部署不变(§5.2)。
- `load_frozen_config(ckpt_dir)` 工具供评估/部署;Phase 1 之后它必须是
  `deploy/model_loader.py` 的唯一配置来源。

### 4.3 遗留 checkpoint 兼容(规范性)

当前 OpenWAM 保存的 checkpoint 含带 `model.architecture.{framework,variant}`
与注册表时代键的 `config.yaml`。这些配置继续通过**原样保留**的
`model/architectures/registry.py::resolve_architecture_config` 映射加载 —
旧 checkpoint 继续训练(finetune/resume 经 `train/utils/ckpt_model_loader.py`)
并零迁移部署。这是一条长期测试:"加载一个黄金遗留 checkpoint 目录"
自 Phase 1 起在 CI 运行。

---

## 5. 数据集层(双轨,D3)

### 5.1 Track A — LeRobot v3(机器人域;已有,保留)

`dataloader/bases/lerobot_v3_reader.py` 及其四个预训练读取器保持原样。
一个薄适配器把它们的规范样本字典包进 `PipelineContext`(映射见 §3.1)。
它们现有的 transforms(`transforms/`)继续运行,直到预处理器链对 Track A
达到对等(阶段闸口,§10 P2);对等之后 transforms 包删除,不留 shim。

### 5.2 Track B — 驾驶数据存储(新增)

```text
<storage_root>/raw_clips/<clip_id>/
    cams/<cam_name>.mp4                  # ~7V 环视;每相机分辨率层级
                                         #   (front > side > rear,§3.1),H.264/H.265
    topics/ego_motion.parquet            # 高频车辆总线
    topics/localization.parquet          # 世界系 ego 位姿
    topics/navigation.parquet            # 指令
    calib.json                           # 每相机内参/外参(相机->ego-rig)
<storage_root>/lance_indices/<dataset_name>.lance
    sample_id, clip_uri, anchor_timestamp_us,
    cam_frame_offsets(在 ingest 时由 tools/ingest_lance.py 预计算 — 读取器
        绝不从 num_frames/video_stride 重新推导),
    nav_instruction, cot_text, scene_tags, sample_weight,
    gt_future_traj_ego (fixed_size_list<float32>), valid_mask
```

- `LanceClipDataset`:SQL 过滤下推(`lance.dataset(uri).to_table(filter=...)`)、
  列投影、动态标签/子集选择。
- 统计:由 `tools/ingest_lance.py` 离线计算或首次加载时计算
  (rank-0 计算,其他等待 — 与今天 auto-stats 相同协议)→ `dataset_statistics.json`。
- 数据治理:ingestion 负责去标识化(人脸/车牌打码)与 `clip_uri` 访问控制;
  本仓库假设只读、已脱敏的存储(§12 O5)。

### 5.3 可组合 PreProcessor 链(`dataloader/preprocessors/`)

`BasePreprocessor.process(ctx: PipelineContext) -> PipelineContext`,
属性 `is_training_only: bool`。链顺序由配置声明
(`configs/dataloader/preprocessors/*.yaml`)。

| 处理器 | 仅训练 | 职责 |
|---|---|---|
| `MultiCamVideoDecoder` | 是 | 从 MP4 读取 `cam_frame_offsets` 帧。decord/PyAV,每 worker 单例句柄(10 workers × 11 相机不得炸句柄/RSS — 每 worker 有界 LRU)。颜色抖动(仅训练、时间一致)从 `transforms/video.py` 移到这里。 |
| `TopicTimeSynchronizer` | 是 | 在相机曝光时间戳处插值 CAN/ego topics → ego-rig 坐标系的 `raw_ego_history` / `raw_gt_trajectory`(使用 `calib`)。插值类型、容差闸口、超闸策略(丢弃 vs 无效掩码)是配置字段;**默认值是未决项(§12 O1)**。旋转工具复用 `dataloader/utils/poses.py` / `rotation.py`(6D/四元数)。 |
| `EgoStateProcessor` | 否 | 用 `dataset_statistics.json` 归一化历史/轨迹,可选 Fourier 嵌入 → `ego_state_tensor`、`gt_trajectory_norm`、`trajectory_mask`。提供 `denormalize_trajectory`。还提供 π0.5 风格的**状态即文本**渲染(来自 `vla/base_vla.py` 的 `state2str`),由 prompt 处理器消费。 |
| `MultiViewImageProcessor` | 否 | 配置:`per_cam_resolution: {cam: [H, W]}`(驾驶层级:front > side > rear)+ `mode`。三种模式:`wam_canvas`(视图合成到一个画布,槽位尺寸跟随相机层级 — Track A 用现有 L 形,驾驶泛化为 7 槽)、`standard_tensor`(每相机张量 `{cam: [T_obs, C, H_cam, W_cam]}` 保持各自层级分辨率 — **不上采样**;合并进单一 token 流发生在模型侧,§6.2)、`qwen_vl_patch`(每相机缩放到其层级,Qwen 动态分辨率分块 + `image_grid_thw` — token 级合并在此原生)。缩放/归一化参数 checkpoint 冻结(§4.2)。 |
| `VLAPromptTokenizeProcessor` | 否 | 渲染模板(导航指令、相机 token、可选状态即文本),分词 → `input_ids`/`attention_mask`,存在 `cot_text` 时构建 `-100` 掩码的 `vlm_labels`(协同训练,§6.5)。 |

双模式管线:

- **训练**:在 DataLoader worker 内运行所配置的链;经 `contracts/collate.py` 组批。
- **推理**:`InferenceProcessorPipeline.from_checkpoint(ckpt_dir)` 从冻结的
  `config.yaml` 重建链,过滤掉 `is_training_only=True` 的处理器,并把实时输入
  (`{cam: frame}` 字典、最新 ego 读数、指令)映射进 `PipelineContext`,
  再让保留的处理器运行 — 通过**同一代码路径**保证训练/部署对等,
  而非重新实现(这是 starVLA `PolicyNormProcessor` 的明确教训)。

### 5.4 混合与加权

- `MixtureDataset` 升级:条目可指向任意 Track A 读取器**或** Track B 表;
  现有 `weight_strategy`(`proportional/manual/uniform/inverse_size`)语义保留;
  逐 epoch 确定性重洗牌保留(带种子的索引映射 — 现状行为)。
- `sample_weight`:数据集侧由权重器产出(先做 `constant`;FluxVLA 风格的
  `success_rate` / `progress_delta` 是 RL 相邻数据到来时的参照),
  损失侧消费并做**跨 rank 与累积步的全局归一化**(FluxVLA
  `reduce_action_bc_loss` 模式),使有效逐批损失量级保持稳定。

### 5.5 吞吐

- 多相机 × 10 Hz × T_obs 解码是主导成本;链必须从第一天起在 step 日志中
  暴露逐阶段计时计数器。
- **离线 latent 预编码**(VAE latent 缓存在片段旁,解码处理器被绕过)
  提议为可选模式 — 待决策 §12 O2。若采纳,以 `LatentCacheReader` 处理器落地,
  短路解码+编码,架构接受预计算 latent(已支持:`inputs["input_latents"]` 路径)。
  缓存的 latent 遵循同样的每相机层级分辨率(异构 latent 网格),
  因此 §6.2 的 token 合并适配器也是那里的消费路径。

---

## 6. 模型层

### 6.1 `BaseArchitecture` 接口

所有顶层架构实现(现有 WAM 类把当前
`forward(noisy_actions, action_timestep, **pipeline_inputs)` 改道到该契约):

```python
# 主训练入口:批进,损失 / 指标出。
def forward(self, batch: ModelInputBatch) -> SystemOutput

# 低频场景编码 — 编译/缓存单元(TensorRT-LLM 目标):
# 每次观测计算一次,跨所有去噪步复用。
def encode_scene_tokens(self, batch: ModelInputBatch) -> ConnectorOutput

# 高频单步去噪 — 静态张量形状;CUDA-Graph / 实时分块(RTC)单元,
# 每 chunk 调用 len(schedule)-1 次。
def denoise_action_step(self, cond: ConnectorOutput, noisy_action: Tensor,
                        timestep: Tensor) -> Tensor

# 端到端推理便利入口:编码一次,按 schedule 迭代 denoise_action_step,
# 返回(反归一化的)轨迹。
def predict_trajectory(self, batch: ModelInputBatch) -> Tensor
```

现有 `generate()` 去噪循环保留为两个钩子由 schedule
(`deploy/denoise_schedule.py`)驱动的组合;CFG、dit-cache 与非激活维处理
原样移入循环。`prepare_inputs`(GPU,no_grad)把 `pixel_values`/`prompt`
桥接到 latent/上下文,与今天一致 — 它是 R2 边界的模型侧半边。

### 6.2 WAM 系统

`dual_system`(joint_cross_attn / joint_self_attn / idm)、`tri_system`、
`single_system`(vanilla / moe)保留其块级逻辑;重构是契约形状
(批进,`SystemOutput` 出),组件装配继续走注册表(§4.1)。`action_backbone`
(ActionDiT / shared)与 `video_backbone` 内部不动。

**异构多相机视觉路径(规范性)。** 当批携带 `pixel_values_by_cam`
(standard_tensor 模式)时,视觉编码器包装按各相机自身层级分辨率编码,
把每个相机特征网格展平为 token,并沿序列维按配置相机顺序拼接 token 流;
构建统一注意力掩码,并在**合并前**附加每相机 2-D RoPE / 位置嵌入。
下游(video DiT 块、连接器、动作头)看到单一 token 流 —
分辨率异构性从不在合并点之后泄漏。对 WAM video backbone 这是适配器层扩展:
Wan/Cosmos DiT 原生消费一个规则 latent 网格,因此适配器必须
(a) 逐相机运行视觉编码器,(b) 逐相机分块,
(c) token 拼接 + 每相机 RoPE + 统一掩码,可选(d) 插入每相机分隔符嵌入。
在该适配器落地之前,`wam_canvas`(一个规则网格、层级尺寸槽位)
是默认 WAM 驾驶路径;一个对等测试必须断言画布与 token 合并路径
在等分辨率 rig 上产生相同的注意力语义。

### 6.3 `VLASystem` — 吸收而非重建

VLA 家族**已存在**(`BaseVLAArchitecture`、`qwen_pi/oft/groot`,
头 MLP / FlowMatching-DiT / LayerwiseFM)。重构把隐式条件化逻辑提取为
显式**连接器**层,使 YAML 组合变体而无需子类增殖:

| `connector` | 映射到(今天) | 机制 |
|---|---|---|
| `layerwise_hidden` | QwenPI | 最后 N 层 VLM 隐藏状态(消费 `BackboneOutput.hidden_states`,仅配置了该连接器时填充),逐层交叉注意力进 LayerwiseFM DiT |
| `last_hidden_tokens` | QwenOFT | 从最后隐藏状态收集动作 token 位置(`_gather_action_token_embeddings`)→ MLP L1 头 |
| `last_hidden_cross_attn` | QwenGR00T | 最后隐藏状态(+ 状态嵌入、未来 token)→ DiT-B/L FlowMatching 头 |

`VLASystem` 组合:`vlm_backbone`(现有 Qwen3-VL backbone;冻结或协同训练)+
`ego_proj`(状态 → VLM 维;或配置时的状态即文本)+ `connector` + `action_head`
(复用与 WAM **相同**的 FlowMatching 头家族 — 共享 `fm_components.py`)。
绑定 VLM 的形状字段在运行时自动对齐(`configs/model/vla.yaml` 现有行为)。

### 6.4 FAST 头(词表内嵌,D4)

- 工具移植:`tools/fit_fast_tokenizer.py`(对目标动作空间拟合/保存 FAST BPE)
  与 `vlm_backbone/fast_vocab/`(特殊 token 区段扩展 — starVLA
  `add_qwen_special_tokens` 的移植,含词表大小填充与 resize 规则)。
- 训练:动作 chunk → FAST token → `<robot_action_xx>` id 追加进 labels;
  损失是 VLM 自身在这些位置的 next-token CE(`vlm_labels` 已建模此机制)。
- 推理:`model.generate` + token 区段提取 → FAST 解码 → 轨迹。
  `ActionHeadOutput.pred_action_tokens` 携带 id;`pred_trajectory_norm` 为 `None`。
- `FASTTokenHead` 作为独立头**被否决**(它会强迫 LM 头、generate 循环与
  kv-cache 穿过 ActionHead 抽象 — 对两者的扭曲)。

### 6.5 协同训练规则

- 损失:`total = action_loss + cfg.vlm_loss_weight * vlm_loss`,**一次反向**
  (兼容 Accelerate `accumulate` 与两个后端)。starVLA 的两次反向方案刻意不采纳。
- 构建时校验:`vlm_loss_weight > 0` 要求 VLM(或其声明部分)可训练;
  完全冻结 `vlm_backbone` 却请求 CE 损失的配置必须快速失败。
- CE 归一化:token 求和,跨 rank/累积全局归一化(FluxVLA
  `losses/causal_lm.py` 模式),避免序列长度偏置。
- `train_starvla_cotrain` 的双 dataloader 模式初期不需要(labels 同批携带);
  仅当加入纯语言语料阶段时再考虑。

### 6.6 RTC(Real-Time Chunking)

- 训练侧**延迟条件化**(每样本采样一个推理延迟;逐索引时间步条件化相应偏移 —
  starVLA `apply_rtc_time_conditioning` / FluxVLA `rtc_training_config` 的移植)
  进入 FlowMatching 头,受 `rtc_training_config.enabled` 控制。
  阶段位置:§12 O3。
- 推理侧固定前缀重采样移植进部署循环(`deploy/`),
  跨步复用 `encode_scene_tokens`。

---

## 7. 训练引擎(双后端,D2)

### 7.1 后端 A — DeepSpeed(已有,默认)

`scripts/train.py` + `openwam_trainer.py` 保留默认路径
(`training.backend: deepspeed`)。行为、种子处理(`per_step_seed`)、
ZeRO 1/2、`action_lr`/`video_lr`、`keep_last_k`、finetune/resume — 全部保留。
`training.backend` 默认 `deepspeed`,现有命令逐比特一致。

### 7.2 后端 B — FSDP runner(`train/fsdp_runner.py`,FluxVLA 衍生)

- torchrun 启动(`scripts/train_fsdp.sh`,环境变量与云调度器兼容,
  如 `scripts/train.sh`)。
- `sharding_strategy ∈ {full_shard, hybrid_shard(默认), shard_grad_op,
  hybrid_shard_zero2, no_shard}`;`use_orig_params=True`、`limit_all_gathers=True`;
  按 transformer 块包装(显式 wrap 策略,粒度 model/block/root —
  无自动探测意外)。
- 混合精度:FSDP `MixedPrecision` bf16 + 选项 `reduce_in_full_precision` 与
  `keep_params_fp32`;梯度累积经 `no_sync()` 除最后一个微步;
  `FSDP.clip_grad_norm_`。
- **冻结子树排除在分片之外**(对现有 `freeze_modules` 列表执行
  `get_fsdp_ignored_modules`):完全冻结的模块保持源 dtype 复制 —
  否则冻结的 5B 编码器每次前向都无谓 all-gather。冻结列表语义
  (点路径 + `no_grad` 前向包装,`model/architectures/base.py:373`)不变。
- 差分 LR:`lr_map: {prefix: lr}`,剥离 `module.`/`_fsdp_wrapped_module.`
  后最长前缀匹配 — 泛化现有 `action_lr`/`video_lr` 对(后者继续可用,
  向后兼容)。可选 `GroupwiseFreezeWarmupCosine` 调度。
- EMA(可选开关,decay 可配置)在 runner 内跟踪并写入 checkpoint。

### 7.3 Checkpoint 管理器(`train/checkpoint_manager.py`)

双后端共享(DeepSpeed 路径从 `train/utils/checkpointing.py` 增量迁移;
Phase 4 起两者调用同一管理器):

- 保存(rank-0 聚合):权重 → `safetensors`(**保留 VLM 排除约定** —
  `vlm_backbone.*` 键另存为独立目录,见
  `model/architectures/base.py::VLM_STATE_DICT_PREFIX`);
  优化器状态(完整聚合 dict,恢复时经 `optim_state_dict_to_load` 重分片);
  LR 调度器;EMA;`global_step`/`epoch`;`latest` 符号链接;
  `keep_last_k` 滚动删除。
- 恢复:权重 + 优化器重分片 + 调度器快进;数据位置从
  `(seed, epoch, global_step)` **确定性重推导**(逐步种子 +
  MixtureDataset 带种子索引映射已保证此性质 — 如实记录:
  不 pickle 采样器位置,与 FluxVLA 实践一致;
  现有"epoch 中途恢复"警告保留)。
- 每个 checkpoint 目录自包含:`config.yaml`(已解析,§4.2)+
  `git_meta.json` + `dataset_statistics.json`(或遗留 `.npy`)+
  部署资产(`save_assets_for_deployment` 不变)。
- `torch.distributed.checkpoint` / 异步保存**明确不在**首个 FSDP 发布范围内
  (FluxVLA 本身就是聚合路径);当 >1 TB checkpoint 成为常态时,
  作为未来优化再议。

### 7.4 确定性

`project.seed` + `per_step_seed(run_seed, rank, step)` 是双后端唯一的
确定性机制;FSDP runner 原样复用。`git_meta.json` + 已解析 `config.yaml` +
统计闭合可复现性回路。

### 7.5 激活检查点与显存

- `training.gradient_checkpointing` / `..._offload`(现有开关)保留;
  新增 `gradient_checkpointing_layers: int`,对 Video DiT 与 VLM 块做
  部分逐块检查点(现有 `wan/shared/core/gradient/gradient_checkpoint.py`
  是实现基础)。
- 冻结视觉编码器 / 文本编码器 CPU offload 使用现有
  `wan/shared/core/vram` 管理;不引入新框架。

---

## 8. 部署与接口

### 8.1 部署栈 — 契约不变

`deploy/`(WebSocket 策略服务器、执行器、dit-cache、obs_preprocess、
自包含加载器)保持其通信协议与 checkpoint 契约。变更仅限内部:
模型构建不变(走注册表,§4.1);预处理对等来自
`InferenceProcessorPipeline.from_checkpoint`(§5.3);归一化器挂载路径不变。

### 8.2 桩(`openwam/interfaces/`)

- `rl_stub.py`:`RLPolicyMixin` — `sample_trajectory_with_logprob(batch)`,
  构建于 `denoise_action_step`(静态形状)+
  `ActionHeadOutput.action_logprobs`(FM 密度或概率流 ODE 对数似然;
  精确估计器延后)。
- `eval_server_stub.py`:ZMQ 批推理包装(参照 FluxVLA
  `distributed_zmq_server.py`),围绕
  `InferenceProcessorPipeline.from_checkpoint` + 架构;
  将现有 WebSocket 服务器的角色适配给闭环仿真器。
  目标仿真器协议:开放(§12 O4)。
- `deploy_trt_stub.py`:`OrinThorExporter` — `encode_scene_tokens` →
  TensorRT-LLM(INT8/FP8);`denoise_action_step` →
  TensorRT + CUDA Graph + RTC 平滑。仅接口,无重依赖。

---

## 9. 冒烟测试(`scripts/smoke_test.py`)

```bash
python scripts/smoke_test.py --config configs/train.yaml model=vla_system connector=layerwise_hidden
```

检查:(1) schema 校验 + 已解析 `config.yaml` 生成;(2) 合成 `PipelineContext`
上的预处理器链 → 契约合规的 `ModelInputBatch`;(3) 所选架构的前向+反向;
(4) `encode_scene_tokens` / `denoise_action_step` 的静态形状验证;
(5) 遗留 checkpoint 加载(§4.3)。必须 CPU-only,使用
`tests/_assets.py` 现有 fakes — 不新建 fake 基础设施。

约定(采自 starVLA):每个新模型/处理器模块可独立运行
(`python -m openwam.… --config …`)作为自己的冒烟测试。

---

## 10. 重构阶段与验收标准

| 阶段 | 范围 | 验收 |
|---|---|---|
| P0 | 在重构后套件上做基线冻结(2026-11 测试重构轮:镜像树重组、套件合并 + 参数化、setup 去重、删除 docker 与 robotics 数据集套件 — 87 个文件 / 约 1.1k 用例):完整 pytest 绿色快照记录 | 套件绿色,快照已存档 |
| P1 | `contracts/`(含 `config_schema.py`)+ `config_freezer.py` + 遗留兼容测试;deploy loader 切换到冻结的已解析配置 | 所有现有测试绿色;**遗留黄金 checkpoint 可加载**;`config.yaml` 现在完全解析 |
| P2 | PreProcessor 链 + Track B 数据集 + `MixtureDataset` 跨格式 + `contracts/collate.py`;Track A 适配器;`configs/dataloader/` 重组为 `robot/` + `driving/` 子组(Hydra 包重映射,cfg 树回归测试) | 固定种子下 Track A 经新链运行,**损失与 transforms 路径逐比特一致**(对等闸口);配置移动不改变组合出的 cfg 树;闸口通过后删除 transforms 包 |
| P3 | VLASystem 吸收(连接器)+ FAST 词表路径 + 协同训练规则 + RTC 训练配置 | `vla.yaml` 变体复现先前损失;smoke_test 覆盖所有连接器;协同训练快速失败校验已测 |
| P4 | FSDP runner + checkpoint 管理器统一 + `lr_map` + EMA | 固定种子下 hybrid-shard 8-GPU 冒烟 vs DeepSpeed 参照:损失曲线在容差内匹配;mid-epoch checkpoint 恢复确定性地还原步数与数据位置 |
| P5 | 驾驶链硬化:ingest 工具、时间同步默认值(O1)、可选 latent 缓存(O2)、标定流程 | 样本片段上的驾驶冒烟运行;合成抖动的同步闸口单元测试 |
| P6 | 部署/评估集成:`from_checkpoint` 管线进 deploy、ZMQ 评估桩、TRT 桩 | 在现有服务器 harness 中端到端部署驾驶 checkpoint |

transforms 包删除仅发生在注明的阶段闸口(P2);注册表保留(§1.3 规则 1)。
每次删除是独立的 PR。

---

## 11. 配���兼容性矩阵

| 产物 | 旧 | 新 | 规则 |
|---|---|---|---|
| Checkpoint `config.yaml` | 注册表时代键 | 已解析 + 同文件名 | 现有注册表加载器原样读旧键(§4.3);旧代码路径忽略新键 |
| 归一化统计 | `normalization_stats.npy`(`{joint, eef}`) | `dataset_statistics.json` | 加载器双读两者 |
| `action_lr`/`video_lr` | 平键 | `lr_map` 前缀表 | 两者都接受;平键映射到两个前缀 |
| `freeze:` 列表 | 点路径 | 同语义 + FSDP ignore | 不变 |
| `training.zero_stage` | DeepSpeed | `backend=fsdp` 下忽略 | `backend` 选择相关块 |

---

## 12. 未决项

- **O1 — 时间同步参数**:插值类型(线性 vs 样条)、容差闸口
  (提议:±5 ms)、超闸策略(无效掩码 vs 丢弃)。需要生产精度要求。
- **O2 — 离线 VAE latent 预编码**:推荐(11 相机在线编码是吞吐瓶颈);
  决策须在 P2 结束前做出,因为它影响 Lance schema 与 ingest 工具。
- **O3 — RTC 训练侧条件化**:P3 或延后到 post-P4
  (推理侧 RTC 只有在匹配的训练下才有意义)。
- **O4 — 闭环评估目标**:ZMQ 评估服务器首先面向哪个驾驶仿真器协议。
- **O5 — 数据治理**:去标识化与访问控制是 ingest 侧职责;
  在本仓库之外确认所有权。
