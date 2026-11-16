# Technical Specification: OpenWAM-AutoDrive — Refactoring & Extension (Rev 3)

> **Rev 3 (2026-11-13)** — post-review revision: the decorator-registry mechanism is **retained**
> (§1.3 rule 1 rewritten; the builder replacements and registry deletion targets are dropped) and
> test-suite references are updated to the post-refactor suite. See §12 item 14.
>
> **Rev 2 (2026-11-11)** — revised after a full source-level audit of the OpenWAM codebase and of
> `starVLA` / `FluxVLA`. Changes vs Rev 1 are summarized in §12; confirmed decisions in §1.2;
> unresolved questions in §13.

## 1. Overview

### 1.1 Goal

Turn **OpenWAM** (a robot-manipulation WAM/VLA framework) into a production-oriented
autonomous-driving VLA/WAM suite by reusing:

- **OpenWAM (base repo)**: `dual_system` / `tri_system` / `single_system` WAM architectures,
  video backbones (Wan2.1/2.2, Cosmos-Predict2.5, Cosmos3-Edge), visual encoders (DINOv3,
  V-JEPA 2.1, Wan22-VAE / SVAE / Flux2-VAE), VLM backbone (Qwen3-VL), the Accelerate+DeepSpeed
  training runtime, and the deploy stack (WebSocket policy server + self-contained checkpoints).
- **starVLA (absorbed)**: VLA variants. QwenPI / QwenOFT / QwenGR00T are **already ported**
  (`openwam/model/architectures/vla/`, `openwam/model/action_backbone/vla_heads/`, `configs/model/vla.yaml`).
  Still to port: the **vocabulary-embedded FAST head**, **RTC (Real-Time Chunking)** training &
  inference, and the Qwen special-token vocabulary-extension tooling.
- **FluxVLA (absorbed, engine only)**: FSDP cluster engineering — hybrid sharding, prefix-wise
  per-module LR, aggregated checkpoint + optimizer resharding resume, deterministic multi-source
  balanced sampling, and sample weighters (RA-BC / AW-BC).

### 1.2 Confirmed Decisions

| ID | Decision |
|----|----------|
| D1 | **Package names are frozen.** `openwam/model`, `openwam/dataloader`, `openwam/train`, `openwam/deploy` keep their names; the refactor only **adds** subpackages/files (`contracts/`, `dataloader/preprocessors/`, `interfaces/`, new files under `train/`). No repo-wide import churn, no `model → models` renames. |
| D2 | **Dual training backend.** The existing Accelerate+DeepSpeed ZeRO path is retained and stays the default; a new torchrun+FSDP runner is added, selected by `training.backend: deepspeed \| fsdp`. Both share the same architecture, configs, and checkpoint contract. |
| D3 | **Dual-track data layer.** LeRobot v3 readers (robot domain, four pretraining sources) stay; a new Lance-index + raw-clips store (driving domain) is added. Both tracks emit the same `PipelineContext`; `MixtureDataset` is upgraded to cross-format mixing. |
| D4 | **FAST head is vocabulary-embedded** (starVLA original design: FAST BPE → `<robot_action_xx>` special tokens inside the VLM vocabulary → next-token CE training → `model.generate` inference + FAST decode). It is **not** modeled as an independent `ActionHead`. |
| D5 | **Remove `presets/fast.env` / "RBA" references** — they do not exist in this repository. The acceleration spec (§7.5) cites real mechanisms: per-layer gradient checkpointing, frozen-encoder CPU offload (existing `wan/shared/core/vram` management), and the freeze `no_grad` forward wrapper. |

### 1.3 Mandatory Engineering Rules

1. **Registry-Based Instantiation (kept).** Component instantiation continues through the repo's
   decorator registries (`model/architectures/registry.py`, `model/video_backbone/registry.py`,
   `model/video_backbone/encoder/registry.py`, `model/vlm_backbone/registry.py`,
   `dataloader/registry.py`). Review verdict: the current registries do **not** exhibit the
   debug/navigation pain that motivates MMEngine-style bans elsewhere — registration is static
   (one class per file, at import time), misconfiguration fails at build time, and call sites are
   greppable — so they stay the authoritative mechanism and are **not** deletion targets. New
   component classes (connectors, preprocessors, Track B readers) register through the same
   pattern — in the existing registry for their component kind, or a new registry file beside
   them. The `framework/variant → class` resolution logic of
   `model/architectures/registry.py::resolve_architecture_config` stays where it is, because
   legacy checkpoint configs depend on it (§4.3).
2. **Pure-Tensor Model Graph.** `nn.Module` architectures (`WAMDualSystem`, `WAMTriSystem`,
   `VLASystem`, …) accept and return only tensors (and frozen dataclass contracts).
   **Boundary clarification (important):** the PreProcessor chain owns video decoding, coordinate
   transforms, state normalization, image resize/patching, prompt rendering, and *tokenization* —
   all CPU worker-side ops. GPU-side encoders stay **inside the model**: VAE/visual-encoder latent
   encoding, frozen text-encoder embedding (WAM), and the VLM visual forward (VLA). Consequently
   `ModelInputBatch` carries **raw-scale normalized pixel tensors** and, for WAM, the **prompt as a
   raw string list** (encoded on-GPU by the frozen text encoder, as today); for VLA it carries
   `input_ids` / `attention_mask` (tokenized in dataloader workers). VAE latents, text embeddings,
   and VACE conditioning are never dataloader-side.
3. **High Reuse / Minimal Modification Radius (DRY).** Shared logic (normalizers, attention-mask
   builders, loss aggregation, Flow-Matching components, schedulers) lives in shared base classes
   or `*_components` modules (existing `vla_heads/fm_components.py`, `action_backbone/components.py`
   are the pattern). No glue-code duplication between WAM and VLA paths.
4. **Composable Hydra Configs + Self-Contained Checkpoints.** Keep Hydra composition during
   development. On startup, rank 0 resolves all interpolations and saves the fully-resolved config
   into **every checkpoint directory as `config.yaml`** (filename unchanged — `deploy/model_loader.py`
   and the pytest suite depend on it), plus `git_meta.json` and normalization statistics (§5.2).
5. **Backward Compatibility with OpenWAM Core.** `dual_system` (3 variants), `tri_system`,
   `single_system` (vanilla/moe), `vla` (oftp/pi/groot), all video backbones, all visual encoders,
   the Qwen3-VL backbone, the deploy self-contained checkpoint contract
   (`config.yaml` + `checkpoint_step_*.safetensors` + `normalization_stats.npy` + deploy assets),
   and the existing pytest suite must remain functional throughout.

---

## 2. Target Directory Structure (additive)

```text
OpenWAM/
├── configs/
│   ├── train.yaml                            # [existing] + training.backend: deepspeed|fsdp
│   ├── model/
│   │   ├── dual_system.yaml                  # [existing]
│   │   ├── tri_system.yaml                   # [existing]
│   │   ├── single_system.yaml                # [existing — Rev 1 omitted it; kept]
│   │   ├── vla.yaml                          # [existing; extended with connector/cotraining keys]
│   │   ├── video_backbone/                   # [existing, untouched]
│   │   ├── vlm_backbone/                     # [existing]
│   │   ├── action_backbone/                  # [one group — WAM ActionDiT and VLA heads are ONE
│   │   │                                     #   concept; mirrors code layout model/action_backbone/
│   │   │                                     #   incl. vla_heads/]
│   │   │   ├── separate_action_dit.yaml      # [existing] dual/tri ActionDiT
│   │   │   ├── shared_action_backbone.yaml   # [existing]
│   │   │   ├── mlp_l1.yaml                   # [VLA OFT head]
│   │   │   ├── layerwise_fm.yaml             # [VLA PI head]
│   │   │   └── dit_fm.yaml                   # [VLA GR00T head]
│   │   └── connector/                        # [NEW] layerwise_hidden.yaml, last_hidden_tokens.yaml,
│   │                                         #       last_hidden_cross_attn.yaml
│   ├── dataloader/
│   │   ├── mixture.yaml                      # [existing; upgraded: cross-format entries]
│   │   ├── robot/                            # [moved] all 11 existing LeRobot v3 reader configs
│   │   │   └── robotwin.yaml, agibotworld.yaml, …
│   │   ├── driving/                          # [NEW] Lance dataset configs — expected to scale to
│   │   │   ├── _base.yaml                    #       dozens/hundreds; shared template, per-dataset
│   │   │   └── <dataset>.yaml                #       files stay thin (storage root / filter / tags)
│   │   └── preprocessors/                    # [NEW] wam_canvas_chain.yaml, vla_qwen_chain.yaml
│   ├── runner/
│   │   └── fsdp.yaml                         # [NEW] FSDP sharding / wrapping / mixed precision
│   └── deploy.yaml                           # [existing]
├── openwam/
│   ├── contracts/                            # [NEW]
│   │   ├── config_schema.py                  # typed dataclasses validating new YAML blocks (connector:, preprocessor chain)
│   │   ├── data_contract.py                  # PipelineContext, ModelInputBatch
│   │   ├── model_contract.py                 # BackboneOutput, ConnectorOutput, ActionHeadOutput, SystemOutput
│   │   └── collate.py                        # PipelineContext -> ModelInputBatch collation
│   ├── dataloader/
│   │   ├── bases/                            # [existing] LeRobot v3 readers (Track A, kept)
│   │   ├── *.py                              # [existing] per-dataset readers
│   │   ├── transforms/                       # [existing] kept until preprocessors reach parity (§10 P2)
│   │   ├── lance_dataset.py                  # [NEW] Lance slice / tag / filter-pushdown reader (Track B)
│   │   ├── clip_reader.py                    # [NEW] raw multi-cam MP4 + topics parquet reader
│   │   ├── mixture.py                        # [existing; upgraded to cross-format]
│   │   ├── registry.py                       # [existing, kept] Track B readers register here (§1.3 rule 1)
│   │   └── preprocessors/                    # [NEW]
│   │       ├── base.py                       # BasePreprocessor, PreProcessorPipeline, InferenceProcessorPipeline
│   │       ├── video_decoder.py              # MultiCamVideoDecoder
│   │       ├── time_sync.py                  # TopicTimeSynchronizer
│   │       ├── ego_processor.py              # EgoStateProcessor
│   │       ├── image_processor.py            # MultiViewImageProcessor
│   │       └── prompt_processor.py           # VLAPromptTokenizeProcessor
│   ├── model/
│   │   ├── architectures/
│   │   │   ├── base.py                       # [existing] BaseArchitecture / BaseWAMArchitecture
│   │   │   ├── dual_system/ tri_system/ single_system/   # [existing]
│   │   │   ├── vla/                          # [existing] base_vla.py, qwen_pi/oft/groot (absorbed into §6.3 design)
│   │   │   └── connectors/                   # [NEW] layerwise_hidden.py, last_hidden.py (joint_self_attn stays in WAM)
│   │   ├── action_backbone/
│   │   │   ├── vla_heads/                    # [existing] MLP / FlowMatching / LayerwiseFM / DiT heads
│   │   │   └── scheduler.py components.py    # [existing]
│   │   ├── video_backbone/                   # [existing, untouched]
│   │   └── vlm_backbone/
│   │       ├── qwen3_vl_backbone.py          # [existing]
│   │       └── fast_vocab/                   # [NEW] port of starVLA add_qwen_special_tokens + FAST BPE wrapper
│   ├── train/
│   │   ├── openwam_trainer.py                # [existing] Accelerate+DeepSpeed path (default backend)
│   │   ├── fsdp_runner.py                    # [NEW] torchrun FSDP runner
│   │   ├── checkpoint_manager.py             # [NEW] unified save/resume for both backends
│   │   ├── config_freezer.py                 # [NEW] resolved config.yaml + git_meta.json writer
│   │   └── utils/                            # [existing]
│   ├── deploy/                               # [existing] unchanged contract (§8.1)
│   └── interfaces/                           # [NEW stubs]
│       ├── rl_stub.py
│       ├── eval_server_stub.py
│       └── deploy_trt_stub.py
├── scripts/
│   ├── train.py / train.sh                   # [existing] DeepSpeed path
│   ├── train_fsdp.sh                         # [NEW]
│   └── smoke_test.py                         # [NEW] component contract & forward/backward tester
└── tools/                                    # [NEW] offline data utilities
    ├── ingest_lance.py                       # build Lance tables from raw clips (+ cam_frame_offsets precompute)
    └── fit_fast_tokenizer.py                 # FAST BPE fit/save per action space
```

Existing files not listed are untouched, with two exceptions, each landing at its phase gate (§10),
never a same-PR add+delete: (a) `configs/dataloader/robot/*.yaml` are **moves** — `train.yaml`
defaults and `mixture.yaml` must remap the Hydra package (`- dataloader/robot@dataloader: robotwin`,
`- robot/agibotworld@datasets.agibotworld`, …) so `cfg.dataloader.*` — and therefore
`build_dataset(cfg.dataloader)` and every reader — stay source-compatible; a regression test
asserts the composed config tree is byte-identical before/after the move. (b) Removal target at
end of plan: `dataloader/transforms/` once preprocessors reach parity. The five registry files are
retained (§1.3 rule 1) and are **not** removal targets.

---

## 3. Contracts (`openwam/contracts/`)

All cross-module communication uses these types. `PipelineContext` is the mutable unit inside
DataLoader workers; `ModelInputBatch` is the frozen unit crossing into the model.

### 3.1 `data_contract.py`

**Design intent.** Exactly ONE context class serves both tracks. There are no robot-specific or
driving-specific field *slots* — domain differences are expressed by *who fills a slot and how*
(Track A adapter vs Track B preprocessor), never by extra types. "Required" below is evaluated
per-run: the run config declares which capabilities it uses (state input? action loss?
co-training?), and the dataset layer must then guarantee the corresponding fields are populated.

```python
@dataclass
class PipelineContext:
    """Mutable, chain-passed context. Exactly one training slice (one window of one clip)."""

    # ── 1. Index metadata — SHARED, always required ──────────────────────────
    #    Filled by the dataset layer (Lance row / LeRobot window adapter).
    #    Purpose: identity + storage addressing + the read plan. Nothing here
    #    is derived at read time (precompute at ingest, §5.2).

    # Join key for logs / eval / resume auditing.
    sample_id: str

    # Storage locator of the source clip (driving: raw_clips/<id>/; robot:
    # LeRobot repo root). Only the decoder/reader stage dereferences it.
    clip_uri: str

    # Sync origin. Driving: anchor camera exposure time; robot: window start.
    # All topic interpolation is anchored here.
    anchor_timestamp_us: int

    # Per-camera frame indices to read — the complete read plan. Driving:
    # precomputed at ingest into Lance. Robot: synthesized by the adapter
    # from window bounds (LeRobot stores per-camera MP4 chunks, so the same
    # read plan exists). The decoder performs ZERO index math.
    cam_frame_offsets: Dict[str, List[int]]

    # Which dataset/table this slice came from (mixture logging, per-source
    # metrics, stats resolution).
    source_name: str

    # Task instruction. Driving: navigation text; robot: its prompt (see
    # Track A mapping below). "" only for pure world-model samples (no
    # action loss, no language conditioning).
    nav_instruction: str = ""

    # ── 2. Optional metadata — SHARED ─────────────────────────────────────────

    # From weighters (constant / success_rate / progress_delta); consumed by
    # the loss with global normalization (§5.4). 1.0 = plain BC.
    sample_weight: float = 1.0

    # Filtering / curriculum / mixture stratification (Lance SQL pushdown).
    scene_tags: Optional[List[str]] = None

    # Chain-of-thought target. Present ⇒ prompt processor builds vlm_labels
    # (co-training §6.5); absent ⇒ vlm_labels stays None.
    cot_text: Optional[str] = None

    # ── 3. Calibration & timing — DRIVING-ONLY ────────────────────────────────
    #    Robot rigs are fixed and implicit in the canvas layout, so Track A
    #    leaves both slots None.

    # REQUIRED for every Track B run. Per cam: {"extrinsics": [4,4]
    # cam->ego-rig, "intrinsics": [3,3]}. Read from calib.json (§5.2);
    # consumed by TopicTimeSynchronizer (coordinate transforms).
    calib: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    # REQUIRED for every Track B run. Real per-cam exposure stamps; the sync
    # tolerance gate (§13 O1) reads these. Track A: None (uniform fps is a
    # LeRobot guarantee, no gate needed).
    frame_timestamps_ns: Optional[Dict[str, List[int]]] = None

    # ── 4. Raw decoded physical quantities — SHARED slots, per-track fillers ──
    #    Consumed by the tensor-izing preprocessors; never leaves the worker.

    # {cam: [T_obs, H_cam, W_cam, 3] uint8} at NATIVE per-cam resolution.
    # Resolution tiers (front > side > rear, §3.1 batch note) are preserved
    # here; rescaling is owned exclusively by the image processor.
    raw_images: Optional[Dict[str, np.ndarray]] = None

    # [T_hist, D_state] float32, ego-rig frame. Driving: ego kinematic
    # history interpolated at the anchor. Robot: T_hist=1 proprio (current
    # state). None = state-free run.
    raw_ego_history: Optional[np.ndarray] = None

    # [T_pred, D_action] float32, ego-rig frame. Driving: future ego plan;
    # robot: action window. REQUIRED when the run trains an action head.
    raw_gt_trajectory: Optional[np.ndarray] = None

    # [T_pred] or [T_pred, D_action] bool. REQUIRED whenever
    # raw_gt_trajectory is set. The 2-D form preserves PER-DIM validity and
    # is mandatory for the robot 80-D unified space (reserved/unmapped dims
    # must stay out of the loss); the 1-D form suffices for driving plans.
    raw_trajectory_valid_mask: Optional[np.ndarray] = None

    # ── 5. Processed tensors — SHARED, produced by the preprocessor chain ─────

    # wam_canvas / qwen_vl_patch layouts (batch note §3.1).
    pixel_values: Optional[Tensor] = None

    # standard_tensor mode, per sample: {cam: [T_obs, C, H_cam, W_cam]} at the
    # cam's tier resolution. Collate stacks PER CAM → [B, T_obs, C, H_cam, W_cam].
    pixel_values_by_cam: Optional[Dict[str, Tensor]] = None

    # qwen_vl_patch only, [N_img, 3].
    image_grid_thw: Optional[Tensor] = None

    # WAM path: raw string; model-side text encode.
    prompt: Optional[str] = None

    # VLA path (tokenized in the worker).
    input_ids: Optional[Tensor] = None

    # ditto (per-sample [L_seq]; padded at collate).
    attention_mask: Optional[Tensor] = None

    # -100 on prompt tokens when cot_text present.
    vlm_labels: Optional[Tensor] = None

    # [T_hist, D_enc] (T_hist=1 for robot proprio).
    ego_state_tensor: Optional[Tensor] = None

    # [T_pred, D_action].
    gt_trajectory_norm: Optional[Tensor] = None

    # [T_pred] or [T_pred, D_action] bool.
    trajectory_mask: Optional[Tensor] = None
```

**Field classification summary**

| Block | Fields | Domain | Required when |
|---|---|---|---|
| 1 Index | `sample_id, clip_uri, anchor_timestamp_us, cam_frame_offsets, source_name` | shared | always |
| 1 Index | `nav_instruction` | shared | language-conditioned or action-loss runs |
| 2 Optional | `sample_weight, scene_tags, cot_text` | shared | weighters / filtering / co-training, respectively |
| 3 Calibration | `calib, frame_timestamps_ns` | **driving-only** | all Track B runs (None on Track A) |
| 4 Raw | `raw_images` | shared | any visual model |
| 4 Raw | `raw_ego_history` | shared (driving `T_hist>1`; robot `T_hist=1` proprio) | state-conditioned runs |
| 4 Raw | `raw_gt_trajectory, raw_trajectory_valid_mask` | shared | action-loss runs (2-D per-dim mask on robot) |
| 5 Processed | `pixel_values … trajectory_mask` | shared | produced by the configured chain |

**Track A mapping (normative).** Robot readers fill the SAME slots — proprio →
`raw_ego_history` (T_hist=1), action window → `raw_gt_trajectory`, action/proprio per-dim masks →
`raw_trajectory_valid_mask` (2-D form), prompt → `nav_instruction`. Robot-only bookkeeping that
does NOT become context data: the 80-D `unify_action_map` (static per-source config in the reader
yaml), color-jitter state (processor-internal). Track A must not fork a second context class.

```python
@dataclass(frozen=True)
class ModelInputBatch:
    """Pure-tensor batch. Only these cross into BaseArchitecture.forward().

    Multi-camera resolution tiers (driving): a frame carries ~7 surround cams
    and compute is bounded by tiering resolutions — front cam largest, side
    cams smaller, rear smallest. Per-camera target resolutions are STATIC
    config (MultiViewImageProcessor.per_cam_resolution, frozen into
    config.yaml — never per-sample batch data).

    NORMATIVE: resolution heterogeneity survives THROUGH the visual encoder.
    Each camera is encoded at its own tier; per-cam feature grids differ in
    spatial size; the streams merge ONLY after per-cam flatten-to-tokens,
    concatenated along the sequence dim in the config camera order, with a
    unified attention mask and per-cam RoPE / positional embeddings attached
    before the merge. The merge point is MODEL-SIDE (visual-encoder wrapper /
    video-backbone adapter, §6.2) — never in the preprocessor chain. Three
    layouts, exactly one per batch:

      wam_canvas       pixel_values [B, T_obs, C, H_canvas, W_canvas]
                       views composited into one canvas; slot size follows
                       the cam tier (front big, rear small) — generalizes
                       the robot L-shape; one regular grid, zero padding
                       waste. Tier-1 path: works with today's backbones
                       unchanged.
      standard_tensor  pixel_values_by_cam {cam: [B, T_obs, C, H_cam, W_cam]}
                       per-cam tensors at their tier resolutions — NO
                       upscaling of small cams; the backbone adapter
                       patchifies each cam separately and merges at token
                       level (§6.2).
      qwen_vl_patch    pixel_values packed per-image patches (variable count);
                       native variable resolution; image_grid_thw records
                       each image's [T, H/patch, W/patch] — zero padding
                       waste. Format = HF Qwen-VL convention, byte-compatible
                       with qwen3_vl_backbone.batch_vlm_inputs (§3.1 note).
    """

    # Visuals (exactly one of the three layouts above, per preprocessor chain):

    # wam_canvas or qwen_vl_patch.
    pixel_values: Optional[Tensor] = None

    # standard_tensor only. Dict insertion order = config camera order;
    # H/W differ ACROSS cams by tier, uniform WITHIN a cam across the batch.
    pixel_values_by_cam: Optional[Dict[str, Tensor]] = None

    # qwen_vl_patch only: [N_img_total, 3] long, (t, h, w) per image in 14-px
    # patch units, CONCATENATED across frames/cams/batch — no batch dim, no
    # padding (HF convention; qwen3_vl_backbone.batch_vlm_inputs cats dim=0).
    # Image boundaries recover via cumsum(t*h*w); cam identity/order via the
    # prompt processor's placeholder expansion order.
    image_grid_thw: Optional[Tensor] = None

    # [B, T_obs] bool, frame validity (WAM).
    video_mask: Optional[Tensor] = None

    # Language (exactly one path):

    # WAM: model-side text encoding (GPU, frozen).
    prompt: Optional[List[str]] = None

    # VLA: [B, L].
    input_ids: Optional[Tensor] = None

    # [B, L].
    attention_mask: Optional[Tensor] = None

    # [B, L], -100-masked.
    vlm_labels: Optional[Tensor] = None

    # State / action supervision:

    # [B, T_hist, D_enc]; legacy [B, D] accepted.
    ego_state_tensor: Optional[Tensor] = None

    # [B, T_pred, D_action].
    gt_trajectory_norm: Optional[Tensor] = None

    # [B, T_pred] bool (subsumes action_is_pad).
    trajectory_mask: Optional[Tensor] = None

    # [B] float32.
    sample_weights: Optional[Tensor] = None
```

> Deliberately **absent** (model-side, never dataloader-side): VAE latents
> (`input_latents` / `first_frame_latents`), text embeddings (`context`/`context_mask`),
> VACE conditioning, `condition_mask`, `num_clean_prefix_frames`. The WAM architectures'
> existing `prepare_inputs`/`preprocess` (GPU, no_grad) consume `pixel_values`/`prompt` and
> produce them, unchanged.

### 3.2 `model_contract.py`

**Naming convention (normative).** Two orthogonal axes: *state* fields are named by provenance
(`last_hidden_state`, `hidden_states` — which layer); *mask* fields are named by the sequence
they mask (`attention_mask`, `context_mask`, `condition_mask`, `video_mask`,
`trajectory_mask`). A mask is a property of the token SEQUENCE, never of one layer — one mask
serves every layer's states, so mask names never follow layer naming. (HF defines no
output-side mask convention; internal consistency rules here.)

```python
@dataclass
class BackboneOutput:
    """Standardized output of VLMBackbone / VideoBackbone.

    Naming follows the HF convention everyone already knows: last_hidden_state
    is the final layer; hidden_states is the per-layer stack and
    hidden_states[-1] IS last_hidden_state (aliased, never copied).
    """

    # [B, L_ctx, D_model] — final-layer hidden states; the default
    # conditioning source and the low-freq compile unit consumed by
    # encode_scene_tokens.
    last_hidden_state: Tensor

    # [B, L_ctx] — validity over the token sequence (shared by every layer).
    context_mask: Tensor

    # Optional: hidden states of the transformer layers, HF
    # output_hidden_states convention — embeddings first, one [B, L_ctx, D]
    # entry per layer, hidden_states[-1] IS last_hidden_state. len()
    # self-describes the layer count — no config lookup needed downstream.
    # Populated ONLY when the run config asks for it (the connector declares
    # its layer needs, e.g. connector=layerwise_hidden with num_layers=N —
    # the backbone then returns exactly those layers). Consumers:
    #   - QwenPI layer-wise cross-attention conditioning (§6.3)
    #   - DeepStack-style multi-layer routing (existing in Qwen3-VL)
    #   - auxiliary deep-supervision losses (future)
    # None by default: the full stack costs ~N_layers × L_ctx × D × 2 bytes
    # (hundreds of MB/sample for a 4B VLM) — never materialize speculatively.
    hidden_states: Optional[List[Tensor]] = None

    # Video flow-matching loss (WAM world stream).
    world_pred_loss: Optional[Tensor] = None

    # Language CE (VLA co-training, §6.5).
    vlm_loss: Optional[Tensor] = None

    # Decode-phase cache (inference only).
    kv_cache: Optional[Dict[str, Tensor]] = None

@dataclass
class ConnectorOutput:
    """Standardized output of the connector, consumed by the action head.

    One condition stream per VLM layer consumed; the list length
    self-describes the layer count (mirrors BackboneOutput.hidden_states).
    """

    # Per-layer condition streams, every entry [B, L_src, D_head] — all
    # entries live on the SAME token sequence (the VLM's), so identical
    # length and identical per-entry masks:
    #   - standard routes (last_hidden_tokens, last_hidden_cross_attn):
    #     len == 1; single-stream heads take condition_tokens[0].
    #   - layer-wise route (layerwise_hidden / QwenPI): len == N_layers;
    #     DiT layer i cross-attends condition_tokens[i]. A wrong layer count
    #     fails loudly at the head's len()/shape assert — nothing is packed
    #     into a shape that requires external config to decode.
    # The layer→DiT-layer mapping ("last N in order" vs "evenly bridged") is
    # connector config; the LIST carries zero packing ambiguity.
    condition_tokens: List[Tensor]

    # [B, L_src] — one mask serves all streams (identical per entry).
    condition_mask: Tensor

@dataclass
class ActionHeadOutput:
    """Standardized output of an action head."""

    # Supervision loss (FM velocity MSE / L1 / FAST token CE).
    action_loss: Optional[Tensor] = None

    # [B, T_pred, D_action] — continuous heads (FM / MLP).
    pred_trajectory_norm: Optional[Tensor] = None

    # [B, L_tok] — FAST path token ids (§6.4); trajectory is None there.
    pred_action_tokens: Optional[Tensor] = None

    # Reserved for RL post-training (§8.2).
    action_logprobs: Optional[Tensor] = None

@dataclass
class SystemOutput:
    """Top-level return of BaseArchitecture.forward()."""

    total_loss: Optional[Tensor] = None

    action_loss: Optional[Tensor] = None

    world_pred_loss: Optional[Tensor] = None

    vlm_loss: Optional[Tensor] = None

    pred_trajectory_norm: Optional[Tensor] = None

    pred_action_tokens: Optional[Tensor] = None

    # Scalar logging metrics (detached).
    metrics: Optional[Dict[str, float]] = None
```

### 3.3 Collation & invertibility

- `contracts/collate.py` owns `PipelineContext -> ModelInputBatch`: per-field stacking, padding
  (`attention_mask`, `image_grid_thw` concatenation + padding to `max(N_img)`), mask promotion
  (1-D/2-D mask reconciliation — reuse the reconciliation rules already in
  `BaseArchitecture.prepare_inputs`). `pixel_values_by_cam` stacks **per camera** across the
  batch (cam iteration order = config camera order; a missing cam is a hard error, not a
  zero-fill). No architecture may collate.
- Invertibility is a first-class property: `EgoStateProcessor.denormalize_trajectory(norm) -> ndarray`
  and image ops keep the parameters needed to invert; the deploy path un-normalizes through the
  same object (replaces starVLA's separate "unapply" transform with one normalizer instance,
  loaded from checkpoint-local statistics).

---

## 4. Component Wiring & Config Freezing

### 4.1 Component wiring (registry-based)

- Instantiation routes through the **existing decorator registries** (§1.3 rule 1): architectures
  via `model/architectures/registry.py`, video/vlm backbones via their registries, datasets via
  `dataloader/registry.py`. New component classes join the same pattern — connectors register
  alongside the architectures, preprocessors get a registry in `dataloader/preprocessors/` (or
  entries in `dataloader/registry.py`) following the same decorator pattern.
- The action config group stays **single** — WAM ActionDiT and the VLA heads (MLP / FlowMatching /
  LayerwiseFM) are one concept, share one config group (`configs/model/action_backbone/`,
  mirroring the code layout `model/action_backbone/` incl. `vla_heads/`), and dispatch on the
  existing `action_model_type` discriminator; in-file block keys (`action_head:` in `vla.yaml`,
  `action_backbone:` in `dual_system.yaml`) keep their names for config compatibility.
- New YAML blocks (`connector:`, action configs, preprocessor chain) get dataclasses in
  `contracts/config_schema.py`; validate at build time with `OmegaConf.structured(...)` — schema
  errors fail fast, not at forward. **Scope control**: full DictConfig→dataclass conversion is
  *not* a goal; only the new blocks and the architectures' top-level params are typed.

### 4.2 Config freezing (`train/config_freezer.py`)

- At startup, after Hydra composition and **before the first distributed collective** (so all
  ranks see one truth and rank 0's write cannot diverge), rank 0 writes into `<work_dir>` and each
  future checkpoint dir:
  - `config.yaml` — `OmegaConf.to_yaml(cfg, resolve=True)` (all interpolations and env resolved).
    Filename unchanged from today's `save_config` write-once behavior; this upgrades its content
    to fully-resolved.
  - `git_meta.json` — `{commit, dirty, diff}` (the diff is mandatory: configs drift between
    "clean" commits during development).
  - normalization statistics: new runs write `dataset_statistics.json`; the loader **dual-reads**
    the legacy `normalization_stats.npy` (nested `{joint, eef}` schema) so old checkpoints deploy
    unchanged (§5.2).
- `load_frozen_config(ckpt_dir)` utility for eval/deploy; it must be the only config source in
  `deploy/model_loader.py` after Phase 1.

### 4.3 Legacy checkpoint compatibility (normative)

A checkpoint saved by current OpenWAM contains `config.yaml` with
`model.architecture.{framework,variant}` and registry-era keys. These keep loading through the
**unchanged** `model/architectures/registry.py::resolve_architecture_config` mapping — old
checkpoints keep training (finetune/resume via `train/utils/ckpt_model_loader.py`) and deploying
with zero migration. This is a standing test: "load a golden legacy checkpoint dir" runs in CI
from Phase 1 onward.

---

## 5. Dataset Layer (dual-track, D3)

### 5.1 Track A — LeRobot v3 (robot domain; existing, kept)

`dataloader/bases/lerobot_v3_reader.py` and its four pretraining readers stay as-is. A thin
adapter wraps their canonical sample dict into `PipelineContext` (mapping in §3.1). Their
existing transforms (`transforms/`) keep running until the preprocessor chain reaches parity for
Track A (phase-gated, §10 P2); after parity the transforms package is deleted, not shimmed.

### 5.2 Track B — driving store (new)

```text
<storage_root>/raw_clips/<clip_id>/
    cams/<cam_name>.mp4                  # ~7V surround; per-cam resolution tiers
                                         #   (front > side > rear, §3.1), H.264/H.265
    topics/ego_motion.parquet            # high-frequency vehicle bus
    topics/localization.parquet          # ego pose in world frame
    topics/navigation.parquet            # instructions
    calib.json                           # per-cam intrinsics/extrinsics (cam->ego-rig)
<storage_root>/lance_indices/<dataset_name>.lance
    sample_id, clip_uri, anchor_timestamp_us,
    cam_frame_offsets (PRE-COMPUTED at ingest by tools/ingest_lance.py — the reader
        must never re-derive them from num_frames/video_stride),
    nav_instruction, cot_text, scene_tags, sample_weight,
    gt_future_traj_ego (fixed_size_list<float32>), valid_mask
```

- `LanceClipDataset`: SQL filter pushdown (`lance.dataset(uri).to_table(filter=...)`), column
  projection, dynamic tag/subset selection.
- Statistics: computed offline by `tools/ingest_lance.py` or on first load (rank-0 computes,
  others wait — same protocol as today's auto-stats) → `dataset_statistics.json`.
- Data governance: ingestion owns de-identification (face/plate blurring) and `clip_uri` access
  control; this repo assumes read-only, already-sanitized stores (§13 O5).

### 5.3 Composable PreProcessor chain (`dataloader/preprocessors/`)

`BasePreprocessor.process(ctx: PipelineContext) -> PipelineContext`, property
`is_training_only: bool`. Chain order is config-declared
(`configs/dataloader/preprocessors/*.yaml`).

| Processor | training-only | Responsibility |
|---|---|---|
| `MultiCamVideoDecoder` | yes | Reads `cam_frame_offsets` frames from MP4s. decord/PyAV with per-worker singleton handles (10 workers × 11 cams must not explode handles/RSS — bounded LRU per worker). Color jitter (train-only, temporal-consistent) moves here from `transforms/video.py`. |
| `TopicTimeSynchronizer` | yes | Interpolates CAN/ego topics at camera exposure timestamps → `raw_ego_history` / `raw_gt_trajectory` in the **ego-rig frame** (uses `calib`). Interpolation kind, tolerance gate, and out-of-gate policy (drop vs invalid-mask) are config fields; **defaults are an open item (§13 O1)**. Rotation utilities reuse `dataloader/utils/poses.py` / `rotation.py` (6D/quaternion). |
| `EgoStateProcessor` | no | Normalizes history/trajectory with `dataset_statistics.json`, optional Fourier embedding → `ego_state_tensor`, `gt_trajectory_norm`, `trajectory_mask`. Provides `denormalize_trajectory`. Also offers the π0.5-style **state-as-text** rendering (`state2str` from `vla/base_vla.py`) consumed by the prompt processor. |
| `MultiViewImageProcessor` | no | Config: `per_cam_resolution: {cam: [H, W]}` (driving tiers: front > side > rear) + `mode`. Three modes: `wam_canvas` (views composited into one canvas, slot size follows the cam tier — Track A uses the existing L-shape, driving generalizes it to 7 slots), `standard_tensor` (per-cam tensors `{cam: [T_obs, C, H_cam, W_cam]}` at their tier resolutions — **no upscaling**; merging into one token stream happens model-side, §6.2), `qwen_vl_patch` (per-cam resize to its tier, Qwen dynamic-resolution patchify + `image_grid_thw` — token-level merge is native here). Resize/normalize params are checkpoint-frozen (§4.2). |
| `VLAPromptTokenizeProcessor` | no | Renders template (nav instruction, camera tokens, optional state-as-text), tokenizes → `input_ids`/`attention_mask`, builds `vlm_labels` with `-100` masking when `cot_text` present (co-training, §6.5). |

Dual-mode pipeline:

- **Training**: runs the configured chain inside DataLoader workers; collation via
  `contracts/collate.py`.
- **Inference**: `InferenceProcessorPipeline.from_checkpoint(ckpt_dir)` rebuilds the chain from
  the frozen `config.yaml`, filters out `is_training_only=True` processors, and maps live inputs
  (`{cam: frame}` dict, latest ego reading, instruction) into a `PipelineContext` before the
  retained processors run — guaranteeing train/deploy parity through the **same code path**, not a
  reimplementation (this is the explicit lesson from starVLA's `PolicyNormProcessor`).

### 5.4 Mixture & weighting

- `MixtureDataset` upgraded: entries may point at any Track A reader **or** Track B tables;
  existing `weight_strategy` (`proportional/manual/uniform/inverse_size`) semantics preserved;
  per-epoch deterministic reshuffle retained (seeded index map — current behavior).
- `sample_weight`: produced dataset-side by weighters (start with `constant`; FluxVLA-style
  `success_rate` / `progress_delta` are the reference when RL-adjacent data arrives), consumed
  loss-side with **global normalization across ranks and accumulation steps** (FluxVLA
  `reduce_action_bc_loss` pattern), so effective per-batch loss magnitude stays stable.

### 5.5 Throughput

- Multi-cam × 10 Hz × T_obs decode is the dominant cost; the chain must expose per-stage timing
  counters in the step log from day one.
- **Offline latent pre-encoding** (VAE latents cached next to clips, decoder processor bypassed)
  is proposed as an optional mode — pending decision §13 O2. If adopted, it lands as a
  `LatentCacheReader` processor that short-circuits decode+encode, with the architecture accepting
  pre-computed latents (already supported: `inputs["input_latents"]` path). Cached latents follow
  the same per-cam tier resolutions (heterogeneous latent grids), so the token-merge adapter of
  §6.2 is the consumption path there too.

---

## 6. Model Layer

### 6.1 `BaseArchitecture` interface

All top-level architectures implement (existing WAM classes re-route their current
`forward(noisy_actions, action_timestep, **pipeline_inputs)` through this contract):

```python
# Main training entry: batch in, losses / metrics out.
def forward(self, batch: ModelInputBatch) -> SystemOutput

# Low-frequency scene encoding — the compile/caching unit (TensorRT-LLM
# target): computed once per observation, reused across all denoising steps.
def encode_scene_tokens(self, batch: ModelInputBatch) -> ConnectorOutput

# High-frequency single denoising step — static tensor shapes; the
# CUDA-Graph / Real-Time-Chunking unit, called len(schedule)-1 times per chunk.
def denoise_action_step(self, cond: ConnectorOutput, noisy_action: Tensor,
                        timestep: Tensor) -> Tensor

# End-to-end inference convenience: encode once, iterate denoise_action_step
# with the schedule, return the (denormalized) trajectory.
def predict_trajectory(self, batch: ModelInputBatch) -> Tensor
```

The existing `generate()` denoising loop stays as the composition of the two hooks driven by a
schedule (`deploy/denoise_schedule.py`); CFG, dit-cache, and inactive-dim handling move inside the
loop unchanged. `prepare_inputs` (GPU, no_grad) bridges `pixel_values`/`prompt` to latents/contexts
as today — it is the model-side half of the R2 boundary.

### 6.2 WAM systems

`dual_system` (joint_cross_attn / joint_self_attn / idm), `tri_system`, `single_system`
(vanilla / moe) keep their block-level logic; the refactor is contract-shaped (batch in,
`SystemOutput` out) and component wiring stays registry-based (§4.1). `action_backbone`
(ActionDiT / shared) and `video_backbone` are untouched internally.

**Heterogeneous multi-cam visual path (normative).** When the batch carries
`pixel_values_by_cam` (standard_tensor mode), the visual-encoder wrapper encodes each camera at
its own tier resolution, flattens each per-cam feature grid into tokens, and concatenates the
token streams along the sequence dim in the config camera order; a unified attention mask is
built and per-cam 2-D RoPE / positional embeddings are attached **before** the merge. Downstream
(video DiT blocks, connectors, action heads) see a single token stream — resolution heterogeneity
never leaks past the merge point. For WAM video backbones this is an adapter-tier extension:
Wan/Cosmos DiTs natively consume one regular latent grid, so the adapter must (a) run the visual
encoder per cam, (b) patchify per cam, (c) token-concat + per-cam RoPE + unified mask, optionally
(d) insert per-cam separator embeddings. Until this adapter lands, `wam_canvas` (one regular
grid, tier-sized slots) is the default WAM driving path; a parity test must assert the canvas and
token-merge paths produce identical attention semantics on an equal-resolution rig.

### 6.3 `VLASystem` — absorb, don't rebuild

The VLA family **already exists** (`BaseVLAArchitecture`, `qwen_pi/oft/groot`, heads MLP /
FlowMatching-DiT / LayerwiseFM). The refactor extracts the implicit conditioning logic into an
explicit **connector** layer so YAML composes variants without subclass proliferation:

| `connector` | Maps to (today) | Mechanism |
|---|---|---|
| `layerwise_hidden` | QwenPI | last-N VLM hidden states (consumes `BackboneOutput.hidden_states`, populated only when this connector is configured), per-layer cross-attention into LayerwiseFM DiT |
| `last_hidden_tokens` | QwenOFT | action-token positions gathered from last hidden state (`_gather_action_token_embeddings`) → MLP L1 head |
| `last_hidden_cross_attn` | QwenGR00T | last hidden state (+ state embedding, future tokens) → DiT-B/L FlowMatching head |

`VLASystem` composition: `vlm_backbone` (existing Qwen3-VL backbone; frozen or co-trained) +
`ego_proj` (state → VLM dim; or state-as-text when configured) + `connector` + `action_head`
(reusing the **same** FlowMatching head family as WAM — shared `fm_components.py`). Shape fields
bound to the VLM stay auto-aligned at runtime (existing behavior in `configs/model/vla.yaml`).

### 6.4 FAST head (vocabulary-embedded, D4)

- Tooling port: `tools/fit_fast_tokenizer.py` (FAST BPE fit/save over a target action space) and
  `vlm_backbone/fast_vocab/` (special-token range extension — port of starVLA
  `add_qwen_special_tokens`, with vocab-size padding and resize rules).
- Training: action chunk → FAST tokens → `<robot_action_xx>` ids appended to labels; the loss is
  the VLM's own next-token CE restricted to those positions (`vlm_labels` already models this).
- Inference: `model.generate` with token-range extraction → FAST decode → trajectory.
  `ActionHeadOutput.pred_action_tokens` carries the ids; `pred_trajectory_norm` is `None`.
- `FASTTokenHead` as an independent head is **rejected** (it would force the LM head, generate
  loop, and kv-cache through the ActionHead abstraction — a distortion of both).

### 6.5 Co-training rules

- Loss: `total = action_loss + cfg.vlm_loss_weight * vlm_loss`, **one backward** (compatible with
  Accelerate `accumulate` and both backends). starVLA's two-backward scheme is deliberately not
  adopted.
- Validation at build time: `vlm_loss_weight > 0` requires the VLM (or a declared part of it) to
  be trainable; a config that fully freezes `vlm_backbone` while requesting CE loss must fail fast.
- CE normalization: token-sum with cross-rank/accumulation global normalization (FluxVLA
  `losses/causal_lm.py` pattern) to avoid sequence-length bias.
- `train_starvla_cotrain`'s dual-dataloader pattern is not needed initially (labels ride in the
  same batch); revisit only if a language-only corpus phase is added.

### 6.6 RTC (Real-Time Chunking)

- Training-side **delay conditioning** (sample an inference delay per sample; per-index timestep
  conditioning shifts accordingly — port of starVLA `apply_rtc_time_conditioning` / FluxVLA
  `rtc_training_config`) goes into the FlowMatching heads behind `rtc_training_config.enabled`.
  Phase placement: §13 O3.
- Inference-side fixed-prefix re-sampling ports into the deploy loop (`deploy/`), reusing
  `encode_scene_tokens` across steps.

---

## 7. Training Engine (dual backend, D2)

### 7.1 Backend A — DeepSpeed (existing, default)

`scripts/train.py` + `openwam_trainer.py` stay the default path (`training.backend: deepspeed`).
Behavior, seed handling (`per_step_seed`), ZeRO 1/2, `action_lr`/`video_lr`, `keep_last_k`,
finetune/resume — all preserved. `training.backend` defaults to `deepspeed` so existing commands
are bit-identical.

### 7.2 Backend B — FSDP runner (`train/fsdp_runner.py`, FluxVLA-derived)

- torchrun launch (`scripts/train_fsdp.sh`, env-var compatible with cloud schedulers as in
  `scripts/train.sh`).
- `sharding_strategy ∈ {full_shard, hybrid_shard (default), shard_grad_op, hybrid_shard_zero2,
  no_shard}`; `use_orig_params=True`, `limit_all_gathers=True`; wrapping per transformer block
  (explicit wrap policy, granularity model/block/root — no auto-probing surprises).
- Mixed precision: FSDP `MixedPrecision` bf16 + options `reduce_in_full_precision` and
  `keep_params_fp32`; grad accumulation via `no_sync()` except the final micro-step;
  `FSDP.clip_grad_norm_`.
- **Frozen subtrees are excluded from sharding** (`get_fsdp_ignored_modules` over the existing
  `freeze_modules` list): fully-frozen modules stay replicated in source dtype — without this, a
  frozen 5B encoder pointlessly all-gathers every forward. The freeze list semantics (dotted path +
  `no_grad` forward wrap, `model/architectures/base.py:373`) are unchanged.
- Differential LR: `lr_map: {prefix: lr}` with longest-prefix matching after stripping
  `module.`/`_fsdp_wrapped_module.` — generalizes the existing `action_lr`/`video_lr` pair (which
  stays working for backward compat). Optional `GroupwiseFreezeWarmupCosine` schedule.
- EMA (optional flag, decay configurable) tracked in the runner and checkpointed.

### 7.3 Checkpoint manager (`train/checkpoint_manager.py`)

Shared by both backends (DeepSpeed path migrates from `train/utils/checkpointing.py` incrementally;
both call the same manager by Phase 4):

- Save (rank-0 aggregation): weights → `safetensors` (**VLM exclusion convention retained** —
  `vlm_backbone.*` keys are saved as a separate directory, per
  `model/architectures/base.py::VLM_STATE_DICT_PREFIX`); optimizer state (full aggregated dict,
  resharded on resume via `optim_state_dict_to_load`); LR scheduler; EMA; `global_step`/`epoch`;
  `latest` symlink; `keep_last_k` rolling deletion.
- Resume: weights + optimizer reshard + scheduler fast-forward; data position is **re-derived
  deterministically** from `(seed, epoch, global_step)` (per-step seed + MixtureDataset's seeded
  index map already guarantee this — documented honestly: sampler position is not pickled, matching
  FluxVLA practice; the existing mid-epoch-resume warning stays).
- Every checkpoint dir is self-contained: `config.yaml` (resolved, §4.2) + `git_meta.json` +
  `dataset_statistics.json` (or legacy `.npy`) + deploy assets (`save_assets_for_deployment`
  unchanged).
- `torch.distributed.checkpoint` / async save is **explicitly out of scope** for the first FSDP
  release (FluxVLA itself ships the aggregated path); revisited as a future optimization when
  >1 TB checkpoints become routine.

### 7.4 Determinism

`project.seed` + `per_step_seed(run_seed, rank, step)` remain the single determinism mechanism for
both backends; FSDP runner reuses them verbatim. `git_meta.json` + resolved `config.yaml` +
statistics close the reproducibility loop.

### 7.5 Activation checkpointing & memory (replaces Rev-1 "RBA")

- `training.gradient_checkpointing` / `..._offload` (existing flags) retained; add
  `gradient_checkpointing_layers: int` for partial per-block checkpointing across Video DiT and VLM
  blocks (the existing `wan/shared/core/gradient/gradient_checkpoint.py` is the implementation
  base).
- Frozen visual encoder / text encoder CPU offload uses the existing `wan/shared/core/vram`
  management; no new framework.

---

## 8. Deploy & Interfaces

### 8.1 Deploy stack — contract unchanged

`deploy/` (WebSocket policy server, executors, dit-cache, obs_preprocess, self-contained loader)
keeps its wire protocol and checkpoint contract. Changes are internal only: model construction is
unchanged (registry-based, §4.1); preprocessing parity comes from
`InferenceProcessorPipeline.from_checkpoint` (§5.3); normalizer attach path unchanged.

### 8.2 Stubs (`openwam/interfaces/`)

- `rl_stub.py`: `RLPolicyMixin` — `sample_trajectory_with_logprob(batch)` built on
  `denoise_action_step` (static shapes) + `ActionHeadOutput.action_logprobs` (FM density or
  probability-flow ODE log-likelihood; exact estimator deferred).
- `eval_server_stub.py`: ZMQ batch-inference wrapper (reference: FluxVLA
  `distributed_zmq_server.py`) around `InferenceProcessorPipeline.from_checkpoint` +
  architecture; adapts the existing WebSocket server's role for closed-loop simulators. Target
  simulator protocol: open (§13 O4).
- `deploy_trt_stub.py`: `OrinThorExporter` — `encode_scene_tokens` → TensorRT-LLM (INT8/FP8);
  `denoise_action_step` → TensorRT + CUDA Graph + RTC smoothing. Interfaces only, no heavy deps.

---

## 9. Smoke Testing (`scripts/smoke_test.py`)

```bash
python scripts/smoke_test.py --config configs/train.yaml model=vla_system connector=layerwise_hidden
```

Checks: (1) schema validation + resolved `config.yaml` generation; (2) preprocessor chain on a
synthetic `PipelineContext` → contract-compliant `ModelInputBatch`; (3) forward+backward for the
selected architecture; (4) static-shape verification of `encode_scene_tokens` /
`denoise_action_step`; (5) legacy-checkpoint load (§4.3). Must run CPU-only using the existing
fakes in `tests/_assets.py` — no new fake infrastructure.

Convention (adopted from starVLA): every new model/processor module is runnable standalone
(`python -m openwam.… --config …`) as its own smoke test.

---

## 10. Refactoring Phases & Acceptance Criteria

| Phase | Scope | Acceptance |
|---|---|---|
| P0 | Baseline freeze on the post-refactor suite (2026-11 tests round: mirror-tree reorg, merged/parametrized cases, deduped setup, docker & robotics-dataset suites dropped — 87 files / ~1.1k cases): full pytest green snapshot recorded | suite green, snapshot stored |
| P1 | `contracts/` (incl. `config_schema.py`) + `config_freezer.py` + legacy-compat tests; deploy loader switches to the frozen resolved config | all existing tests green; **legacy golden checkpoint loads**; `config.yaml` now fully resolved |
| P2 | PreProcessor chain + Track B datasets + `MixtureDataset` cross-format + `contracts/collate.py`; Track A adapter; `configs/dataloader/` reorg into `robot/` + `driving/` subgroups (Hydra package remap, cfg-tree regression test) | Track A runs through the new chain with **bit-identical loss** to transforms path on a fixed seed (parity gate); composed `cfg` tree unchanged by the config move; transforms package deleted at gate pass |
| P3 | VLASystem absorption (connectors) + FAST vocab path + co-training rules + RTC training config | `vla.yaml` variants reproduce prior losses; smoke_test covers all connectors; co-training fail-fast validation tested |
| P4 | FSDP runner + checkpoint manager unification + `lr_map` + EMA | hybrid-shard 8-GPU smoke vs DeepSpeed reference: loss curves match within tolerance on fixed seed; resume from mid-epoch checkpoint restores step & data position deterministically |
| P5 | Driving chain hardening: ingest tools, time-sync defaults (O1), optional latent cache (O2), calibration flow | driving smoke run on sample clips; sync-gate unit tests with synthetic jitter |
| P6 | Deploy/eval integration: `from_checkpoint` pipeline in deploy, ZMQ eval stub, TRT stubs | deploy of a driving checkpoint end-to-end in the existing server harness |

The transforms-package deletion happens only at the phase gate noted (P2); the registries are
retained (§1.3 rule 1). Each deletion is its own PR.

---

## 11. Config Compatibility Matrix

| Artifact | Old | New | Rule |
|---|---|---|---|
| Checkpoint `config.yaml` | registry-era keys | resolved + same filename | existing registry loader reads old keys unchanged (§4.3); new keys ignored by old code paths |
| Normalization stats | `normalization_stats.npy` (`{joint, eef}`) | `dataset_statistics.json` | loader dual-reads both |
| `action_lr`/`video_lr` | flat keys | `lr_map` prefix table | both accepted; flat keys map to two prefixes |
| `freeze:` list | dotted paths | same semantics + FSDP ignore | unchanged |
| `training.zero_stage` | DeepSpeed | ignored under `backend=fsdp` | `backend` selects the relevant block |

---

## 12. Revision Summary (vs Rev 1)

1. **Fact fixes**: removed non-existent `presets/fast.env` / "RBA" (D5); VLASystem re-scoped from
   "build" to "absorb" (the port already landed upstream); `single_system` added back to scope.
2. **D1**: directory plan changed from renames (`models/`, `dataset/`, `engine/`) to additive
   subpackages — ~200-file import churn and a full-suite regression traded for zero.
3. **Contracts**: added calibration/timestamps, robot-domain mapping, WAM-side prompt & video
   fields; clarified the GPU-encoder boundary (latents/text-embeddings are model-side, never
   dataloader-side).
4. **Checkpoint**: kept the `config.yaml` filename (deploy/tests depend on it); added
   `git_meta.json`; scoped DCP/async save out of the first release; documented sampler
   re-derivation instead of pickled sampler state.
5. **FSDP section grounded** in FluxVLA's actual implementation (sharding modes, ignored modules,
   prefix LR, aggregated checkpoint + reshard), with DeepSpeed retained as default (D2).
6. **FAST** re-modeled as vocabulary-embedded (D4); **co-training** pinned to weighted single
   backward with fail-fast freeze validation; **RTC training-side** conditioning added to head spec.
7. **Data layer** made dual-track (D3) with `cam_frame_offsets` ingest-time precompute, decoder
   resource bounds, offline-latent-cache option (O2), and sample-weight global normalization.
8. Added §4.3 legacy compatibility (normative), §10 phase plan with parity gates, §11 compat
   matrix, §13 open items.
9. Review round 2 — config layout: `action_head/` merged into `action_backbone/` (one concept,
   one config group, mirrors code layout); `configs/dataloader/` split into `robot/` and `driving/`
   subgroups (driving expected to scale to hundreds), with Hydra package remap keeping
   `cfg.dataloader` stable.
10. Review round 3 — contracts clarity: `PipelineContext` fields annotated with design intent +
    a domain/required classification table (driving-only slots: `calib`,
    `frame_timestamps_ns`); trajectory validity mask extended to the 2-D per-dim form (robot
    80-D reserved dims). `ModelInputBatch` explicitly models heterogeneous per-camera resolution
    tiers (front > side > rear) absorbed by image-processor mode (`wam_canvas` slot sizing /
    `qwen_vl_patch` native variable resolution / `standard_tensor` tier equalization), with
    per-cam resolutions as static frozen config.
11. Review round 4 — heterogeneous resolution THROUGH the encoder: `standard_tensor` re-scoped
    from "uniform working resolution (tier equalization)" to per-cam tensors
    (`pixel_values_by_cam`, no upscaling); the merge point is normatively placed MODEL-SIDE —
    per-cam encode → flatten-to-tokens → concat in config cam order + unified mask + per-cam
    RoPE, never earlier; WAM DiT token-merge adapter specified in §6.2 with canvas as default
    until it lands; latent cache (O2) inherits per-cam tiered grids.
12. Review round 5 — comment style unified to comment-before-definition across all contract
    code blocks (`PipelineContext`, `ModelInputBatch`, §3.2 outputs, §6.1 interface);
    a per-layer hidden-states output added to `BackboneOutput` for layer-wise consumers
    (QwenPI connector, DeepStack routing, deep supervision) — populated only on connector
    request, with the memory rationale documented.
13. Review round 6 — output contracts redesigned after review: (a) layer-wise representation
    switched from a packed flatten (`[B, N×L_src, D]`, layer count not inferable from the
    tensor, silent-misconfig risk) to per-layer streams — `ConnectorOutput.condition_tokens:
    List[Tensor]` (len == N_layers, DiT layer i takes entry i; standard routes len == 1,
    one shared mask); (b) `BackboneOutput` naming unified to the HF convention —
    `last_hidden_state` + `hidden_states: Optional[List[Tensor]]` (`hidden_states[-1]`
    aliases `last_hidden_state`) replace the unrelatedly-named context_tokens /
    all_hidden_states pair.
14. Review round 7 (2026-11-13) — **registry mechanism retained**: review found the five
    decorator registries free of the debug/navigation pain that motivated the ban (static
    import-time registration, one class per file, build-time failure, greppable call sites), so
    §1.3 rule 1 is rewritten as "Registry-Based Instantiation (kept)"; the `model/builder.py` /
    `dataloader/builder.py` replacements and the registry deletion targets are dropped, and new
    component classes (connectors, preprocessors) register through the same pattern. Test-suite
    references updated after the 2026-11 tests refactor round (mirror-tree reorg, suite merges +
    parametrization, setup dedup, docker & robotics-dataset suites dropped, pruned to core
    logic): stale "90+ tests" counts replaced and the P0 baseline pinned to the post-refactor
    suite.

## 13. Open Items

- **O1 — Time-sync parameters**: interpolation kind (linear vs spline), tolerance gate (proposal:
  ±5 ms), out-of-gate policy (invalid-mask vs drop). Needs production precision requirements.
- **O2 — Offline VAE latent pre-encoding**: recommended (11-cam online encode is the throughput
  bottleneck); decision required by end of P2 since it affects the Lance schema and ingest tool.
- **O3 — RTC training-side conditioning**: in P3 or deferred to post-P4 (inference-side RTC is
  only meaningful with matching training).
- **O4 — Closed-loop eval target**: which driving simulator protocol the ZMQ eval server targets
  first.
- **O5 — Data governance**: de-identification and access control are ingest-side responsibilities;
  confirm ownership outside this repo.
