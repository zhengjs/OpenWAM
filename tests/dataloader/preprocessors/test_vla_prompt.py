"""VLAPromptTokenizeProcessor (spec_data_zh.md §5): token path + CoT masking."""

import pytest

from openwam.contracts.config_schema import PreprocessorChainConfig, ProcessorSpec
from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.preprocessors import PreProcessorPipeline


class StubTokenizer:
    """Whitespace tokenizer: stable ids per word."""

    def __call__(self, text):
        return {"input_ids": [abs(hash(w)) % 1000 for w in text.split()]}


def make_ctx(nav="turn left", cot=None, ego=None):
    return PipelineContext(sample_id="", clip_uri="", anchor_timestamp_us=0, cam_frame_offsets={},
                           source_name="d", nav_instruction=nav, cot_text=cot,
                           ego_state_tensor=ego)


def build(**kwargs):
    return PreProcessorPipeline.from_config(PreprocessorChainConfig(
        chain=[ProcessorSpec(name="vla_prompt", kwargs={"tokenizer": StubTokenizer(), **kwargs})]))


def test_prompt_tokens_masked_when_cot_present():
    out = build(camera_tokens="<CAM><CAM>").process(make_ctx(nav="turn left", cot="slow then turn"))
    n_prompt = len(StubTokenizer()("<CAM><CAM> turn left")["input_ids"])
    assert (out.vlm_labels[:n_prompt] == -100).all()   # prompt suppressed
    assert (out.vlm_labels[n_prompt:] != -100).all()   # CoT supervised
    assert out.prompt is None and out.input_ids is not None
    assert (out.attention_mask == 1).all()


def test_full_supervision_without_cot():
    out = build().process(make_ctx(nav="go straight"))
    assert (out.vlm_labels != -100).all()


def test_camera_tokens_alone_still_emit_stream():
    out = build(camera_tokens="<CAM>").process(make_ctx(nav=""))
    assert out.input_ids is not None


def test_no_language_passthrough():
    out = build().process(make_ctx(nav=""))
    assert out.input_ids is None and out.vlm_labels is None


def test_state2str_requires_callable():
    with pytest.raises(ValueError, match="state2str_fn"):
        build(use_state2str=True).process(make_ctx(nav="go", ego=None))
