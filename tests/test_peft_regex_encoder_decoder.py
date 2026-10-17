"""Encoder-decoder models must LoRA both stacks; decoder-only models stay unchanged."""

import os
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import re
from types import SimpleNamespace
import unittest

from torch import nn

from unsloth_zoo.peft_utils import get_peft_regex


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(4, 4)


def _model(is_encoder_decoder):
    model = nn.Module()
    model.config = SimpleNamespace(
        is_encoder_decoder = is_encoder_decoder,
        model_type = "t5gemma2",
        architectures = [],
    )
    model.model = nn.Module()
    model.model.encoder = nn.Module()
    model.model.encoder.text_model = nn.Module()
    model.model.encoder.text_model.layers = nn.ModuleList([_Block()])
    model.model.decoder = nn.Module()
    model.model.decoder.layers = nn.ModuleList([_Block()])
    model.model.layers = nn.ModuleList([_Block()])
    return model


def _pattern(model):
    return get_peft_regex(
        model,
        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
    )


class EncoderDecoderTargets(unittest.TestCase):
    def test_encoder_and_decoder_both_match(self):
        pattern = _pattern(_model(True))
        self.assertIsNotNone(re.fullmatch(
            pattern, "model.encoder.text_model.layers.0.self_attn.q_proj"))
        self.assertIsNotNone(re.fullmatch(
            pattern, "model.decoder.layers.0.self_attn.q_proj"))
        self.assertIsNotNone(re.fullmatch(
            pattern, "model.decoder.layers.0.mlp.down_proj"))

    def test_decoder_only_does_not_gain_decoder_tag(self):
        pattern = _pattern(_model(False))
        self.assertIsNone(re.fullmatch(
            pattern, "model.decoder.layers.0.self_attn.q_proj"))


if __name__ == "__main__":
    unittest.main()
