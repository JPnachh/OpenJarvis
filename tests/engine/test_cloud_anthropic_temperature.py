"""Claude 5.x models reject an explicit ``temperature``."""

from openjarvis.engine.cloud import _anthropic_temperature_kwargs


def test_claude_5_models_omit_temperature():
    for model in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"):
        assert _anthropic_temperature_kwargs(model, 0.7) == {}


def test_older_models_keep_temperature():
    assert _anthropic_temperature_kwargs("claude-haiku-4-5", 0.3) == {
        "temperature": 0.3
    }
