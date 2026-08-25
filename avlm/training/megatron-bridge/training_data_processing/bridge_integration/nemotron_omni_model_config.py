"""Keep Nemotron Omni's outer model on the config owned by Bridge training."""

from __future__ import annotations

from functools import wraps

from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import NemotronOmniModelProvider


def apply_nemotron_omni_model_config_override() -> None:
    """Attach the original provider config to each constructed outer Omni model.

    The provider deep-copies itself for the inner language model. Without this
    override, the outer model also inherits that copy, while Bridge installs
    gradient-finalization callbacks on the original provider. DDP and the
    pipeline schedule must use the original provider so those callbacks run.
    """

    if getattr(NemotronOmniModelProvider, "_avlm_model_config_override", False):
        return

    original_provide = NemotronOmniModelProvider.provide

    @wraps(original_provide)
    def provide(self, *args, **kwargs):
        model = original_provide(self, *args, **kwargs)
        model.config = self
        assert model.config is self
        return model

    NemotronOmniModelProvider.provide = provide
    NemotronOmniModelProvider._avlm_model_config_override = True
