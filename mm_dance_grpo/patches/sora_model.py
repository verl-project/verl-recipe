import torch
from mindspeed_mm.models.sora_model import SoRAModel


class MMSoRAModel(SoRAModel):
    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                prompt: torch.Tensor,
                prompt_mask: torch.Tensor = None,
                i2v_clip_feature: torch.Tensor = None,
                i2v_vae_feature: torch.Tensor = None,
                **kwargs):
        return self.predictor(x, timestep, prompt, prompt_mask, i2v_clip_feature, i2v_vae_feature, **kwargs)[0]
