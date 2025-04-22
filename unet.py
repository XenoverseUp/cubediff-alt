from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from attention import convert_attention_to_inflated
from synchronized_norm import convert_groupnorm_to_synchronized

class CubeDiffUNet(nn.Module):
    def __init__(self,
                 pretrained_model_path: str,
                 num_frames: int = 6,
                 device: Optional[torch.device] = None):
        super().__init__()

        self.pretrained_model_path = pretrained_model_path
        self.num_frames = num_frames
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._load_pretrained_unet()

    def _load_pretrained_unet(self):
        try:
            unet = UNet2DConditionModel.from_pretrained(
                self.pretrained_model_path,
                subfolder="unet"
            )

            self.time_embedding = unet.time_embedding
            self.time_proj = unet.time_proj
            self.conv_in = unet.conv_in  # Use pretrained conv_in directly
            self.down_blocks = unet.down_blocks
            self.mid_block = unet.mid_block
            self.up_blocks = unet.up_blocks
            self.conv_norm_out = unet.conv_norm_out
            self.conv_out = unet.conv_out

            self.config = unet.config
            self.in_channels = unet.config.in_channels
            self.cross_attention_dim = unet.config.cross_attention_dim

            convert_groupnorm_to_synchronized(self, self.num_frames)
            convert_attention_to_inflated(self, self.num_frames)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained UNet: {e}")

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: Optional[torch.Tensor] = None,
                class_labels: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        # 1. Time embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

        # Broadcast timestep to batch dimension
        batch_size = sample.shape[0] // self.num_frames  # e.g., 24 // 6 = 4
        timestep = timestep.expand(batch_size)  # [4]

        t_emb = self.time_proj(timestep)  # [4, dim]
        temb = self.time_embedding(t_emb)  # [4, dim']

        # Repeat temb for each face to match effective batch size (B * num_frames)
        temb = temb.repeat_interleave(self.num_frames, dim=0)  # [4, dim'] -> [24, dim']

        # 2. Process input sample
        hidden_states = self.conv_in(sample)  # [24, 4, 64, 64] -> [24, 320, 64, 64]

        # 3. Down blocks
        down_block_res_samples = []
        for downsample_block in self.down_blocks:
            hidden_states, res_samples = downsample_block(
                hidden_states=hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask
            )
            down_block_res_samples.extend(res_samples)

        # 4. Mid block
        hidden_states = self.mid_block(
            hidden_states=hidden_states,
            temb=temb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask
        )

        # 5. Up blocks
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            hidden_states = upsample_block(
                hidden_states=hidden_states,
                temb=temb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask
            )

        # 6. Output block
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        if not return_dict:
            return hidden_states

        return {"sample": hidden_states}

    def process_cubemap_batch(self,
                             latents: List[torch.Tensor],
                             timesteps: torch.Tensor,
                             encoder_hidden_states: Optional[torch.Tensor] = None,
                             attention_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Process a batch of cubemap faces with synchronized normalization.

        Args:
            latents: List of 6 latent tensors, each [B, C, H, W]
            timesteps: Diffusion timesteps
            encoder_hidden_states: Text embeddings
            attention_mask: Optional attention mask

        Returns:
            List of processed latent tensors
        """
        batch_size = latents[0].shape[0]

        # Stack faces to utilize synchronized normalization
        stacked_latents = torch.cat(latents, dim=0)  # [B*6, C, H, W]

        # Process with UNet
        output = self.forward(
            sample=stacked_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask
        )

        if isinstance(output, dict):
            output = output["sample"]

        output_faces = []
        for i in range(self.num_frames):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            output_faces.append(output[start_idx:end_idx])

        return output_faces

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        attention_params = []

        def is_inflated_attention(module):
            return (
                "inflated" in module.__class__.__name__.lower() or
                "inflatedattention" in module.__class__.__name__.lower()
            )

        for name, module in self.named_modules():
            if is_inflated_attention(module):
                attention_params.extend(module.parameters())

        return attention_params
