from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D
from attention import convert_attention_to_inflated
from synchronized_norm import convert_groupnorm_to_synchronized

class CustomCrossAttnDownBlock2D(CrossAttnDownBlock2D):
    def forward(self,
                hidden_states: torch.Tensor,
                temb: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        res_samples = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb=temb)
            res_samples.append(hidden_states)

            if attn is not None:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **kwargs
                )
                res_samples.append(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states, res_samples

class CubeDiffUNet(nn.Module):
    def __init__(self,
                 pretrained_model_path: str,
                 num_frames: int = 6,
                 device: Optional[torch.device] = None,
                 use_inflated_attention: bool = True,
                 use_synchronized_norm: bool = True):
        super().__init__()

        self.pretrained_model_path = pretrained_model_path
        self.num_frames = num_frames
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_inflated_attention = use_inflated_attention
        self.use_synchronized_norm = use_synchronized_norm

        self._load_pretrained_unet()

    def _load_pretrained_unet(self):
        try:
            unet = UNet2DConditionModel.from_pretrained(
                self.pretrained_model_path,
                subfolder="unet"
            )

            self.time_embedding = unet.time_embedding
            self.time_proj = unet.time_proj
            self.conv_in = unet.conv_in
            self.down_blocks = nn.ModuleList([
                CustomCrossAttnDownBlock2D(
                    num_layers=block.num_layers,
                    in_channels=block.in_channels,
                    out_channels=block.out_channels,
                    temb_channels=block.temb_channels,
                    add_downsample=hasattr(block, 'downsamplers') and block.downsamplers is not None,
                    resnet_eps=block.resnets[0].norm1.eps,
                    resnet_act_fn=block.resnets[0].act_fn.__class__.__name__.lower(),
                    resnet_groups=block.resnets[0].groups,
                    cross_attention_dim=block.attentions[0].cross_attention_dim if block.attentions else None,
                    attn_num_head_channels=block.attentions[0].num_heads if block.attentions else None,
                    downsample_padding=block.downsamplers[0].padding if block.downsamplers else 1,
                ) if isinstance(block, CrossAttnDownBlock2D) else block
                for block in unet.down_blocks
            ])
            self.mid_block = unet.mid_block
            self.up_blocks = unet.up_blocks
            self.conv_norm_out = unet.conv_norm_out
            self.conv_out = unet.conv_out

            self.config = unet.config
            self.in_channels = unet.config.in_channels
            self.cross_attention_dim = unet.config.cross_attention_dim

            print("Down blocks before conversion:", len(self.down_blocks), [len(block.resnets) for block in self.down_blocks])
            print("Up blocks before conversion:", len(self.up_blocks), [len(block.resnets) for block in self.up_blocks])

            if self.use_synchronized_norm:
                convert_groupnorm_to_synchronized(self, self.num_frames)
            if self.use_inflated_attention:
                convert_attention_to_inflated(self, self.num_frames)

            print("Down blocks after conversion:", len(self.down_blocks), [len(block.resnets) for block in self.down_blocks])
            print("Up blocks after conversion:", len(self.up_blocks), [len(block.resnets) for block in self.up_blocks])

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

        batch_size = sample.shape[0] // self.num_frames
        timestep = timestep.expand(batch_size)

        t_emb = self.time_proj(timestep)
        temb = self.time_embedding(t_emb)
        temb = temb.repeat_interleave(self.num_frames, dim=0)
        print("temb shape:", temb.shape)  # Debug: Should be [24, dim']

        # 2. Process input sample
        hidden_states = self.conv_in(sample)
        print("hidden_states after conv_in:", hidden_states.shape)  # Debug: Should be [24, 320, 64, 64]

        # 3. Down blocks
        down_block_res_samples = []
        for i, downsample_block in enumerate(self.down_blocks):
            hidden_states, res_samples = downsample_block(
                hidden_states=hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask
            )
            print(f"Down block {i} res_samples count:", len(res_samples), [s.shape for s in res_samples])  # Debug
            down_block_res_samples.extend(res_samples)

        print("Total down_block_res_samples:", len(down_block_res_samples))  # Debug

        # 4. Mid block
        hidden_states = self.mid_block(
            hidden_states=hidden_states,
            temb=temb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask
        )
        print("hidden_states after mid_block:", hidden_states.shape)  # Debug

        # 5. Up blocks
        for i, upsample_block in enumerate(self.up_blocks):
            expected_resnets = len(upsample_block.resnets)
            if len(down_block_res_samples) < expected_resnets:
                print(f"Warning: Not enough residual connections for up block {i}. Expected {expected_resnets}, got {len(down_block_res_samples)}. Using empty residuals.")
                res_samples = []
            else:
                res_samples = down_block_res_samples[-expected_resnets:]
                down_block_res_samples = down_block_res_samples[:-expected_resnets]

            print(f"Up block {i} res_samples count:", len(res_samples), [s.shape for s in res_samples] if res_samples else [])  # Debug
            hidden_states = upsample_block(
                hidden_states=hidden_states,
                temb=temb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask
            )
            print(f"hidden_states after up block {i}:", hidden_states.shape)  # Debug

        # 6. Output block
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        print("Final hidden_states:", hidden_states.shape)  # Debug

        if not return_dict:
            return hidden_states

        return {"sample": hidden_states}

    def process_cubemap_batch(self,
                             latents: List[torch.Tensor],
                             timesteps: torch.Tensor,
                             encoder_hidden_states: Optional[torch.Tensor] = None,
                             attention_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        batch_size = latents[0].shape[0]
        stacked_latents = torch.cat(latents, dim=0)

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
