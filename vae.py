import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from diffusers import AutoencoderKL

from synchronized_norm import convert_groupnorm_to_synchronized

class SynchronizedVAE(nn.Module):
    """
    VAE with synchronized GroupNorm for consistent color tones across cubemap faces.
    Based on pretrained Stable Diffusion VAE but with synchronized normalization.
    """

    def __init__(self,
                pretrained_model_path: str,
                num_frames: int = 6,
                device: Optional[torch.device] = None):

        super().__init__()

        self.pretrained_model_path = pretrained_model_path
        self.num_frames = num_frames
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._load_pretrained_vae()

        self.encoder = convert_groupnorm_to_synchronized(self.encoder, num_frames)
        self.decoder = convert_groupnorm_to_synchronized(self.decoder, num_frames)

    def _load_pretrained_vae(self):
        try:
            vae = AutoencoderKL.from_pretrained(
                self.pretrained_model_path,
                subfolder="vae"
            )

            self.encoder = vae.encoder
            self.decoder = vae.decoder
            self.quant_conv = vae.quant_conv
            self.post_quant_conv = vae.post_quant_conv

            self.scaling_factor = vae.config.scaling_factor
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained VAE: {e}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def encode_cubemap(self, faces: List[torch.Tensor]) -> List[torch.Tensor]:
        # Stack faces to utilize synchronized normalization
        batch_size = faces[0].shape[0]
        stacked_faces = torch.cat(faces, dim=0)  # [B*6, C, H, W]

        stacked_latents = self.encode(stacked_faces)  # [B*6, 4, H/8, W/8]

        # Unstack latents
        latent_faces = []
        for i in range(self.num_frames):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            latent_faces.append(stacked_latents[start_idx:end_idx])

        return latent_faces

    def decode_cubemap(self, latent_faces: List[torch.Tensor]) -> List[torch.Tensor]:
        batch_size = latent_faces[0].shape[0]
        stacked_latents = torch.cat(latent_faces, dim=0)  # [B*6, 4, H/8, W/8]

        stacked_reconstructed = self.decode(stacked_latents)  # [B*6, 3, H, W]

        reconstructed_faces = []
        for i in range(self.num_frames):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            reconstructed_faces.append(stacked_reconstructed[start_idx:end_idx])

        return reconstructed_faces

    def encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.shape

        # Reshape to [B*F, C, H, W]
        x_reshaped = x.reshape(batch_size * num_frames, channels, height, width)

        latent = self.encode(x_reshaped)

        # Reshape back to [B, F, 4, H/8, W/8]
        latent_height, latent_width = latent.shape[2:]
        latent = latent.reshape(batch_size, num_frames, 4, latent_height, latent_width)

        return latent

    def decode_batch(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, latent_height, latent_width = z.shape

        # Reshape to [B*F, 4, H/8, W/8]
        z_reshaped = z.reshape(batch_size * num_frames, channels, latent_height, latent_width)
        reconstructed = self.decode(z_reshaped)

        # Reshape back to [B, F, 3, H, W]
        rec_channels, rec_height, rec_width = reconstructed.shape[1:]
        reconstructed = reconstructed.reshape(batch_size, num_frames, rec_channels, rec_height, rec_width)

        return reconstructed

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize: [-1, 1]
        x = 2 * x - 1

        latent = self.encode(x)
        return  latent * self.scaling_factor

    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.scaling_factor
        x = self.decode(z)

        # Normalize: [0, 1]
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        return x
