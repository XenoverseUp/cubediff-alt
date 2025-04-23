from typing import List, Tuple, Dict, Optional, Union, Any

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from unet import CubeDiffUNet
from vae import SynchronizedVAE
from config import CubeDiffConfig
from projection import CubeProjection
from prompt_generator import OllamaPromptGenerator
from positional_encoding import PositionalEncoding, PositionalProjectionLayer

class CubeDiff(nn.Module):
    def __init__(self,
                 config,
                 device: Optional[torch.device] = None,
                 prompt_generator: Optional[OllamaPromptGenerator] = None):
        super().__init__()

        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_frames = config.cube_faces
        self.prompt_generator = prompt_generator or OllamaPromptGenerator()

        # Initialize components
        self.vae = SynchronizedVAE(
            pretrained_model_path=config.pretrained_model_path,
            num_frames=self.num_frames,
            device=self.device
        )

        self.unet = CubeDiffUNet(
            pretrained_model_path=config.pretrained_model_path,
            num_frames=self.num_frames,
            device=self.device
        )

        self.positional_encoding = PositionalEncoding(
            cube_size=config.cube_size // 8,  # VAE downsamples by 8
            fov=config.fov,
            overlap=config.overlap
        )

        self.pos_projection = PositionalProjectionLayer(
            in_channels=self.unet.conv_in.in_channels + 2,  # 4 + 2 = 6
            out_channels=self.unet.conv_in.in_channels      # 4
        )

        self.cubemap_utils = CubeProjection()

        # Diffusion parameters
        self.num_train_timesteps = config.num_timesteps
        self.min_beta = config.beta_start
        self.max_beta = config.beta_end

        # Linear noise schedule
        self.betas = torch.linspace(
            self.min_beta, self.max_beta, self.num_train_timesteps, dtype=torch.float32
        )

        # Define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _extract_into_tensor(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def encode_faces(self, faces: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.vae.encode_cubemap(faces)

    def decode_faces(self, latent_faces: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.vae.decode_cubemap(latent_faces)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            x_start: Initial latent [B, C, H, W]
            t: Timestep [B]
            noise: Random noise [B, C, H, W]

        Returns:
            Noisy latent at timestep t
        """
        sqrt_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_v_prediction(self, model_output: torch.Tensor,
                            x: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        alpha_t = self._extract_into_tensor(self.alphas_cumprod, t, x.shape)
        sigma_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        v_prediction = alpha_t.sqrt() * model_output - sigma_t * x
        return v_prediction

    def compute_prediction_x0(self, model_output: torch.Tensor,
                             x: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        """
        Compute x0 prediction from model output.

        Args:
            model_output: UNet output
            x: Input latent
            t: Timestep

        Returns:
            Predicted x0
        """
        alpha_t = self._extract_into_tensor(self.alphas_cumprod, t, x.shape)
        sigma_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        if self.config.prediction_type == "v":
            x0_prediction = alpha_t.sqrt() * x - sigma_t * model_output
        elif self.config.prediction_type == "epsilon":
            x0_prediction = (x - sigma_t * model_output) / alpha_t.sqrt()
        else:
            x0_prediction = model_output

        return x0_prediction

    def get_text_embedding(self,
                          prompts: Union[List[str], List[List[str]]],
                          batch_size: int) -> torch.Tensor:
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.config.pretrained_model_path, subfolder="tokenizer"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                self.config.pretrained_model_path, subfolder="text_encoder"
            ).to(self.device)

            if isinstance(prompts[0], list):
                all_embeddings = []

                for face_idx in range(self.num_frames):
                    face_prompts = [prompt[face_idx] for prompt in prompts]

                    # Tokenize text
                    text_input = tokenizer(
                        face_prompts,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    # Generate embeddings
                    with torch.no_grad():
                        face_embeddings = text_encoder(
                            text_input.input_ids.to(self.device)
                        )[0]

                    all_embeddings.append(face_embeddings)

                embeddings = torch.zeros(
                    batch_size * self.num_frames,
                    face_embeddings.shape[1],
                    face_embeddings.shape[2],
                    device=self.device
                )

                for batch_idx in range(batch_size):
                    for face_idx, face_emb in enumerate(all_embeddings):
                        embeddings[face_idx * batch_size + batch_idx] = face_emb[batch_idx]

            else:
                # Common prompt for all faces
                text_input = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                with torch.no_grad():
                    text_embeddings = text_encoder(
                        text_input.input_ids.to(self.device)
                    )[0]

                embeddings = text_embeddings.repeat_interleave(self.num_frames, dim=0)

            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to generate text embeddings: {e}")

    def get_null_embedding(self, batch_size: int) -> torch.Tensor:
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.config.pretrained_model_path, subfolder="tokenizer"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                self.config.pretrained_model_path, subfolder="text_encoder"
            ).to(self.device)

            uncond_input = tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                uncond_embeddings = text_encoder(
                    uncond_input.input_ids.to(self.device)
                )[0]

            uncond_embeddings = uncond_embeddings.repeat_interleave(self.num_frames, dim=0)

            return uncond_embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to generate null embeddings: {e}")

    @torch.no_grad()
    def ddim_sample(self,
                    shape: Tuple[int, ...],
                    cond_face_latent: Optional[torch.Tensor] = None,
                    cond_face_mask: Optional[torch.Tensor] = None,
                    text_embeddings: Optional[torch.Tensor] = None,
                    uncond_embeddings: Optional[torch.Tensor] = None,
                    guidance_scale: float = 7.5,
                    ddim_steps: int = 50,
                    eta: float = 0.0) -> torch.Tensor:
        device = self.device
        batch_size = shape[0] // self.num_frames  # Effective batch size (e.g., 24 // 6 = 4)

        latents = torch.randn(shape, device=device)

        if cond_face_latent is not None and cond_face_mask is not None:
            latents = latents * (1 - cond_face_mask) + cond_face_latent * cond_face_mask

        timesteps = torch.linspace(self.num_train_timesteps - 1, 0, ddim_steps, device=device).long()

        alphas_cumprod = self.alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            timestep = t.expand(batch_size)

            # For classifier-free guidance: do 2 forward passes
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents

            # Apply positional encoding
            pos_enc = self.positional_encoding.add_positional_encoding_flat(latent_model_input, num_frames=self.num_frames)
            latent_model_input_with_pos = torch.cat([latent_model_input, pos_enc], dim=1)
            latent_model_input_projected = self.pos_projection(latent_model_input_with_pos)

            if guidance_scale > 1.0:
                embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
            else:
                embeddings = text_embeddings

            noise_pred = self.unet(
                sample=latent_model_input_projected,
                timestep=timestep,
                encoder_hidden_states=embeddings
            )

            if isinstance(noise_pred, dict):
                noise_pred = noise_pred["sample"]

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            x0_pred = self.compute_prediction_x0(noise_pred, latents, timestep)

            if cond_face_latent is not None and cond_face_mask is not None:
                x0_pred = x0_pred * (1 - cond_face_mask) + cond_face_latent * cond_face_mask

            prev_timestep = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor([0], device=device)

            alpha_t = self._extract_into_tensor(alphas_cumprod, timestep, latents.shape)
            alpha_prev = self._extract_into_tensor(alphas_cumprod, prev_timestep, latents.shape)

            sigma_t = self._extract_into_tensor(sqrt_one_minus_alphas_cumprod, timestep, latents.shape)
            sigma_prev = self._extract_into_tensor(sqrt_one_minus_alphas_cumprod, prev_timestep, latents.shape)

            latents_prev_deterministic = alpha_prev.sqrt() * x0_pred + sigma_prev * (latents - alpha_t.sqrt() * x0_pred) / sigma_t

            if eta > 0:
                noise = torch.randn_like(latents)
                sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
                latents = latents_prev_deterministic + sigma * noise
            else:
                latents = latents_prev_deterministic

        return latents

    @torch.no_grad()
    def generate_panorama(self,
                         prompts: Union[str, List[str], List[List[str]]],
                         cond_face_idx: Optional[int] = None,
                         cond_face_image: Optional[torch.Tensor] = None,
                         guidance_scale: float = 7.5,
                         height: int = 512,
                         width: int = 512,
                         num_inference_steps: int = 50,
                         eta: float = 0.0,
                         output_type: str = "latent",
                         return_dict: bool = True) -> Union[Dict[str, Any], List[torch.Tensor]]:
        device = self.device

        if isinstance(prompts, str):
            prompts = [prompts]

        batch_size = len(prompts)

        text_embeddings = self.get_text_embedding(prompts, batch_size)
        uncond_embeddings = self.get_null_embedding(batch_size)

        cond_face_latent = None
        cond_face_mask = None

        if cond_face_image is not None and cond_face_idx is not None:
            cond_face_latent = self.vae.encode_to_latent(cond_face_image)

            cond_face_mask = torch.zeros(
                batch_size * self.num_frames,
                1,
                height // 8,
                width // 8,
                device=device
            )

            for i in range(batch_size):
                start_idx = i * self.num_frames + cond_face_idx
                cond_face_mask[start_idx] = 1.0

        latent_shape = (
            batch_size * self.num_frames,
            4,
            height // 8,
            width // 8
        )

        latents = self.ddim_sample(
            shape=latent_shape,
            cond_face_latent=cond_face_latent,
            cond_face_mask=cond_face_mask,
            text_embeddings=text_embeddings,
            uncond_embeddings=uncond_embeddings,
            guidance_scale=guidance_scale,
            ddim_steps=num_inference_steps,
            eta=eta
        )

        face_latents = []
        for i in range(self.num_frames):
            face_batch = []
            for b in range(batch_size):
                face_batch.append(latents[b * self.num_frames + i])
            face_latents.append(torch.stack(face_batch))

        if output_type == "latent":
            if return_dict:
                return {"latent_faces": face_latents}
            else:
                return face_latents

        decoded_faces = []
        for face_batch in face_latents:
            decoded_faces.append(self.vae.decode_from_latent(face_batch))

        if output_type == "faces":
            if return_dict:
                return {"faces": decoded_faces}
            else:
                return decoded_faces

        elif output_type == "equirectangular":
            panoramas = []
            for b in range(batch_size):
                batch_faces = [face[b:b+1] for face in decoded_faces]
                cubemap = torch.stack(batch_faces, dim=1)
                panorama = self.cubemap_utils.forward(
                    cubemap,
                    input_type='cubemap',
                    output_type='equirect'
                )
                panoramas.append(panorama[0])

            panoramas = torch.stack(panoramas)

            if return_dict:
                return {"panorama": panoramas, "faces": decoded_faces}
            else:
                return panoramas

        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

    def prepare_latents_for_training(self,
                                    images: List[torch.Tensor],
                                    timesteps: torch.Tensor,
                                    noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            latent_faces = self.encode_faces(images)

        latents = torch.cat(latent_faces, dim=0)  # [B*6, 4, H/8, W/8]

        if noise is None:
            noise = torch.randn_like(latents)

        noisy_latents = self.q_sample(latents, timesteps.repeat(self.num_frames), noise)

        # Apply positional encoding
        pos_enc = self.positional_encoding.add_positional_encoding_flat(noisy_latents, num_frames=self.num_frames)
        noisy_latents_with_pos = torch.cat([noisy_latents, pos_enc], dim=1)
        noisy_latents_projected = self.pos_projection(noisy_latents_with_pos)

        return noisy_latents_projected, latents, noise

    def forward(self,
                batch: Dict[str, Any],
                timesteps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        images = batch["faces"]
        text = batch["text"]
        batch_size = images[0].shape[0]
        device = images[0].device

        if timesteps is None:
            timesteps = torch.randint(
                0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long
            )

        noisy_latents, original_latents, noise = self.prepare_latents_for_training(images, timesteps)

        use_text_cond = torch.rand(batch_size) >= self.config.classifier_free_guidance_prob

        text_embeddings = self.get_text_embedding(text, batch_size)
        null_embeddings = self.get_null_embedding(batch_size)

        embeddings = text_embeddings.clone()
        for i in range(batch_size):
            if not use_text_cond[i]:
                for j in range(self.num_frames):
                    embeddings[i * self.num_frames + j] = null_embeddings[i * self.num_frames + j]

        model_output = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=embeddings
        )

        if isinstance(model_output, dict):
            model_output = model_output["sample"]

        if self.config.prediction_type == "v":
            target = self.compute_v_prediction(model_output, original_latents, timesteps.repeat(self.num_frames))
            loss = F.mse_loss(model_output, target)
        elif self.config.prediction_type == "epsilon":
            loss = F.mse_loss(model_output, noise)
        else:
            loss = F.mse_loss(model_output, original_latents)

        return {"loss": loss}

    def generate_per_face_prompts(self,
                                 prompt: str,
                                 num_samples: int = 1) -> List[List[str]]:
        return self.prompt_generator.generate_per_face_prompts(
            prompt=prompt,
            num_samples=num_samples
        )

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        unet_dir = os.path.join(output_dir, "unet")
        os.makedirs(unet_dir, exist_ok=True)
        torch.save(self.unet.state_dict(), os.path.join(unet_dir, "pytorch_model.bin"))

        vae_dir = os.path.join(output_dir, "vae")
        os.makedirs(vae_dir, exist_ok=True)
        torch.save(self.vae.state_dict(), os.path.join(vae_dir, "pytorch_model.bin"))

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(vars(self.config), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, config=None, device=None, prompt_generator=None):
        if config is None:
            config_path = os.path.join(pretrained_model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)

                config = CubeDiffConfig()
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            else:
                config = CubeDiffConfig()

        model = cls(config, device, prompt_generator)

        unet_path = os.path.join(pretrained_model_path, "unet", "pytorch_model.bin")
        if os.path.exists(unet_path):
            model.unet.load_state_dict(torch.load(unet_path, map_location=model.device))

        vae_path = os.path.join(pretrained_model_path, "vae", "pytorch_model.bin")
        if os.path.exists(vae_path):
            model.vae.load_state_dict(torch.load(vae_path, map_location=model.device))

        return model

    def fine_grained_generate(self,
                             prompts: List[List[str]],
                             cond_face_idx: Optional[int] = None,
                             cond_face_image: Optional[torch.Tensor] = None,
                             guidance_scale: float = 7.5,
                             height: int = 512,
                             width: int = 512,
                             num_inference_steps: int = 50,
                             eta: float = 0.0,
                             output_type: str = "equirectangular",
                             return_dict: bool = True) -> Union[Dict[str, Any], torch.Tensor]:
        if not isinstance(prompts, list) or not isinstance(prompts[0], list):
            raise ValueError("Prompts must be a list of lists, with one list per sample and 6 prompts per face")

        for sample_prompts in prompts:
            if len(sample_prompts) != 6:
                raise ValueError(f"Each sample must have exactly 6 prompts, got {len(sample_prompts)}")

        return self.generate_panorama(
            prompts=prompts,
            cond_face_idx=cond_face_idx,
            cond_face_image=cond_face_image,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            eta=eta,
            output_type=output_type,
            return_dict=return_dict
        )

    @torch.no_grad()
    def evaluate(self, eval_dataset, metrics=None, batch_size=1):
        device = self.device
        self.eval()

        if metrics is None:
            metrics = {
                "fid": FrechetInceptionDistance(feature=64).to(device),
                "kid": KernelInceptionDistance(subset_size=100).to(device)
            }

        dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        all_generated = []
        all_real = []

        for batch in tqdm(dataloader, desc="Evaluation"):
            real_faces = batch["faces"]
            real_faces = [face.to(device) for face in real_faces]
            text_prompts = batch.get("text", ["a panorama"] * batch_size)

            generated = self.generate_panorama(
                prompts=text_prompts,
                output_type="equirectangular",
                return_dict=True
            )

            real_cubemap = torch.stack([face for face in real_faces], dim=1)
            real_equirect = self.cubemap_utils.forward(
                real_cubemap,
                input_type='cubemap',
                output_type='equirect'
            )

            generated_pano = generated["panorama"]
            generated_uint8 = (generated_pano * 255).to(torch.uint8)
            real_uint8 = (real_equirect * 255).to(torch.uint8)

            all_generated.append(generated_uint8)
            all_real.append(real_uint8)

            for metric_name, metric in metrics.items():
                if hasattr(metric, "update"):
                    metric.update(generated_uint8, real=False)
                    metric.update(real_uint8, real=True)

        all_generated = torch.cat(all_generated, dim=0)
        all_real = torch.cat(all_real, dim=0)

        results = {}
        for metric_name, metric in metrics.items():
            if hasattr(metric, "compute"):
                results[metric_name] = metric.compute()
            else:
                results[metric_name] = metric(all_generated, all_real)

        return results
