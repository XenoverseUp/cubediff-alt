from typing import Dict

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding:
    face_vectors = {
        'front': np.array([0, 0, 1]),
        'right': np.array([1, 0, 0]),
        'back': np.array([0, 0, -1]),
        'left': np.array([-1, 0, 0]),
        'up': np.array([0, 1, 0]),
        'down': np.array([0, -1, 0])
    }

    face_indices = {
        'front': 0,
        'right': 1,
        'back': 2,
        'left': 3,
        'up': 4,
        'down': 5
    }

    def __init__(self, cube_size: int = 512, fov: float = 95.0, overlap: float = 2.5):
        self.cube_size = cube_size
        self.fov = fov
        self.overlap = overlap

        self.position_encodings = self._precompute_position_encodings()

    def _precompute_position_encodings(self) -> Dict[str, torch.Tensor]:
        encodings = {}

        for face_name, face_vector in self.face_vectors.items():
            encoding = self._compute_face_encoding(face_name, face_vector)
            encodings[face_name] = encoding

        return encodings

    def _compute_face_encoding(self, face_name: str, face_vector: np.ndarray) -> torch.Tensor:

        grid_x = np.linspace(-1, 1, self.cube_size)
        grid_y = np.linspace(-1, 1, self.cube_size)
        X, Y = np.meshgrid(grid_x, grid_y)

        if face_name == 'front':
            x, y, z = X, Y, np.ones_like(X)
        elif face_name == 'right':
            x, y, z = np.ones_like(X), Y, -X
        elif face_name == 'back':
            x, y, z = -X, Y, -np.ones_like(X)
        elif face_name == 'left':
            x, y, z = -np.ones_like(X), Y, X
        elif face_name == 'up':
            x, y, z = X, np.ones_like(X), -Y
        elif face_name == 'down':
            x, y, z = X, -np.ones_like(X), Y

        fov_rad = np.radians(self.fov)
        scale = np.tan(fov_rad / 2)
        x *= scale
        y *= scale

        norm = np.sqrt(x**2 + y**2 + z**2)
        x /= norm
        y /= norm
        z /= norm

        u = np.arctan2(x, z)
        v = np.arctan2(y, np.sqrt(x**2 + z**2))

        # Normalize to [0, 1]
        u = (u / (2 * np.pi)) + 0.5
        v = (v / np.pi) + 0.5

        encoding = np.stack([u, v], axis=0)
        encoding = torch.from_numpy(encoding).float()

        return encoding

    def get_positional_encoding(self,
                              batch_size: int,
                              device: torch.device) -> torch.Tensor:
        """
        Get positional encoding for all cubemap faces.

        Args:
            batch_size: Batch size
            device: Device to place tensors on

        Returns:
            torch.Tensor: Position encoding tensor [B, 6, 2, H, W]
        """
        encodings_list = []

        for face_name in ['front', 'right', 'back', 'left', 'up', 'down']:
            encoding = self.position_encodings[face_name].to(device)
            encodings_list.append(encoding.unsqueeze(0))

        encodings = torch.stack(encodings_list, dim=0)
        encodings = encodings.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        return encodings

    def add_positional_encoding(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to latent representations.

        Args:
            latents: Latent tensor [B, 6, C, H, W]

        Returns:
            torch.Tensor: Latents with positional encoding [B, 6, C+2, H, W]
        """
        batch_size, num_faces, channels, height, width = latents.shape
        device = latents.device

        # Get positional encodings
        encodings = self.get_positional_encoding(batch_size, device)

        # Reshape encodings to match latents
        encodings = encodings.reshape(batch_size, num_faces, 2, height, width)

        # Concatenate along channel dimension
        latents_with_pos = torch.cat([latents, encodings], dim=2)

        return latents_with_pos

    def add_positional_encoding_flat(self, latents: torch.Tensor, num_frames: int = 6) -> torch.Tensor:
        """
        Add positional encoding to flattened latent representations.

        Args:
            latents: Latent tensor [B*6, C, H, W] or [B, 6*C, H, W]
            num_frames: Number of frames (should be 6 for cubemap)

        Returns:
            torch.Tensor: Latents with positional encoding [B*6, C+2, H, W] or [B, 6*(C+2), H, W]
        """
        if latents.shape[0] % num_frames == 0:
            # Case [B*6, C, H, W]
            batch_size = latents.shape[0] // num_frames
            channels, height, width = latents.shape[1:]

            # Reshape to [B, 6, C, H, W]
            latents_reshaped = latents.reshape(batch_size, num_frames, channels, height, width)

            # Add positional encoding
            latents_with_pos = self.add_positional_encoding(latents_reshaped)

            # Reshape back to [B*6, C+2, H, W]
            return latents_with_pos.reshape(batch_size * num_frames, channels + 2, height, width)

        else:
            # Case [B, 6*C, H, W]
            batch_size = latents.shape[0]
            channels_total = latents.shape[1]
            channels = channels_total // num_frames
            height, width = latents.shape[2:]

            # Reshape to [B, 6, C, H, W]
            latents_reshaped = latents.reshape(batch_size, num_frames, channels, height, width)

            # Add positional encoding
            latents_with_pos = self.add_positional_encoding(latents_reshaped)

            # Reshape back to [B, 6*(C+2), H, W]
            return latents_with_pos.reshape(batch_size, num_frames * (channels + 2), height, width)

    def spatial_uv_encoding(self, batch_size: int, height: int, width: int,
                         device: torch.device) -> torch.Tensor:
        """
        Generate UV positional encoding for all spatial positions across all faces.

        Args:
            batch_size: Batch size
            height: Height of each face
            width: Width of each face
            device: Device to place tensors on

        Returns:
            torch.Tensor: Spatial UV encoding tensor [B, 6, 2, H, W]
        """
        pos_enc = self.get_positional_encoding(batch_size, device)

        if height != self.cube_size or width != self.cube_size:
            pos_enc_resized = []
            for i in range(6):
                face_enc = pos_enc[:, i]  # [B, 2, H, W]
                face_enc_resized = F.interpolate(
                    face_enc, size=(height, width), mode='bilinear', align_corners=True
                )
                pos_enc_resized.append(face_enc_resized)
            pos_enc = torch.stack(pos_enc_resized, dim=1)  # [B, 6, 2, H, W]

        return pos_enc

class PositionalProjectionLayer(nn.Module):
    def __init__(self, in_channels=6, out_channels=4):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.projection(x)
