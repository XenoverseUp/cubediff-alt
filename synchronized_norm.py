import torch
import torch.nn as nn

class SynchronizedGroupNorm2d(nn.GroupNorm):
    """
    Synchronized Group Normalization for Cubemap faces.
    This class extends GroupNorm to normalize across both spatial and frame dimensions,
    ensuring consistent color tones across all faces of the cubemap.
    """

    def __init__(self,
                num_groups: int,
                num_channels: int,
                eps: float = 1e-5,
                affine: bool = True,
                num_frames: int = 6):

        super().__init__(num_groups, num_channels, eps, affine)
        self.num_frames = num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_frames = x.shape[0]

        if batch_frames % self.num_frames == 0:
            batch_size = batch_frames // self.num_frames

            # Handle both 3D and 4D tensors
            if len(x.shape) == 3:
                # For 3D tensors [B*F, C, L]
                channels, seq_length = x.shape[1:]
                height, width = 1, seq_length
            else:
                # For 4D tensors [B*F, C, H, W]
                channels, height, width = x.shape[1:]

            original_shape = x.shape

            try:
                # Reshape to [B, F, C, H, W] then to [B, F*C, H, W]
                x_reshaped = x.reshape(batch_size, self.num_frames, channels, height, width)
                x_sync = x_reshaped.reshape(batch_size, self.num_frames * channels, height, width)

                # Calculate adjusted groups
                adjusted_groups = self.num_groups * self.num_frames

                # Manual synchronized normalization implementation
                c_per_group = (self.num_frames * channels) // adjusted_groups
                x_groups = x_sync.reshape(batch_size, adjusted_groups, c_per_group, height, width)
                x_groups_flat = x_groups.reshape(batch_size, adjusted_groups, c_per_group, -1)

                mean = x_groups_flat.mean(dim=3, keepdim=True)
                var = x_groups_flat.var(dim=3, keepdim=True, unbiased=False)

                x_groups_norm = (x_groups_flat - mean) / torch.sqrt(var + self.eps)

                # Reshape and apply affine transform
                x_sync_norm = x_groups_norm.reshape(batch_size, self.num_frames * channels, height, width)

                if self.affine:
                    x_sync_norm = x_sync_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

                # Reshape back to original shape
                x_out = x_sync_norm.reshape(original_shape)

                return x_out

            except Exception as e:
                return super().forward(x)
        else:
            # Fallback to standard group norm if batch size is not divisible by num_frames
            return super().forward(x)

class SyncGroupNormBlock(nn.Module):
    """
    A block that applies synchronized group normalization.
    """

    def __init__(self,
                num_channels: int,
                num_groups: int = 32,
                eps: float = 1e-5,
                num_frames: int = 6):
        """
        Initialize the SyncGroupNormBlock.

        Args:
            num_channels: Number of channels in the input
            num_groups: Number of groups for group normalization
            eps: Small constant for numerical stability
            num_frames: Number of frames to synchronize across
        """
        super().__init__()
        self.norm = SynchronizedGroupNorm2d(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=True,
            num_frames=num_frames
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying synchronized group normalization"""
        return self.norm(x)


def convert_groupnorm_to_synchronized(
    model: nn.Module,
    num_frames: int = 6,
    inplace: bool = True
) -> nn.Module:
    """
    Convert all GroupNorm layers in a model to SynchronizedGroupNorm2d.

    Args:
        model: The model to convert
        num_frames: Number of frames (cubemap faces) to synchronize across
        inplace: If True, modifies the model in-place

    Returns:
        Model with synchronized group normalization
    """
    if not inplace:
        model = model.copy()

    for name, module in model.named_children():
        if isinstance(module, nn.GroupNorm):
            setattr(
                model,
                name,
                SynchronizedGroupNorm2d(
                    module.num_groups,
                    module.num_channels,
                    module.eps,
                    module.affine,
                    num_frames
                )
            )
            if module.affine:
                getattr(model, name).weight.data.copy_(module.weight.data)
                getattr(model, name).bias.data.copy_(module.bias.data)
        else:
            convert_groupnorm_to_synchronized(module, num_frames, True)

    return model
