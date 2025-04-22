from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

USE_FLASH_ATTENTION = False

if hasattr(F, "scaled_dot_product_attention"):
    USE_FLASH_ATTENTION = True

class InflatedCrossAttention(nn.Module):
    """
    Inflated cross attention module for cubemap faces.
    This extends standard cross attention to allow attention across all faces.
    Uses flash attention when available for memory efficiency.
    """

    def __init__(self,
                query_dim: int,
                context_dim: Optional[int] = None,
                heads: int = 8,
                dim_head: int = 64,
                dropout: float = 0.0,
                num_frames: int = 6):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.num_frames = num_frames
        self.use_flash_attn = USE_FLASH_ATTENTION

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def _flash_attention(self, q, k, v, mask=None):
        if hasattr(F, "scaled_dot_product_attention"):
            q_pt = q.transpose(1, 2)
            k_pt = k.transpose(1, 2)
            v_pt = v.transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                q_pt, k_pt, v_pt,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False
            )

            # [b, n, h, d] -> [b, h, n, d]
            return attn_output.transpose(1, 2)
        return None

    def _vanilla_attention(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), -torch.finfo(attn.dtype).max)

        attn = F.softmax(attn, dim=-1)

        return torch.matmul(attn, v)

    def forward(self,
               x: torch.Tensor,
               context: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, sequence_length, dim = x.shape

        if batch_size % self.num_frames == 0:
            # Case: [b*f, n, c]
            actual_batch = batch_size // self.num_frames

            # Reshape to [b, f*n, c]
            x = x.reshape(actual_batch, self.num_frames * sequence_length, dim)

            # Process attention
            output = self._attention_forward(x, context, mask)

            # Reshape back to [b*f, n, c]
            output = output.reshape(batch_size, sequence_length, dim)

        else:
            # Case: [b, f*n, c]
            output = self._attention_forward(x, context, mask)

        return output

    def _attention_forward(self,
                         x: torch.Tensor,
                         context: Optional[torch.Tensor],
                         mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, sequence_length, dim = x.shape

        if context is None:
            context = x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.reshape(batch_size, sequence_length, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)

        if self.use_flash_attn:
            out = self._flash_attention(q, k, v, mask)
        else:
            out = self._vanilla_attention(q, k, v, mask)

        out = out.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)

        return self.to_out(out)


class InflatedSelfAttention(nn.Module):
    """
    Inflated self attention module for cubemap faces.
    Allows attention operation to span across all six faces.
    """

    def __init__(self,
                dim: int,
                heads: int = 8,
                dim_head: int = 64,
                dropout: float = 0.0,
                num_frames: int = 6):
        super().__init__()
        self.inflated_attn = InflatedCrossAttention(
            query_dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            num_frames=num_frames
        )

    def forward(self,
               x: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.inflated_attn(x, None, mask)


class InflatedAttentionBlock(nn.Module):
    """
    Inflated attention block for cubemap processing.
    Includes both self-attention and optional cross-attention.
    """

    def __init__(self,
                dim: int,
                context_dim: Optional[int] = None,
                num_heads: int = 8,
                head_dim: int = 64,
                dropout: float = 0.0,
                num_frames: int = 6,
                use_cross_attention: bool = True):
        super().__init__()

        self.use_cross_attention = use_cross_attention
        self.num_frames = num_frames

        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if use_cross_attention else None
        self.norm3 = nn.LayerNorm(dim)

        # Self-attention
        self.self_attn = InflatedSelfAttention(
            dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            num_frames=num_frames
        )

        # Cross-attention
        self.cross_attn = InflatedCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            num_frames=num_frames
        ) if use_cross_attention else None

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self,
               x: torch.Tensor,
               context: Optional[torch.Tensor] = None,
               self_mask: Optional[torch.Tensor] = None,
               cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Self-attention
        x = x + self.self_attn(self.norm1(x), self_mask)

        # Cross-attention
        if self.use_cross_attention and context is not None:
            x = x + self.cross_attn(self.norm2(x), context, cross_mask)

        # Feed-forward
        x = x + self.ff(self.norm3(x))

        return x


def convert_attention_to_inflated(
    module: nn.Module,
    num_frames: int = 6,
    inplace: bool = True
) -> nn.Module:
    """
    Convert regular attention layers to inflated attention.

    Args:
        module: PyTorch module to convert
        num_frames: Number of frames (6 for cubemap)
        inplace: Whether to modify the module in-place

    Returns:
        Module with inflated attention layers
    """
    if not inplace:
        module = module.copy()

    for name, child in module.named_children():
        convert_attention_to_inflated(child, num_frames, True)

        if not any(x in name.lower() for x in ["attn", "attention"]):
            continue

        if isinstance(child, nn.Module) and "self" in name.lower():
            try:
                if hasattr(child, "heads") and hasattr(child, "dim_head"):
                    heads = child.heads
                    dim_head = child.dim_head
                else:
                    heads = 8
                    dim_head = 64

                dim = child.to_q.in_features if hasattr(child, "to_q") else child.to_out[0].out_features

                inflated_self_attn = InflatedSelfAttention(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=0.0,
                    num_frames=num_frames
                )

                if hasattr(child, "to_q") and hasattr(inflated_self_attn.inflated_attn, "to_q"):
                    inflated_self_attn.inflated_attn.to_q.weight.data.copy_(child.to_q.weight.data)
                    inflated_self_attn.inflated_attn.to_k.weight.data.copy_(child.to_k.weight.data)
                    inflated_self_attn.inflated_attn.to_v.weight.data.copy_(child.to_v.weight.data)
                    inflated_self_attn.inflated_attn.to_out[0].weight.data.copy_(child.to_out[0].weight.data)
                    if hasattr(child.to_out[0], "bias") and child.to_out[0].bias is not None:
                        inflated_self_attn.inflated_attn.to_out[0].bias.data.copy_(child.to_out[0].bias.data)

                setattr(module, name, inflated_self_attn)
            except Exception as e:
                print(f"Failed to convert self-attention {name}: {e}")
        elif isinstance(child, nn.Module) and "cross" in name.lower():
            if hasattr(child, "heads") and hasattr(child, "dim_head"):
                heads = child.heads
                dim_head = child.dim_head
            else:
                heads = 8
                dim_head = 64

            query_dim = child.to_q.in_features if hasattr(child, "to_q") else child.to_out[0].out_features
            context_dim = child.to_k.in_features if hasattr(child, "to_k") else query_dim

            inflated_cross_attn = InflatedCrossAttention(
                query_dim=query_dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=0.0,
                num_frames=num_frames
            )

            if hasattr(child, "to_q") and hasattr(inflated_cross_attn, "to_q"):
                inflated_cross_attn.to_q.weight.data.copy_(child.to_q.weight.data)
                inflated_cross_attn.to_k.weight.data.copy_(child.to_k.weight.data)
                inflated_cross_attn.to_v.weight.data.copy_(child.to_v.weight.data)
                inflated_cross_attn.to_out[0].weight.data.copy_(child.to_out[0].weight.data)
                if hasattr(child.to_out[0], "bias") and child.to_out[0].bias is not None:
                    inflated_cross_attn.to_out[0].bias.data.copy_(child.to_out[0].bias.data)

            setattr(module, name, inflated_cross_attn)

    return module
