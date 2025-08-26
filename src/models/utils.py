import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


def get_attention_mask(features, lengths):
    B, T, D = features.shape
    device = features.device
    # Create an index tensor of shape (T,) and compare it with lengths (after unsqueezing)
    attention_mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) < lengths.unsqueeze(1)
    return attention_mask

def swish(x):
    return x * torch.sigmoid(x)

# ------------------ Common 1D Modules ------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int]] = 3,
                 causal: bool = False, stride: int = 1, bias: bool = True):
        super().__init__()
        self.causal = causal
        if causal:
            # Pad only on the left.
            self.padding = kernel_size - 1 if isinstance(kernel_size, int) else kernel_size[0] - 1
        else:
            self.padding = kernel_size // 2 if isinstance(kernel_size, int) else kernel_size[0] // 2
        # Set padding=0 in the conv because we apply it manually.
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=bias, padding=0)

    def forward(self, x):
        # x: (B, C, T)
        if self.causal:
            x = F.pad(x, (self.padding, 0))
        else:
            x = F.pad(x, (self.padding, self.padding))
        return self.conv(x)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, use_conv_shortcut: bool = False):
        super().__init__()
        num_groups_in = 8 if in_channels >= 8 else 1
        num_groups_out = 8 if out_channels >= 8 else 1
        self.norm1 = nn.GroupNorm(num_groups_in, in_channels, eps=1e-6)
        self.conv1 = ConvBlock1D(in_channels, out_channels, kernel_size=kernel_size, causal=True, bias=False)
        self.norm2 = nn.GroupNorm(num_groups_out, out_channels, eps=1e-6)
        self.conv2 = ConvBlock1D(out_channels, out_channels, kernel_size=kernel_size, causal=True, bias=False)
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = ConvBlock1D(in_channels, out_channels, kernel_size=3, causal=True, bias=False)
            else:
                self.shortcut = ConvBlock1D(in_channels, out_channels, kernel_size=1, causal=True, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = swish(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = swish(out)
        out = self.conv2(out)
        if not isinstance(self.shortcut, nn.Identity):
            residual = self.shortcut(residual)
        return out + residual
    

# ------------------ Speech Encoder (1D) ------------------
class SpeechEncoder1D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, 
                 num_res_blocks: int, channel_mults: Tuple[int, ...], 
                 d_model: int):
        """
        Args:
            in_channels: Input feature dimension (e.g., 120).
            base_channels: Initial number of channels (e.g., 128).
            num_res_blocks: Number of residual blocks per stage.
            channel_mults: Tuple of multipliers, e.g., (1, 2) so channels go from 128 to 256.
            d_model: Final projection dimension (e.g., 1024).
        """
        super().__init__()
        self.linear_in = nn.Linear(in_channels, base_channels)
        self.conv_in = ConvBlock1D(base_channels, base_channels, kernel_size=3, causal=True, bias=False)
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        for mult in channel_mults:
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.blocks.append(ResBlock1D(current_channels, out_channels))
                current_channels = out_channels
        self.linear_out = nn.Linear(current_channels, d_model)

    def forward(self, x):
        # x: (B, T, in_channels)
        x = self.linear_in(x)            # (B, T, base_channels)
        x = x.transpose(1, 2)            # (B, base_channels, T)
        x = self.conv_in(x)              # (B, base_channels, T)

        for block in self.blocks:
            x = block(x)                 # (B, new_channels, T)

        x = x.transpose(1, 2)            # (B, T, new_channels)
        x = self.linear_out(x)           # (B, T, d_model)
        return x
    

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, bias: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=bias, dtype=dtype
        )
    def forward(self, x):
        # x: (B, C, H, W)
        return self.conv(x)
    

# Example 2D Residual Block
class ResBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_conv_shortcut: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.norm1 = nn.GroupNorm(8 if in_channels >= 8 else 1, in_channels, eps=1e-6, dtype=dtype)
        self.conv1 = ConvBlock2D(in_channels, out_channels, kernel_size=3, stride=1, bias=False, dtype=dtype)
        self.norm2 = nn.GroupNorm(8 if out_channels >= 8 else 1, out_channels, eps=1e-6, dtype=dtype)
        self.conv2 = ConvBlock2D(out_channels, out_channels, kernel_size=3, stride=1, bias=False, dtype=dtype)
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = ConvBlock2D(in_channels, out_channels, kernel_size=3, stride=1, bias=False, dtype=dtype)
            else:
                self.shortcut = ConvBlock2D(in_channels, out_channels, kernel_size=1, stride=1, bias=False, dtype=dtype)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        if not isinstance(self.shortcut, nn.Identity):
            residual = self.shortcut(residual)
        return out + residual
    

    
# ---------------- 2D Visual Encoder ----------------
class VisualEncoder2D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, 
                 num_res_blocks: int, channel_mults: Tuple[float, ...], 
                 d_model: int, dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            in_channels: Dimensionality of the visual feature per frame.
            base_channels: Initial number of channels for the visual encoder (e.g., 256 or 4096).
            num_res_blocks: Number of residual blocks per stage.
            channel_divs: Factors to reduce channels; for example, (2, 4, 8) will gradually reduce 
                          the number of channels by division.
            d_model: Final output dimension per frame (e.g., 1024).
        """
        super().__init__()
        # Instead of a linear layer (as in the 1D version), we use a 1x1 conv to map in_channels to base_channels.
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False, dtype=dtype)

        self.blocks = nn.ModuleList()
        current_channels = base_channels
        # Here channel_divs are expected to be floats (or ints) that reduce the channel count.
        for mult in channel_mults:
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.blocks.append(ResBlock2D(current_channels, out_channels, dtype=dtype))
                current_channels = out_channels
            # additional spatial downsampling
            self.blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.proj = nn.Linear(current_channels * 4 * 4, d_model, dtype=dtype)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B * T, in_channels, H, W)
        Returns:
            Tensor of shape (B, T, d_model)
        """

        x = self.conv_in(x)         # (B*T, base_channels, H, W)
        for block in self.blocks:
            x = block(x)            # Process through each ResBlock2D.
        x = x.view(x.size(0), -1)
        x = self.proj(x)        # (B*T, d_model, H, W)
        return x
    


# ------------------ Visual Encoder as 1D ------------------
class VisualEncoder1D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, 
                 num_res_blocks: int, channel_divs: Tuple[int, ...], 
                 d_model: int):
        """
        Args:
            in_channels: Dimensionality of the visual feature per frame (from VIT).
            base_channels: Base number of channels for the visual encoder (e.g., 256).
            num_res_blocks: Number of residual blocks per stage.
            channel_mults: Multiplicative factors to gradually adjust channels.
            d_model: Final output dimension per frame (e.g., 1024).
        """
        super().__init__()
        self.linear_in = nn.Linear(in_channels, base_channels)
        self.conv_in = ConvBlock1D(base_channels, base_channels, kernel_size=5, causal=True, bias=False)
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        for mult in channel_divs:
            out_channels = int(base_channels // mult)
            for _ in range(num_res_blocks):
                self.blocks.append(ResBlock1D(current_channels, out_channels))
                current_channels = out_channels
        self.linear_out = nn.Linear(current_channels, d_model)

    def forward(self, x):
        # x: (B, T, in_channels)
        x = self.linear_in(x)            # (B, T, base_channels)
        x = x.transpose(1, 2)            # (B, base_channels, T)
        x = self.conv_in(x)              # (B, base_channels, T)
        x = x.transpose(1, 2)            # (B, T, base_channels)
        for block in self.blocks:
            x = x.transpose(1, 2)
            x = block(x)
            x = x.transpose(1, 2)
        x = self.linear_out(x)           # (B, T, d_model)
        return x
    

