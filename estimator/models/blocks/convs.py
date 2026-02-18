import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class DoubleConv(nn.Module):
    """(convolution => [GELU]) * 2 # used in decoder"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU())

    def forward(self, x):
        return self.double_conv(x)

class DoubleResConv(nn.Module):
    """(res convolution => [GELU]) * 2 # used in decoder"""

    def __init__(self, feature, activation):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False),
            activation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False),
            activation)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x + out

class SingleConvCNNLN(nn.Module):
    """(convolution => [LayerNorm] => GELU) * 1 # used in encoder (for input features)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            LayerNorm([out_channels]),
            nn.GELU())

    def forward(self, x):
        return self.single_conv(x)
    
class SingleConv(nn.Module):
    """(convolution => GELU) * 1 # used in encoder (for input features)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU())

    def forward(self, x):
        return self.single_conv(x)

class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x
