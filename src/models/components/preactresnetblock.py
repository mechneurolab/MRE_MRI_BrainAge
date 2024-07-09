import torch.nn as nn

class PreActResNetBlock(nn.Module):

    def __init__(self, c_in, act_fn, subsample=False, c_out=-1, grouped_maps=1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm3d(c_in),
            act_fn(),
            nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False, groups=grouped_maps),
            nn.BatchNorm3d(c_out),
            act_fn(),
            nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False, groups=grouped_maps)
        )

        # 1x1 convolution can apply non-linearity as well, but not strictly necessary
        self.downsample = nn.Sequential(
            nn.BatchNorm3d(c_in),
            act_fn(),
            nn.Conv3d(c_in, c_out, kernel_size=1, stride=2, bias=False, groups=grouped_maps)
        ) if subsample else None

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out