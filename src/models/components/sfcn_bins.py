#%%
import torch.nn as nn
import torch
from types import SimpleNamespace
# from coral_pytorch.layers import CoralLayer
# from src.models.components.resnetblock import ResNetBlock
# from src.models.components.preactresnetblock import PreActResNetBlock
from src.models.components.sfcnblock import SFCNBlock
# from sfcnblock import SFCNBlock
# resnet_blocks_by_name = {
#     "ResNetBlock": ResNetBlock,
#     "PreActResNetBlock": PreActResNetBlock
# }
#%%
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}
class SFCN_Bins(nn.Module):

    # def __init__(self, num_classes=1, num_blocks=[3,4,6,4,3], c_hidden=[8,16,32,64,128], act_fn_name="relu", num_input_maps=1, block_name="PreActResNetBlock", **kwargs):
    def __init__(self, num_classes=100, c_hidden=[32, 64, 128, 256, 256, 64], act_fn_name="relu", num_input_maps=1, **kwargs):
        """
        Inputs:
            num_classes - Number of classification outputs (10 for CIFAR10)
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
            act_fn_name - Name of the activation function to use, looked up in "act_fn_by_name"
            block_name - Name of the ResNet block, looked up in "resnet_blocks_by_name"
        """
        super().__init__()
        # assert block_name in resnet_blocks_by_name
        self.hparams = SimpleNamespace( num_input_maps=num_input_maps,
                                        num_classes=num_classes,
                                        c_hidden=c_hidden,
                                        # num_blocks=num_blocks,
                                        act_fn_name=act_fn_name,
                                        act_fn=act_fn_by_name[act_fn_name],
                                        # block_class=resnet_blocks_by_name[block_name]
                                        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden
        num_input_maps = self.hparams.num_input_maps

        # A first convolution on the original image to scale up the channel size
        self.input_net = SFCNBlock(c_in=num_input_maps, c_out=c_hidden[0], act_fn=self.hparams.act_fn, grouped_maps=num_input_maps)

        # if self.hparams.block_class == PreActResNetBlock: # => Don't apply non-linearity on output
        #     self.input_net = nn.Sequential(
        #         nn.Conv3d(num_input_maps, c_hidden[0], kernel_size=3, padding=1, bias=False)
        #     )
        # else:
        #     self.input_net = nn.Sequential(
        #         nn.Conv3d(num_input_maps, c_hidden[0], kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm3d(c_hidden[0]),
        #         self.hparams.act_fn()
        #     )

        # Creating the ResNet blocks
        blocks = []
        for hidden_idx in range(len(c_hidden)-2):
            # for bc in range(block_count):
                # subsample = (bc == 0 and block_idx > 0) # Subsample the first block of each group, except the very first one.
            blocks.append(
                SFCNBlock(c_in=c_hidden[hidden_idx],
                                            act_fn=self.hparams.act_fn,
                                            # subsample=subsample,
                                            c_out=c_hidden[hidden_idx+1],
                                            grouped_maps=num_input_maps)
            )
        self.blocks = nn.Sequential(*blocks)

        #
        self.penultimate_block = nn.Sequential(
                nn.Conv3d(c_hidden[-2], c_hidden[-1], padding=0, kernel_size=3, groups=num_input_maps),
                nn.BatchNorm3d(c_hidden[-1]),
                self.hparams.act_fn()
            )

        # Mapping to classification output
        self.output_net_age = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Dropout(0.5),
            nn.Conv3d(c_hidden[-1], self.hparams.num_classes, padding=0, kernel_size=1),
            # self.hparams.act_fn(),
            nn.Flatten(),
            # nn.Linear(c_hidden[-1], self.hparams.num_classes)
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_cat):
        # print(x.shape)
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.penultimate_block(x)
        x = self.output_net_age(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

# %%
# import torch
# input = torch.randn(8, 1, 91, 109, 91)
# model = SFCN_Bins()
# model(input,0).shape
# %%
