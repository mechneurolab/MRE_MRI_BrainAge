import torch.nn as nn
from types import SimpleNamespace
# from coral_pytorch.layers import CoralLayer
# from src.models.components.resnetblock import ResNetBlock
# from src.models.components.preactresnetblock import PreActResNetBlock

# resnet_blocks_by_name = {
#     "ResNetBlock": ResNetBlock,
#     "PreActResNetBlock": PreActResNetBlock
# }

# act_fn_by_name = {
#     "tanh": nn.Tanh,
#     "relu": nn.ReLU,
#     "leakyrelu": nn.LeakyReLU,
#     "gelu": nn.GELU
# }
class OutputHead(nn.Module):

    def __init__(self, num_input_activations=128, num_classes=1, **kwargs):
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
        self.hparams = SimpleNamespace( num_input_activations=num_input_activations,
                                        num_classes=num_classes,
                                        )
        self._create_network()
        # self._init_params()

    def _create_network(self):
        # c_hidden = self.hparams.c_hidden
        # num_input_maps = self.hparams.num_input_maps

        # # A first convolution on the original image to scale up the channel size
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

        # # Creating the ResNet blocks
        # blocks = []
        # for block_idx, block_count in enumerate(self.hparams.num_blocks):
        #     for bc in range(block_count):
        #         subsample = (bc == 0 and block_idx > 0) # Subsample the first block of each group, except the very first one.
        #         blocks.append(
        #             self.hparams.block_class(c_in=c_hidden[block_idx if not subsample else (block_idx-1)],
        #                                      act_fn=self.hparams.act_fn,
        #                                      subsample=subsample,
        #                                      c_out=c_hidden[block_idx])
        #         )
        # self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        self.output_net_age = nn.Sequential(
            nn.Linear(self.hparams.num_input_activations, self.hparams.num_classes)
        )

    # def _init_params(self):
    #     # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
    #     # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.input_net(x)
        # x = self.blocks(x)
        x_age = self.output_net_age(x)
        return x_age