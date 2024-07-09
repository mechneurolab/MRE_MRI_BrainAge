#%%
import pyrootutils
pyrootutils.setup_root(search_from='/work/cniel/sw/BrainAge_torch/project/', indicator=".project-root", pythonpath=True)
#%%
import torch.nn as nn
from types import SimpleNamespace
import tltorch
# from coral_pytorch.layers import CoralLayer
from src.models.components.resnetblock import ResNetBlock
# from resnetblock import ResNetBlock
from src.models.components.preactresnetblock import PreActResNetBlock
# from preactresnetblock import PreActResNetBlock
from src.models.components.resnet3d import ResNet3D
# from resnet3d import ResNet3D

#%%
resnet_blocks_by_name = {
    "ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock
}

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}
class ResNet3DFT2(nn.Module):

    def __init__(self, path_1=None, path_2=None, num_classes=1, num_blocks=[3,4,6,4,3], c_hidden=[8,16,32,64,128], act_fn_name="relu", num_input_maps=1, block_name="PreActResNetBlock", **kwargs):
        """
        Inputs:
            num_classes - Number of classification outputs (10 for CIFAR10)
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
            act_fn_name - Name of the activation function to use, looked up in "act_fn_by_name"
            block_name - Name of the ResNet block, looked up in "resnet_blocks_by_name"
        """
        super().__init__()
        assert block_name in resnet_blocks_by_name
        self.hparams = SimpleNamespace( model=ResNet3D,
                                        path_1=path_1,
                                        path_2=path_2,
                                        num_input_maps=num_input_maps,
                                        # num_classes=num_classes,
                                        # c_hidden=c_hidden,
                                        # num_blocks=num_blocks,
                                        # act_fn_name=act_fn_name,
                                        # act_fn=act_fn_by_name[act_fn_name],
                                        # block_class=resnet_blocks_by_name[block_name]
                                        )
        self._create_network()
        # self._init_params()

    def _create_network(self):
        path_1 = '/work/cniel/sw/BrainAge_torch/project/logs/train/multiruns/2023-11-30_16-01-08/48/checkpoints/epoch_074.ckpt'
        path_2 = '/work/cniel/sw/BrainAge_torch/project/logs/train/multiruns/2023-12-02_09-32-57/67/checkpoints/epoch_081.ckpt'
        model = self.hparams.model
        # model_2 = self.hparams.model
        num_input_maps = self.hparams.num_input_maps
        self.trained_model_1 = model().load_state_dict(torch.load(path_1)['state_dict'])
        self.trained_model_2 = model().load_state_dict(torch.load(path_2)['state_dict'])

        # A first convolution on the original image to scale up the channel size
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
        # self.output_net_age = nn.Sequential(
        #     nn.AdaptiveAvgPool3d((1,1,1)),
        #     nn.Flatten(),
        #     nn.Linear(c_hidden[-1], self.hparams.num_classes)
        # )
        # self.output_net_age = tltorch.TRL(input_shape=(c_hidden[-1], 6, 7, 4), output_shape=(1,), rank=(int(c_hidden[-1]*0.5),3,4,2,1), factorization='tucker' )

    # def _init_params(self):
    #     # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
    #     # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x, x_cat):
        x_1 = self.trained_model_1(x,0)
        x_2 = self.trained_model_2(x,0)
        # print(x.shape)
        # x_age = self.output_net_age(x)
        return x_1,x_2

#%%
model = ResNet3DFT2()
#%%
import torch

input_shape = (1, 91, 109, 55)
# output_shape = (6, 2)
batch_size = 8

device = 'cpu'

x = torch.randn((batch_size,) + input_shape,
                dtype=torch.float32, device=device)
#%%
model(x,0)
# %%
# input_shape = (128, 6, 7, 4)
# output_shape = (1)
# batch_size = 2

# device = 'cpu'

# x = torch.randn((batch_size,) + input_shape,
#                 dtype=torch.float32, device=device)
# trl = tltorch.TRL(input_shape, output_shape, rank='same')
# result = trl(x)
# %%
