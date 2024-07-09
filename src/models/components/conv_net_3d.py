from torch import nn
import torch
from torch.nn import functional as F

class ConvNet3D(nn.Module):
    def __init__(
        self,
        num_input_maps: int = 1,
        cat_input_type: str = 'sex_study',
        arc_type: str = '1',
    ):
        super().__init__()
        self.num_input_maps = num_input_maps
        self.cat_input_type = cat_input_type
        self.arc_type = arc_type
        self.layers_in = self.in_out_layers()
        self._create_network()
        self._init_params()
    
    def _create_network(self):
        n_chan = 8
        block_parameters = zip(range(4), [1,1,1,1], [[0,1,0],[1,0,0],[0,0,1],[0,1,0]])
        self.initial_block = self.make_block(self.num_input_maps, n_chan, 0, 1, 1)
        self.sequential_blocks = nn.ModuleList([self.make_block(2**(i)*n_chan, n_chan, i+1, j, k) for i,j,k in block_parameters])
        self.flatten = nn.Flatten()
        if self.arc_type=='1' and self.cat_input_type!='none':
            self.dense_layer_1 = nn.Linear(self.layers_in[0], self.num_input_maps*91*109*55)
        self.dense_layer_2 = nn.Linear(self.layers_in[1], 640) #3072
        self.dense_layer_3 = nn.Linear(self.layers_in[2], 100) #640
        self.dense_layer_4 = nn.Linear(self.layers_in[3], 1) #100
    
    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(
                    m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_block(self, in_chan, out_chan, n_layer, padding_conv, padding_maxpool):
        # padding = [1,1]
        conv_block = nn.Sequential(
            nn.ConvTranspose3d(in_chan, 2**n_layer*out_chan, kernel_size=(3,3,3), stride=(1,1,1), padding=padding_conv),
            nn.ReLU(),
            nn.ConvTranspose3d(2**n_layer*out_chan, 2**n_layer*out_chan, kernel_size=(3,3,3), stride=(1,1,1), padding=padding_conv),
            nn.BatchNorm3d(2**n_layer*out_chan),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=padding_maxpool)
            )
        return conv_block

    def in_out_layers(self):
        layer_1_in = 1
        layer_2_in = 3072 
        layer_3_in = 640 
        layer_4_in = 100
        if self.cat_input_type=='sex':
            if self.arc_type == '1':
                layer_1_in = 2
            elif self.arc_type == '2':
                layer_2_in = 3072+2
            elif self.arc_type == '3':
                layer_3_in = 640+2
            elif self.arc_type == '4':
                layer_4_in = 100+2
            else:
                raise ValueError('Invalid architecture option')
        elif self.cat_input_type=='study':
            if self.arc_type == '1':
                layer_1_in = 8
            elif self.arc_type == '2':
                layer_2_in = 3072+8
            elif self.arc_type == '3':
                layer_3_in = 640+8
            elif self.arc_type == '4':
                layer_4_in = 100+8
            else:
                raise ValueError('Invalid architecture option')
        elif self.cat_input_type=='sex_study':
            if self.arc_type == '1':
                layer_1_in = 10
            elif self.arc_type == '2':
                layer_2_in = 3072+10
            elif self.arc_type == '3':
                layer_3_in = 640+10
            elif self.arc_type == '4':
                layer_4_in = 100+10
            else:
                raise ValueError('Invalid architecture option')
        elif self.cat_input_type=='none' and self.arc_type in ('1','2','3','4'):
            print('architecture is not modified')
        else:
            raise ValueError('Invalid categorical variable or achitecture option')
        return layer_1_in, layer_2_in, layer_3_in, layer_4_in

    def slice_categorical(self, x_cat):
        if self.cat_input_type=='sex':
            x_cat = x_cat[:,:2]
        elif self.cat_input_type=='study':
            x_cat = x_cat[:,2:]
        elif self.cat_input_type=='sex_study':
            x_cat = x_cat
        else:
            raise ValueError('Invalid categorical variable option')
        return x_cat
    
    def forward(self, x, x_cat):
        # batch_size, channels, width, height, slices = x.size()
        # Max pooling over a (2, 2) window
        if self.cat_input_type!='none':
            x_cat = self.slice_categorical(x_cat)
        if self.arc_type=='1' and self.cat_input_type!='none':
            y = self.dense_layer_1(x_cat)
            y = y.view(-1,int(self.num_input_maps),91,109,55)
            x = torch.add(x,y)
        x = self.initial_block(x)
        for i in range(4):
            x = self.sequential_blocks[i](x)
        x = self.flatten(x)
        if self.arc_type=='2' and self.cat_input_type!='none': 
            x = torch.cat([x, x_cat], axis=-1)    
        x = self.dense_layer_2(x)
        x = F.relu(x)
        if self.arc_type=='3'and self.cat_input_type!='none':
            x = torch.cat([x, x_cat], axis=-1)
        x = self.dense_layer_3(x)
        x = F.relu(x)
        if self.arc_type=='4'and self.cat_input_type!='none':
            x = torch.cat([x, x_cat], axis=-1)
        x = self.dense_layer_4(x)
        return x


if __name__ == "__main__":
    _ = ConvNet3D()
