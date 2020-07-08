import torch
import numpy as np
import yaml, logging, math

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Embedding'] = torch.nn.Embedding(
            num_embeddings= hp_Dict['Encoder']['Embedding_Tokens'],
            embedding_dim= hp_Dict['Encoder']['Channels'],
            )
        self.layer_Dict['Prenet'] = Prenet(hp_Dict['Encoder']['Prenet']['Stacks'])
        self.layer_Dict['Transformer'] = Transformer(hp_Dict['Encoder']['Transformer']['Stacks'])

        self.layer_Dict['Project'] = torch.nn.Conv1d(
            in_channels= hp_Dict['Encoder']['Channels'],
            out_channels= hp_Dict['Sound']['Mel_Dim'] * 2,
            kernel_size= 1
            )
        self.layer_Dict['Duration_Predictor'] = Duration_Predictor()

    def forward(self, x, lengths):
        '''
        x: [Batch, Time]
        lengths: [Batch]
        '''
        x = self.layer_Dict['Embedding'](x).transpose(2, 1) * math.sqrt(hp_Dict['Encoder']['Channels']) # [Batch, Dim, Time]
        
        mask = torch.arange(x.size(2))[None, :] < lengths[:, None]  # [Batch, Time]
        mask = mask.unsqueeze(1)    # [Batch, 1, Time]
        mask = mask.to(x.dtype) # [Batch, 1, Time]

        x = self.layer_Dict['Prenet'](x, mask)
        x = self.layer_Dict['Transformer'](x, mask)

        mean, log_S = torch.split(
            self.layer_Dict['Project'](x) * mask,
            [hp_Dict['Sound']['Mel_Dim'], hp_Dict['Sound']['Mel_Dim']],
            dim= 1
            )
        log_W = self.layer_Dict['Duration_Predictor'](x.detach(), mask)

        return mean, log_S, log_W, mask

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Squeeze'] = Squeeze(num_squeeze= hp_Dict['Decoder']['Num_Squeeze'])
        self.layer_Dict['Unsqueeze'] = Unsqueeze(num_squeeze= hp_Dict['Decoder']['Num_Squeeze'])

        for index in range(hp_Dict['Decoder']['Stack']):
            self.layer_Dict['Flow_{}']

        self.flows = torch.nn.ModuleList()

    def forward(self, x, mask, reverse= False):
        x, mask = self.layer_Dict['Squeeze'](x, mask)

        logdets = []
        for flow in reversed(self.flows) if reverse else self.flows:




            
        pass

class         





class Prenet(torch.nn.Module):
    def __init__(self, stacks):
        super(Prenet, self).__init__()
        self.stacks = stacks

        self.layer_Dict = torch.nn.ModuleDict()
        for index in range(stacks):            
            self.layer_Dict['CLRD_{}'.format(index)] = CLRD()

        self.layer_Dict['Conv1x1'] = torch.nn.Conv1d(
            in_channels= hp_Dict['Encoder']['Channels'],
            out_channels= hp_Dict['Encoder']['Channels'],
            kernel_size= 1
            )

    def forward(self, x, mask):
        residual = x
        for index in range(self.stacks):
            x = self.layer_Dict['CLRD_{}'.format(index)](x, mask)
        x = self.layer_Dict['Conv1x1'](x) + residual

        return x * mask

class CLRD(torch.nn.Module):
    def __init__(self):
        super(CLRD, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Conv'] = torch.nn.Conv1d(
            in_channels= hp_Dict['Encoder']['Channels'],
            out_channels= hp_Dict['Encoder']['Channels'],
            kernel_size= hp_Dict['Encoder']['Prenet']['Kernel_Size'],
            padding= (hp_Dict['Encoder']['Prenet']['Kernel_Size'] - 1) // 2
            )
        self.layer_Dict['LayerNorm'] = torch.nn.LayerNorm(
            hp_Dict['Encoder']['Channels']
            )
        self.layer_Dict['ReLU'] = torch.nn.ReLU(
            inplace= True
            )
        self.layer_Dict['Dropout'] = torch.nn.Dropout(
            p= hp_Dict['Encoder']['Prenet']['Dropout_Rate']
            )

    def forward(self, x, mask):
        x = self.layer_Dict['Conv'](x * mask)   # [Batch, Dim, Time]
        x = self.layer_Dict['LayerNorm'](x.transpose(2, 1)).transpose(2, 1)
        x = self.layer_Dict['ReLU'](x)
        x = self.layer_Dict['Dropout'](x)

        return x


class Transformer(torch.nn.Module):
    def __init__(self, stacks):
        super(Transformer, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.stacks = stacks
        
        self.layer_Dict = torch.nn.ModuleDict()
        for index in range(stacks):
            self.layer_Dict['ANCRDCN_{}'.format(index)] = ANCRDCN()

    def forward(self, x, mask):
        for index in range(self.stacks):
            x = self.layer_Dict['ANCRDCN_{}'.format(index)](x, mask)

        return x * mask

class ANCRDCN(torch.nn.Module):
    def __init__(self):
        super(ANCRDCN, self).__init__()
        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Attention'] = torch.nn.MultiheadAttention(     # Require [Time, Batch, Dim]
                embed_dim= hp_Dict['Encoder']['Channels'],
                num_heads= hp_Dict['Encoder']['Transformer']['Attention_Head']
                )
        self.layer_Dict['LayerNorm_0'] = torch.nn.LayerNorm(    # This normalize last dim...
            normalized_shape= hp_Dict['Encoder']['Channels']
            )   
        
        self.layer_Dict['Conv_0'] = torch.nn.Conv1d(
            in_channels= hp_Dict['Encoder']['Channels'],
            out_channels= hp_Dict['Encoder']['Transformer']['Conv']['Calc_Channels'],
            kernel_size= hp_Dict['Encoder']['Transformer']['Conv']['Kernel_Size'],
            padding= (hp_Dict['Encoder']['Transformer']['Conv']['Kernel_Size'] - 1) // 2
            )
        self.layer_Dict['Conv_1'] = torch.nn.Conv1d(
            in_channels= hp_Dict['Encoder']['Transformer']['Conv']['Calc_Channels'],
            out_channels= hp_Dict['Encoder']['Channels'],
            kernel_size= hp_Dict['Encoder']['Transformer']['Conv']['Kernel_Size'],
            padding= (hp_Dict['Encoder']['Transformer']['Conv']['Kernel_Size'] - 1) // 2
            )
        
        self.layer_Dict['LayerNorm_1'] = torch.nn.LayerNorm(    # This normalize last dim...
            normalized_shape= hp_Dict['Encoder']['Channels']
            )
        
        self.layer_Dict['ReLU'] = torch.nn.ReLU(
            inplace= True
            )
        self.layer_Dict['Dropout'] = torch.nn.Dropout(
            p= hp_Dict['Encoder']['Transformer']['Dropout_Rate']
            )

    def forward(self, x, mask):
        x = x.permute(2, 0, 1)  # [Time, Batch, Dim]
        residual = x
        x = self.layer_Dict['Attention'](  # [Time, Batch, Dim]
            query= x,
            key= x,
            value= x,
            attn_mask= torch.repeat_interleave(     # Ver1.4 does not support this.
                input= (mask * mask.transpose(2, 1)),
                repeats= hp_Dict['Encoder']['Transformer']['Attention_Head'],
                dim= 0
                )
            )[0]
        x = self.layer_Dict['Dropout'](x)
        x = self.layer_Dict['LayerNorm_0'](x + residual).permute(1, 2, 0) # [Batch, Dim, Time]

        residual = x
        x = self.layer_Dict['Conv_0'](x * mask)
        x = self.layer_Dict['ReLU'](x)
        x = self.layer_Dict['Dropout'](x)
        x = self.layer_Dict['Conv_1'](x * mask)        
        x = self.layer_Dict['Dropout'](x)

        x = self.layer_Dict['LayerNorm_1']((x * mask + residual).transpose(2, 1)).transpose(2, 1)

        return x


class Duration_Predictor(torch.nn.Module):
    def __init__(self):
        super(Duration_Predictor, self).__init__()
        self.layer_Dict = torch.nn.ModuleDict()

        previous_channels = hp_Dict['Encoder']['Channels']
        for index in range(hp_Dict['Encoder']['Duration_Predictor']['Stacks']):
            self.layer_Dict['CRND_{}'.format(index)] = CRND(in_channels= previous_channels)
            previous_channels = hp_Dict['Encoder']['Duration_Predictor']['Channels']

        self.layer_Dict['Projection'] = torch.nn.Conv1d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= 1
            )

    def forward(self, x, x_mask):
        for index in range(hp_Dict['Encoder']['Duration_Predictor']['Stacks']):
            x = self.layer_Dict['CRND_{}'.format(index)](x, x_mask)
        x = self.layer_Dict['Projection'](x * x_mask)

        return x * x_mask

class CRND(torch.nn.Module):
    def __init__(self, in_channels):
        super(CRND, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Conv'] = torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= hp_Dict['Encoder']['Duration_Predictor']['Channels'],
            kernel_size= hp_Dict['Encoder']['Duration_Predictor']['Kernel_Size'],
            padding= (hp_Dict['Encoder']['Duration_Predictor']['Kernel_Size'] - 1) // 2
            )
        self.layer_Dict['ReLU'] = torch.nn.ReLU(
            inplace= True
            )
        self.layer_Dict['LayerNorm'] = torch.nn.LayerNorm(
            hp_Dict['Encoder']['Duration_Predictor']['Channels']
            )
        self.layer_Dict['Dropout'] = torch.nn.Dropout(
            p= hp_Dict['Encoder']['Prenet']['Dropout_Rate']
            )

    def forward(self, x, mask):
        x = self.layer_Dict['Conv'](x * mask)   # [Batch, Dim, Time]
        x = self.layer_Dict['ReLU'](x)
        x = self.layer_Dict['LayerNorm'](x.transpose(2, 1)).transpose(2, 1)        
        x = self.layer_Dict['Dropout'](x)

        return x



class AIA(torch.nn.Module):
    def __init__(self):
        super(AIA, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(Activation_Norm())
        self.layers.append(Invertible_1x1_Conv(
            channels=,
            num_split=
            ))
        self.layers.append(Affine_Coupling_Layer())

    def forward(self, x, mask, reverse= False):        
        for layer in (reversed(self.layers) if reverse else self.layers):
            x = layer(x, mask, reverse= reverse)
        
        return x, logdet

class Activation_Norm(torch.nn.Module):
    def __init__(self):
        super(Activation_Norm, self).__init__()
        self.initialized = False

        self.logs = torch.nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, mask, reverse= False):
        if mask is None:
            mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype= x.dtype)
        if not self.initialized:
            self.initialize(x, mask)
            self.initialized = True
        
        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * mask
            logdet = None
        else:
            z = (self.bias + torch.exp(self.logs) * x) * mask
            logedt = torch.sum(self.logs) * torch.sum(mask, [1, 2])

        return z, logdet

    def initialize(self, x, mask):
        with torch.no_grad():
            denorm = torch.sum(mask, [0, 2])
            mean = torch.sum(x * mask, [0, 2]) / denorm
            square = torch.sum(x * x * mask, [0, 2]) / denorm
            variance = squre - (mean ** 2)
            logs = 0.5 * torch.log(v + 1e-7)

            self.logs.data.copy_(
                (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)
                )
            self.bias.data.copy_(
                (-mean * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
                )

class Invertible_1x1_Conv(torch.nn.Module):
    def __init__(self, channels, num_split= 4):
        super(Invertible_1x1_Conv, self).__init__()
        assert num_split % 2 == 0

        self.channels = channels
        self.num_split = num_split

        weight = torch.qr(torch.FloatTensor(self.num_split, self.num_split).normal_())[0]
        if torch.det(weight) < 0:
            weight[:, 0] = -weight[:, 0]

        self.weight = torch.nn.Parameter(weight)

    def forwrad(self, x, mask= None, reverse= False):
        batch, channels, time = x.size()
        assert channels % self.num_split == 0

        if mask is None:
            mask = 1
            length = torch.ones((batch,), device=x.device, dtype= x.dtype) * time
        else:
            length = torch.sum(mask, [1, 2])

        x = x.view(batch, 2, channels // self.num_split, self.num_split // 2, t)    # [Batch, 2, Dim/split, split/2, Time]
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(batch, self.num_split, channels // self.num_split, t)   # [Batch, 2, split/2, Dim/split, Time] -> [Batch, split, Dim/split, Time]

        if reverse:
            weight = torch.inverse(self.weight).to(dtype= self.weight.dtype)
            print(torch.inverse(self.weight))
            print(torch.inverse(self.weight.float()))
            #Check the reason why float() is used.
            assert False
            logdet = None
        else:
            weight = self.weight
            logdet = torch.logdet(self.weight) * (c / self.num_split) * length
        
        z = torch.nn.functional.conv2d(
            input= x,
            weight= weight.unsqueeze(-1).unsqueeze(-1)            
            )
        z = z.view(batch, 2, self.num_split // 2, channels // self.num_split, time) # [Batch, 2, Split/2, Dim/Split, Time]
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(batch, channels, time) * mask # [Batch, 2, Dim/Split, Split/2, Time] -> [Batch, Dim, Time]

        return z, logdet





    def forward(self, x):
        pass

class Affine_Coupling_Layer(torch.nn.Module):
    def __init__(self):
        super(Affine_Coupling_Layer, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Start'] = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels=,
            out_channels=,
            kernel_size= 1,            
            ))
        self.layer_Dict['WaveNet']
        self.layer_Dict['End'] = torch.nn.Conv1d(
            in_channels=,
            out_channels=,
            kernel_size= 1,            
            )
        self.layer_Dict['End'].weight.data.zeros_()
        self.layer_Dict['End'].bias.data.zeros_()

    def forward(self, x, mask, reverse= False):
        batch, channels, time = x.size()
        if mask is None:
            mask = 1
        
        x_0, x_1 = torch.split(
            tensor= x,
            split_size_or_sections= [self.in_channels // 2] * 2,
            dim= 1
            )
        
        x = self.layer_Dict['Start'](x_0) * mask
        x = self.layer_Dict['WaveNet'](x, mask)
        out = self.layer_Dict['End'](x)

        z_0 = x_0
        mean, logs = torch.split(
            tensor= outs,
            split_size_or_sections= [self.in_channels // 2] * 2,
            dim= 1
            )

        if reverse:
            z_1 = (x_1 - mean) * torch.exp(-logs) * mask
            logdet = None
        else:
            z_1 = (mean + torch.exp(logs) * x_1) * mask
            logdet = torch.sum(logs * mask, [1, 2])

        z = torch.cat([z_0, z_1], 1)

        return z, logdet

class WaveNet()


class Squeeze(torch.nn.Module):
    def __init__(self, num_squeeze= 2):
        super(Squeeze, self).__init__()
        self.num_Squeeze = num_squeeze

    def forward(self, x, mask):
        batch, channels, time = x.size()
        time = (time // self.num_Squeeze) * self.num_Squeeze
        x = [:, :, :time]
        x = x.view(batch, channels, time // self.num_Squeeze, self.num_Squeeze)
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, channels * self.num_Squeeze, time // self.num_Squeeze)

        if not mask is None:
            mask = mask[:, :, self.num_Squeeze - 1::self.num_Squeeze]
        else:
            mask = torch.ones(batch, 1, time // self.num_Squeeze).to(device= x.device, dtype= x.dtype)

        return x * mask, mask

class Unsqueeze(torch.nn.Module):
    def __init__(self, num_squeeze= 2):
        super(Squeeze, self).__init__()
        self.num_Squeeze = num_squeeze

    def forward(self, x, mask):
        batch, channels, time = x.size()
        x = x.view(batch, self.num_Squeeze,  hannels // self.num_Squeeze, time)
        x = x.permute(0, 2, 3, 1).contiguous().view(batch, channels // self.num_Squeeze, time * self.num_Squeeze)

        if not mask is None:
            mask = mask.unsqueeze(-1).repeat(1,1,1,self.num_Squeeze).view(batch, 1, time * self.num_Squeeze)
        else:
            mask = torch.ones(batch, 1, time * self.num_Squeeze).to(device= x.device, dtype= x.dtype)

        return x * mask, mask




    
        

if __name__ == "__main__":
    encoder = Encoder()

    x = torch.LongTensor([
        [6,3,4,6,1,3,26,5,7,3,14,6,3,3,6,22,3],
        [7,3,2,16,1,13,26,25,7,3,14,6,23,3,0,0,0],
        ])
    lengths = torch.LongTensor([15, 12])

    encoder(x, lengths)