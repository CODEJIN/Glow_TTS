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

    def forward(self, x):
        pass

        





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



class Activation_Norm(torch.nn.Module):
    def __init__(self):
        super(Activation_Norm, self).__init__()

    def forward(self, x):
        pass

class Invertible_1x1_Conv(torch.nn.Module):
    def __init__(self):
        super(Invertible_1x1_Conv, self).__init__()

    def forward(self, x):
        pass

class Affine_Coupling_Layer(torch.nn.Module):
    def __init__(self):
        super(Affine_Coupling_Layer, self).__init__()

    def forward(self, x):
        pass
    


if __name__ == "__main__":
    encoder = Encoder()

    x = torch.LongTensor([
        [6,3,4,6,1,3,26,5,7,3,14,6,3,3,6,22,3],
        [7,3,2,16,1,13,26,25,7,3,14,6,23,3,0,0,0],
        ])
    lengths = torch.LongTensor([15, 12])

    encoder(x, lengths)