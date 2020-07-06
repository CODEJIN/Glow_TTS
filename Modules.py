import torch
import numpy as np
import yaml, logging, math

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class Duration_Predictor(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,        
        kernel_size,
        stacks,
        dropout_rate
        ):
        super(Duration_Predictor, self).__init__()
        self.layer_Dict = torch.nn.ModuleDict()
        self.stacks = stacks

        previous_channels = in_channels
        for index in range(stacks):
            self.layer_Dict['Conv_{}'.format(index)] = torch.nn.Sequential([
                torch.nn.Conv1d(
                    in_channels= previous_channels,
                    out_channels= out_channels,
                    kernel_size= kernel_size,
                    padding= (kernel_size - 1 // 2)
                    ),
                torch.nn.ReLU(inplace= True),
                torch.nn.LayerNorm(out_channels),
                torch.nn.Dropout(p= dropout_rate)
                ])
            previous_channels = out_channels

        self.layer_Dict['Projection'] = torch.nn.Conv1d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= 1
            )        

    def forward(self, x, x_mask):
        for index in range(self.stacks):
            x = self.layer_Dict['Conv_{}'.format(index)](x * x_mask)
        x = self.layer_Dict['Projection'](x * x_mask)

        return x * x_mask

class Text_Encoder(torch.nn.Module):
    def __init__(
        self,
        num_tokens,
        out_channels,
        hidden_channels,
        filter_channels,
        filter_channels_dp,
        num_heads,
        num_layers,
        kernel_size,
        dropout_rate,
        window_size= None,
        block_length= None,
        mean_only= False,
        use_prenet= False,
        gin_channels= 0
        ):
        super(Text_Encoder, self).__init__()
        
        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Embedding'] = torch.nn.Embedding(
            num_embeddings= num_tokens,
            embedding_dim=hidden_channels
            )
        torch.nn.init.normal_(
            tensor= self.layer_Dict['Embedding'].weight,
            mean= 0.0,
            std= hidden_channels ** -0.5
            )

        if use_prenet:
            self.layer_Dict['Prenet'] = ConvReLUNorm(hidden_channels, kernel_size= 5, stacks= 3, dropout_rate= 0.5)
        


class ConvReLUNorm(torch.nn.Module):
    def __init__(
        self,
        channels,   # In, hidden, and out channels must be same for mask and residual.
        kernel_size,
        stacks,
        dropout_rate
        ):
        super(ConvReLUNorm, self).__init__()
        self.layer_Dict = torch.nn.ModuleDict()
        self.stacks = stacks

        for index in range(stacks):
            self.layer_Dict['Conv_{}'.format(index)] = torch.nn.Sequential([
                torch.nn.Conv1d(
                    in_channels= channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    padding= (kernel_size - 1 // 2)
                    ),
                torch.nn.ReLU(inplace= True),
                torch.nn.LayerNorm(channels),
                torch.nn.Dropout(p= dropout_rate)
                ])

        self.layer_Dict['Projection'] = torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= 1
            )
        self.layer_Dict['Projection'].weight.data.zero_()
        self.layer_Dict['Projection'].bias.data.zero_()

    def forward(self, x, x_mask):
        residual = x
        for index in range(self.stacks):
            x = self.layer_Dict['Conv_{}'.format(index)](x * x_mask)
        x = self.layer_Dict['Projection'](x) + residual

        return x * x_mask
        