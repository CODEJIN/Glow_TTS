import torch
import numpy as np
import yaml, logging, math

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class GlowTTS(torch.nn.Module):
    def __init__(self):
        super(GlowTTS, self).__init__()

        self.layer_Dict = {}
        self.layer_Dict['Encoder'] = Encoder()
        self.layer_Dict['Decoder'] = Decoder()

    def forward(self, tokens, token_lengths, mels= None, mel_lengths= None, noise_scale= 1.0, length_scale= 1.0, is_training= False):
        if is_training:
            return self.train(tokens, token_lengths, mels, mel_lengths)
        else:
            return self.inference(tokens, token_lengths, noise_scale, length_scale)

    def train(self, tokens, token_lengths, mels, mel_lengths):
        assert all(mel_lengths % hp_Dict['Decoder']['Num_Squeeze'] == 0), 'Mel lengths must be diviable by Num_Squeeze.'
        token_Masks = self.Mask_Generate(token_lengths)
        mean, log_Std, log_Durations, x_Mask = self.layer_Dict['Encoder'](tokens, token_Masks)

        mel_Masks = self.Mask_Generate(mel_lengths)

        attention_Masks = torch.unsqueeze(token_Masks, -1) * torch.unsqueeze(mel_Masks, 2)
        attention_Masks = attention_Masks.squeeze(1)

        z, log_Dets = self.layer_Dict['Decoder'](mels, mel_Masks)
        
        with torch.no_grad():
            std_Square_R = torch.exp(-2 * log_Std)
            # [Batch, Token_t, 1] [Batch, Token_t, Mel_t] [Batch, Token_t, Mel_t] [Batch, Token_t, 1]
            log_P = \
                torch.sum(-0.5 * math.log(2 * math.pi) - log_Std, dim= 1).unsqueeze(-1) + \
                std_Square_R.transpose(2, 1) @ (-0.5 * (z ** 2)) + \
                (mean * std_Square_R).transpose(2, 1) @ z + \
                torch.sum(-0.5 * (mean ** 2) * std_Square_R, dim= 1).unsqueeze(-1)

            attentions = 0?

        mel_Mean = mean @ attentions    # [Batch, Mel_Dim, Token_t] @ [Batch, Token_t, Mel_t] -> [Batch, Mel_dim, Mel_t]
        mel_Log_Std = log_Std @ attentions    # [Batch, Mel_Dim, Token_t] @ [Batch, Token_t, Mel_t] -> [Batch, Mel_dim, Mel_t]
        log_Duration_Targets = torch.log(torch.sum(attentions.unsqueeze(1), dim= -1) + 1e-8) * token_Masks

        return z, mel_Mean, mel_Log_Std, log_Dets, log_Durations, log_Duration_Targets
        
        
    def inference(self, tokens, token_lengths, noise_scale= 1.0, length_scale= 1.0):
        token_Masks = self.Mask_Generate(token_lengths)
        mean, log_Std, log_Durations, mask = self.layer_Dict['Encoder'](tokens, token_Masks)
        
        durations = torch.ceil(torch.exp(log_Durations) * mask * length_scale).squeeze(1)
        mel_Lengths = torch.clamp_min(torch.sum(durations, dim= 1), 1.0).long()
        mel_Masks = self.Mask_Generate(mel_Lengths)

        attention_Masks = torch.unsqueeze(token_Masks, -1) * torch.unsqueeze(mel_Masks, 2)
        attention_Masks = attention_Masks.squeeze(1)

        attentions = self.Path_Generate(durations, attention_Masks) # [Batch, Token_t, Mel_t]

        mel_Mean = mean @ attentions    # [Batch, Mel_Dim, Token_t] @ [Batch, Token_t, Mel_t] -> [Batch, Mel_dim, Mel_t]
        mel_Log_Std = log_Std @ attentions    # [Batch, Mel_Dim, Token_t] @ [Batch, Token_t, Mel_t] -> [Batch, Mel_dim, Mel_t]
        noises = torch.randn_like(mel_Mean) * noise_scale
        
        z = (mel_Mean + torch.exp(mel_Log_Std)) * noises * mel_Masks
        
        mels, _ = self.layer_Dict['Decoder'](z, mel_Masks, reverse= True)

        return mels

    def Mask_Generate(self, lengths, max_lengths= None, dtype= torch.float):
        '''
        lengths: [Batch]
        '''
        mask = torch.arange(max_lengths or torch.max(lengths))[None, :] < lengths[:, None]    # [Batch, Time]
        return mask.unsqueeze(1).to(dtype)  # [Batch, 1, Time]

    def Path_Generate(self, durations, masks):
        '''
        durations: [Batch, Token_t]
        masks: [Batch, Token_t, Mel_t]
        '''
        batch, token_Time, mel_Time = masks.size()
        durations = torch.cumsum(durations, dim= 1)        
        paths = self.Mask_Generate(
            lengths= durations.view(-1),
            max_lengths= mel_Time,
            dtype= masks.dtype
            ).to(device= masks.device)
        paths = paths.view(batch, token_Time, mel_Time)
        paths = paths - torch.nn.functional.pad(paths, [0,0,1,0,0,0])[:, :-1]
        paths = paths * masks

        return paths


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

    def forward(self, x, mask):
        '''
        x: [Batch, Time]
        lengths: [Batch]
        '''
        x = self.layer_Dict['Embedding'](x).transpose(2, 1) * math.sqrt(hp_Dict['Encoder']['Channels']) # [Batch, Dim, Time]        
        x = self.layer_Dict['Prenet'](x, mask)
        x = self.layer_Dict['Transformer'](x, mask)

        mean, log_Std = torch.split(
            self.layer_Dict['Project'](x) * mask,
            [hp_Dict['Sound']['Mel_Dim'], hp_Dict['Sound']['Mel_Dim']],
            dim= 1
            )
        log_Durations = self.layer_Dict['Duration_Predictor'](x.detach(), mask)

        return mean, log_Std, log_Durations, mask

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Squeeze'] = Squeeze(num_squeeze= hp_Dict['Decoder']['Num_Squeeze'])
        self.layer_Dict['Unsqueeze'] = Unsqueeze(num_squeeze= hp_Dict['Decoder']['Num_Squeeze'])

        self.layer_Dict['Flows'] = torch.nn.ModuleList()
        for index in range(hp_Dict['Decoder']['Stack']):
            self.layer_Dict['Flows'].append(AIA())

    def forward(self, x, mask, reverse= False):
        x, mask = self.layer_Dict['Squeeze'](x, mask)

        log_Dets = []
        for flow in  reversed(self.layer_Dict['Flows']) if reverse else self.layer_Dict['Flows']:
            x, logdet = flow(x, mask, reverse= reverse)            
            log_Dets.extend(logdet)
        
        x, mask = self.layer_Dict['Unsqueeze'](x, mask)

        return x, (None if reverse else torch.sum(torch.stack(log_Dets), dim= 0))

# class Loss(torch.nn.Module):
#     def __init__(self):
#         pass

#     def forward(self):
#         0.5 * math.log(2 * math.pi) + \
#             (
#                 torch.sum(y_logs) +
#                 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m) ** 2) - torch.sum(logdet)
#                 ) / (torch.sum(y_lengths // hps.model.n_sqz) * hps.model.n_sqz * hps.data.n_mel_channels) 


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
        self.layers.append(Invertible_1x1_Conv())
        self.layers.append(Affine_Coupling_Layer())

    def forward(self, x, mask, reverse= False):
        logdets = []
        for layer in (reversed(self.layers) if reverse else self.layers):
            x, logdet = layer(x, mask, reverse= reverse)
            logdets.append(logdet)
        
        return x, logdets

class Activation_Norm(torch.nn.Module):
    def __init__(self):
        super(Activation_Norm, self).__init__()
        self.initialized = False

        self.logs = torch.nn.Parameter(
            torch.zeros(1, hp_Dict['Sound']['Mel_Dim'] * hp_Dict['Decoder']['Num_Squeeze'], 1)
            )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, hp_Dict['Sound']['Mel_Dim'] * hp_Dict['Decoder']['Num_Squeeze'], 1)
            )

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
            logdet = torch.sum(self.logs) * torch.sum(mask, [1, 2])

        return z, logdet

    def initialize(self, x, mask):
        with torch.no_grad():
            denorm = torch.sum(mask, [0, 2])
            mean = torch.sum(x * mask, [0, 2]) / denorm
            square = torch.sum(x * x * mask, [0, 2]) / denorm
            variance = square - (mean ** 2)
            logs = 0.5 * torch.log(variance + 1e-7)

            self.logs.data.copy_(
                (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)
                )
            self.bias.data.copy_(
                (-mean * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
                )

class Invertible_1x1_Conv(torch.nn.Module):
    def __init__(self):
        super(Invertible_1x1_Conv, self).__init__()
        assert hp_Dict['Decoder']['Num_Split'] % 2 == 0

        weight = torch.qr(torch.FloatTensor(
            hp_Dict['Decoder']['Num_Split'],
            hp_Dict['Decoder']['Num_Split']
            ).normal_())[0]
        if torch.det(weight) < 0:
            weight[:, 0] = -weight[:, 0]

        self.weight = torch.nn.Parameter(weight)

    def forward(self, x, mask= None, reverse= False):
        batch, channels, time = x.size()
        assert channels % hp_Dict['Decoder']['Num_Split'] == 0

        if mask is None:
            mask = 1
            length = torch.ones((batch,), device=x.device, dtype= x.dtype) * time
        else:
            length = torch.sum(mask, [1, 2])

        # [Batch, 2, Dim/split, split/2, Time]
        x = x.view(batch, 2, channels // hp_Dict['Decoder']['Num_Split'], hp_Dict['Decoder']['Num_Split'] // 2, time)
        # [Batch, 2, split/2, Dim/split, Time] -> [Batch, split, Dim/split, Time]
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(batch, hp_Dict['Decoder']['Num_Split'], channels // hp_Dict['Decoder']['Num_Split'], time)

        if reverse:
            weight = torch.inverse(self.weight).to(dtype= self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            logdet = torch.logdet(self.weight) * (channels / hp_Dict['Decoder']['Num_Split']) * length
        
        z = torch.nn.functional.conv2d(
            input= x,
            weight= weight.unsqueeze(-1).unsqueeze(-1)            
            )
        # [Batch, 2, Split/2, Dim/Split, Time]
        z = z.view(batch, 2, hp_Dict['Decoder']['Num_Split'] // 2, channels // hp_Dict['Decoder']['Num_Split'], time)
        # [Batch, 2, Dim/Split, Split/2, Time] -> [Batch, Dim, Time]
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(batch, channels, time) * mask

        return z, logdet

class Affine_Coupling_Layer(torch.nn.Module):
    def __init__(self):
        super(Affine_Coupling_Layer, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Start'] = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels= hp_Dict['Sound']['Mel_Dim'] * hp_Dict['Decoder']['Num_Squeeze'] // 2,
            out_channels= hp_Dict['Decoder']['Affine_Coupling']['Calc_Channels'],
            kernel_size= 1,
            ))
        self.layer_Dict['WaveNet'] = WaveNet()
        self.layer_Dict['End'] = torch.nn.Conv1d(
            in_channels= hp_Dict['Decoder']['Affine_Coupling']['Calc_Channels'],
            out_channels= hp_Dict['Sound']['Mel_Dim'] * hp_Dict['Decoder']['Num_Squeeze'],
            kernel_size= 1,
            )
        self.layer_Dict['End'].weight.data.zero_()
        self.layer_Dict['End'].bias.data.zero_()

    def forward(self, x, mask, reverse= False):
        batch, channels, time = x.size()
        if mask is None:
            mask = 1
        
        x_a, x_b = torch.split(
            tensor= x,
            split_size_or_sections= [channels // 2] * 2,
            dim= 1
            )
        
        x = self.layer_Dict['Start'](x_a) * mask
        x = self.layer_Dict['WaveNet'](x, mask)
        outs = self.layer_Dict['End'](x)

        mean, logs = torch.split(
            tensor= outs,
            split_size_or_sections= [outs.size(1) // 2] * 2,
            dim= 1
            )

        if reverse:
            x_b = (x_b - mean) * torch.exp(-logs) * mask
            logdet = None
        else:
            x_b = (mean + torch.exp(logs) * x_b) * mask
            logdet = torch.sum(logs * mask, [1, 2])

        z = torch.cat([x_a, x_b], 1)

        return z, logdet

class WaveNet(torch.nn.Module):
    def __init__(self):
        super(WaveNet, self).__init__()
        self.layer_Dict = torch.nn.ModuleDict()

        for index in range(hp_Dict['Decoder']['Affine_Coupling']['WaveNet']['Num_Layers']):
            self.layer_Dict['In_{}'.format(index)] = torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= hp_Dict['Decoder']['Affine_Coupling']['Calc_Channels'],
                out_channels= hp_Dict['Decoder']['Affine_Coupling']['Calc_Channels'] * 2,
                kernel_size= hp_Dict['Decoder']['Affine_Coupling']['WaveNet']['Kernel_Size'],
                padding= (hp_Dict['Decoder']['Affine_Coupling']['WaveNet']['Kernel_Size'] - 1) // 2
                ))
            self.layer_Dict['Res_Skip_{}'.format(index)] = torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= hp_Dict['Decoder']['Affine_Coupling']['Calc_Channels'],
                out_channels= hp_Dict['Decoder']['Affine_Coupling']['Calc_Channels'] * (2 if index < hp_Dict['Decoder']['Affine_Coupling']['WaveNet']['Num_Layers'] - 1 else 1),
                kernel_size= 1
                ))

        self.layer_Dict['Dropout'] = torch.nn.Dropout(
            p= hp_Dict['Decoder']['Affine_Coupling']['WaveNet']['Dropout_Rate']
            )


    def forward(self, x, mask):
        output = torch.zeros_like(x)

        for index in range(hp_Dict['Decoder']['Affine_Coupling']['WaveNet']['Num_Layers']):
            ins = self.layer_Dict['In_{}'.format(index)](x)
            ins = self.layer_Dict['Dropout'](ins)
            acts = self.fused_gate(ins)
            res_Skips = self.layer_Dict['Res_Skip_{}'.format(index)](acts)
            if index < hp_Dict['Decoder']['Affine_Coupling']['WaveNet']['Num_Layers'] - 1:
                res, outs = torch.split(
                    tensor= res_Skips,
                    split_size_or_sections= [res_Skips.size(1) // 2] * 2,
                    dim= 1
                    )
                x = (x + res) * mask
                output += outs
            else:
                output += res_Skips

        return output * mask



    def fused_gate(self, x):
        tanh, sigmoid = torch.split(
            tensor= x,
            split_size_or_sections=[x.size(1) // 2] * 2,
            dim= 1
            )
        return torch.tanh(tanh) * torch.sigmoid(sigmoid)


class Squeeze(torch.nn.Module):
    def __init__(self, num_squeeze= 2):
        super(Squeeze, self).__init__()
        self.num_Squeeze = num_squeeze

    def forward(self, x, mask):
        batch, channels, time = x.size()
        time = (time // self.num_Squeeze) * self.num_Squeeze
        x = x[:, :, :time]
        x = x.view(batch, channels, time // self.num_Squeeze, self.num_Squeeze)
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, channels * self.num_Squeeze, time // self.num_Squeeze)

        if not mask is None:
            mask = mask[:, :, self.num_Squeeze - 1::self.num_Squeeze]
        else:
            mask = torch.ones(batch, 1, time // self.num_Squeeze).to(device= x.device, dtype= x.dtype)

        return x * mask, mask

class Unsqueeze(torch.nn.Module):
    def __init__(self, num_squeeze= 2):
        super(Unsqueeze, self).__init__()
        self.num_Squeeze = num_squeeze

    def forward(self, x, mask):
        batch, channels, time = x.size()
        x = x.view(batch, self.num_Squeeze, channels // self.num_Squeeze, time)
        x = x.permute(0, 2, 3, 1).contiguous().view(batch, channels // self.num_Squeeze, time * self.num_Squeeze)

        if not mask is None:
            mask = mask.unsqueeze(-1).repeat(1,1,1,self.num_Squeeze).view(batch, 1, time * self.num_Squeeze)
        else:
            mask = torch.ones(batch, 1, time * self.num_Squeeze).to(device= x.device, dtype= x.dtype)

        return x * mask, mask




if __name__ == "__main__":
    glowTTS = GlowTTS()

    tokens = torch.LongTensor([
        [6,3,4,6,1,3,26,5,7,3,14,6,3,3,6,22,3],
        [7,3,2,16,1,13,26,25,7,3,14,6,23,3,0,0,0],
        ])
    token_lengths = torch.LongTensor([15, 17])
    mels = torch.randn(2, 80, 156)
    mel_lengths = torch.LongTensor([86, 156])

    x = glowTTS(tokens, token_lengths, mels, mel_lengths, is_training= True)
    print(x)


    # decoder = Decoder()
    # # x = torch.randn(3, 80, 156)
    # # y = decoder(x, None)
    # x = torch.randn(3, 80, 156)
    # lengths = torch.LongTensor([141, 156, 92])
    # y = decoder(x, lengths)

    # print(y[0].shape, y[1])