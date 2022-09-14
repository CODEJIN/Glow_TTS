import torch
import numpy as np
import yaml, logging, math

from RPR_MHA import RPR_Multihead_Attention
from Gradient_Reversal_Layer import GRL
from Speaker_Embedding.Modules import Encoder as GE2E, Normalize as GE2E_Normalize

from Arg_Parser import Recursive_Parse
hp = Recursive_Parse(yaml.load(
    open('Hyper_Parameters.yaml', encoding='utf-8'),
    Loader=yaml.Loader
    ))


class GlowTTS(torch.nn.Module):
    def __init__(self):
        super(GlowTTS, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()

        if hp.Mode.upper() in ['SE', 'GR']:
            if hp.Speaker_Embedding.Type.upper() == 'LUT':
                self.layer_Dict['LUT'] = torch.nn.Embedding(
                    num_embeddings= hp.Speaker_Embedding.Num_Speakers,
                    embedding_dim= hp.Speaker_Embedding.Embedding_Size,
                    )
                torch.nn.init.uniform_(self.layer_Dict['LUT'].weight, -1.0, 1.0)
            elif hp.Speaker_Embedding.Type.upper() == 'GE2E':
                self.layer_Dict['GE2E'] = GE2E(
                    mel_dims= hp.Sound.Mel_Dim,
                    lstm_size= hp.Speaker_Embedding.GE2E.LSTM.Sizes,
                    lstm_stacks= hp.Speaker_Embedding.GE2E.LSTM.Stacks,
                    embedding_size= hp.Speaker_Embedding.Embedding_Size,
                    )
            else:
                raise ValueError('Unsupported Speaker embedding type: {}'.format(hp.Speaker_Embedding.Type))
        
        if hp.Mode.upper() in ['PE', 'GR']:
            self.layer_Dict['Prosody_Encoder'] = Prosody_Encoder()

        if hp.Mode.upper() == 'GR':
            self.layer_Dict['Speaker_Classifier_GR'] = Speaker_Classifier_GR()
            self.layer_Dict['Pitch_Interpolater'] = Pitch_Interpolater()
        
        self.layer_Dict['Encoder'] = Encoder()
        self.layer_Dict['Decoder'] = Decoder()
        self.layer_Dict['Maximum_Path_Generater'] = Maximum_Path_Generater()

    def forward(
        self,
        tokens,
        token_lengths,
        mels,
        mel_lengths,
        speakers,
        mels_for_ge2e,
        pitches
        ):
        '''
        For train.

        token: [Batch, Token_t] # Input text
        token_lengths: [Batch]  # Length of input text
        mels: [Batch, Mel_d, Mel_t] # Target and input of prosody encoder
        mel_lengths: [Batch]    # Length of target/prosody encoder
        speakers: [Batch]   # Indice of speaker.
        mels_for_ge2e: [Batch * Samples, Mel_d, Mel_SE_t]    # Input of speaker embedding
        pitches: [Batch, Mel_t] # Input of pitch quantinizer (Mel_t == Pitch_t)
        '''
        assert all(mel_lengths % hp.Decoder.Num_Squeeze == 0), 'Mel lengths must be diviable by Num_Squeeze.'
        
        if 'LUT' in self.layer_Dict.keys():
            speakers = self.layer_Dict['LUT'](speakers)
        elif 'GE2E' in self.layer_Dict.keys():
            speakers = self.layer_Dict['GE2E'](mels_for_ge2e)
            speakers = GE2E_Normalize(speakers).detach()    # GE2E is pre-trained.
        else:
            speakers = None

        if 'Prosody_Encoder' in self.layer_Dict.keys():
            prosodies = self.layer_Dict['Prosody_Encoder'](mels, mel_lengths)
        else:
            prosodies = None

        if 'Speaker_Classifier_GR' in self.layer_Dict.keys():
            classified_Speakers = self.layer_Dict['Speaker_Classifier_GR'](prosodies)
        else:
            classified_Speakers = None

        if not 'Pitch_Interpolater' in self.layer_Dict.keys():
            pitches = None

        if hp.Device != '-1': torch.cuda.synchronize()

        token_Masks = self.Mask_Generate(token_lengths)
        mel_Masks = self.Mask_Generate(mel_lengths)

        mean, log_Std, log_Durations, token_Masks = self.layer_Dict['Encoder'](tokens, token_Masks, speakers, prosodies)
        z, log_Dets, mel_Masks = self.layer_Dict['Decoder'](mels, mel_Masks, speakers, prosodies, pitches)
        
        attention_Masks = torch.unsqueeze(token_Masks, -1) * torch.unsqueeze(mel_Masks, 2)
        attention_Masks = attention_Masks.squeeze(1)

        if hp.Device != '-1': torch.cuda.synchronize()

        with torch.no_grad():
            std_Square_R = torch.exp(-2 * log_Std)
            # [Batch, Token_t, 1] [Batch, Token_t, Mel_t] [Batch, Token_t, Mel_t] [Batch, Token_t, 1]
            log_P = \
                torch.sum(-0.5 * math.log(2 * math.pi) - log_Std, dim= 1).unsqueeze(-1) + \
                std_Square_R.transpose(2, 1) @ (-0.5 * (z ** 2)) + \
                (mean * std_Square_R).transpose(2, 1) @ z + \
                torch.sum(-0.5 * (mean ** 2) * std_Square_R, dim= 1).unsqueeze(-1)

            attentions = self.layer_Dict['Maximum_Path_Generater'](log_P, attention_Masks)

        if hp.Device != '-1': torch.cuda.synchronize()

        mel_Mean = mean @ attentions    # [Batch, Mel_Dim, Token_t] @ [Batch, Token_t, Mel_t] -> [Batch, Mel_dim, Mel_t]
        mel_Log_Std = log_Std @ attentions    # [Batch, Mel_Dim, Token_t] @ [Batch, Token_t, Mel_t] -> [Batch, Mel_dim, Mel_t]
        log_Duration_Targets = torch.log(torch.sum(attentions.unsqueeze(1), dim= -1) + 1e-7) * token_Masks

        if hp.Device != '-1': torch.cuda.synchronize()

        return z, mel_Mean, mel_Log_Std, log_Dets, log_Durations, log_Duration_Targets, attentions, classified_Speakers

    def inference(
        self,
        tokens,
        token_lengths,
        mels_for_prosody,
        mel_lengths_for_prosody,
        speakers,
        mels_for_ge2e,
        pitches,
        pitch_lengths,
        noise_scale= 1.0,
        length_scale= 1.0
        ):
        '''
        For inference.
        token: [Batch, Token_t] # Input text
        token_lengths: [Batch]  # Length of input text
        mels_for_prosody: [Batch, Mel_d, Mel_t] # Input of prosody encoder
        mel_lengths_for_prosody: [Batch]    # Length of input mel for prosody
        speakers: [Batch] or None   # Indice of speaker. Only when hp.Speaker_Embedding.Type.upper() == 'LUT'
        mels_for_ge2e: [Batch * Samples, Mel_d, Mel_SE_t]    # Input of speaker embedding
        noise_scale: scalar of float
        length_scale: scalar of float or [Batch]. (I may change this to matrix to control speed letter by letter later)
        '''        
        if 'LUT' in self.layer_Dict.keys():
            speakers = self.layer_Dict['LUT'](speakers)
        elif 'GE2E' in self.layer_Dict.keys():
            speakers = self.layer_Dict['GE2E'](mels_for_ge2e)
            speakers = GE2E_Normalize(speakers)
        else:
            speakers = None

        if 'Prosody_Encoder' in self.layer_Dict.keys():
            prosodies = self.layer_Dict['Prosody_Encoder'](mels_for_prosody, mel_lengths_for_prosody)
        else:
            prosodies = None

        if hp.Device != '-1': torch.cuda.synchronize()

        token_Masks = self.Mask_Generate(token_lengths)
        mean, log_Std, log_Durations, mask = self.layer_Dict['Encoder'](tokens, token_Masks, speakers, prosodies)
        length_scale = length_scale.unsqueeze(-1).unsqueeze(-1)

        if hp.Device != '-1': torch.cuda.synchronize()

        durations = torch.ceil(torch.exp(log_Durations) * mask * length_scale).squeeze(1)
        mel_Lengths = torch.clamp_min(torch.sum(durations, dim= 1), 1.0).long()
        mel_Masks = self.Mask_Generate(mel_Lengths)

        attention_Masks = torch.unsqueeze(token_Masks, -1) * torch.unsqueeze(mel_Masks, 2)
        attention_Masks = attention_Masks.squeeze(1)


        attentions = self.Path_Generate(durations, attention_Masks) # [Batch, Token_t, Mel_t]

        if hp.Device != '-1': torch.cuda.synchronize()

        mel_Mean = mean @ attentions    # [Batch, Mel_Dim, Token_t] @ [Batch, Token_t, Mel_t] -> [Batch, Mel_dim, Mel_t]
        mel_Log_Std = log_Std @ attentions    # [Batch, Mel_Dim, Token_t] @ [Batch, Token_t, Mel_t] -> [Batch, Mel_dim, Mel_t]
        noises = torch.randn_like(mel_Mean) * noise_scale

        if hp.Device != '-1': torch.cuda.synchronize()

        z = (mel_Mean + torch.exp(mel_Log_Std) * noises) * mel_Masks

        if 'Pitch_Interpolater' in self.layer_Dict.keys():
            pitches = self.layer_Dict['Pitch_Interpolater'](pitches, pitch_lengths, mel_Lengths)
        else:
            pitches = None

        mels, _, mel_Masks = self.layer_Dict['Decoder'](z, mel_Masks, speakers, prosodies, pitches, reverse= True)

        if hp.Device != '-1': torch.cuda.synchronize()

        mels.masked_fill_(mel_Masks == 0.0, -hp.Sound.Max_Abs_Mel)

        return mels, mel_Lengths, attentions

    def Mask_Generate(self, lengths, max_lengths= None, dtype= torch.float):
        '''
        lengths: [Batch]
        '''
        mask = torch.arange(max_lengths or torch.max(lengths))[None, :].to(lengths.device) < lengths[:, None]    # [Batch, Time]
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
    '''
    Don't apply the xavier_uniform_ to submodules.
    I tried to apply the initializer to all of them, but failed. If you have any advice, please let me know by the issue.
    '''
    def __init__(self):
        super(Encoder, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Embedding'] = torch.nn.Embedding(
            num_embeddings= hp.Encoder.Embedding_Tokens,
            embedding_dim= hp.Encoder.Channels,
            )        
        torch.nn.init.normal_(
            self.layer_Dict['Embedding'].weight,
            mean= 0.0,
            std= hp.Encoder.Channels ** -0.5
            )
            
        self.layer_Dict['Prenet'] = Prenet(hp.Encoder.Prenet.Stacks)
        self.layer_Dict['Transformer'] = Transformer(hp.Encoder.Transformer.Stacks)

        self.layer_Dict['Project'] = torch.nn.Conv1d(   # xavier_uniform_ could be appiled to this only...
            in_channels= hp.Encoder.Channels,
            out_channels= hp.Sound.Mel_Dim * 2,
            kernel_size= 1
            )
        self.layer_Dict['Duration_Predictor'] = Duration_Predictor()

    def forward(self, x, mask, speakers= None, prosodies= None):
        '''
        x: [Batch, Time]
        lengths: [Batch]
        '''
        x = self.layer_Dict['Embedding'](x).transpose(2, 1) * math.sqrt(hp.Encoder.Channels) # [Batch, Dim, Time]
        x = self.layer_Dict['Prenet'](x, mask)
        x = self.layer_Dict['Transformer'](x, mask)

        mean, log_Std = torch.split(
            self.layer_Dict['Project'](x) * mask,
            [hp.Sound.Mel_Dim, hp.Sound.Mel_Dim],
            dim= 1
            )

        if not speakers is None:
            speakers = speakers.detach()
        if not prosodies is None:
            prosodies = prosodies.detach()

        log_Durations = self.layer_Dict['Duration_Predictor'](x.detach(), mask, speakers, prosodies)

        return mean, log_Std, log_Durations, mask

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Squeeze'] = Squeeze(num_squeeze= hp.Decoder.Num_Squeeze)
        self.layer_Dict['Unsqueeze'] = Unsqueeze(num_squeeze= hp.Decoder.Num_Squeeze)

        self.layer_Dict['Flows'] = torch.nn.ModuleList()
        for index in range(hp.Decoder.Stack):
            self.layer_Dict['Flows'].append(AIA())

    def forward(self, x, mask, speakers= None, prosodies= None, pitches= None, reverse= False):
        x, squeezed_Mask = self.layer_Dict['Squeeze'](x, mask)
        if not pitches is None:
            pitches, _ = self.layer_Dict['Squeeze'](pitches.unsqueeze(1), mask)
        log_Dets = []
        for flow in  reversed(self.layer_Dict['Flows']) if reverse else self.layer_Dict['Flows']:
            x, logdet = flow(x, squeezed_Mask, speakers, prosodies, pitches, reverse= reverse)
            log_Dets.extend(logdet)

        x, mask = self.layer_Dict['Unsqueeze'](x, squeezed_Mask)

        return x, (None if reverse else torch.sum(torch.stack(log_Dets), dim= 0)), mask


class Prosody_Encoder(torch.nn.Module):
    '''
    This is GST layer
    '''
    def __init__(self):
        super(Prosody_Encoder, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()

        previous_Channels = 1
        height = hp.Sound.Mel_Dim
        for index, (kernel_Size, channels, strides) in enumerate(zip(
            hp.Prosody_Encoder.Reference_Encoder.Conv.Kernel_Size,
            hp.Prosody_Encoder.Reference_Encoder.Conv.Channels,
            hp.Prosody_Encoder.Reference_Encoder.Conv.Strides
            )):
            self.layer_Dict['Conv_{}'.format(index)] = torch.nn.Sequential()
            self.layer_Dict['Conv_{}'.format(index)].add_module('Conv', Conv2d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= kernel_Size,
                stride= strides,
                padding= (kernel_Size - 1) // 2,
                bias= False,
                w_init_gain= 'relu'
                ))
            self.layer_Dict['Conv_{}'.format(index)].add_module('ReLU', torch.nn.ReLU(inplace= True))
            previous_Channels = channels
            height = math.ceil(height /  strides)
            
        self.layer_Dict['GRU'] = torch.nn.GRU(
            input_size= previous_Channels * height,
            hidden_size= hp.Prosody_Encoder.Reference_Encoder.GRU.Size,
            num_layers= hp.Prosody_Encoder.Reference_Encoder.GRU.Stacks,
            batch_first= True
            )

        self.layer_Dict['Attention'] = RPR_Multihead_Attention(     # Normal MHA
            query_channels= hp.Prosody_Encoder.Reference_Encoder.GRU.Size,
            key_channels= hp.Prosody_Encoder.Style_Token.Size,
            calc_channels= hp.Prosody_Encoder.Size,
            out_channels= hp.Prosody_Encoder.Size,
            num_heads= hp.Prosody_Encoder.Style_Token.Attention_Head
            )

        self.gst_Tokens = torch.nn.Parameter(
            data= torch.FloatTensor(
                hp.Prosody_Encoder.Style_Token.Size,
                hp.Prosody_Encoder.Style_Token.Num_Tokens
                )
            )
        torch.nn.init.normal_(self.gst_Tokens, mean= 0.0, std= 0.5)

    def forward(self, x, lengths):
        x = x.unsqueeze(1)  # [Batch, 1, Mel_d, Time]
        for index in range(len(hp.Prosody_Encoder.Reference_Encoder.Conv.Kernel_Size)):
            x = self.layer_Dict['Conv_{}'.format(index)](x)
        
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))     # [Batch, Dim, Compressed_Time]
        x = self.layer_Dict['GRU'](x.transpose(2, 1))[0].transpose(2, 1)

        indices = torch.ceil(lengths / np.prod(hp.Prosody_Encoder.Reference_Encoder.Conv.Strides, dtype=float)).to(dtype= lengths.dtype) - 1
        x = torch.stack([x[batch_Index, :, step] for batch_Index, step in enumerate(indices)], dim= 0)  # [Batch, Dim]

        x, _ = self.layer_Dict['Attention'](    # [Batch, Dim, 1]
            queries= x.unsqueeze(2),    # [Batch, Dim, 1(Time)]
            keys= torch.tanh(self.gst_Tokens).unsqueeze(0).expand(
                x.size(0),
                self.gst_Tokens.size(0),
                self.gst_Tokens.size(1)
                )  # [Batch, GST_dim, N_GST]
            )
        
        return x.squeeze(2)

class Pitch_Interpolater(torch.nn.Module):
    def forward(self, pitches, base_lengths, new_lengths):
        new_Max_Length = torch.max(new_lengths)

        pitches = [
            torch.nn.functional.interpolate(
                input= pitch[:base_Length].unsqueeze(0).unsqueeze(0),
                size= new_Length,
                mode= 'linear',
                align_corners= True
                ).squeeze(0).squeeze(0)
            for pitch, base_Length, new_Length in zip(pitches, base_lengths, new_lengths)
            ]
        pitches = torch.stack([
            torch.nn.functional.pad(pitch, [0, new_Max_Length - pitch.size(0)])
            for pitch in pitches
            ])
        
        return pitches #[Batch, Pitch_t]

class Speaker_Classifier_GR(torch.nn.Module):
    def __init__(self):
        super(Speaker_Classifier_GR, self).__init__()

        self.layer = torch.nn.Sequential()
        self.layer.add_module('GRL', GRL(weight= hp.Train.Adversarial_Speaker_Weight))

        previous_Channels = hp.Prosody_Encoder.Size
        for index, channels in enumerate(hp.Speaker_Classifier_GR.Channels):
            self.layer.add_module('Hidden_{}'.format(index), Conv1d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= 1,
                bias= True,
                w_init_gain= 'relu'
                ))
            self.layer.add_module('ReLU_{}'.format(index), torch.nn.ReLU())
            previous_Channels = channels
        
        self.layer.add_module('Output_{}'.format(index), Conv1d(
                in_channels= previous_Channels,
                out_channels= hp.Speaker_Embedding.Num_Speakers,
                kernel_size= 1,
                bias= True,
                w_init_gain= 'linear'
                ))

    def forward(self, x):
        return self.layer(x.unsqueeze(2)).squeeze(2)


class Prenet(torch.nn.Module):
    def __init__(self, stacks):
        super(Prenet, self).__init__()
        self.stacks = stacks

        self.layer_Dict = torch.nn.ModuleDict()
        for index in range(stacks):            
            self.layer_Dict['CLRD_{}'.format(index)] = CLRD()

        self.layer_Dict['Conv1x1'] = torch.nn.Conv1d(
            in_channels= hp.Encoder.Channels,
            out_channels= hp.Encoder.Channels,
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
            in_channels= hp.Encoder.Channels,
            out_channels= hp.Encoder.Channels,
            kernel_size= hp.Encoder.Prenet.Kernel_Size,
            padding= (hp.Encoder.Prenet.Kernel_Size - 1) // 2
            )
        self.layer_Dict['LayerNorm'] = torch.nn.LayerNorm(
            hp.Encoder.Channels,
            eps= 1e-4
            )
        self.layer_Dict['ReLU'] = torch.nn.ReLU(
            inplace= True
            )
        self.layer_Dict['Dropout'] = torch.nn.Dropout(
            p= hp.Encoder.Prenet.Dropout_Rate
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

        self.layer_Dict['Attention'] = RPR_Multihead_Attention(  # [Batch, Dim, Time]
            query_channels = hp.Encoder.Channels,
            calc_channels= hp.Encoder.Channels,
            out_channels= hp.Encoder.Channels,
            num_heads= hp.Encoder.Transformer.Attention.Heads,
            relative_postion_clipping_distance= hp.Encoder.Transformer.Attention.Window_Size,
            dropout_rate= hp.Encoder.Transformer.Dropout_Rate,
            )

        self.layer_Dict['LayerNorm_0'] = torch.nn.LayerNorm(    # This normalize last dim...
            normalized_shape= hp.Encoder.Channels,
            eps= 1e-4
            )
        
        self.layer_Dict['Conv_0'] = torch.nn.Conv1d(
            in_channels= hp.Encoder.Channels,
            out_channels= hp.Encoder.Transformer.Conv.Calc_Channels,
            kernel_size= hp.Encoder.Transformer.Conv.Kernel_Size,
            padding= (hp.Encoder.Transformer.Conv.Kernel_Size - 1) // 2
            )
        self.layer_Dict['Conv_1'] = torch.nn.Conv1d(
            in_channels= hp.Encoder.Transformer.Conv.Calc_Channels,
            out_channels= hp.Encoder.Channels,
            kernel_size= hp.Encoder.Transformer.Conv.Kernel_Size,
            padding= (hp.Encoder.Transformer.Conv.Kernel_Size - 1) // 2
            )
        
        self.layer_Dict['LayerNorm_1'] = torch.nn.LayerNorm(    # This normalize last dim...
            normalized_shape= hp.Encoder.Channels,
            eps= 1e-4
            )
        
        self.layer_Dict['ReLU'] = torch.nn.ReLU(
            inplace= True
            )
        self.layer_Dict['Dropout'] = torch.nn.Dropout(
            p= hp.Encoder.Transformer.Dropout_Rate
            )

    def forward(self, x, mask):
        x *= mask
        residual = x
        x, _ = self.layer_Dict['Attention'](  # [Batch, Dim, Time]
            queries= x,
            masks= (mask * mask.transpose(2, 1)).unsqueeze(1)
            )
        
        x = self.layer_Dict['Dropout'](x)
        x = self.layer_Dict['LayerNorm_0']((x + residual).transpose(2, 1)).transpose(2, 1) # [Batch, Dim, Time]

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

        previous_Channels = hp.Encoder.Channels
        
        if hp.Mode.upper() == 'SE':
            previous_Channels += hp.Speaker_Embedding.Embedding_Size
        elif hp.Mode.upper() == 'PE':
            previous_Channels += hp.Prosody_Encoder.Size
        elif hp.Mode.upper() == 'GR':
            assert hp.Speaker_Embedding.Embedding_Size == hp.Prosody_Encoder.Size, \
                'In GR mode, the size of speaker embeding and prosody encoder must be same.'
            previous_Channels += hp.Speaker_Embedding.Embedding_Size
        
        for index in range(hp.Encoder.Duration_Predictor.Stacks):
            self.layer_Dict['CRND_{}'.format(index)] = CRND(in_channels= previous_Channels)
            previous_Channels = hp.Encoder.Duration_Predictor.Channels

        self.layer_Dict['Projection'] = torch.nn.Conv1d(
            in_channels= previous_Channels,
            out_channels= 1,
            kernel_size= 1
            )

    def forward(self, x, x_mask, speakers= None, prosodies= None):
        step = x.size(2)
        x = [x]
        
        if any([not speakers is None, not prosodies is None]):
            conditions = 0
            conditions += speakers if not speakers is None else 0
            conditions += prosodies if not prosodies is None else 0
            x.append(conditions.unsqueeze(2).expand(-1, -1, step))

        x = torch.cat(x, dim= 1)

        for index in range(hp.Encoder.Duration_Predictor.Stacks):
            x = self.layer_Dict['CRND_{}'.format(index)](x, x_mask)
        x = self.layer_Dict['Projection'](x * x_mask)

        return x * x_mask

class CRND(torch.nn.Module):
    def __init__(self, in_channels):
        super(CRND, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Conv'] = torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= hp.Encoder.Duration_Predictor.Channels,
            kernel_size= hp.Encoder.Duration_Predictor.Kernel_Size,
            padding= (hp.Encoder.Duration_Predictor.Kernel_Size - 1) // 2
            )
        self.layer_Dict['ReLU'] = torch.nn.ReLU(
            inplace= True
            )
        # self.layer_Dict['LayerNorm'] = torch.nn.LayerNorm(
        #     hp.Encoder.Duration_Predictor.Channels,
        #     eps= 1e-4
        #     )
        self.layer_Dict['Dropout'] = torch.nn.Dropout(
            p= hp.Encoder.Duration_Predictor.Dropout_Rate
            )

    def forward(self, x, mask):
        x = self.layer_Dict['Conv'](x * mask)   # [Batch, Dim, Time]
        x = self.layer_Dict['ReLU'](x)
        # x = self.layer_Dict['LayerNorm'](x.transpose(2, 1)).transpose(2, 1)
        x = self.layer_Dict['Dropout'](x)

        return x




class AIA(torch.nn.Module):
    def __init__(self):
        super(AIA, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(Activation_Norm())
        self.layers.append(Invertible_1x1_Conv())
        self.layers.append(Affine_Coupling_Layer())

    def forward(self, x, mask, speakers, prosodies, pitches, reverse= False):
        logdets = []
        for layer in (reversed(self.layers) if reverse else self.layers):
            x, logdet = layer(x, mask, speakers= speakers, prosodies= prosodies, pitches= pitches, reverse= reverse)
            logdets.append(logdet)
        
        return x, logdets

class Activation_Norm(torch.nn.Module):
    def __init__(self):
        super(Activation_Norm, self).__init__()
        self.initialized = False

        self.logs = torch.nn.Parameter(
            torch.zeros(1, hp.Sound.Mel_Dim * hp.Decoder.Num_Squeeze, 1)
            )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, hp.Sound.Mel_Dim * hp.Decoder.Num_Squeeze, 1)
            )

    def forward(self, x, mask, reverse= False, **kwargs):   # kwargs is to skip speaker embedding
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
            logs = 0.5 * torch.log(torch.clamp_min(variance, 1e-7))

            self.logs.data.copy_(
                (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)
                )
            self.bias.data.copy_(
                (-mean * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
                )

class Invertible_1x1_Conv(torch.nn.Module):
    def __init__(self):
        super(Invertible_1x1_Conv, self).__init__()
        assert hp.Decoder.Num_Split % 2 == 0

        weight = torch.qr(torch.FloatTensor(
            hp.Decoder.Num_Split,
            hp.Decoder.Num_Split
            ).normal_())[0]
        if torch.det(weight) < 0:
            weight[:, 0] = -weight[:, 0]

        self.weight = torch.nn.Parameter(weight)

    def forward(self, x, mask= None, reverse= False, **kwargs):   # kwargs is to skip speaker embedding
        batch, channels, time = x.size()
        assert channels % hp.Decoder.Num_Split == 0

        if mask is None:
            mask = 1
            length = torch.ones((batch,), device=x.device, dtype= x.dtype) * time
        else:
            length = torch.sum(mask, [1, 2])

        # [Batch, 2, Dim/split, split/2, Time]
        x = x.view(batch, 2, channels // hp.Decoder.Num_Split, hp.Decoder.Num_Split // 2, time)
        # [Batch, 2, split/2, Dim/split, Time] -> [Batch, split, Dim/split, Time]
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(batch, hp.Decoder.Num_Split, channels // hp.Decoder.Num_Split, time)

        if reverse:
            weight = torch.inverse(self.weight).to(dtype= self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            logdet = torch.logdet(self.weight) * (channels / hp.Decoder.Num_Split) * length
        
        z = torch.nn.functional.conv2d(
            input= x,
            weight= weight.unsqueeze(-1).unsqueeze(-1)
            )
        # [Batch, 2, Split/2, Dim/Split, Time]
        z = z.view(batch, 2, hp.Decoder.Num_Split // 2, channels // hp.Decoder.Num_Split, time)
        # [Batch, 2, Dim/Split, Split/2, Time] -> [Batch, Dim, Time]
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(batch, channels, time) * mask

        return z, logdet

class Affine_Coupling_Layer(torch.nn.Module):
    def __init__(self):
        super(Affine_Coupling_Layer, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Start'] = torch.nn.utils.weight_norm(Conv1d(
            in_channels= hp.Sound.Mel_Dim * hp.Decoder.Num_Squeeze // 2,
            out_channels= hp.Decoder.Affine_Coupling.Calc_Channels,
            kernel_size= 1,
            w_init_gain= 'linear'
            ))
        self.layer_Dict['WaveNet'] = WaveNet()
        self.layer_Dict['End'] = Conv1d(
            in_channels= hp.Decoder.Affine_Coupling.Calc_Channels,
            out_channels= hp.Sound.Mel_Dim * hp.Decoder.Num_Squeeze,
            kernel_size= 1,
            w_init_gain= 'zero'
            )

    def forward(self, x, mask, speakers= None, prosodies= None, pitches= None, reverse= False):
        batch, channels, time = x.size()
        if mask is None:
            mask = 1
        
        x_a, x_b = torch.split(
            tensor= x,
            split_size_or_sections= [channels // 2] * 2,
            dim= 1
            )
        
        x = self.layer_Dict['Start'](x_a) * mask
        x = self.layer_Dict['WaveNet'](x, mask, speakers, prosodies, pitches)
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

        for index in range(hp.Decoder.Affine_Coupling.WaveNet.Num_Layers):
            self.layer_Dict['In_{}'.format(index)] = torch.nn.utils.weight_norm(Conv1d(
                in_channels= hp.Decoder.Affine_Coupling.Calc_Channels,
                out_channels= hp.Decoder.Affine_Coupling.Calc_Channels * 2,
                kernel_size= hp.Decoder.Affine_Coupling.WaveNet.Kernel_Size,
                padding= (hp.Decoder.Affine_Coupling.WaveNet.Kernel_Size - 1) // 2,
                w_init_gain= ['tanh', 'sigmoid']
                ))
            self.layer_Dict['Res_Skip_{}'.format(index)] = torch.nn.utils.weight_norm(Conv1d(
                in_channels= hp.Decoder.Affine_Coupling.Calc_Channels,
                out_channels= hp.Decoder.Affine_Coupling.Calc_Channels * (2 if index < hp.Decoder.Affine_Coupling.WaveNet.Num_Layers - 1 else 1),
                kernel_size= 1,
                w_init_gain= 'linear'
                ))

            if hp.Mode.upper() in ['SE', 'GR']:
                self.layer_Dict['Speaker_{}'.format(index)] = torch.nn.utils.weight_norm(Conv1d(
                    in_channels= hp.Speaker_Embedding.Embedding_Size,
                    out_channels= hp.Decoder.Affine_Coupling.Calc_Channels * 2,
                    kernel_size= 1,
                    w_init_gain= ['tanh', 'sigmoid']
                    ))
            if hp.Mode.upper() in ['PE', 'GR']:
                self.layer_Dict['Prosody_{}'.format(index)] = torch.nn.utils.weight_norm(Conv1d(
                    in_channels= hp.Prosody_Encoder.Size,
                    out_channels= hp.Decoder.Affine_Coupling.Calc_Channels * 2,
                    kernel_size= 1,
                    w_init_gain= ['tanh', 'sigmoid']
                    ))
            if hp.Mode.upper() == 'GR':
                self.layer_Dict['Pitch_{}'.format(index)] = torch.nn.utils.weight_norm(Conv1d(
                    in_channels= hp.Decoder.Num_Squeeze,
                    out_channels= hp.Decoder.Affine_Coupling.Calc_Channels * 2,
                    kernel_size= 1,
                    w_init_gain= ['tanh', 'sigmoid']
                    ))

        self.layer_Dict['Dropout'] = torch.nn.Dropout(
            p= hp.Decoder.Affine_Coupling.WaveNet.Dropout_Rate
            )

    def forward(self, x, mask, speakers= None, prosodies= None, pitches= None):
        output = torch.zeros_like(x)
        for index in range(hp.Decoder.Affine_Coupling.WaveNet.Num_Layers):
            ins = self.layer_Dict['In_{}'.format(index)](x)     # [Batch, Channels, Time]
            ins = self.layer_Dict['Dropout'](ins)
            if not speakers is None:
                ins += self.layer_Dict['Speaker_{}'.format(index)](speakers.unsqueeze(2))     # [Batch, Channels, Time] + [Batch, Channels, 1] -> [Batch, Channels, Time]
            if not prosodies is None:
                ins += self.layer_Dict['Prosody_{}'.format(index)](prosodies.unsqueeze(2))     # [Batch, Channels, Time] + [Batch, Channels, 1] -> [Batch, Channels, Time]

            if not pitches is None:
                ins += self.layer_Dict['Pitch_{}'.format(index)](pitches)     # [Batch, Channels, Time] + [Batch, Channels, Time] -> [Batch, Channels, Time]
            acts = self.fused_gate(ins)
            res_Skips = self.layer_Dict['Res_Skip_{}'.format(index)](acts)
            if index < hp.Decoder.Affine_Coupling.WaveNet.Num_Layers - 1:
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
        tanh, sigmoid = x.chunk(chunks= 2, dim= 1)
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


class Maximum_Path_Generater(torch.nn.Module):
    def __init__(self):
        super(Maximum_Path_Generater, self).__init__()
        if hp.Use_Cython_Alignment:
            import monotonic_align
            self.forward = monotonic_align.maximum_path

    def forward(self, log_p, mask):
        '''
        x: [Batch, Token_t, Mel_t]
        mask: [Batch, Token_t, Mel_t]
        '''
        log_p *= mask
        device, dtype = log_p.device, log_p.dtype
        log_p = log_p.data.cpu().numpy().astype(np.float32)
        mask = mask.data.cpu().numpy()

        token_Lengths = np.sum(mask, axis= 1)[:, 0].astype(np.int32)   # [Batch]
        mel_Lengths = np.sum(mask, axis= 2)[:, 0].astype(np.int32)   # [Batch]

        paths = self.calc_paths(log_p, token_Lengths, mel_Lengths)

        return torch.from_numpy(paths).to(device= device, dtype= dtype)

    def calc_paths(self, log_p, token_lengths, mel_lengths):
        return np.stack([
            self.calc_path(x, token_Length, mel_Length)
            for x, token_Length, mel_Length in zip(log_p, token_lengths, mel_lengths)
            ], axis= 0)

    def calc_path(self, x, token_length, mel_length):
        path = np.zeros_like(x).astype(np.int32)
        for mel_Index in range(mel_length):
            for token_Index in range(max(0, token_length + mel_Index - mel_length), min(token_length, mel_Index + 1)):
                if mel_Index == token_Index:
                    current_Q = -1e+7
                else:
                    current_Q = x[token_Index, mel_Index - 1]   # Stayed current token
                if token_Index == 0:
                    if mel_Index == 0:
                        prev_Q = 0.0
                    else:
                        prev_Q = -1e+7
                else:
                    prev_Q = x[token_Index - 1, mel_Index - 1]  # Moved to next token
                x[token_Index, mel_Index] = max(current_Q, prev_Q) + x[token_Index, mel_Index]

        token_Index = token_length - 1
        for mel_Index in range(mel_length - 1, -1, -1):
            path[token_Index, mel_Index] = 1
            if token_Index == mel_Index or x[token_Index, mel_Index - 1] < x[token_Index - 1, mel_Index - 1]:
                token_Index = max(0, token_Index - 1)

        return path


class Conv1d(torch.nn.Conv1d):
    def __init__(self, w_init_gain= 'relu', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        gains = self.w_init_gain
        if isinstance(gains, str):
            gains = [gains]
        
        weights = torch.chunk(self.weight, len(gains), dim= 0)
        for gain, weight in zip(gains, weights):
            if gain == 'zero':
                torch.nn.init.zeros_(weight)
            elif gain in ['relu', 'leaky_relu']:
                torch.nn.init.kaiming_uniform_(weight, nonlinearity= gain)
            else:
                torch.nn.init.xavier_uniform_(weight, gain= torch.nn.init.calculate_gain(gain))
        
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Conv2d(torch.nn.Conv2d):
    def __init__(self, w_init_gain= 'relu', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)



class MLE_Loss(torch.nn.modules.loss._Loss):
    def forward(self, z, mean, std, log_dets, lengths):
        '''
        https://github.com/jaywalnut310/glow-tts/issues/6
        '''
        loss = torch.sum(std) + 0.5 * torch.sum(torch.exp(-2 * std) * (z - mean) ** 2) - torch.sum(log_dets)
        loss /= torch.sum(lengths // hp.Decoder.Num_Squeeze) * hp.Decoder.Num_Squeeze * hp.Sound.Mel_Dim
        loss += 0.5 * math.log(2 * math.pi)    # I ignore this part because this does not affect to the gradients.

        return loss


if __name__ == "__main__":
    # glowTTS = GlowTTS()

    # tokens = torch.LongTensor([
    #     [6,3,4,6,1,3,26,5,7,3,14,6,3,3,6,22,3],
    #     [7,3,2,16,1,13,26,25,7,3,14,6,23,3,0,0,0],
    #     ])
    # token_lengths = torch.LongTensor([15, 17])
    # mels = torch.randn(2, 80, 156)
    # mel_lengths = torch.LongTensor([86, 156])

    # # x = glowTTS(tokens, token_lengths, mels, mel_lengths, is_training= True)
    # # print(x)

    # mels, attentions = glowTTS(tokens, token_lengths, is_training= False)

    # import matplotlib.pyplot as plt
    # plt.subplot(211)    
    # plt.imshow(mels.detach().cpu().numpy()[0], aspect='auto', origin='lower')
    # plt.subplot(212)
    # plt.imshow(attentions.cpu().numpy()[0], aspect='auto', origin='lower')
    # plt.show()

    # decoder = Decoder()
    # # x = torch.randn(3, 80, 156)
    # # y = decoder(x, None)
    # x = torch.randn(3, 80, 156)
    # lengths = torch.LongTensor([141, 156, 92])
    # y = decoder(x, lengths)

    # print(y[0].shape, y[1])

    pe = Prosody_Encoder()
    q = pe(torch.randn(3, 80, 562), torch.LongTensor([345, 467, 562]))
    print(q.shape)
