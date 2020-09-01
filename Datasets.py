import torch
import numpy as np
import yaml, pickle, os, math, logging
from random import shuffle

from Pattern_Generator import Pattern_Generate, Text_Filtering

from Arg_Parser import Recursive_Parse
hp = Recursive_Parse(yaml.load(
    open('Hyper_Parameters.yaml', encoding='utf-8'),
    Loader=yaml.Loader
    ))

with open(hp.Token_Path) as f:
    token_Dict = yaml.load(f, Loader=yaml.Loader)

def Text_to_Token(text):
    return np.array([
        token_Dict[letter]
        for letter in ['<S>'] + list(text) + ['<E>']
        ], dtype= np.int32)

def Token_Stack(tokens):
    max_Token_Length = max([token.shape[0] for token in tokens])
    tokens = np.stack(
        [np.pad(token, [0, max_Token_Length - token.shape[0]], constant_values= token_Dict['<E>']) for token in tokens],
        axis= 0
        )
    return tokens

def Mel_Stack(mels):
    max_Mel_Length = max([mel.shape[0] for mel in mels])
    mels = np.stack(
        [np.pad(mel, [[0, max_Mel_Length - mel.shape[0]], [0, 0]], constant_values= -hp.Sound.Max_Abs_Mel) for mel in mels],
        axis= 0
        )

    return mels

def Mel_for_GE2E_Stack(mels):
    mels_for_embeddig = []
    for mel in mels:
        overlap_Length = hp.Speaker_Embedding.GE2E.Inference.Overlap_Length
        slice_Length = hp.Speaker_Embedding.GE2E.Inference.Slice_Length
        required_Length = hp.Speaker_Embedding.GE2E.Inference.Samples * (slice_Length - overlap_Length) + overlap_Length

        if mel.shape[0] > required_Length:
            offset = np.random.randint(0, mel.shape[0] - required_Length)
            mel = mel[offset:offset + required_Length]
        else:
            pad = (required_Length - mel.shape[0]) / 2
            mel = np.pad(
                mel,
                [[int(np.floor(pad)), int(np.ceil(pad))], [0, 0]],
                mode= 'reflect'
                )

        mel = np.stack([
            mel[index:index + slice_Length]
            for index in range(0, required_Length - overlap_Length, slice_Length - overlap_Length)
            ])
        mels_for_embeddig.append(mel)

    return np.vstack(mels_for_embeddig)

def Pitch_Stack(pitches):
    max_Pitch_Length = max([pitch.shape[0] for pitch in pitches])    
    pitches = np.stack(
        [np.pad(pitch, [0, max_Pitch_Length - pitch.shape[0]], constant_values= 0.0) for pitch in pitches],
        axis= 0
        )

    return pitches



class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path,
        metadata_file,
        accumulated_dataset_epoch= 1,
        mel_length_min= -math.inf,
        mel_length_max= math.inf,
        text_length_min= -math.inf,
        text_length_max= math.inf,
        use_cache = False
        ):
        super(Dataset, self).__init__()

        self.pattern_Path = pattern_path
        self.use_cache = use_cache

        metadata_Dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'),
            mode= 'rb'
            ))
            
        self.file_List = [
            x for x in metadata_Dict['File_List']
            if all([
                metadata_Dict['Mel_Length_Dict'][x] >= mel_length_min,
                metadata_Dict['Mel_Length_Dict'][x] <= mel_length_max,
                metadata_Dict['Text_Length_Dict'][x] >= text_length_min,
                metadata_Dict['Text_Length_Dict'][x] <= text_length_max
                ])
            ]
        self.base_Length = len(self.file_List)
        self.file_List *= accumulated_dataset_epoch
            
        self.cache_Dict = {}

    def __getitem__(self, idx):
        idx = idx % self.base_Length

        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        file = self.file_List[idx]
        path = os.path.join(self.pattern_Path, file).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = Text_to_Token(pattern_Dict['Text']), pattern_Dict['Mel'], pattern_Dict['Speaker_ID'], pattern_Dict['Pitch']

        if self.use_cache:
            self.cache_Dict[idx] = pattern
        
        return pattern

    def __len__(self):
        return len(self.file_List)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self, pattern_path, use_cache = False):
        super(Inference_Dataset, self).__init__()
        self.use_cache = use_cache

        self.pattern_List = []
        for index, line in enumerate(open(pattern_path, 'r').readlines()[1:]):
            label, text, length_Scale, speaker, wav_for_GE2E, wav_for_Prosody, wav_for_Pitch = [x.strip() for x in line.strip().split('\t')]

            text = Text_Filtering(text)
            length_Scale = float(length_Scale)
            speaker = int(speaker)

            self.pattern_List.append((label, text, length_Scale, speaker, wav_for_GE2E, wav_for_Prosody, wav_for_Pitch))

        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        label, text, length_Scale, speaker, wav_for_GE2E, wav_for_Prosody, wav_for_Pitch = self.pattern_List[idx]
        token = Text_to_Token(text)

        _, mel_for_GE2E, _ = Pattern_Generate(wav_for_GE2E, top_db= 30)
        _, mel_for_Prosody, _ = Pattern_Generate(wav_for_Prosody, top_db= 30)
        _, _, pitch = Pattern_Generate(wav_for_Pitch, top_db= 30)
        pattern = token, length_Scale, speaker, mel_for_GE2E, mel_for_Prosody, pitch, label, text

        if self.use_cache:
            self.cache_Dict[idx] = pattern
 
        return pattern

    def __len__(self):
        return len(self.pattern_List)


class Collater:
    def __call__(self, batch):
        tokens, mels, speakers, pitches = zip(*batch)

        mels_for_GE2E = mels
        mels = [
            mel[:(mel.shape[0] // hp.Decoder.Num_Squeeze * hp.Decoder.Num_Squeeze)]
            for mel in mels
            ]
        token_Lengths = [token.shape[0] for token in tokens]
        mel_Lengths = [mel.shape[0] for mel in mels]

        tokens = Token_Stack(tokens)
        mels = Mel_Stack(mels)
        mels_for_GE2E = Mel_for_GE2E_Stack(mels_for_GE2E)
        pitches = Pitch_Stack(pitches)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths)   # [Batch]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Mel_dim, Time]
        mel_Lengths = torch.LongTensor(mel_Lengths)   # [Batch]        
        speakers = torch.LongTensor(speakers)
        mels_for_GE2E = torch.FloatTensor(mels_for_GE2E).transpose(2, 1)   # [Batch, Mel_dim, Time]
        pitches = torch.FloatTensor(pitches)    # [Batch, Time] Mel_t == Pitch_t

        return tokens, token_Lengths, mels, mel_Lengths, speakers, mels_for_GE2E, pitches

class Inference_Collater:
    def __call__(self, batch):
        tokens, length_Scales, speakers, mels_for_GE2E, mels_for_Prosody, pitches, labels, texts = zip(*batch)

        token_Lengths = [token.shape[0] for token in tokens]
        mel_Lengths_for_Prosody = [mel.shape[0] for mel in mels_for_Prosody]
        pitch_Lengths = [pitch.shape[0] for pitch in pitches]

        tokens = Token_Stack(tokens)
        mels_for_Prosody = Mel_Stack(mels_for_Prosody)
        mels_for_GE2E = Mel_for_GE2E_Stack(mels_for_GE2E)
        pitches = Pitch_Stack(pitches)

        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths)   # [Batch]
        mels_for_Prosody = torch.FloatTensor(mels_for_Prosody).transpose(2, 1)   # [Batch, Mel_dim, Time]
        mel_Lengths_for_Prosody = torch.LongTensor(mel_Lengths_for_Prosody)   # [Batch]
        speakers = torch.LongTensor(speakers)   # [Batch]
        mels_for_GE2E = torch.FloatTensor(mels_for_GE2E).transpose(2, 1)   # [Batch, Mel_dim, Time]
        pitches = torch.FloatTensor(pitches)    # [Batch, Time]
        pitch_Lengths = torch.LongTensor(pitch_Lengths)   # [Batch]
        length_Scales = torch.FloatTensor(length_Scales)    # [Batch]
        
        return tokens, token_Lengths, mels_for_Prosody, mel_Lengths_for_Prosody, speakers, mels_for_GE2E, pitches, pitch_Lengths, length_Scales, labels, texts

# if __name__ == '__main__':
#     dataset = Dev_Dataset()
#     collater = Collater()
    
#     dataLoader = torch.utils.data.DataLoader(
#         dataset= dataset,
#         shuffle= False,
#         collate_fn= collater,
#         batch_size= hp.Train.Batch_Size,
#         num_workers= hp.Train.Num_Workers,
#         pin_memory= True
#         )

#     import time
#     for x in dataLoader:
#         tokens, token_Lengths, mels, mel_Lengths, mels_for_Embedding = x
#         print(tokens.shape)
#         print(token_Lengths.shape)
#         print(mels.shape)
#         print(mel_Lengths.shape)
#         print(mels_for_Embedding.shape)
#         time.sleep(2.0)