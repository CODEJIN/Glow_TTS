import torch
import numpy as np
import yaml, pickle, os, math, logging
from random import shuffle

from Pattern_Generator import Pattern_Generate, Text_Filtering

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

with open(hp_Dict['Token_Path']) as f:
    token_Dict = yaml.load(f, Loader=yaml.Loader)

def Text_to_Token(text):
    return np.array([
        token_Dict[letter]
        for letter in ['<S>'] + list(text) + ['<E>']
        ], dtype= np.int32)

def Speaker_Embedding_Stack(mels):
    mels_for_embeddig = []
    for mel in mels:
        overlap_Length = hp_Dict['Speaker_Embedding']['GE2E']['Inference']['Overlap_Length']
        slice_Length = hp_Dict['Speaker_Embedding']['GE2E']['Inference']['Slice_Length']
        required_Length = hp_Dict['Speaker_Embedding']['GE2E']['Inference']['Samples'] * (slice_Length - overlap_Length) + overlap_Length

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


class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Train_Dataset, self).__init__()

        metadata_Dict = pickle.load(open(
            os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], hp_Dict['Train']['Train_Pattern']['Metadata_File']).replace('\\', '/'), 'rb'
            ))
        self.file_List = [
            x for x in metadata_Dict['File_List']
            if all([
                metadata_Dict['Mel_Length_Dict'][x] >= hp_Dict['Train']['Train_Pattern']['Mel_Length']['Min'],
                metadata_Dict['Mel_Length_Dict'][x] <= hp_Dict['Train']['Train_Pattern']['Mel_Length']['Max'],
                metadata_Dict['Text_Length_Dict'][x] >= hp_Dict['Train']['Train_Pattern']['Text_Length']['Min'],
                metadata_Dict['Text_Length_Dict'][x] <= hp_Dict['Train']['Train_Pattern']['Text_Length']['Max']
                ])            
            ] * hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']        
            
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        file = self.file_List[idx]
        path = os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], file).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = Text_to_Token(pattern_Dict['Text']), pattern_Dict['Mel'], pattern_Dict['Speaker_ID']

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[path] = pattern
        
        return pattern

    def __len__(self):
        return len(self.file_List)

class Dev_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dev_Dataset, self).__init__()

        metadata_Dict = pickle.load(open(
            os.path.join(hp_Dict['Train']['Eval_Pattern']['Path'], hp_Dict['Train']['Eval_Pattern']['Metadata_File']).replace('\\', '/'), 'rb'
            ))
        self.file_List = [
            x for x in metadata_Dict['File_List']
            if all([
                metadata_Dict['Mel_Length_Dict'][x] >= hp_Dict['Train']['Eval_Pattern']['Mel_Length']['Min'],
                metadata_Dict['Mel_Length_Dict'][x] <= hp_Dict['Train']['Eval_Pattern']['Mel_Length']['Max'],
                metadata_Dict['Text_Length_Dict'][x] >= hp_Dict['Train']['Eval_Pattern']['Text_Length']['Min'],
                metadata_Dict['Text_Length_Dict'][x] <= hp_Dict['Train']['Eval_Pattern']['Text_Length']['Max']
                ])            
            ]
            
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        file = self.file_List[idx]
        path = os.path.join(hp_Dict['Train']['Eval_Pattern']['Path'], file).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = Text_to_Token(pattern_Dict['Text']), pattern_Dict['Mel'], pattern_Dict['Speaker_ID']

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[path] = pattern
        
        return pattern

    def __len__(self):
        return len(self.file_List)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self, pattern_path= 'Inference_Text.txt'):
        super(Inference_Dataset, self).__init__()

        self.pattern_List = []
        for index, line in enumerate(open(pattern_path, 'r').readlines()[1:]):
            line = line.strip().split('\t')
            label, text, length_Scale = line[0], Text_Filtering(line[1]), float(line[2])

            if text is None or text == '':
                logging.warn('The text of line {} in \'{}\' is incorrect. This line is ignoired.'.format(index + 1, pattern_path))
                continue

            path, speaker = None, None
            if hp_Dict['Speaker_Embedding']['Type'] == 'GE2E':
                path = line[3]
                if not os.path.exists(path):
                    logging.warn('There is no wav file of line {} in \'{}\'. This line is ignoired.'.format(index + 1, pattern_path))
                    continue
            elif hp_Dict['Speaker_Embedding']['Type'] == 'LUT':
                speaker = int(line[3])
                if speaker >= hp_Dict['Speaker_Embedding']['Num_Speakers']:
                    logging.warn('The speaker ID index ({}) of line {} is over the limit ({}). This line is ignoired.'.format(speaker, index + 1, hp_Dict['Speaker_Embedding']['Num_Speakers']))
                    continue

            self.pattern_List.append((label, text, length_Scale, path, speaker))

        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        label, text, length_Scale, path, speaker = self.pattern_List[idx]
        token = Text_to_Token(text)

        if not path is None:
            _, mel_for_Embedding, _ = Pattern_Generate(path, top_db= 30)
        else:
            mel_for_Embedding = None
        
        pattern = token, length_Scale, mel_for_Embedding, label, text, speaker

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[idx] = pattern
 
        return pattern

    def __len__(self):
        return len(self.pattern_List)


class Collater:
    def __call__(self, batch):
        tokens, mels, speakers = zip(*[
            (token, mel, speaker)
            for token, mel, speaker  in batch
            ])
        mels_for_Embedding = mels
        
        mels = [
            mel[:(mel.shape[0] // hp_Dict['Decoder']['Num_Squeeze'] * hp_Dict['Decoder']['Num_Squeeze'])]
            for mel in mels
            ]
        token_Lengths = [token.shape[0] for token in tokens]
        mel_Lengths = [mel.shape[0] for mel in mels]

        tokens, mels = self.Stack(tokens, mels)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths)   # [Batch]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Mel_dim, Time]
        mel_Lengths = torch.LongTensor(mel_Lengths)   # [Batch]
        speakers = torch.LongTensor(speakers)

        if hp_Dict['Speaker_Embedding']['Type'] == 'GE2E':
            mels_for_Embedding = Speaker_Embedding_Stack(mels_for_Embedding)
            mels_for_Embedding = torch.FloatTensor(mels_for_Embedding).transpose(2, 1)   # [Batch, Mel_dim, Time]
        else:
            mels_for_Embedding = None

        
        return tokens, token_Lengths, mels, mel_Lengths, mels_for_Embedding, speakers
    
    def Stack(self, tokens, mels):
        max_Token_Length = max([token.shape[0] for token in tokens])
        max_Mel_Length = max([mel.shape[0] for mel in mels])

        tokens = np.stack(
            [np.pad(token, [0, max_Token_Length - token.shape[0]], constant_values= token_Dict['<E>']) for token in tokens],
            axis= 0
            )
        mels = np.stack(
            [np.pad(mel, [[0, max_Mel_Length - mel.shape[0]], [0, 0]], constant_values= -hp_Dict['Sound']['Max_Abs_Mel']) for mel in mels],
            axis= 0
            )

        return tokens, mels

class Inference_Collater:
    def __call__(self, batch):
        tokens, length_Scales, mels_for_Embedding, labels, texts, speakers = zip(*[
            (token, length_Scale, mel_for_Embedding, label, text, speaker)
            for token, length_Scale, mel_for_Embedding, label, text, speaker  in batch
            ])
        token_Lengths = [token.shape[0] for token in tokens]

        tokens = self.Stack(tokens)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths)   # [Batch]
        length_Scales = torch.FloatTensor(length_Scales)

        mels_for_Embedding = None        
        if hp_Dict['Speaker_Embedding']['Type'] == 'GE2E':
            mels_for_Embedding = Speaker_Embedding_Stack(mels_for_Embedding)
            mels_for_Embedding = torch.FloatTensor(mels_for_Embedding).transpose(2, 1)   # [Batch, Mel_dim, Time]
        elif hp_Dict['Speaker_Embedding']['Type'] == 'LUT':
            speakers = torch.LongTensor(speakers)
            

        return tokens, token_Lengths, length_Scales, mels_for_Embedding, speakers, labels, texts

    def Stack(self, tokens):
        max_Token_Length = max([token.shape[0] for token in tokens])
        tokens = np.stack(
            [np.pad(token, [0, max_Token_Length - token.shape[0]], constant_values= token_Dict['<E>']) for token in tokens],
            axis= 0
            )

        return tokens

if __name__ == '__main__':
    dataset = Dev_Dataset()
    collater = Collater()
    
    dataLoader = torch.utils.data.DataLoader(
        dataset= dataset,
        shuffle= False,
        collate_fn= collater,
        batch_size= hp_Dict['Train']['Batch_Size'],
        num_workers= hp_Dict['Train']['Num_Workers'],
        pin_memory= True
        )

    import time
    for x in dataLoader:
        tokens, token_Lengths, mels, mel_Lengths, mels_for_Embedding = x
        print(tokens.shape)
        print(token_Lengths.shape)
        print(mels.shape)
        print(mel_Lengths.shape)
        print(mels_for_Embedding.shape)
        time.sleep(2.0)