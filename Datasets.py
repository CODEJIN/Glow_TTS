import torch
import numpy as np
import yaml, pickle, os, math
from random import shuffle
#from Pattern_Generator import Pattern_Generate
from Pattern_Generator_Replication import Pattern_Generate

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

def Text_to_Token(text):
    return
def Token_to_Text(token):
    return

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
        self.dataset_Dict = metadata_Dict['Dataset_Dict']
            
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        file = self.file_List[idx]
        dataset = self.dataset_Dict[file]
        path = os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], dataset, file).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = Text_to_Token(pattern_Dict['Text']), pattern_Dict['Mel']

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
        self.dataset_Dict = metadata_Dict['Dataset_Dict']
            
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        file = self.file_List[idx]
        dataset = self.dataset_Dict[file]
        path = os.path.join(hp_Dict['Train']['Eval_Pattern']['Path'], dataset, file).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = Text_to_Token(pattern_Dict['Text']), pattern_Dict['Mel']

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[path] = pattern
        
        return pattern

    def __len__(self):
        return len(self.file_List)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self, pattern_path= 'Inference_Text.txt'):
        super(Inference_Dataset, self).__init__()

        self.pattern_List = [
            (line.strip().split('\t')[0], float(line.strip().split('\t')[1]))
            for line in open(pattern_path, 'r').readlines()[1:]
            ]
        
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        text, length_Scale = self.pattern_List[idx]

        pattern = text, length_Scale

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[idx] = pattern
 
        return pattern

    def __len__(self):
        return len(self.pattern_List)


class Collater:
    def __call__(self, batch):
        tokens, mels = zip(*[
            (token, mel)
            for token, mel  in batch
            ])
        token_Lengths = [token.shape[0] for token in tokens]
        mel_Lengths = [mel.shape[0] for mel in mels]

        tokens, mels = self.Stack(tokens, mels)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths)   # [Batch]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Mel_dim, Time]
        mel_Lengths = torch.LongTensor(mel_Lengths)   # [Batch]
        
        return tokens, token_Lengths, mels, mel_Lengths
    
    def Stack(self, tokens, mels):
        max_Token_Length = max([token.shape[0] for token in tokens])
        max_Mel_Length = max([mel.shape[0] for mel in mels])

        tokens = np.stack(
            [np.pad(token, [0, max_Token_Length - token.shape[0]]) for token in tokens],
            axis= 0
            )
        mels = np.stack(
            [np.pad(mel, [[0, max_Mel_Length - mel.shape[0]], [0, 0]], constant_values= token_Index_Dict['<E>']) for mel in mels],
            axis= 0
            )

        return tokens, mels

class Inference_Collater:
    def __call__(self, batch):
        tokens, length_Scales = zip(*[
            (token, length_Scale)
            for token, length_Scale  in batch
            ])
        token_Lengths = [token.shape[0] for token in tokens]

        tokens = self.Stack(tokens)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths)   # [Batch]
        length_Scales = torch.FloatTensor(length_Scales)

        return tokens, token_Lengths, length_Scales

    def Stacks(self, tokens):
        max_Token_Length = max([token.shape[0] for token in tokens])        
        tokens = np.stack(
            [np.pad(token, [0, max_Token_Length - token.shape[0]]) for token in tokens],
            axis= 0
            )

        return tokens