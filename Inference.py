import torch
import numpy as np
import logging, yaml, os, sys, argparse, time, math
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from random import sample

from Modules import GlowTTS
from Datasets import Text_to_Token, Token_Stack, Mel_Stack, Mel_for_GE2E_Stack, Pitch_Stack

from Pattern_Generator import Pattern_Generate, Text_Filtering

from Speaker_Embedding.Modules import Encoder as Speaker_Embedding, Normalize

from Arg_Parser import Recursive_Parse
hp = Recursive_Parse(yaml.load(
    open('Hyper_Parameters.yaml', encoding='utf-8'),
    Loader=yaml.Loader
    ))

if not hp.Device is None:
    os.environ['CUDA_VISIBLE_DEVICES']= hp.Device

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

if hp.Use_Mixed_Precision:
    try:
        from apex import amp
    except:
        logging.info('There is no apex modules in the environment. Mixed precision does not work.')
        hp.Use_Mixed_Precision = False


class Dataset(torch.utils.data.Dataset):
    def __init__(self, labels, texts, scales, speakers= None, references= None):
        super(Dataset, self).__init__()
        speakers = speakers or [None] * len(texts)
        references = references or [None] * len(texts)

        self.patterns = [x for x in zip(labels, texts, scales, speakers, references)]

    def __getitem__(self, idx):
        label, text, scale, speaker, reference = self.patterns[idx]
        
        text = Text_Filtering(text)
        token = Text_to_Token(text)

        if not reference is None:
            _, reference, pitch = Pattern_Generate(reference, top_db= 30)
        else:
            pitch = None

        return token, scale, speaker, reference, pitch, label, text

    def __len__(self):
        return len(self.patterns)

class Collater:
    def __call__(self, batch):
        tokens, scales, speakers, references, pitches, labels, texts = zip(*batch)
        
        token_Lengths = [token.shape[0] for token in tokens]
        tokens = Token_Stack(tokens)
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths)   # [Batch]

        scales = torch.FloatTensor(scales)    # [Batch]

        if any([(x is None) for x in references]):
            prosodies = None
            prosody_Lengths = None
            ge2es = None
            pitches = None
            pitch_Lengths = None
        else:
            prosody_Lengths = [mel.shape[0] for mel in references]
            pitch_Lengths = [pitch.shape[0] for pitch in pitches]

            prosodies = Mel_Stack(references)
            prosodies = torch.FloatTensor(prosodies).transpose(2, 1)   # [Batch, Mel_dim, Time]
            prosody_Lengths = torch.LongTensor(prosody_Lengths)   # [Batch]

            ge2es = Mel_for_GE2E_Stack(references)
            ge2es = torch.FloatTensor(ge2es).transpose(2, 1)   # [Batch, Mel_dim, Time]
            
            pitches = Pitch_Stack(pitches)
            pitches = torch.FloatTensor(pitches)    # [Batch, Time]
            pitch_Lengths = torch.LongTensor(pitch_Lengths)   # [Batch]

        if any([(x is None) for x in speakers]):
            speakers = None
        else:
            speakers = torch.LongTensor(speakers)   # [Batch]

        return tokens, token_Lengths, prosodies, prosody_Lengths, speakers, ge2es, pitches, pitch_Lengths, scales, labels, texts


class Inferencer:
    def __init__(self, checkpoint_path):
        self.Model_Generate()
        self.Load_Checkpoint(checkpoint_path)

    def Model_Generate(self):
        self.model_Dict = {
            'GlowTTS': GlowTTS().to(device)
            }

        if not hp.Speaker_Embedding.GE2E.Checkpoint_Path is None:
            self.model_Dict['Speaker_Embedding'] = Speaker_Embedding(
                mel_dims= hp.Sound.Mel_Dim,
                lstm_size= hp.Speaker_Embedding.GE2E.LSTM.Sizes,
                lstm_stacks= hp.Speaker_Embedding.GE2E.LSTM.Stacks,
                embedding_size= hp.Speaker_Embedding.Embedding_Size,
                ).to(device)

        if hp.Use_Mixed_Precision:
            self.model_Dict['GlowTTS'] = amp.initialize(
                models= self.model_Dict['GlowTTS']
                )

        for model in self.model_Dict.values():
            model.eval()


    @torch.no_grad()
    def Inference_Step(self, tokens, token_lengths, prosodies, prosody_lengths, speakers, ge2es, pitches, pitch_lengths, length_scales, labels, texts, start_index= 0, tag_index= False, inference_path= './inference'):
        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)
        
        prosodies = prosodies if prosodies is None else prosodies.to(device)
        prosody_lengths = prosody_lengths if prosody_lengths is None else prosody_lengths.to(device)
        speakers = speakers if speakers is None else speakers.to(device)
        ge2es = ge2es if ge2es is None else ge2es.to(device)
        pitches = pitches if pitches is None else pitches.to(device)
        pitch_lengths = pitch_lengths if pitch_lengths is None else pitch_lengths.to(device)

        length_scales = length_scales.to(device)

        mels, mel_Lengths, attentions = self.model_Dict['GlowTTS'].inference(
            tokens= tokens,
            token_lengths= token_lengths,
            mels_for_prosody= prosodies,
            mel_lengths_for_prosody= prosody_lengths,
            speakers= speakers,
            mels_for_ge2e= ge2es,
            pitches= pitches,
            pitch_lengths= pitch_lengths,
            length_scale= length_scales
            )

        files = []
        for index, label in enumerate(labels):
            tags = []
            tags.append(label)
            if tag_index: tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(inference_path, 'PNG').replace('\\', '/'), exist_ok= True)
        for index, (mel, mel_Length, attention, label, text, length_Scale, file) in enumerate(zip(
            mels.cpu().numpy(),
            mel_Lengths.cpu().numpy(),
            attentions.cpu().numpy(),
            labels,
            texts,
            length_scales,
            files
            )):
            mel = mel[:, :mel_Length]
            attention = attention[:len(text) + 2, :mel_Length]
            
            new_Figure = plt.figure(figsize=(20, 5 * 3), dpi=100)
            plt.subplot2grid((3, 1), (0, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title('Mel    Label: {}    Text: {}    Length scale: {:.3f}'.format(label, text if len(text) < 70 else text[:70] + '…', length_Scale))
            plt.colorbar()
            plt.subplot2grid((3, 1), (1, 0), rowspan= 2)
            plt.imshow(attention, aspect='auto', origin='lower', interpolation= 'none')
            plt.title('Attention    Label: {}    Text: {}    Length scale: {:.3f}'.format(label, text if len(text) < 70 else text[:70] + '…', length_Scale))
            plt.yticks(
                range(len(text) + 2),
                ['<S>'] + list(text) + ['<E>'],
                fontsize = 10
                )
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(inference_path, 'PNG', '{}.PNG'.format(file)).replace('\\', '/'))
            plt.close(new_Figure)

        os.makedirs(os.path.join(inference_path, 'NPY').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(inference_path, 'NPY', 'Mel').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(inference_path, 'NPY', 'Attention').replace('\\', '/'), exist_ok= True)
        
        for index, (mel, mel_Length, file) in enumerate(zip(
            mels.cpu().numpy(),
            mel_Lengths.cpu().numpy(),
            files
            )):
            mel = mel[:, :mel_Length]
            attention = attention[:len(text) + 2, :mel_Length]

            np.save(
                os.path.join(inference_path, 'NPY', 'Mel', file).replace('\\', '/'),
                mel.T,
                allow_pickle= False
                )
            np.save(
                os.path.join(inference_path, 'NPY', 'Attention', file).replace('\\', '/'),
                attentions.cpu().numpy()[index],
                allow_pickle= False
                )

    def Inference(
        self,
        labels,
        texts,
        scales,
        speakers= None,
        references= None,
        inference_path= './inference'
        ):
        logging.info('Start inference.')
        dataLoader = torch.utils.data.DataLoader(
            dataset= Dataset(
                labels= labels,
                texts= texts,
                scales= scales,
                speakers= speakers,
                references= references
                ),
            shuffle= False,
            collate_fn= Collater(),
            batch_size= hp.Inference_Batch_Size or hp.Train.Batch_Size,
            num_workers= hp.Train.Num_Workers,
            pin_memory= True
            )
        logging.info('The number of inference patterns = {}.'.format(len(dataLoader.dataset)))

        for step, (tokens, token_Lengths, prosodies, prosody_Lengths, speakers, ge2es, pitches, pitch_Lengths, scales, labels, texts) in tqdm(
            enumerate(dataLoader),
            desc='[Inference]',
            total= math.ceil(len(dataLoader.dataset) / (hp.Inference_Batch_Size or hp.Train.Batch_Size))
            ):
            self.Inference_Step(tokens, token_Lengths, prosodies, prosody_Lengths, speakers, ge2es, pitches, pitch_Lengths, scales, labels, texts, start_index= step * (hp.Inference_Batch_Size or hp.Train.Batch_Size), inference_path= inference_path)

    def Load_Checkpoint(self, checkpoint_path):
        state_Dict = torch.load(checkpoint_path, map_location= 'cpu')
        self.model_Dict['GlowTTS'].load_state_dict(state_Dict['Model'])
        
        if hp.Use_Mixed_Precision:
            if not 'AMP' in state_Dict.keys():
                logging.info('No AMP state dict is in the checkpoint. Model regards this checkpoint is trained without mixed precision.')
            else:                
                amp.load_state_dict(state_Dict['AMP'])

        for flow in self.model_Dict['GlowTTS'].layer_Dict['Decoder'].layer_Dict['Flows']:
            flow.layers[0].initialized = True   # Activation_Norm is already initialized when checkpoint is loaded.

        logging.info('Checkpoint loaded at {} steps.'.format(state_Dict['Steps']))

        if 'GE2E' in self.model_Dict['GlowTTS'].layer_Dict.keys():
            self.GE2E_Load_Checkpoint()

    def GE2E_Load_Checkpoint(self):
        state_Dict = torch.load(
            hp.Speaker_Embedding.GE2E.Checkpoint_Path,
            map_location= 'cpu'
            )
        self.model_Dict['GlowTTS'].layer_Dict['GE2E'].load_state_dict(state_Dict['Model'])
        logging.info('Speaker embedding checkpoint \'{}\' loaded.'.format(hp.Speaker_Embedding.GE2E.Checkpoint_Path))


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-c', '--checkpoint', required= True)
    args = argParser.parse_args()

    labels = [
        'Alpha',
        'Bravo'
        ]
    texts = [
        'Birds of a feather flock together.',
        'A creative artist works on his next composition because he was not satisfied with his previous one.'
        ]
    scales = [1.0, 0.9]
    speakers = [0, 1]
    refereces = [
        './Wav_for_Inference/LJ.LJ050-0278.wav',
        './Wav_for_Inference/VCTK.p361_209.wav'
        ]

    inferencer = Inferencer(checkpoint_path= args.checkpoint)
    inferencer.Inference(
        labels= labels,
        texts= texts,
        scales= scales,
        speakers= speakers,
        references= refereces,
        inference_path= 'XXX'
        )