import torch
import numpy as np
import logging, yaml, os, sys, argparse, time, math
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from random import sample

from Modules import GlowTTS, MLE_Loss
from Datasets import Train_Dataset, Dev_Dataset, Inference_Dataset, Collater, Inference_Collater
from Noam_Scheduler import Modified_Noam_Scheduler
from Radam import RAdam

from PWGAN.Modules import Generator as PWGAN
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
        

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, steps= 0):
        self.steps = steps
        self.epochs = 0

        self.Datset_Generate()
        self.Model_Generate()

        self.scalar_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        self.writer_Dict = {
            'Train': SummaryWriter(os.path.join(hp.Log_Path, 'Train')),
            'Evaluation': SummaryWriter(os.path.join(hp.Log_Path, 'Evaluation')),
            }
        
        self.Load_Checkpoint()

    def Datset_Generate(self):
        train_Dataset = Train_Dataset()
        dev_Dataset = Dev_Dataset()
        inference_Dataset = Inference_Dataset()
        logging.info('The number of train patterns = {}.'.format(len(train_Dataset) // hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
        logging.info('The number of development patterns = {}.'.format(len(dev_Dataset)))
        logging.info('The number of inference patterns = {}.'.format(len(inference_Dataset)))

        collater = Collater()
        inference_Collater = Inference_Collater()

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            shuffle= True,
            collate_fn= collater,
            batch_size= hp.Train.Batch_Size,
            num_workers= hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataLoader_Dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_Dataset,
            shuffle= False,
            collate_fn= collater,
            batch_size= hp.Train.Batch_Size,
            num_workers= hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataLoader_Dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_Dataset,
            shuffle= False,
            collate_fn= inference_Collater,
            batch_size= hp.Inference_Batch_Size or hp.Train.Batch_Size,
            num_workers= hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model_Dict = {
            'GlowTTS': GlowTTS().to(device)
            }

        if not hp.WaveNet.Checkpoint_Path is None:
            self.model_Dict['PWGAN'] = PWGAN().to(device)

        if not hp.Speaker_Embedding.GE2E.Checkpoint_Path is None:
            self.model_Dict['Speaker_Embedding'] = Speaker_Embedding(
                mel_dims= hp.Sound.Mel_Dim,
                lstm_size= hp.Speaker_Embedding.GE2E.LSTM.Sizes,
                lstm_stacks= hp.Speaker_Embedding.GE2E.LSTM.Stacks,
                embedding_size= hp.Speaker_Embedding.Embedding_Size,
                ).to(device)

        self.criterion_Dict = {
            'MSE': torch.nn.MSELoss().to(device),
            'MLE': MLE_Loss().to(device)
            }
        self.optimizer = RAdam(
            params= self.model_Dict['GlowTTS'].parameters(),
            lr= hp.Train.Learning_Rate.Initial,
            betas=(hp.Train.ADAM.Beta1, hp.Train.ADAM.Beta2),
            eps= hp.Train.ADAM.Epsilon,
            )
        self.scheduler = Modified_Noam_Scheduler(
            optimizer= self.optimizer,
            step_size= hp.Train.Learning_Rate.Decay_Step,
            gamma= hp.Train.Learning_Rate.Decay_Rate,
            )

        if hp.Use_Mixed_Precision:
            models = [self.model_Dict['GlowTTS']]
            if not hp.WaveNet.Checkpoint_Path is None:
                models.append(self.model_Dict['PWGAN'])
            if not hp.Speaker_Embedding.GE2E.Checkpoint_Path is None:
                models.append(self.model_Dict['Speaker_Embedding'])
                
            models, self.optimizer = amp.initialize(
                models= models,
                optimizers=self.optimizer
                )
            
            self.model_Dict['GlowTTS'] = models[0]
            if not hp.WaveNet.Checkpoint_Path is None:
                self.model_Dict['PWGAN'] = models[1]
            if not hp.Speaker_Embedding.GE2E.Checkpoint_Path is None:
                self.model_Dict['Speaker_Embedding'] = models[-1]

        logging.info(self.model_Dict['GlowTTS'])


    def Train_Step(self, tokens, token_lengths, mels, mel_lengths, mels_for_embedding= None, speakers= None):
        loss_Dict = {}

        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)
        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)

        if not mels_for_embedding is None:
            with torch.no_grad():
                mels_for_embedding = mels_for_embedding.to(device)
                embeddings = Normalize(
                    self.model_Dict['Speaker_Embedding'](mels_for_embedding),
                    samples= hp.Speaker_Embedding.GE2E.Inference.Samples
                    )
            speakers= None
        else:
            embeddings = None
            speakers= speakers.to(device)

        z, mel_Mean, mel_Log_Std, log_Dets, log_Durations, log_Duration_Targets = self.model_Dict['GlowTTS'](
            tokens= tokens,
            token_lengths= token_lengths,
            mels= mels,
            mel_lengths= mel_lengths,
            speaker_embeddings= embeddings,
            speakers= speakers,
            is_training= True
            )

        loss_Dict['MLE'] = self.criterion_Dict['MLE'](
            z= z,
            mean= mel_Mean,
            std= mel_Log_Std,
            log_dets= log_Dets,
            lengths= mel_lengths
            )
        loss_Dict['Length'] = self.criterion_Dict['MSE'](log_Durations, log_Duration_Targets)
        loss_Dict['Loss'] = loss_Dict['MLE'] + loss_Dict['Length']

        self.optimizer.zero_grad()
        if hp.Use_Mixed_Precision:
            with amp.scale_loss(loss_Dict['Loss'], self.optimizer) as scaled_loss:
                scaled_loss.backward()            
            torch.nn.utils.clip_grad_norm_(
                parameters= amp.master_params(self.optimizer),
                max_norm= hp.Train.Gradient_Norm
                )
        else:
            loss_Dict['Loss'].backward()        
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_Dict['GlowTTS'].parameters(),
                max_norm= hp.Train.Gradient_Norm
                )
        self.optimizer.step()
        self.scheduler.step()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Train.Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for tokens, token_Lengths, mels, mel_Lengths, mels_for_Embedding, speakers in self.dataLoader_Dict['Train']:
            self.Train_Step(tokens, token_Lengths, mels, mel_Lengths, mels_for_Embedding, speakers)
            
            if self.steps % hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % hp.Train.Logging_Interval == 0:                
                self.scalar_Dict['Train'] = {
                    tag: loss / hp.Train.Logging_Interval
                    for tag, loss in self.scalar_Dict['Train'].items()
                    }
                self.scalar_Dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()
                self.Write_to_Tensorboard('Train', self.scalar_Dict['Train'])
                self.scalar_Dict['Train'] = defaultdict(float)

            if self.steps % hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= hp.Train.Max_Step:
                return

        self.epochs += hp.Train.Train_Pattern.Accumulated_Dataset_Epoch

    @torch.no_grad()
    def Evaluation_Step(self, tokens, token_lengths, mels, mel_lengths, mels_for_embedding= None, speakers= None):
        loss_Dict = {}

        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)
        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)

        if not mels_for_embedding is None:
            with torch.no_grad():
                mels_for_embedding = mels_for_embedding.to(device)
                embeddings = Normalize(
                    self.model_Dict['Speaker_Embedding'](mels_for_embedding),
                    samples= hp.Speaker_Embedding.GE2E.Inference.Samples
                    )
        else:
            embeddings = None

        z, mel_Mean, mel_Log_Std, log_Dets, log_Durations, log_Duration_Targets = self.model_Dict['GlowTTS'](
            tokens= tokens,
            token_lengths= token_lengths,
            mels= mels,
            mel_lengths= mel_lengths,
            speaker_embeddings= embeddings,
            is_training= True
            )

        loss_Dict['MLE'] = self.criterion_Dict['MLE'](
            z= z,
            mean= mel_Mean,
            std= mel_Log_Std,
            log_dets= log_Dets,
            lengths= mel_lengths
            )
        loss_Dict['Length'] = self.criterion_Dict['MSE'](log_Durations, log_Duration_Targets)
        loss_Dict['Loss'] = loss_Dict['MLE'] + loss_Dict['Length']

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Evaluation.Loss/{}'.format(tag)] += loss
    
    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (tokens, token_Lengths, mels, mel_Lengths, mels_for_Embedding, speakers) in tqdm(
            enumerate(self.dataLoader_Dict['Dev'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataLoader_Dict['Dev'].dataset) / hp.Train.Batch_Size)
            ):
            self.Evaluation_Step(tokens, token_Lengths, mels, mel_Lengths, mels_for_Embedding, speakers)

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
            }
        self.Write_to_Tensorboard('Evaluation', self.scalar_Dict['Evaluation'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)

        for model in self.model_Dict.values():
            model.train()


    @torch.no_grad()
    def Inference_Step(self, tokens, token_lengths, length_scales, mels_for_embedding, speakers, labels, texts, start_index= 0, tag_step= False, tag_index= False):
        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)
        length_scales = length_scales.to(device)

        if not mels_for_embedding is None:
            with torch.no_grad():
                mels_for_embedding = mels_for_embedding.to(device)
                embeddings = Normalize(
                    self.model_Dict['Speaker_Embedding'](mels_for_embedding),
                    samples= hp.Speaker_Embedding.GE2E.Inference.Samples
                    )
        else:
            embeddings = None

        mels, attentions = self.model_Dict['GlowTTS'](
            tokens= tokens,
            token_lengths= token_lengths,
            length_scale= length_scales,
            speaker_embeddings= embeddings,
            is_training= False
            )

        files = []
        for label in labels:
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append(label)
            if tag_index: tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        for index, (mel, attention, label, text, length_Scale, file) in enumerate(zip(
            mels.cpu().numpy(),
            attentions.cpu().numpy(),
            labels,
            texts,
            length_scales,
            files
            )):
            new_Figure = plt.figure(figsize=(20, 5 * 3), dpi=100)
            plt.subplot2grid((3, 1), (0, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title('Mel    Label: {}    Text: {}    Length scale: {:.3f}'.format(label, text if len(text) < 90 else text[:90] + '…', length_Scale))
            plt.colorbar()
            plt.subplot2grid((3, 1), (1, 0), rowspan= 2)
            plt.imshow(attention[:len(text) + 2], aspect='auto', origin='lower')
            plt.title('Attention    Label: {}    Text: {}    Length scale: {:.3f}'.format(label, text if len(text) < 90 else text[:90] + '…', length_Scale))
            plt.yticks(
                range(len(text) + 2),
                ['<S>'] + list(text) + ['<E>'],
                fontsize = 10
                )
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.PNG'.format(file)).replace('\\', '/'))
            plt.close(new_Figure)

        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'NPY').replace('\\', '/'), exist_ok= True)
        for index, (mel, file) in enumerate(zip(
            mels.cpu().numpy(),
            files
            )):
            np.save(
                os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'NPY', file).replace('\\', '/'),
                mel.T,
                allow_pickle= False
                )        
            np.save(
                os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), file).replace('\\', '/'),
                attentions.cpu().numpy()[index],
                allow_pickle= False
                )

        if 'PWGAN' in self.model_Dict.keys():
            os.makedirs(os.path.join(hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)

            noises = torch.randn(mels.size(0), mels.size(2) * hp.Sound.Frame_Shift).to(device)
            mels = torch.nn.functional.pad(
                mels,
                pad= (hp.WaveNet.Upsample.Pad, hp.WaveNet.Upsample.Pad),
                mode= 'replicate'
                )
            mels.clamp_(min= -hp.Sound.Max_Abs_Mel, max= hp.Sound.Max_Abs_Mel)            

            for index, (audio, file) in enumerate(zip(
                self.model_Dict['PWGAN'](noises, mels).cpu().numpy(),
                files
                )):
                wavfile.write(
                    filename= os.path.join(hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.WAV'.format(file)).replace('\\', '/'),
                    data= (np.clip(audio, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
                    rate= hp.Sound.Sample_Rate
                    )

    def Inference_Epoch(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (tokens, token_lengths, length_scales, mels_for_Embedding, speakers, labels, texts) in tqdm(
            enumerate(self.dataLoader_Dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataLoader_Dict['Inference'].dataset) / (hp.Inference_Batch_Size or hp.Train.Batch_Size))
            ):
            self.Inference_Step(tokens, token_lengths, length_scales, mels_for_Embedding, speakers, labels, texts, start_index= step * (hp.Inference_Batch_Size or hp.Train.Batch_Size))

        for model in self.model_Dict.values():
            model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')                
                for root, _, files in os.walk(hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_Dict = torch.load(path, map_location= 'cpu')
        self.model_Dict['GlowTTS'].load_state_dict(state_Dict['Model.GlowTTS'])
        self.optimizer.load_state_dict(state_Dict['Optimizer'])
        self.scheduler.load_state_dict(state_Dict['Scheduler'])
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        if hp.Use_Mixed_Precision:
            if not 'AMP' in state_Dict.keys():
                logging.info('No AMP state dict is in the checkpoint. Model regards this checkpoint is trained without mixed precision.')
            else:                
                amp.load_state_dict(state_Dict['AMP'])

        for flow in self.model_Dict['GlowTTS'].layer_Dict['Decoder'].layer_Dict['Flows']:
            flow.layers[0].initialized = True   # Activation_Norm is already initialized when checkpoint is loaded.

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

        if not hp.WaveNet.Checkpoint_Path is None:
            self.PWGAN_Load_Checkpoint()
        if not hp.Speaker_Embedding.GE2E.Checkpoint_Path is None:
            self.Speaker_Embedding_Load_Checkpoint()

    def Save_Checkpoint(self):
        os.makedirs(hp.Checkpoint_Path, exist_ok= True)

        state_Dict = {
            'Model': {
                'GlowTTS': self.model_Dict['GlowTTS'].state_dict()
                },
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),            
            'Steps': self.steps,
            'Epochs': self.epochs,
            }
        if hp.Use_Mixed_Precision:
            state_Dict['AMP'] = amp.state_dict()

        torch.save(
            state_Dict,
            os.path.join(hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

    def PWGAN_Load_Checkpoint(self):
        state_Dict = torch.load(
            hp.WaveNet.Checkpoint_Path,
            map_location= 'cpu'
            )
        self.model_Dict['PWGAN'].load_state_dict(state_Dict['Model.Generator'])

        logging.info('PWGAN checkpoint \'{}\' loaded.'.format(hp.WaveNet.Checkpoint_Path))

    def Speaker_Embedding_Load_Checkpoint(self):
        state_Dict = torch.load(
            hp.Speaker_Embedding.GE2E.Checkpoint_Path,
            map_location= 'cpu'
            )
        self.model_Dict['Speaker_Embedding'].load_state_dict(state_Dict['Model'])

        logging.info('Speaker embedding checkpoint \'{}\' loaded.'.format(hp.Speaker_Embedding.GE2E.Checkpoint_Path))

    def Train(self):
        self.tqdm = tqdm(
            initial= self.steps,
            total= hp.Train.Max_Step,
            desc='[Training]'
            )
        
        hp_Path = os.path.join(hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_Path):
            os.makedirs(hp.Checkpoint_Path, exist_ok= True)
            yaml.dump(hp_Dict, open(hp_Path, 'w'))

        if hp.Train.Initial_Inference:
            self.Evaluation_Epoch()
            self.Inference_Epoch()

        for model in self.model_Dict.values():
            model.train()

        while self.steps < hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

    def Write_to_Tensorboard(self, category, scalar_Dict):
        for tag, scalar in scalar_Dict.items():
            self.writer_Dict[category].add_scalar(tag, scalar, self.steps)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    args = argParser.parse_args()

    new_Trainer = Trainer(steps= args.steps)
    new_Trainer.Train()