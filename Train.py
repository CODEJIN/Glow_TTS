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

# from PWGAN.Modules import Generator as PWGAN

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

if not hp_Dict['Device'] is None:
    os.environ['CUDA_VISIBLE_DEVICES']= hp_Dict['Device']

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

if hp_Dict['Use_Mixed_Precision']:
    try:
        from apex import amp
    except:
        logging.info('There is no apex modules in the environment. Mixed precision does not work.')
        hp_Dict['Use_Mixed_Precision'] = False
        

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

        self.writer = SummaryWriter(hp_Dict['Log_Path'])

        if not hp_Dict['WaveNet']['Checkpoint_Path'] is None:
            self.PWGAN_Load_Checkpoint()
        
        self.Load_Checkpoint()

    def Datset_Generate(self):
        train_Dataset = Train_Dataset()
        dev_Dataset = Dev_Dataset()
        inference_Dataset = Inference_Dataset()
        logging.info('The number of train patterns = {}.'.format(len(train_Dataset) // hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']))
        logging.info('The number of development patterns = {}.'.format(len(dev_Dataset)))
        logging.info('The number of inference patterns = {}.'.format(len(inference_Dataset)))

        collater = Collater()
        inference_Collater = Inference_Collater()

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            shuffle= True,
            collate_fn= collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_Dataset,
            shuffle= False,
            collate_fn= collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_Dataset,
            shuffle= False,
            collate_fn= inference_Collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )

    def Model_Generate(self):
        self.model_Dict = {
            'GlowTTS': GlowTTS().to(device)
            }
        self.criterion_Dict = {
            'MSE': torch.nn.MSELoss().to(device),
            'MLE': MLE_Loss().to(device)
            }
        self.optimizer = torch.optim.Adam(
            params= self.model_Dict['GlowTTS'].parameters(),
            lr= hp_Dict['Train']['Learning_Rate']['Initial'],
            betas=(hp_Dict['Train']['ADAM']['Beta1'], hp_Dict['Train']['ADAM']['Beta2']),
            eps= hp_Dict['Train']['ADAM']['Epsilon'],
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer= self.optimizer,
            step_size= hp_Dict['Train']['Learning_Rate']['Decay_Step'],
            gamma= hp_Dict['Train']['Learning_Rate']['Decay_Rate'],
            )

        if hp_Dict['Use_Mixed_Precision']:
            self.model_Dict['GlowTTS'], self.optimizer = amp.initialize(
                models=self.model_Dict['GlowTTS'],
                optimizers=self.optimizer
                )

        logging.info(self.model_Dict['GlowTTS'])


    def Train_Step(self, tokens, token_lengths, mels, mel_lengths):
        loss_Dict = {}

        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)
        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)

        z, mel_Mean, mel_Log_Std, log_Dets, log_Durations, log_Duration_Targets = self.model_Dict['GlowTTS'](
            tokens= tokens,
            token_lengths= token_lengths,
            mels= mels,
            mel_lengths= mel_lengths,
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
        if hp_Dict['Use_Mixed_Precision']:            
            with amp.scale_loss(loss_Dict['Loss'], self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_Dict['Loss'].backward()        
        torch.nn.utils.clip_grad_norm_(
            parameters= self.model_Dict['GlowTTS'].parameters(),
            max_norm= hp_Dict['Train']['Gradient_Norm']
            )
        self.optimizer.step()
        self.scheduler.step()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Train'][tag] += loss

    def Train_Epoch(self):
        for tokens, token_Lengths, mels, mel_Lengths in self.dataLoader_Dict['Train']:
            self.Train_Step(tokens, token_Lengths, mels, mel_Lengths)
            
            if self.steps % hp_Dict['Train']['Checkpoint_Save_Interval'] == 0:
                self.Save_Checkpoint()

            if self.steps % hp_Dict['Train']['Logging_Interval'] == 0:                
                self.scalar_Dict['Train'] = {
                    tag: loss / hp_Dict['Train']['Logging_Interval']
                    for tag, loss in self.scalar_Dict['Train'].items()
                        }
                self.scalar_Dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()
                self.Write_to_Tensorboard('Train', self.scalar_Dict['Train'])
                self.scalar_Dict['Train'] = defaultdict(float)

            if self.steps % hp_Dict['Train']['Evaluation_Interval'] == 0:
                self.Evaluation_Epoch()

            if self.steps % hp_Dict['Train']['Inference_Interval'] == 0:
                self.Inference_Epoch()
            
            if self.steps >= hp_Dict['Train']['Max_Step']:
                return

        self.epochs += hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']

    @torch.no_grad()
    def Evaluation_Step(self, tokens, token_lengths, mels, mel_lengths):
        loss_Dict = {}

        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)
        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)

        z, mel_Mean, mel_Log_Std, log_Dets, log_Durations, log_Duration_Targets = self.model_Dict['GlowTTS'](
            tokens= tokens,
            token_lengths= token_lengths,
            mels= mels,
            mel_lengths= mel_lengths,
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
            self.scalar_Dict['Evaluation'][tag] += loss
    
    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (tokens, token_Lengths, mels, mel_Lengths) in tqdm(
            enumerate(self.dataLoader_Dict['Dev'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataLoader_Dict['Dev'].dataset) / hp_Dict['Train']['Batch_Size'])
            ):
            self.Evaluation_Step(tokens, token_Lengths, mels, mel_Lengths)

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
            }
        self.Write_to_Tensorboard('Evaluation', self.scalar_Dict['Evaluation'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)

        for model in self.model_Dict.values():
            model.train()


    @torch.no_grad()
    def Inference_Step(self, tokens, token_lengths, length_scales, texts, start_index= 0, tag_step= False):
        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)
        length_scales = length_scales.to(device)

        mels, attentions = self.model_Dict['GlowTTS'](
            tokens= tokens,
            token_lengths= token_lengths,
            length_scale= length_scales,
            is_training= False
            )

        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        for index, (mel, attention, text, length_Scale) in enumerate(zip(
            mels.cpu().numpy(),
            attentions.cpu().numpy(),
            texts,
            length_scales
            )):
            new_Figure = plt.figure(figsize=(20, 5 * 3), dpi=100)
            plt.subplot2grid((3, 1), (0, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title('Mel    Text: {}    Length scale: {:.3f}'.format(text if len(text) < 90 else text[:90] + '…', length_Scale))
            plt.colorbar()
            plt.subplot2grid((3, 1), (1, 0), rowspan= 2)
            plt.imshow(attention[:len(text) + 2], aspect='auto', origin='lower')
            plt.title('Attention    Text: {}    Length scale: {:.3f}'.format(text if len(text) < 90 else text[:90] + '…', length_Scale))
            plt.yticks(
                range(len(text) + 2),
                ['<S>'] + list(text) + ['<E>'],
                fontsize = 10
                )
            plt.colorbar()
            plt.tight_layout()
            file = '{}IDX_{}.PNG'.format(
                'Step-{}.'.format(self.steps) if tag_step else '',
                index + start_index
                )
            plt.savefig(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'PNG', file).replace('\\', '/'))
            plt.close(new_Figure)

        if 'PWGAN' in self.model_Dict.keys():
            os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)

            noises = torch.randn(mels.size(0), mels.size(2) * hp_Dict['Sound']['Frame_Shift']).to(device)
            mels = torch.nn.functional.pad(
                mels,
                pad= (hp_Dict['WaveNet']['Upsample']['Pad'], hp_Dict['WaveNet']['Upsample']['Pad']),
                mode= 'replicate'
                )

            for index, audio in enumerate(self.model_Dict['PWGAN'](noises, mels).cpu().numpy()):
                file = '{}IDX_{}.WAV'.format(
                    'Step-{}.'.format(self.steps) if tag_step else '',
                    index + start_index
                    )
                wavfile.write(
                    filename= os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'WAV', file).replace('\\', '/'),
                    data= (np.clip(audio, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
                    rate= hp_Dict['Sound']['Sample_Rate']
                    )
        else:
            os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'NPY').replace('\\', '/'), exist_ok= True)

            for index, mel in enumerate(mels.cpu().numpy()):
                file = '{}IDX_{}.NPY'.format(
                    'Step-{}.'.format(self.steps) if tag_step else '',
                    index + start_index
                    )
                np.save(
                    os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'NPY', file).replace('\\', '/'),
                    mel.T,
                    allow_pickle= False
                    )

    def Inference_Epoch(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (tokens, token_lengths, length_scales, texts) in tqdm(
            enumerate(self.dataLoader_Dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataLoader_Dict['Inference'].dataset) / hp_Dict['Train']['Batch_Size'])
            ):
            self.Inference_Step(tokens, token_lengths, length_scales, texts, start_index= step * hp_Dict['Train']['Batch_Size'])

        for model in self.model_Dict.values():
            model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')                
                for root, _, files in os.walk(hp_Dict['Checkpoint_Path'])
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_Dict = torch.load(path, map_location= 'cpu')
        self.model_Dict['GlowTTS'].load_state_dict(state_Dict['Model']['GlowTTS'])
        self.optimizer.load_state_dict(state_Dict['Optimizer'])
        self.scheduler.load_state_dict(state_Dict['Scheduler'])
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        if hp_Dict['Use_Mixed_Precision']:
            if not 'AMP' in state_Dict.keys():
                logging.info('No AMP state dict is in the checkpoint. Model regards this checkpoint is trained without mixed precision.')
            else:                
                amp.load_state_dict(state_Dict['AMP'])

        for flow in self.model_Dict['GlowTTS'].layer_Dict['Decoder'].layer_Dict['Flows']:
            flow.layers[0].initialized = True   # Activation_Norm is already initialized when checkpoint is loaded.

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)

        state_Dict = {
            'Model': {
                'GlowTTS': self.model_Dict['GlowTTS'].state_dict()
                },
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),            
            'Steps': self.steps,
            'Epochs': self.epochs,
            }
        if hp_Dict['Use_Mixed_Precision']:
            state_Dict['AMP'] = amp.state_dict()

        torch.save(
            state_Dict,
            os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

    def PWGAN_Load_Checkpoint(self):
        self.model_Dict['PWGAN'] = PWGAN().to(device)

        if hp_Dict['Use_Mixed_Precision']:
            self.model_Dict['PWGAN'] = amp.initialize(
                models=self.model_Dict['PWGAN']
                )

        state_Dict = torch.load(
            hp_Dict['WaveNet']['Checkpoint_Path'],
            map_location= 'cpu'
            )
        self.model_Dict['PWGAN'].load_state_dict(state_Dict['Model']['Generator'])

        logging.info('PWGAN checkpoint \'{}\' loaded.'.format(hp_Dict['WaveNet']['Checkpoint_Path']))

    def Train(self):
        self.tqdm = tqdm(
            initial= self.steps,
            total= hp_Dict['Train']['Max_Step'],
            desc='[Training]'
            )
        
        hp_Path = os.path.join(hp_Dict['Checkpoint_Path'], 'Hyper_Parameter.yaml').replace('\\', '/')
        if not os.path.exists(hp_Path):
            os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)
            yaml.dump(hp_Dict, open(hp_Path, 'w'))

        if hp_Dict['Train']['Initial_Inference']:
            self.Evaluation_Epoch()
            self.Inference_Epoch()

        for model in self.model_Dict.values():
            model.train()

        while self.steps < hp_Dict['Train']['Max_Step']:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

    def Write_to_Tensorboard(self, category, loss_Dict):
        for tag, loss in loss_Dict.items():
            self.writer.add_scalar('{}/{}'.format(category, tag), loss, self.steps)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    args = argParser.parse_args()
    
    new_Trainer = Trainer(steps= args.steps)
    new_Trainer.Train()