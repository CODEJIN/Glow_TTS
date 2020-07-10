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
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
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

        self.Datset_Generate()
        self.Model_Generate()

        self.loss_Dict = {
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
            'SpeechSplit': SpeechSplit().to(device)
            }
        self.criterion_Dict = {
            'MSE': torch.nn.MSELoss().to(device)
            }
        self.optimizer = torch.optim.Adam(
            params= self.model_Dict['SpeechSplit'].parameters(),
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
            self.model_Dict['SpeechSplit'], self.optimizer = amp.initialize(
                models=self.model_Dict['SpeechSplit'],
                optimizers=self.optimizer
                )

        logging.info(self.model_Dict['SpeechSplit'])


    def Train_Step(self, speakers, mels, pitches, factors):
        loss_Dict = {}

        speakers = speakers.to(device)
        mels = mels.to(device)
        pitches = pitches.to(device)

        reconstructions = self.model_Dict['SpeechSplit'](
            rhymes= mels,
            contents= mels,
            pitches= pitches,
            speakers= speakers,
            random_resampling_factors= factors
            )

        loss_Dict['Loss'] = self.criterion_Dict['MSE'](mels, reconstructions)

        self.optimizer.zero_grad()
        if hp_Dict['Use_Mixed_Precision']:            
            with amp.scale_loss(loss_Dict['Loss'], self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_Dict['Loss'].backward()        
        torch.nn.utils.clip_grad_norm_(
            parameters= self.model_Dict['SpeechSplit'].parameters(),
            max_norm= hp_Dict['Train']['Gradient_Norm']
            )
        self.optimizer.step()
        self.scheduler.step()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.loss_Dict['Train'][tag] += loss

    def Train_Epoch(self):
        for speakers, mels, pitches, factors in self.dataLoader_Dict['Train']:
            self.Train_Step(speakers, mels, pitches, factors)
            
            if self.steps % hp_Dict['Train']['Checkpoint_Save_Interval'] == 0:
                self.Save_Checkpoint()

            if self.steps % hp_Dict['Train']['Logging_Interval'] == 0:
                self.loss_Dict['Train'] = {
                    tag: loss / hp_Dict['Train']['Logging_Interval']
                    for tag, loss in self.loss_Dict['Train'].items()
                    }
                self.Write_to_Tensorboard('Train', self.loss_Dict['Train'])
                self.loss_Dict['Train'] = defaultdict(float)

            if self.steps % hp_Dict['Train']['Evaluation_Interval'] == 0:
                self.Evaluation_Epoch()

            if self.steps % hp_Dict['Train']['Inference_Interval'] == 0:
                self.Inference_Epoch()
            
            if self.steps >= hp_Dict['Train']['Max_Step']:
                return

        self.epochs += hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']

    @torch.no_grad()
    def Evaluation_Step(self, speakers, mels, pitches, factors):
        loss_Dict = {}

        speakers = speakers.to(device)
        mels = mels.to(device)
        pitches = pitches.to(device)

        reconstructions = self.model_Dict['SpeechSplit'](
            rhymes= mels,
            contents= mels,
            pitches= pitches,
            speakers= speakers,
            random_resampling_factors= factors
            )

        loss_Dict['Loss'] = self.criterion_Dict['MSE'](mels, reconstructions)

        for tag, loss in loss_Dict.items():
            self.loss_Dict['Evaluation'][tag] += loss
    
    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (speakers, mels, pitches, factors) in tqdm(
            enumerate(self.dataLoader_Dict['Dev'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataLoader_Dict['Dev'].dataset) / hp_Dict['Train']['Batch_Size'])
            ):
            self.Evaluation_Step(speakers, mels, pitches, factors)

        self.loss_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.loss_Dict['Evaluation'].items()
            }
        self.Write_to_Tensorboard('Evaluation', self.loss_Dict['Evaluation'])
        self.loss_Dict['Evaluation'] = defaultdict(float)

        for model in self.model_Dict.values():
            model.train()


    @torch.no_grad()
    def Inference_Step(self, speakers, rhymes, contents, pitches, rhyme_Labels, content_Labels, pitch_Labels, lengths, start_Index= 0, tag_Step= False, tag_Index= False):
        speakers = speakers.to(device)
        rhymes = rhymes.to(device)
        contents = contents.to(device)
        pitches = pitches.to(device)

        reconstructions = self.model_Dict['SpeechSplit'](
            rhymes= rhymes,
            contents= contents,
            pitches= pitches,
            speakers= speakers
            )

        
        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'PNG').replace("\\", "/"), exist_ok= True)
        for index, (speaker, rhyme, content, pitch, reconstruction, rhyme_Label, content_Label, pitch_Label, length) in enumerate(zip(
            speakers.cpu().numpy(),
            rhymes.cpu().numpy(),
            contents.cpu().numpy(),
            pitches.cpu().numpy(),
            reconstructions.cpu().numpy(),
            rhyme_Labels,
            content_Labels,
            pitch_Labels,
            lengths
            )):
            title = 'Converted_Speaker: {}    Rhyme: {}    Content: {}    Pitch: {}'.format(speaker, rhyme_Label, content_Label, pitch_Label)            
            new_Figure = plt.figure(figsize=(20, 5 * 4), dpi=100)
            plt.subplot(411)
            plt.imshow(rhyme[:, :length], aspect='auto', origin='lower')
            plt.title('Rhyme mel    {}'.format(title))
            plt.colorbar()
            plt.subplot(412)
            plt.imshow(content[:, :length], aspect='auto', origin='lower')
            plt.title('Content mel    {}'.format(title))
            plt.colorbar()
            plt.subplot(413)
            plt.plot(pitch[:length])
            plt.margins(x=0)
            plt.title('Pitch    {}'.format(title))
            plt.colorbar()
            plt.subplot(414)
            plt.imshow(reconstruction[:, :length], aspect='auto', origin='lower')
            plt.title('Reconstruction mel    {}'.format(title))
            plt.colorbar()
            plt.tight_layout()
            file = '{}S_{}.R_{}.C_{}.P_{}{}.PNG'.format(
                'Step-{}.'.format(self.steps) if tag_Step else '',
                speaker,
                rhyme_Label,
                content_Label,
                pitch_Label,
                '.IDX_{}'.format(index + start_Index) if tag_Index else ''
                )
            plt.savefig(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'PNG', file).replace("\\", "/"))
            plt.close(new_Figure)

        if 'PWGAN' in self.model_Dict.keys():
            os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'WAV').replace("\\", "/"), exist_ok= True)

            noises = torch.randn(reconstructions.size(0), reconstructions.size(2) * hp_Dict['Sound']['Frame_Shift']).to(device)
            reconstructions = torch.nn.functional.pad(
                reconstructions,
                pad= (hp_Dict['WaveNet']['Upsample']['Pad'], hp_Dict['WaveNet']['Upsample']['Pad']),
                mode= 'replicate'
                )

            for index, (audio, speaker, rhyme_Label, content_Label, pitch_Label, length) in enumerate(zip(
                self.model_Dict['PWGAN'](noises, reconstructions).cpu().numpy(),
                speakers.cpu().numpy(),
                rhyme_Labels,
                content_Labels,
                pitch_Labels,
                lengths
                )):
                file = '{}S_{}.R_{}.C_{}.P_{}{}.WAV'.format(
                    'Step-{}.'.format(self.steps) if tag_Step else '',
                    speaker,
                    rhyme_Label,
                    content_Label,
                    pitch_Label,
                    '.IDX_{}'.format(index + start_Index) if tag_Index else ''
                    )
                wavfile.write(
                    filename= os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'WAV', file).replace("\\", "/"),
                    data= (np.clip(audio[:length * hp_Dict['Sound']['Frame_Shift']], -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
                    rate= hp_Dict['Sound']['Sample_Rate']
                    )
        else:
            os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'NPY').replace("\\", "/"), exist_ok= True)

            for index, (reconstruction, speaker, rhyme_Label, content_Label, pitch_Label, length) in enumerate(zip(
                reconstructions.cpu().numpy(),
                speakers.cpu().numpy(),
                rhyme_Labels,
                content_Labels,
                pitch_Labels,
                lengths
                )):
                file = '{}S_{}.R_{}.C_{}.P_{}{}.NPY'.format(
                    'Step-{}.'.format(self.steps) if tag_Step else '',
                    speaker,
                    rhyme_Label,
                    content_Label,
                    pitch_Label,
                    '.IDX_{}'.format(index + start_Index) if tag_Index else ''
                    )
                np.save(
                    os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'NPY', file).replace("\\", "/"),
                    reconstruction,
                    allow_pickle= False
                    )

    def Inference_Epoch(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (speakers, rhymes, contents, pitches, rhyme_Labels, content_Labels, pitch_Labels, lengths) in tqdm(
            enumerate(self.dataLoader_Dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataLoader_Dict['Inference'].dataset) / hp_Dict['Train']['Batch_Size'])
            ):
            self.Inference_Step(speakers, rhymes, contents, pitches, rhyme_Labels, content_Labels, pitch_Labels, lengths, start_Index= step * hp_Dict['Train']['Batch_Size'])

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
        self.model_Dict['SpeechSplit'].load_state_dict(state_Dict['Model']['SpeechSplit'])
        self.optimizer.load_state_dict(state_Dict['Optimizer'])
        self.scheduler.load_state_dict(state_Dict['Scheduler'])
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        if hp_Dict['Use_Mixed_Precision']:
            if not 'AMP' in state_Dict.keys():
                logging.info('No AMP state dict is in the checkpoint. Model regards this checkpoint is trained without mixed precision.')
            else:                
                amp.load_state_dict(state_Dict['AMP'])

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)

        state_Dict = {
            'Model': {
                'SpeechSplit': self.model_Dict['SpeechSplit'].state_dict()
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