import numpy as np
import yaml, os, pickle, librosa, argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as PE
from threading import Thread
from random import shuffle

from Audio import Audio_Prep, Mel_Generate
from yin import pitch_calc

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
top_DB_Dict = {'VCTK': 15, 'VC1': 23, 'VC2': 23, 'Libri': 23, 'CMUA': 60}  # VC1 and Libri is from 'https://github.com/CorentinJ/Real-Time-Voice-Cloning'

def Pitch_Generate(audio):
    pitch = pitch_calc(
        sig= audio,
        sr= hp_Dict['Sound']['Sample_Rate'],
        w_len= hp_Dict['Sound']['Frame_Length'],
        w_step= hp_Dict['Sound']['Frame_Shift'],
        confidence_threshold= hp_Dict['Sound']['Confidence_Threshold'],
        gaussian_smoothing_sigma = hp_Dict['Sound']['Gaussian_Smoothing_Sigma']
        )
    return (pitch - np.min(pitch)) / (np.max(pitch) - np.min(pitch) + 1e-7)

def Pattern_Generate(path, top_db= 15):
    audio = Audio_Prep(path, hp_Dict['Sound']['Sample_Rate'], top_db)
    mel = Mel_Generate(
        audio= audio,
        sample_rate= hp_Dict['Sound']['Sample_Rate'],
        num_frequency= hp_Dict['Sound']['Spectrogram_Dim'],
        num_mel= hp_Dict['Sound']['Mel_Dim'],
        window_length= hp_Dict['Sound']['Frame_Length'],
        hop_length= hp_Dict['Sound']['Frame_Shift'],        
        mel_fmin= hp_Dict['Sound']['Mel_F_Min'],
        mel_fmax= hp_Dict['Sound']['Mel_F_Max'],
        max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
        )
    pitch = Pitch_Generate(audio)

    return audio, mel, pitch

def Pattern_File_Generate(path, speaker_Index, speaker, dataset, tag= '', eval= False):
    pattern_Path = hp_Dict['Train']['Eval_Pattern' if eval else 'Train_Pattern']['Path']

    pickle_Path = '{}.{}{}.{}.PICKLE'.format(
        dataset,
        speaker,
        '.{}'.format(tag) if tag != '' else tag,
        os.path.splitext(os.path.basename(path))[0]        
        )
    pickle_Path = os.path.join(pattern_Path, dataset, pickle_Path).replace("\\", "/")

    if os.path.exists(pickle_Path):
        return

    os.makedirs(os.path.join(pattern_Path, dataset).replace('\\', '/'), exist_ok= True)    
    try:
        audio, mel, pitch = Pattern_Generate(path, top_DB_Dict[dataset])
        assert mel.shape[0] == pitch.shape[0], 'Mel_shape != Pitch_shape {} != {}'.format(mel.shape, pitch.shape)
        new_Pattern_Dict = {
            'Audio': audio,
            'Mel': mel,
            'Pitch': pitch,
            'Speaker_Index': speaker_Index,
            'Speaker': speaker,
            'Dataset': dataset,
            }
    except Exception as e:
        print('Error: {} in {}'.format(e, path))
        return

    with open(pickle_Path, 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=4)
            
def VCTK_Info_Load(vctk_Path, num_Speakers):
    vctk_Wav_Path = os.path.join(vctk_Path, 'wav48').replace('\\', '/')
    try:
        with open(os.path.join(vctk_Path, 'VCTK.NonOutlier.txt').replace('\\', '/'), 'r') as f:
            vctk_Non_Outlier_List = [x.strip() for x in f.readlines()]
    except:
        vctk_Non_Outlier_List = None

    vctk_File_Path_List = []
    for root, _, files in os.walk(vctk_Wav_Path):
        for file in files:
            if not vctk_Non_Outlier_List is None and not file in vctk_Non_Outlier_List:
                continue
            wav_File_Path = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue

            vctk_File_Path_List.append(wav_File_Path)
    
    vctk_Speaker_Dict = {
        path: path.split('/')[-2].upper()
        for path in vctk_File_Path_List
        }
    
    # speakers = list(set(vctk_Speaker_Dict.values()))[:num_Speakers]
    speakers = [x for x in list(set(vctk_Speaker_Dict.values())) if not x in ['P243', 'P240', 'P232', 'P277', 'P228', 'P226']]
    speakers = ['P243'] + speakers + ['P232', 'P277', 'P240']
    print(speakers)
    print(len(speakers))
    vctk_File_Path_List = [path for path in vctk_File_Path_List if vctk_Speaker_Dict[path] in speakers]    
    vctk_Speaker_Dict = {path: speaker for path, speaker in vctk_Speaker_Dict.items() if speaker in speakers}
    speaker_Index_Dict = {speaker: index for index, speaker in enumerate(speakers)}

    print('VCTK info generated: {}'.format(len(vctk_File_Path_List)))
    return vctk_File_Path_List, vctk_Speaker_Dict, speaker_Index_Dict

def Split_Eval(path_List, speaker_Dict, eval_Rate):
    path_List_Dict = {spekaer: [] for spekaer in set(speaker_Dict.values())}
    for path in path_List:
        path_List_Dict[speaker_Dict[path]].append(path)
    for speaker in path_List_Dict.keys():
        shuffle(path_List_Dict[speaker])
    
    train_Path_List_Dict = {
        speaker: paths[max(1, int(len(paths) * eval_Rate)):]
        for speaker, paths in path_List_Dict.items()
        }
    eval_Path_List_Dict = {
        speaker: paths[:max(1, int(len(paths) * eval_Rate))]
        for speaker, paths in path_List_Dict.items()
        }

    train_Paths = [path for paths in train_Path_List_Dict.values() for path in paths]
    eval_Paths = [path for paths in eval_Path_List_Dict.values() for path in paths]

    return train_Paths, eval_Paths

def Metadata_Generate(eval= False):
    pattern_Path = hp_Dict['Train']['Eval_Pattern' if eval else 'Train_Pattern']['Path']

    new_Metadata_Dict = {
        'Spectrogram_Dim': hp_Dict['Sound']['Spectrogram_Dim'],
        'Mel_Dim': hp_Dict['Sound']['Mel_Dim'],
        'Frame_Shift': hp_Dict['Sound']['Frame_Shift'],
        'Frame_Length': hp_Dict['Sound']['Frame_Length'],
        'Sample_Rate': hp_Dict['Sound']['Sample_Rate'],
        'Max_Abs_Mel': hp_Dict['Sound']['Max_Abs_Mel'],        
        'File_List': [],        
        'Mel_Length_Dict': {},
        'Pitch_Length_Dict': {},
        'Dataset_Dict': {},
        'Speaker_Dict': {},        
        'File_List_by_Speaker_Dict': {},
        }

    speaker_Index_Set = set()
    for root, _, files in os.walk(pattern_Path):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)
                try:
                    if not all([key in ('Audio', 'Mel', 'Pitch', 'Speaker_Index', 'Speaker', 'Dataset') for key in pattern_Dict.keys()]):
                        continue
                    new_Metadata_Dict['File_List'].append(file)
                    new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                    new_Metadata_Dict['Pitch_Length_Dict'][file] = pattern_Dict['Pitch'].shape[0]
                    new_Metadata_Dict['Dataset_Dict'][file] = pattern_Dict['Dataset']
                    new_Metadata_Dict['Speaker_Dict'][file] = pattern_Dict['Speaker']
                    if not (pattern_Dict['Dataset'], pattern_Dict['Speaker']) in new_Metadata_Dict['File_List_by_Speaker_Dict'].keys():
                        new_Metadata_Dict['File_List_by_Speaker_Dict'][pattern_Dict['Dataset'], pattern_Dict['Speaker']] = []
                    new_Metadata_Dict['File_List_by_Speaker_Dict'][pattern_Dict['Dataset'], pattern_Dict['Speaker']].append(file)

                    speaker_Index_Set.add((pattern_Dict['Speaker'], pattern_Dict['Speaker_Index']))
                except:
                    print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
                
    new_Metadata_Dict['Speaker_Index_Dict'] = {speaker: index for speaker, index in speaker_Index_Set}

    with open(os.path.join(pattern_Path, hp_Dict['Train']['Train_Pattern']['Metadata_File'].upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol=4)

    print('Metadata generate done.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--vctk_path", required=True)
    argParser.add_argument("-s", "--num_speaker", required=True, type= int)
    argParser.add_argument("-e", "--eval_rate", required=True, type= float)
    argParser.add_argument("-mw", "--max_worker", default= 10, type= int)
    args = argParser.parse_args()
    
    path_List, speaker_Dict, speaker_Index_Dict = VCTK_Info_Load(vctk_Path= args.vctk_path, num_Speakers= args.num_speaker)
    dataset_Dict = {path: 'VCTK' for path in path_List}
    tag_Dict = {path: '' for path in path_List}

    train_Path_List, eval_Path_List = Split_Eval(path_List, speaker_Dict, args.eval_rate)

    with PE(max_workers = args.max_worker) as pe:
        for _ in tqdm(
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [(path, speaker_Index_Dict[speaker_Dict[path]], speaker_Dict[path], dataset_Dict[path], tag_Dict[path], False) for path in train_Path_List]
                ),
            total= len(train_Path_List)
            ):
            pass
        for _ in tqdm(
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [(path, speaker_Index_Dict[speaker_Dict[path]], speaker_Dict[path], dataset_Dict[path], tag_Dict[path], True) for path in eval_Path_List]
                ),
            total= len(eval_Path_List)
            ):
            pass

    Metadata_Generate()
    Metadata_Generate(eval= True)