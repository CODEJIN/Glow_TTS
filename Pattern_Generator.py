import numpy as np
import yaml, os, time, pickle, librosa, re, argparse
from concurrent.futures import ThreadPoolExecutor as PE
from collections import deque
from threading import Thread
from random import shuffle
from tqdm import tqdm

from Audio import Audio_Prep, Mel_Generate

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
top_DB_Dict = {'LJ': 60, 'BC2013': 60, 'VCTK': 15, 'VC1': 23, 'VC2': 23, 'Libri': 23, 'CMUA': 60}  # VC1 and Libri is from 'https://github.com/CorentinJ/Real-Time-Voice-Cloning'

def Pattern_Generate(path, top_db= 60):
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

    return audio, mel

def Pattern_File_Generate(path, speaker_ID, speaker, dataset, tag=''):
    audio, mel = Pattern_Generate(path, top_DB_Dict[dataset])
    
    new_Pattern_Dict = {
        'Audio': audio.astype(np.float32),
        'Mel': mel.astype(np.float32),
        'Speaker_ID': speaker_ID,
        'Speaker': speaker,
        'Dataset': dataset,
        }

    file = '{}.{}{}.PICKLE'.format(
        speaker if dataset in speaker else '{}.{}'.format(dataset, speaker),
        '{}.'.format(tag) if tag != '' else '',
        os.path.splitext(os.path.basename(path))[0]
        ).upper()

    os.makedirs(os.path.join(hp_Dict['Train']['Pattern_Path'], dataset).replace('\\', '/'), exist_ok= True)    
    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], dataset, file).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=4)


def LJ_Info_Load(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)
            
    speaker_Dict = {
        path: 'LJ'
        for path in paths
        }

    print('LJ info generated: {}'.format(len(paths)))
    return paths, speaker_Dict

def BC2013_Info_Load(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    speaker_Dict = {
        path: 'BC2013'
        for path in paths
        }

    print('BC2013 info generated: {}'.format(len(paths)))
    return paths, speaker_Dict

def CMUA_Info_Load(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)
            
    speaker_Dict = {
        path: 'CMUA.{}'.format(path.split('/')[-3].split('_')[2].upper())
        for path in paths
        }

    print('CMUA info generated: {}'.format(len(paths)))
    return paths, speaker_Dict

def VCTK_Info_Load(path):
    path = os.path.join(path, 'wav48').replace('\\', '/')
    try:
        with open(os.path.join(path, 'VCTK.NonOutlier.txt').replace('\\', '/'), 'r') as f:
            vctk_Non_Outlier_List = [x.strip() for x in f.readlines()]
    except:
        vctk_Non_Outlier_List = None

    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if not vctk_Non_Outlier_List is None and not file in vctk_Non_Outlier_List:
                continue
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue

            paths.append(file)

    speaker_Dict = {
        path: 'VCTK.{}'.format(path.split('/')[-2].upper())
        for path in paths
        }

    print('VCTK info generated: {}'.format(len(paths)))
    return paths, speaker_Dict

def Libri_Info_Load(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)
            
    speaker_Dict = {
        path: 'Libri.{:04d}'.format(int(path.split('/')[-3].upper()))
        for path in paths
        }

    print('Libri info generated: {}'.format(len(paths)))
    return paths, speaker_Dict

def Speaker_Index_Dict_Generate(speaker_Dict):
    return {
        speaker: index
        for index, speaker in enumerate(sorted(set(speaker_Dict.values())))
        }


def Metadata_Generate():
    new_Metadata_Dict = {
        'Spectrogram_Dim': hp_Dict['Sound']['Spectrogram_Dim'],
        'Mel_Dim': hp_Dict['Sound']['Mel_Dim'],
        'Frame_Shift': hp_Dict['Sound']['Frame_Shift'],
        'Frame_Length': hp_Dict['Sound']['Frame_Length'],
        'Sample_Rate': hp_Dict['Sound']['Sample_Rate'],
        'Max_Abs_Mel': hp_Dict['Sound']['Max_Abs_Mel'],
        'File_List': [],
        'Audio_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Speaker_ID_Dict': {},
        'Speaker_Dict': {},
        'Dataset_Dict': {},
        }

    for root, _, files in os.walk(hp_Dict['Train']['Pattern_Path']):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)

            file = os.path.join(os.path.basename(root), file).replace("\\", "/")
            try:
                if not all([key in ('Audio', 'Mel', 'Speaker_ID', 'Speaker', 'Dataset') for key in pattern_Dict.keys()]):
                    continue
                new_Metadata_Dict['Audio_Length_Dict'][file] = pattern_Dict['Audio'].shape[0]
                new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                new_Metadata_Dict['Speaker_ID_Dict'][file] = pattern_Dict['Speaker_ID']
                new_Metadata_Dict['Speaker_Dict'][file] = pattern_Dict['Speaker']
                new_Metadata_Dict['Dataset_Dict'][file] = pattern_Dict['Dataset']
                new_Metadata_Dict['File_List'].append(file)
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))

    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File'].upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 4)

    print('Metadata generate done.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-lj", "--lj_path", required=False)
    argParser.add_argument("-bc2013", "--bc2013_path", required=False)
    argParser.add_argument("-cmua", "--cmua_path", required=False)
    argParser.add_argument("-vctk", "--vctk_path", required=False)
    argParser.add_argument("-libri", "--libri_path", required=False)
    
    argParser.add_argument("-mw", "--max_worker", default= 10, required=False, type= int)

    args = argParser.parse_args()

    path_List = []
    speaker_Dict = {}
    dataset_Dict = {}
    tag_Dict = {}

    if not args.lj_path is None:
        lj_Paths, lj_Speaker_Dict = LJ_Info_Load(path= args.lj_path)
        path_List.extend(lj_Paths)
        speaker_Dict.update(lj_Speaker_Dict)
        dataset_Dict.update({path: 'LJ' for path in lj_Paths})
        tag_Dict.update({path: '' for path in lj_Paths})
    if not args.bc2013_path is None:
        bc2013_Paths, bc2013_Speaker_Dict = BC2013_Info_Load(path= args.bc2013_path)
        path_List.extend(bc2013_Paths)
        speaker_Dict.update(bc2013_Speaker_Dict)
        dataset_Dict.update({path: 'BC2013' for path in bc2013_Paths})
        tag_Dict.update({path: '' for path in bc2013_Paths})        
    if not args.cmua_path is None:
        cmua_Paths, cmua_Speaker_Dict = CMUA_Info_Load(path= args.cmua_path)
        path_List.extend(cmua_Paths)
        speaker_Dict.update(cmua_Speaker_Dict)
        dataset_Dict.update({path: 'CMUA' for path in cmua_Paths})
        tag_Dict.update({path: '' for path in cmua_Paths})        
    if not args.vctk_path is None:
        vctk_Paths, vctk_Speaker_Dict = VCTK_Info_Load(path= args.vctk_path)
        path_List.extend(vctk_Paths)
        speaker_Dict.update(vctk_Speaker_Dict)
        dataset_Dict.update({path: 'VCTK' for path in vctk_Paths})
        tag_Dict.update({path: '' for path in vctk_Paths})
    if not args.libri_path is None:
        libri_Paths, libri_Speaker_Dict = Libri_Info_Load(path= args.libri_path)
        path_List.extend(libri_Paths)
        speaker_Dict.update(libri_Speaker_Dict)
        dataset_Dict.update({path: 'Libri' for path in libri_Paths})
        tag_Dict.update({path: '' for path in libri_Paths})
    

    if len(path_List) == 0:
        raise ValueError('Total info count must be bigger than 0.')
    
    speaker_Index_Dict = Speaker_Index_Dict_Generate(speaker_Dict)

    with PE(max_workers = args.max_worker) as pe:
        for _ in tqdm(
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [(path, speaker_Index_Dict[speaker_Dict[path]], speaker_Dict[path], dataset_Dict[path], tag_Dict[path]) for path in path_List]
                ),
            total= len(path_List)
            ):
            pass

    Metadata_Generate()



# python Pattern_Generator.py -lj "D:\Pattern\ENG\LJSpeech" -bc2013 "D:\Pattern\ENG\BC2013" -cmua "D:\Pattern\ENG\CMUA" -vctk "D:\Pattern\ENG\VCTK" -libri "D:\Pattern\ENG\LibriTTS"
# python Pattern_Generator.py -vctk "D:\Pattern\ENG\VCTK" -libri "D:\Pattern\ENG\LibriTTS"
