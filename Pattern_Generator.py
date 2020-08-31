import numpy as np
import yaml, os, time, pickle, librosa, re, argparse
from concurrent.futures import ThreadPoolExecutor as PE
from collections import deque
from threading import Thread
from random import shuffle
from tqdm import tqdm

from Audio import Audio_Prep, Mel_Generate
from yin import pitch_calc

from Arg_Parser import Recursive_Parse
hp = Recursive_Parse(yaml.load(
    open('Hyper_Parameters.yaml', encoding='utf-8'),
    Loader=yaml.Loader
    ))

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
regex_Checker = re.compile('[A-Z,.?!\'\-\s]+')
top_DB_Dict = {'LJ': 60, 'BC2013': 60, 'VCTK': 15, 'VC1': 23, 'VC2': 23, 'Libri': 23, 'CMUA': 60}  # VC1 and Libri is from 'https://github.com/CorentinJ/Real-Time-Voice-Cloning'

def Text_Filtering(text):
    remove_Letter_List = ['(', ')', '\"', '[', ']', ':', ';']
    replace_List = [('  ', ' '), (' ,', ','), ('\' ', '\'')]

    text = text.upper().strip()
    for filter in remove_Letter_List:
        text= text.replace(filter, '')
    for filter, replace_STR in replace_List:
        text= text.replace(filter, replace_STR)

    text= text.strip()
    
    if len(regex_Checker.findall(text)) != 1:
        return None
    elif text.startswith('\''):
        return None
    else:
        return regex_Checker.findall(text)[0]

def Pitch_Generate(audio):
    pitch = pitch_calc(
        sig= audio,
        sr= hp.Sound.Sample_Rate,
        w_len= hp.Sound.Frame_Length,
        w_step= hp.Sound.Frame_Shift,
        confidence_threshold= hp.Sound.Confidence_Threshold,
        gaussian_smoothing_sigma = hp.Sound.Gaussian_Smoothing_Sigma
        )
    return (pitch - np.min(pitch)) / (np.max(pitch) - np.min(pitch) + 1e-7)

def Pattern_Generate(path, top_db= 60):
    audio = Audio_Prep(path, hp.Sound.Sample_Rate, top_db)
    mel = Mel_Generate(
        audio= audio,
        sample_rate= hp.Sound.Sample_Rate,
        num_frequency= hp.Sound.Spectrogram_Dim,
        num_mel= hp.Sound.Mel_Dim,
        window_length= hp.Sound.Frame_Length,
        hop_length= hp.Sound.Frame_Shift,
        mel_fmin= hp.Sound.Mel_F_Min,
        mel_fmax= hp.Sound.Mel_F_Max,
        max_abs_value= hp.Sound.Max_Abs_Mel
        )
    pitch = Pitch_Generate(audio)

    return audio, mel, pitch

def Pattern_File_Generate(path, speaker_ID, speaker, dataset, text= None, tag='', eval= False):
    pattern_Path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path

    try:
        audio, mel, pitch = Pattern_Generate(path, top_DB_Dict[dataset])
        assert mel.shape[0] == pitch.shape[0], 'Mel_shape != Pitch_shape {} != {}'.format(mel.shape, pitch.shape)
        new_Pattern_Dict = {
            'Audio': audio.astype(np.float32),
            'Mel': mel.astype(np.float32),
            'Pitch': pitch.astype(np.float32),
            'Speaker_ID': speaker_ID,
            'Speaker': speaker,
            'Dataset': dataset,
            }
        if not text is None:
            new_Pattern_Dict['Text'] = text
    except Exception as e:
        print('Error: {} in {}'.format(e, path))
        return

    file = '{}.{}{}.PICKLE'.format(
        speaker if dataset in speaker else '{}.{}'.format(dataset, speaker),
        '{}.'.format(tag) if tag != '' else '',
        os.path.splitext(os.path.basename(path))[0]
        ).upper()

    os.makedirs(os.path.join(pattern_Path, dataset).replace('\\', '/'), exist_ok= True)
    with open(os.path.join(pattern_Path, dataset, file).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=4)


def LJ_Info_Load(path, use_text= False):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_Dict = {}
    if use_text:        
        for line in open(os.path.join(path, 'metadata.csv').replace('\\', '/'), 'r', encoding= 'utf-8').readlines():
            file, _, text = line.strip().split('|')
            text = Text_Filtering(text)
            if text is None:
                continue            
            text_Dict[os.path.join(path, 'wavs', '{}.wav'.format(file)).replace('\\', '/')] = text
        
        paths = list(text_Dict.keys())

    speaker_Dict = {
        path: 'LJ'
        for path in paths
        }

    print('LJ info generated: {}'.format(len(paths)))
    return paths, text_Dict, speaker_Dict

def BC2013_Info_Load(path, use_text= False):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_Dict = {}
    if use_text:
        for path in paths:
            text = Text_Filtering(open(path.replace('wav', 'txt'), 'r').readlines()[0].strip())
            if not text is None:
                text_Dict[path] = text
        paths = list(text_Dict.keys())

    speaker_Dict = {
        path: 'BC2013'
        for path in paths
        }

    print('BC2013 info generated: {}'.format(len(paths)))
    return paths, text_Dict, speaker_Dict

def CMUA_Info_Load(path, use_text= False):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_Dict = {}
    if use_text:
        for root, _, _ in os.walk(path):
            if not os.path.exists(os.path.join(root, 'txt.done.data')):
                continue
            for line in open(os.path.join(root, 'txt.done.data'), 'r').readlines():
                file, text, _ = line.strip().split('"')
                file = file.strip().split(' ')[1]
                path = os.path.join(root.replace('etc', 'wav'), '{}.wav'.format(file)).replace('\\', '/')
                text = Text_Filtering(text)
                if not text is None:
                    text_Dict[path] = text

        paths = list(text_Dict.keys())

    speaker_Dict = {
        path: 'CMUA.{}'.format(path.split('/')[-3].split('_')[2].upper())
        for path in paths
        }

    print('CMUA info generated: {}'.format(len(paths)))
    return paths, text_Dict, speaker_Dict

def VCTK_Info_Load(path, use_text= False):
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

    text_Dict = {}
    if use_text:
        for path in paths:
            if 'p315'.upper() in path.upper():  #Officially, 'p315' text is lost in VCTK dataset.
                continue
            text = Text_Filtering(open(path.replace('wav48', 'txt').replace('wav', 'txt'), 'r').readlines()[0])
            if not text is None:
                text_Dict[path] = text
        paths = list(text_Dict.keys())

    speaker_Dict = {
        path: 'VCTK.{}'.format(path.split('/')[-2].upper())
        for path in paths
        }

    print('VCTK info generated: {}'.format(len(paths)))
    return paths, text_Dict, speaker_Dict

def Libri_Info_Load(path, use_text= False):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    text_Dict = {}
    if use_text:
        for path in paths:
            text = Text_Filtering(open('{}.normalized.txt'.format(os.path.splitext(path)[0]), 'r', encoding= 'utf-8').readlines()[0])
            if not text is None:
                text_Dict[path] = text
        paths = list(text_Dict.keys())

    speaker_Dict = {
        path: 'Libri.{:04d}'.format(int(path.split('/')[-3].upper()))
        for path in paths
        }

    print('Libri info generated: {}'.format(len(paths)))
    return paths, text_Dict, speaker_Dict

def Speaker_Index_Dict_Generate(speaker_Dict):
    return {
        speaker: index
        for index, speaker in enumerate(sorted(set(speaker_Dict.values())))
        }

def Split_Eval(paths, eval_ratio= 0.001):
    shuffle(paths)
    index = int(len(paths) * eval_ratio)
    return paths[index:], paths[:index]

def Metadata_Generate(eval= False, use_text= False):
    pattern_Path = hp.Train.Eval_Pattern.Path if eval else hp.Train_Pattern.Path
    metadata_File = hp.Train.Eval_Pattern.Metadata_File if eval else hp.Train_Pattern.Metadata_File

    new_Metadata_Dict = {
        'Spectrogram_Dim': hp.Sound.Spectrogram_Dim,
        'Mel_Dim': hp.Sound.Mel_Dim,
        'Frame_Shift': hp.Sound.Frame_Shift,
        'Frame_Length': hp.Sound.Frame_Length,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'Max_Abs_Mel': hp.Sound.Max_Abs_Mel,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Pitch_Length_Dict': {},        
        'Speaker_ID_Dict': {},
        'Speaker_Dict': {},
        'Dataset_Dict': {},
        }
    if use_text:
        new_Metadata_Dict['Text_Length_Dict'] = {}

    for root, _, files in os.walk(pattern_Path):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)

            file = os.path.join(os.path.basename(root), file).replace("\\", "/")
            try:
                if not all([
                    key in ('Audio', 'Mel', 'Pitch', 'Speaker_ID', 'Speaker', 'Dataset', 'Text' if use_text else '')
                    for key in pattern_Dict.keys()
                    ]):
                    continue
                new_Metadata_Dict['Audio_Length_Dict'][file] = pattern_Dict['Audio'].shape[0]
                new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                new_Metadata_Dict['Pitch_Length_Dict'][file] = pattern_Dict['Pitch'].shape[0]                
                new_Metadata_Dict['Speaker_ID_Dict'][file] = pattern_Dict['Speaker_ID']
                new_Metadata_Dict['Speaker_Dict'][file] = pattern_Dict['Speaker']
                new_Metadata_Dict['Dataset_Dict'][file] = pattern_Dict['Dataset']
                new_Metadata_Dict['File_List'].append(file)
                if use_text:
                    new_Metadata_Dict['Text_Length_Dict'][file] = len(pattern_Dict['Text'])
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))

    with open(os.path.join(pattern_Path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 4)

    print('Metadata generate done.')

def Token_Dict_Generate(text_Dict):
    tokens = set()
    for text in text_Dict.values():
        tokens = tokens.union(set(text))

    os.makedirs(os.path.dirname(hp.Token_Path), exist_ok= True)
    #I don't use yaml.dump in this case to sort clearly.
    yaml.dump(
        {token: index for index, token in enumerate(['<S>', '<E>'] + sorted(tokens))},
        open(hp.Token_Path, 'w')
        )
    
    # #I don't use yaml.dump in this case to sort clearly.
    # os.makedirs(os.path.dirname(hp.Token_Path), exist_ok= True)
    # open(hp.Token_Path, 'w').write('\n'.join([
    #     '\'{}\': {}'.format(token, index)
    #     for index, token in enumerate(['<S>', '<E>'] + sorted(tokens))
    #     ]))

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-lj", "--lj_path", required=False)
    argParser.add_argument("-bc2013", "--bc2013_path", required=False)
    argParser.add_argument("-cmua", "--cmua_path", required=False)
    argParser.add_argument("-vctk", "--vctk_path", required=False)
    argParser.add_argument("-libri", "--libri_path", required=False)
    
    argParser.add_argument("-text", "--use_text", action= 'store_true')
    argParser.add_argument("-eval", "--eval_ratio", default= 0.001, type= float)
    argParser.add_argument("-mw", "--max_worker", default= 10, required=False, type= int)

    args = argParser.parse_args()

    paths = []
    text_Dict = {}
    speaker_Dict = {}
    dataset_Dict = {}
    tag_Dict = {}

    if not args.lj_path is None:
        lj_Paths, lj_Text_Dict, lj_Speaker_Dict = LJ_Info_Load(path= args.lj_path, use_text= args.use_text)
        paths.extend(lj_Paths)
        text_Dict.update(lj_Text_Dict)
        speaker_Dict.update(lj_Speaker_Dict)
        dataset_Dict.update({path: 'LJ' for path in lj_Paths})
        tag_Dict.update({path: '' for path in lj_Paths})
    if not args.bc2013_path is None:
        bc2013_Paths, bc2013_Text_Dict, bc2013_Speaker_Dict = BC2013_Info_Load(path= args.bc2013_path, use_text= args.use_text)
        paths.extend(bc2013_Paths)
        text_Dict.update(bc2013_Text_Dict)
        speaker_Dict.update(bc2013_Speaker_Dict)
        dataset_Dict.update({path: 'BC2013' for path in bc2013_Paths})
        tag_Dict.update({path: '' for path in bc2013_Paths})
    if not args.cmua_path is None:
        cmua_Paths, cuma_Text_Dict, cmua_Speaker_Dict = CMUA_Info_Load(path= args.cmua_path, use_text= args.use_text)
        paths.extend(cmua_Paths)
        text_Dict.update(cuma_Text_Dict)
        speaker_Dict.update(cmua_Speaker_Dict)
        dataset_Dict.update({path: 'CMUA' for path in cmua_Paths})
        tag_Dict.update({path: '' for path in cmua_Paths})
    if not args.vctk_path is None:
        vctk_Paths, vctk_Text_Dict, vctk_Speaker_Dict = VCTK_Info_Load(path= args.vctk_path, use_text= args.use_text)
        paths.extend(vctk_Paths)
        text_Dict.update(vctk_Text_Dict)
        speaker_Dict.update(vctk_Speaker_Dict)
        dataset_Dict.update({path: 'VCTK' for path in vctk_Paths})
        tag_Dict.update({path: '' for path in vctk_Paths})
    if not args.libri_path is None:
        libri_Paths, libri_Text_Dict, libri_Speaker_Dict = Libri_Info_Load(path= args.libri_path, use_text= args.use_text)
        paths.extend(libri_Paths)
        text_Dict.update(libri_Text_Dict)
        speaker_Dict.update(libri_Speaker_Dict)
        dataset_Dict.update({path: 'Libri' for path in libri_Paths})
        tag_Dict.update({path: '' for path in libri_Paths})

    if len(paths) == 0:
        raise ValueError('Total info count must be bigger than 0.')

    if args.use_text:
        Token_Dict_Generate(text_Dict)

    speaker_Index_Dict = Speaker_Index_Dict_Generate(speaker_Dict)

    train_Paths, eval_Paths = Split_Eval(paths, args.eval_ratio)
    
    with PE(max_workers = args.max_worker) as pe:
        for _ in tqdm(            
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [
                    (
                        path,
                        speaker_Index_Dict[speaker_Dict[path]],
                        speaker_Dict[path],
                        dataset_Dict[path],
                        text_Dict[path] if args.use_text else None,
                        tag_Dict[path],
                        False
                        )
                    for path in train_Paths
                    ]
                ),
            total= len(train_Paths)
            ):
            pass
        for _ in tqdm(
            pe.map(
                lambda params: Pattern_File_Generate(*params),
                [
                    (
                        path,
                        speaker_Index_Dict[speaker_Dict[path]],
                        speaker_Dict[path],
                        dataset_Dict[path],
                        text_Dict[path] if args.use_text else None,
                        tag_Dict[path],
                        True
                        )
                    for path in eval_Paths
                    ]
                ),
            total= len(eval_Paths)
            ):
            pass

    Metadata_Generate(use_text= args.use_text)
    Metadata_Generate(eval= True, use_text= args.use_text)



# python Pattern_Generator.py -lj "D:\Pattern\ENG\LJSpeech" -bc2013 "D:\Pattern\ENG\BC2013" -cmua "D:\Pattern\ENG\CMUA" -vctk "D:\Pattern\ENG\VCTK" -libri "D:\Pattern\ENG\LibriTTS"
# python Pattern_Generator.py -lj "D:\Pattern\ENG\LJSpeech" -vctk "D:\Pattern\ENG\VCTK" -libri "D:\Pattern\ENG\LibriTTS" -text
# python Pattern_Generator.py -lj "D:\Pattern\ENG\LJSpeech" -text