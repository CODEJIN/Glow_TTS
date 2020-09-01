import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm
from sklearn.manifold import TSNE

from train import load_model
from text import kor_text_to_phoneme, kr_phoneme_symbols
from hparams import create_hparams

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ref_mel_paths):
        super(Dataset, self).__init__()
        self.path_list = [   
            (os.path.join(root, file), dataset)
            for paths, dataset in ref_mel_paths
            for root, _, files in os.walk(paths)
            for file in files
            if os.path.splitext(file)[1].lower() == '.npy'
            ]

    def __getitem__(self, index):
        path, dataset = self.path_list[index]
        return np.load(path), dataset

    def __len__(self):
        return len(self.path_list)

class Collate:
    def __call__(self, batch):
        mels, datasets = zip(*batch)
        lengths = [mel.shape[0] for mel in mels]
        max_length = max(lengths)

        mels = np.stack([
            np.pad(mel, [[0, max_length - mel.shape[0]],[0,0]])
            for mel in mels
            ])
        
        mels = torch.FloatTensor(mels).transpose(2, 1)
        lengths = torch.LongTensor(lengths)

        return mels, lengths, datasets
        

def getColor(c, N, idx):
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    
    return cmap(norm(idx))


kr_phoneme_symbols = {key: value for key, value in enumerate(kr_phoneme_symbols.phoneme_symbols)}

hparams = create_hparams()
checkpoint_path = '/home/heejo/Documents/TMAX/Temp/tsd1_emo_gst/checkpoint_171000'
export_path = os.path.join(
    os.path.dirname(checkpoint_path),
    '{}{}'.format('R', os.path.basename(checkpoint_path).split('_')[1][:-3])
    )

ref_mel_paths= [
    ('/home/heejo/data/tsd1_prep1', 'TSD1'),
    ('/home/heejo/data/emo_data/ada_mel', 'EMO_ADA'),
    ('/home/heejo/data/emo_data/adb_mel', 'EMO_ADB'),
    ('/home/heejo/data/emo_data/adc_mel', 'EMO_ADC'),
    ('/home/heejo/data/emo_data/add_mel', 'EMO_ADD'),
    ('/home/heejo/data/emo_data/ava_mel', 'EMO_AVA'),
    ('/home/heejo/data/emo_data/avb_mel', 'EMO_AVB'),
    ('/home/heejo/data/emo_data/avc_mel', 'EMO_AVC'),
    ('/home/heejo/data/emo_data/avd_mel', 'EMO_AVD'),
    ('/home/heejo/data/emo_data/ema_mel', 'EMO_EMA'),
    ('/home/heejo/data/emo_data/emb_mel', 'EMO_EMB'),
    ('/home/heejo/data/emo_data/emf_mel', 'EMO_EMF'),
    ('/home/heejo/data/emo_data/emg_mel', 'EMO_EMG'),
    ('/home/heejo/data/emo_data/emh_mel', 'EMO_EMH'),
    ('/home/heejo/data/emo_data/lmy_mel', 'EMO_LMY'),
    ('/home/heejo/data/emo_data/nea_mel', 'EMO_NEA'),
    ('/home/heejo/data/emo_data/neb_mel', 'EMO_NEB'),
    ('/home/heejo/data/emo_data/nec_mel', 'EMO_NEC'),
    ('/home/heejo/data/emo_data/ned_mel', 'EMO_NED'),
    ('/home/heejo/data/emo_data/nee_mel', 'EMO_NEE'),
    ('/home/heejo/data/emo_data/nek_mel', 'EMO_NEK'),
    ('/home/heejo/data/emo_data/nel_mel', 'EMO_NEL'),
    ('/home/heejo/data/emo_data/nem_mel', 'EMO_NEM'),
    ('/home/heejo/data/emo_data/nen_mel', 'EMO_NEN'),
    ('/home/heejo/data/emo_data/neo_mel', 'EMO_ENO'),
    ]


model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
model = model.gst.cuda()
model.eval().half()

loader = torch.utils.data.DataLoader(
    dataset= Dataset(ref_mel_paths),
    num_workers=2,
    shuffle= False,
    batch_size=hparams.batch_size,
    pin_memory=False,    
    collate_fn=Collate()
    )




embeddings, datasets = zip(*[
    (model(mels.cuda().half(), lengths.cuda()).cpu().detach().numpy().astype(np.float32), datasets)
    for mels, lengths, datasets in tqdm(loader)
    ])

# embeddings = np.squeeze(np.vstack(embeddings), axis= 1)
embeddings = np.vstack(embeddings)
datasets = [dataset for sub in datasets for dataset in sub]


scatters = TSNE(n_components=2, random_state= 0).fit_transform(embeddings)

fig = plt.figure(figsize=(12, 12))

for index, (_, dataset) in enumerate(ref_mel_paths):
    sub_scatters =[scatter for scatter, scatter_datset in zip(scatters, datasets) if dataset == scatter_datset]
    if len(sub_scatters) == 0:
        continue
    sub_scatters = np.stack(sub_scatters)
    plt.scatter(
        sub_scatters[:, 0],
        sub_scatters[:, 1],
        c= np.array([getColor('gist_ncar', len(ref_mel_paths), index)] * len(sub_scatters)),
        label= dataset
        )
    

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(export_path, 'gst_tsne.png'))
plt.close()