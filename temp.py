# import os
# from itertools import cycle
# from shutil import copy2

# path = 'D:/GoogleDrive/Colab_Test/GlowTTS/Results/SR16K.Result.LJ/Inference/Step-305000/WAV/IDX_{}.WAV'

# texts = [
#     'Birds of a feather flock together.',
#     'A creative artist works on his next composition because he was not satisfied with his previous one.',
#     'Death is like a fisherman who catches fish in his net and leaves them for a while in the water. The fish is still swimming but the net is around him, and the fisherman will draw him up.',
#     'Where do I come from and where are I going.',
#     'Where do I come from and where are I going?',
#     'Where do I come from and where are I going!'
#     ]

# os.makedirs('D:/Teetetetetetete', exist_ok= True)
# for index, text in enumerate(texts * 5):
#     alpha = 0.2 * (index % 5) + 0.6
#     text_Index = index % len(texts)
#     print(alpha, text_Index)
#     copy2(
#         path.format(index),
#         'D:/Teetetetetetete/LS_{:.1f}.T_{}.WAV'.format(alpha, text_Index)
#         )
    
    

    



# t_align = torch.randn(3, 43, 277)
# a = torch.cat((torch.ones((t_align.shape[1], t_align.shape[0], 1), device=t_align.device)*-500, t_align.transpose(1, 0)), dim=2)

# target = list()
# max_mel_len = int(max(x_lengths))
# for length_mel_row in x_lengths:
#     target.append(np.pad([i+1 for i in range(int(length_mel_row))],
#                         (0, max_mel_len-int(length_mel_row)), 'constant', constant_values=int(length_mel_row)))
# target = np.array(target)

# target = torch.from_numpy(target).to(t_align.device)


# print(a.shape)
# print(target.shape)


# import keyword
# import torch
# meta = []
# while len(meta)<1000:
#     meta = meta+keyword.kwlist # get some strings
# meta = meta[:1000]
# print(meta)

# from tensorboardX import SummaryWriter
# writer = SummaryWriter("my_experiment")
# # writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img, global_step=0)
# # writer.add_embedding(torch.randn(100, 5), label_img=label_img, global_step=0)
# writer.add_embedding(torch.randn(1000, 128), metadata=meta, global_step=0)
# writer.flush()
# writer.add_embedding(torch.randn(1000, 128), metadata=meta, global_step=10000)
# writer.flush()
# writer.close()

import pickle

path = 'C:/Pattern/24K.Pattern.LJCMUA/Eval/METADATA.PICKLE'
x = pickle.load(open(path, 'rb'))
print(x)