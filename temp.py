# import re

# regex_Checker = re.compile('[A-Z,.?!\'\-\s]+')

# def Text_Filtering(text):
#     remove_Letter_List = ['(', ')', '\"', '[', ']', ':', ';']
#     replace_List = [('  ', ' '), (' ,', ','), ('\' ', '\'')]

#     text = text.upper().strip()
#     for filter in remove_Letter_List:
#         text= text.replace(filter, '')
#     for filter, replace_STR in replace_List:
#         text= text.replace(filter, replace_STR)

#     text= text.strip()

#     if len(regex_Checker.findall(text)) > 1:
#         return None
#     elif text.startswith('\''):
#         return None
#     else:
#         return regex_Checker.findall(text)[0]



# tokens = set()
# for line in open("C:\Pattern\LJSpeech\metadata.csv", 'r', encoding= 'utf-8').readlines():
#     text = Text_Filtering(line.strip().split('|')[2])    
#     if text is None:
#         continue
#     tokens = tokens.union(set(text))

# print(sorted(tokens))

# import pickle
# from scipy.io import wavfile
# import numpy as np

# x = pickle.load(open("C:\Pattern\GlowTTS.SR16K.Pattern\Train\LJ\LJ.LJ001-0012.PICKLE", 'rb'))

# wavfile.write(
#     filename= 'D:/t.wav',
#     data= (np.clip(x['Audio'], -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
#     rate= 16000
#     )

# import matplotlib.pyplot as plt
# plt.imshow(x['Mel'].T, aspect='auto', origin='lower')
# plt.colorbar()
# plt.show()

# np.save('D:/t.npy', x['Mel'], False)

import torch
import math
import torch.nn.functional as F

relative_postion_clipping_distance = window_size = 4
calc_channels_per_head = 192 // 2

def Pad(x, pad, mode='constant', value= 0):
    return torch.nn.functional.pad(
        input= x,
        pad= [size for sizes in pad[::-1] for size in sizes],
        mode= mode,
        value= value
        )

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def Get_Relative_Embedding(relative_embeddings, length):
    embedding_Length = 2 * length - 1

    pads = max(embedding_Length - relative_postion_clipping_distance * 2 - 1, 0) / 2
    relative_embeddings = Pad(
        relative_embeddings,
        [[0, 0], [math.ceil(pads), math.floor(pads)], [0, 0]]
        )
    index = max(relative_postion_clipping_distance + 1 - length, 0)
    return relative_embeddings[:, index: index + embedding_Length]

def Relative_Position_to_Absolute_Position(x):
    batches, heads, length, _ = x.size()
    
    x = Pad(x, [[0, 0], [0, 0], [0, 0], [0, 1]])    # [Batch, Head, Time, Time * 2]
    x = x.view(batches, heads, length * length * 2)     # [Batch, Head, Time * Time * 2]
    x = Pad(x, [[0, 0], [0, 0], [0, length - 1]])   # [Batch, Head, Time * Time * 2 + Time - 1]
    x = x.view(batches, heads, length + 1, length * 2 - 1)  # [Batch, Head, Time + 1, Time * 2 - 1]
    
    return x[:, :, :length, length - 1:]    # [Batch, Head, Time, Time]

def Absolute_Position_to_Relative_Position(x):
    batches, heads, length, _ = x.size()    # Because this is for self-attention, Time == Key_t == Query_t
    
    x = Pad(x, [[0, 0], [0, 0], [0, 0], [0, length - 1]])    # [Batch, Head, Time, Time * 2 - 1]
    x = x.view(batches, heads, length * (length * 2 - 1))     # [Batch, Head, Time * (Time * 2 - 1)]
    x = Pad(x, [[0, 0], [0, 0], [length, 0]])   # [Batch, Head, Time * Time * 2]
    x = x.view(batches, heads, length, length * 2)  # [Batch, Head, Time, Time * 2]
    
    return x[:, :, :, 1:]    # [Batch, Head, Time, Time * 2 - 1]



def _get_relative_embeddings(relative_embeddings, length):
    max_relative_position = 2 * window_size + 1
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (window_size + 1), 0)
    slice_start_position = max((window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
          relative_embeddings,
          convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
    return used_relative_embeddings

def _matmul_with_relative_keys(x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

def _relative_position_to_absolute_position(x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, convert_pad_shape([[0,0],[0,0],[0,length-1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    return x_final

def _absolute_position_to_relative_position(x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.view([batch, heads, length**2 + length*(length -1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
    return x_final

def _matmul_with_relative_values(x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret


if __name__ == "__main__":
    length= 100
    queries = torch.randn(4, 2, length, calc_channels_per_head)
    keys = torch.randn(4, 2, length, calc_channels_per_head)
    values = torch.randn(4, 2, length, calc_channels_per_head)
    scores = queries @ keys.transpose(3, 2) / math.sqrt(calc_channels_per_head)

    weight_k = torch.randn(1, relative_postion_clipping_distance * 2 + 1, calc_channels_per_head)
    weight_v = torch.randn(1, relative_postion_clipping_distance * 2 + 1, calc_channels_per_head)

    y1 = Get_Relative_Embedding(weight_k, length)
    y1 = queries @ y1.unsqueeze(0).transpose(3, 2)
    y1 = Relative_Position_to_Absolute_Position(y1)
    y1 = scores + y1 / math.sqrt(calc_channels_per_head)
    alignments = torch.nn.functional.softmax(y1, dim= -1)    # [Batch, Head, Query_t, Key_t]
    attentions = y1 @ values    # [Batch, Head, Query_t, Key_t] @ [Batch, Head, Key_t, Channel // Head] -> [Batch, Head, Query_t, Channel // Head]

    y = Absolute_Position_to_Relative_Position(alignments)    
    ye = Get_Relative_Embedding(weight_v, length)

    y = attentions + y @ ye.unsqueeze(0)

    y1 = y.transpose(3, 2).contiguous().view(4, 192, length)
    y2 = y.transpose(3, 2).contiguous().view(4, 192, length)

    

    print(torch.all(y1 == y2))