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

import pickle
from scipy.io import wavfile
import numpy as np

x = pickle.load(open("C:\Pattern\GlowTTS.SR16K.Pattern\Train\LJ\LJ.LJ001-0012.PICKLE", 'rb'))

wavfile.write(
    filename= 'D:/t.wav',
    data= (np.clip(x['Audio'], -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
    rate= 16000
    )

import matplotlib.pyplot as plt
plt.imshow(x['Mel'].T, aspect='auto', origin='lower')
plt.colorbar()
plt.show()

np.save('D:/t.npy', x['Mel'], False)