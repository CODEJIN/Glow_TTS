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


import asyncio

async def f(x):
    x = await (x + 1)
    return x

futures = [f(x) for x in range(100)]
asyncio.get_event_loop().run_until_complete(asyncio.wait(futures))

