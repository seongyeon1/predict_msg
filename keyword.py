### xml을 DataFrame으로 변환하기
from os import name
import xml.etree.ElementTree as ex
import pandas as pd
import bs4
from lxml import html
from urllib.parse import urlencode, quote_plus, unquote
import numpy as np

# 라이브러리 import
import requests 
import pprint

from tqdm import tqdm, trange

import re

df = pd.read_csv('msg.csv')
df.dropna(inplace=True)


from keybert import KeyBERT
tqdm.pandas()

kw_model = KeyBERT()

for i in list(df.cast.value_counts().index.tolist()):
   tmp = df[df.cast == i]
   tmp['keyword'] = tmp.msg.progress_apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(2,4), use_maxsum = True, top_n = 1))
   tmp.to_csv(f'{i}.csv', index=False)
   print(r'{i} complete')
