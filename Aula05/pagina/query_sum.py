import os
from nltk import word_tokenize
import nltk
from string import punctuation
from collections import Counter
import pandas as pd

FILE = "data/buscas.csv"

stopwords = nltk.corpus.stopwords.words('portuguese')
numbers = '0123456789'

def pre_processar(texto):
    texto = ' '.join([t for t in word_tokenize(texto.lower()) if (t not in stopwords) and (t not in punctuation + numbers)])
    return Counter(word_tokenize(texto))

def contar_busca(query):
    if os.path.exists(FILE):
        df = pd.read_csv(FILE)
        dic = pre_processar(query)
        for word in dic.keys():
            if word in df:
                df[word] = df[word] + 1
            else:
                df[word] = 1
        df.to_csv(FILE, index=False)
    else:
        dic = pre_processar(query)
        df = pd.DataFrame(dic, index=['0',])
        df.to_csv(FILE, index=False)

def top_10_buscas():
    if os.path.exists(FILE):
        df = pd.read_csv(FILE)
        dic = df.to_dict(orient='split')
        dic = dict(zip(dic['columns'], dic['data'][0]))
        return sorted(dic.items(), key=lambda kv: kv[1], reverse=True)
    return []