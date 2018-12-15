import pandas as pd
from nltk import word_tokenize
import nltk
from string import punctuation

from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier

stopwords = nltk.corpus.stopwords.words('portuguese')

def pre_processar(noticia):
    global vectorizer
    
    txt = noticia
    
    return' '.join([t for t in word_tokenize(txt.lower()) if (t not in stopwords) and (t not in punctuation) and (not t.isdigit())])

def calcular_noticia(noticia):
    df = pd.read_csv('data/noticias.csv')
    df = df.append(pd.DataFrame({"noticia": pre_processar(noticia)}, index=[0]), ignore_index=True)
    
    vectorizer = TfidfVectorizer(max_features=20000)

    tfidf_matrix = vectorizer.fit_transform(df['noticia'])
    X_train = tfidf_matrix[:-1]
    Y_train = df['target'][:-1]
    X_test = tfidf_matrix[-1:]
    
    # Modelo
    m = XGBClassifier(
        n_jobs=4,
        random_state=1
    )
    
    # Gerando a previsão para a linha de teste
    m.fit(
        X_train,
        Y_train
    )
    
    return m.predict_proba(X_test)[0,1]