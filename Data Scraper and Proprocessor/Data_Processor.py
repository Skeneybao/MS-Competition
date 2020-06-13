import json
import nltk
import numpy as np
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.util import ngrams
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
import re
from nltk.stem.snowball import SnowballStemmer
from gensim.parsing.preprocessing import STOPWORDS
import gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Iterable

class Data_Processor:

    def __init__(self, start_month='2010-06', end_month='2020-06', template="/Users/ethan_bao/Wealth_Management",
                 tokenizer=TweetTokenizer(strip_handles=True, reduce_len=True), stemmer=nltk.stem.porter.PorterStemmer,
                 lemma=nltk.wordnet.WordNetLemmatizer):
        self._S_m = start_month
        self._E_m = end_month
        self._D_list = self.DateList()
        self._Dir = template
        self.Data = []
        self._text = []
        self._Lens = []
        self._tokenizer = tokenizer
        self._stemmer = stemmer()
        self._lemma = lemma()
        self._stopwords = set(nltk.corpus.stopwords.words('english')).union(
            set(['http', 'via', 'ha', 'We', 'I', 'make', 'today', 'A', 'the', 'http', 'one','This','LLC','Inc']))
        self._unigrams=[]

    def DateList(self):
        start_year = int(self._S_m[:4])
        start_month = int(self._S_m[-2:])
        end_year = int(self._E_m[:4])
        end_month = int(self._E_m[-2:])
        if start_year == end_year:
            month_range = range(start_month, end_month + 1)
            date_list = ["{year}-{month:0=2d}".format(year=str(start_year), month=M) for M in month_range]
            return date_list
        year_range = range(start_year + 1, end_year)
        start_year_month_range = range(start_month, 13)
        end_year_month_range = range(1, end_month + 1)
        date_list = ["{year}-{month:0=2d}".format(year=str(start_year), month=M) for M in start_year_month_range]
        date_list += ["{year}-{month:0=2d}".format(year=str(Y), month=M) for Y in year_range for M in range(1,13)]
        date_list += ["{year}-{month:0=2d}".format(year=str(end_year), month=M) for M in end_year_month_range]
        return date_list

    def ReadData(self):
        for date in self._D_list:
            with open("{dir}{D}.json".format(dir=self._Dir, D=date), "r") as read_file:
                foo = "self.data" + date[:4] + date[5:7]
                exec(foo + " = json.load(read_file)")
                exec('self._M_data.append(' + foo + ')')
        for i in self._M_data:
            self._Lens.append(len(i))
        self._text = [[_['text'] for _ in D] for D in self._M_data]

    def DataNums(self):
        return self._Lens, sum(self._Lens)

    def Specify_Lang(self, lang='"en"'):
        self._M_data = [[_ for _ in D if
                         _['text_html'][
                         _['text_html'].find('lang='):_['text_html'].find('lang=') + 9] == 'lang=' + lang]
                        for D in self._M_data]
        self._Lens = []
        for i in self._M_data:
            self._Lens.append(len(i))

    @property
    def Data(self):
        return self._M_data

    @Data.setter
    def Data(self,data):
        self._M_data=data
        self._Lens=[]
        for i in self._M_data:
            self._Lens.append(len(i))
        self._text = [[_['text'] for _ in D] for D in self._M_data]

    def TextData(self):
        return self._text

    def GetNGrams(self, num, lemma=False, stem=False):
        return [[self._extract_ngrams(_, num, lemma, stem) for _ in M] for M in self._text]

    def GetFreq(self, ngrams):
        words_freq = {}
        for month in ngrams:
            for text in month:
                for token in text:
                    if token not in words_freq.keys():
                        words_freq[token] = 1
                    else:
                        words_freq[token] += 1
        words_freq_list = list(words_freq.items())
        words_freq_list = sorted(words_freq_list, key=lambda x: x[1], reverse=True)
        return words_freq_list

    def WordCloud(self, ngrams,Width=3200,Height=1600,Max_Font_Size=160,Max_Words=1000):
        sentence=' '.join([' '.join([' '.join(T) for T in month]) for month in ngrams])
        wordcloud = WordCloud(width=Width, height=Height, max_font_size=Max_Font_Size, stopwords=STOPWORDS, max_words=Max_Words).generate(
            sentence)
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bicubic')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    def AddStopwords(self,words):
        addon=[i for i in words]
        self._stopwords+=addon

    def Clean(self):
        self._remove_link()
        self._remove_sign()

    def TFIDF(self,ngrams):
        T = [[' '.join(T) for T in month] for month in ngrams]
        R = []
        for i in T:
            R += i
        tfidf = TfidfVectorizer(min_df=0.005, max_df=0.5, ngram_range=(1, 2))
        features = tfidf.fit_transform(R)
        DF = pd.DataFrame(
            features.todense(),
            columns=tfidf.get_feature_names()
        )
        return DF

    def _remove_link(self):
        for month in self._M_data:
            for twt in month:
                if '/' in twt['text']:
                    result = re.sub(r"(https|http)\S+", "", twt['text'])
                    # result=re.sub(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b','',i['text'])
                    result = re.sub(r"\S+.com\S+", "", result)
                    twt['text'] = result
        self._text = [[_['text'] for _ in D] for D in self._M_data]

    def _remove_sign(self):
        for month in self._M_data:
            for twt in month:
                #hash tag
                result = re.sub(r'#\S+', '', twt['text'])
                result=re.sub(r'.ealth .anagement','',result)
                # user mention
                # result=re.sub(r'#\S+', '', result)
                # emoji
                result = re.sub(r'[^\x00-\x7F]+', '', result)
                # html tags
                result = re.sub(r'<.*?>', '', result)
                # extra spaces
                result = re.sub(r' +', ' ', result)
                # punctuation
                from string import punctuation as punc
                result = re.sub('[{}]'.format(punc), '', result)
                result=''.join([i for i in result if i.isnumeric()==False])
                twt['text'] = result
        self._text = [[_['text'] for _ in D] for D in self._M_data]

    def _extract_ngrams(self, data, num, lemma=False, stem=False):
        tokens = self._tokenizer.tokenize(data)
        token_lower = [token.lower() for token in tokens if token not in self._stopwords]
        token_stop = [token for token in token_lower if token not in self._stopwords]

        if stem:
            token_stop_stem = [self._stemmer.stem(token) for token in token_stop]
            token_stop_stem_alnum = [word for word in token_stop_stem if word.isalnum()]
            n_grams = ngrams(token_stop_stem_alnum, num)

        if lemma:
            token_stop_lemma = [self._lemma.lemmatize(token) for token in token_stop]
            token_stop_lemma_alnum = [word for word in token_stop_lemma if word.isalnum()]
            n_grams = ngrams(token_stop_lemma_alnum, num)

        return [''.join(grams) for grams in n_grams]