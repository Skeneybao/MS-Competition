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
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Iterable



class Data_Processor:

    def __init__(self, start_month='2010-06', end_month='2020-06', template=["/Users/ethan_bao/Wealth_Management"],
                 tokenizer=TweetTokenizer(strip_handles=True, reduce_len=True), stemmer=nltk.stem.porter.PorterStemmer(),
                 lemma=nltk.wordnet.WordNetLemmatizer()):
        self._S_m = start_month
        self._E_m = end_month
        self._template=template
        self._D_list = self.datelist()
        self._Dir = template
        self.data = []
        self._text = {}
        self._Lens = []
        self._tokenizer = tokenizer
        self._stemmer = stemmer
        self._lemma = lemma
        self._stopwords = set(nltk.corpus.stopwords.words('english')).union(
            set(['http', 'via', 'ha', 'We', 'I', 'make', 'today', 'A', 'the', 'http', 'one', 'This', 'LLC', 'Inc']))
        self._unigrams = []
        self._raw_data=[]
        self._users={}
        self._ban_list=['GoldmanSachBOT','UBS','UBScareers','vandaplas1','ubscenter','UBSf1','UBSglobalart','UBS_France','UBSschweiz','UBSvisionaries',
                        'UBSathletics','ubs_digital','roboadvisorpros','AskRobinhood','UBSOffice','PersonalCapital','WellsFargoGolf','IN2ecosystem','WFAssetMgmt'
                        ,'WellsFargoJobs','WFInvesting','WellsFargoCtr','WFB_Fraud','Ask_WellsFargo','WellsFargo','MorganStanley','GoldmanSachs','Shareworks','ArnoldRKellyms',
                        'kmac_onjohn','stephfinebot','optimaoptionstw','infoguy411','keithcarron','TrumpIdeasBot','Wealthfront','wltheng','WealthfrontJobs',
                        'arachleff','TDAmeritrade','CharlesSchwab','truthsearch1957','VGaykin','Noalpha_allbeta','Noalpha_allbeta','TDAmeritradePR','stdojo',
                        'LJames_TDA','_RobTrader_','bankedits','SIFMAFoundation','etrade','Roxann_Minerals','tmj_CA_finance','tmj_MN_finance','tmj_MI_finance','tmj_sea_adv',
                        'tmj_NH_finance','tmj_OR_finance','BROKERHUNTERcom','tmj_VA_finance','goinhouse','tmj_AZ_finance','tmj_WA_finance','tmj_IL_finance','tmj_KY_finance',
                        'tmj_nct_cstsrv','tmj_NJ_finance','FidelityNews','tmj_FL_finance','tmj_MD_finance','tmj_nwk_cler','tmj_OH_finance','tmj_nwk_acct','tmj_WI_finance',
                        'tmj_MO_finance','tmj_IN_finance','MerrillLynch','Allstocknews','miroslavpitak','jpmorgan','JPMorganAM','wosaikeneriki','qslzpidbenams',
                        'daviddo43706820','reurope_stock','Chrisgebb','BankofAmerica','BofA_News','BofA_Tips','BofAPrivateBank','BofA_Business','BofA_Careers','BofA_Help',
                        'Allstocknews','sheilaballarano','beastsaver','Isabel1170','Vanguard_Group','NoMoreBenjamins']

    def datelist(self):
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
        date_list += ["{year}-{month:0=2d}".format(year=str(Y), month=M) for Y in year_range for M in range(1, 13)]
        date_list += ["{year}-{month:0=2d}".format(year=str(end_year), month=M) for M in end_year_month_range]
        return date_list

    def readdata(self):
        for date in self._D_list:
            month = []
            for key in self._Dir:
                with open("{dir}{D}.json".format(dir=key, D=date), "r") as read_file:
                    temp = json.load(read_file)
                    month += temp
            self._M_data.append(month)
        self._recalc()

    def userdata(self,screenname):
        data=[]
        for m in self.data:
            for i in m:
                if i['screen_name'] == screenname:
                    data.append(i)
        return data

    def datanums(self):
        return self._Lens, sum(self._Lens)

    def specifylang(self, lang='"en"'):
        self._M_data = [[_ for _ in D if
                         _['text_html'][
                         _['text_html'].find('lang='):_['text_html'].find('lang=') + 9] == 'lang=' + lang]
                        for D in self._M_data]
        self._recalc()

    @property
    def data(self):
        return self._M_data

    @data.setter
    def data(self, data):
        self._M_data = data
        self._recalc()

    def users(self):
        return self._users

    def textdata(self):
        return self._text

    def getngrams(self, data=float('inf'), num=1, lemma=True, stem=False):
        if data == float('inf'):
            return [[self._extract_ngrams(_, num, lemma, stem) for _ in M] for nonuse, M in self._text.items()]
        else:
            return self._extract_ngrams(data, num, lemma, stem)

    def getfreq(self, ngrams):
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

    def wordcld(self, ngrams, Width=3200, Height=1600, Max_Font_Size=160, Max_Words=1000):
        sentence = ' '.join([' '.join([' '.join(T) for T in month]) for month in ngrams])
        wordcloud = WordCloud(width=Width, height=Height, max_font_size=Max_Font_Size, stopwords=STOPWORDS,
                              max_words=Max_Words).generate(
            sentence)
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bicubic')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    def addstopwords(self, words):
        addon = [i for i in words]
        self._stopwords += addon

    def removenoise(self):
        temp = []
        other = []
        for data in self._M_data:
            temp.append([])
            other.append([])
            for i in data:
                if i['retweets'] > 0 or i['likes'] > 0 or i['is_replied'] or i['is_reply_to'] or i['replies'] or (not i['links']):
                    temp[-1].append(i)
                else:
                    other[-1].append(i)
        self._M_data = temp
        self._recalc()
        return other

    def clean(self):
        self._remove_link()
        self._remove_sign()
        self._dejob()
        self._remove_dup()
        self._remove_users()
        self._remove_similar()
        self._recalc()
        self._raw_data=self._M_data

    def tfidf(self, ngrams):
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

    def tokenizetext(self, Lemma=True):
        for i in range(len(self._M_data)):
            month = self._D_list[i]
            data = self._M_data[i]
            for j in data:
                j['text'] = ' '.join(self.getngrams(data=j['text'], num=1, lemma=Lemma))
            self._M_data[i] = data
        self._recalc()


    def _remove_similar(self):
        new_data=[]
        for m in self._M_data:
            current_data=[]
            current_text=[]
            for i in m:
                if i['text'] not in current_text:
                    current_text.append(i['text'])
                    current_data.append(i)
            new_data.append(current_data)
        self._M_data=new_data

    def _remove_link(self):
        for month in self._M_data:
            for twt in month:
                if '/' in twt['text']:
                    result = re.sub(r"(https|http)\S+", "", twt['text'])
                    # result=re.sub(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b','',i['text'])
                    result = re.sub(r"\S+\.com\S+", "", result)
                    twt['text'] = result

    def _recalc(self):
        self._Lens = []
        for i in self._M_data:
            self._Lens.append(len(i))
        self._text = {}
        for i in range(len(self._M_data)):
            self._text[self._D_list[i]] = [_['text'] for _ in self._M_data[i]]
        self._users={}
        for m in self._M_data:
            for i in m:
                if i['screen_name'] in self._users.keys():
                    self._users[i['screen_name']]+=1
                else:
                    self._users[i['screen_name']]=1
        self._users =list(sorted(self._users.items(),key=lambda x:x[1],reverse=True))

    def _dejob(self):
        self._M_data = [[_ for _ in D if
                         'job' not in _['username'].lower()]
                        for D in self._M_data]

    def _remove_sign(self):
        for month in self._M_data:
            for twt in month:
                # hash tag
                result = re.sub(r'#\S+', '', twt['text'])
                result = re.sub(r'.ealth .anagement', '', result)
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
                result = re.sub(r'\s+', ' ', result)
                result = ''.join([i for i in result if i.isnumeric() == False])
                twt['text'] = result

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

    def _remove_users(self):
        newdata=[]
        for m in self._M_data:
            cdata=[]
            for i in m:
                if i['screen_name'] not in self._ban_list:
                    cdata.append(i)
            newdata.append(cdata)
        self._M_data=newdata


    def _remove_dup(self):

        D = self._M_data
        newdata=[]
        for m in D:
            current_set= set()
            current_data=[]
            for i in m:
                if i['tweet_id'] in current_set:
                    continue
                else:
                    current_data.append(i)
                    current_set.add(i['tweet_id'])
            newdata.append(current_data)
        self._M_data = D


def clean(Data):
    def C(string):
        result = re.sub(r"(https|http)\S+", "", string)
        result = re.sub(r"\S+\.com\S+", "", result)
        result = re.sub(r'#\S+', '', result)
        result = re.sub(r'.ealth .anagement', '', result)
        result = re.sub(r'[^\x00-\x7F]+', '', result)
        result = re.sub(r'<.*?>', '', result)
        result = re.sub(r' +', ' ', result)
        from string import punctuation as punc
        result = re.sub('[{}]'.format(punc), '', result)
        result = ''.join([i for i in result if i.isnumeric() == False])
        return result

    if isinstance(Data[0], str):
        for i in range(len(Data)):
            Data[i] = C(Data[i])
    else:
        for m in range(len(Data)):
            for i in range(len(Data[m])):
                Data[m][i] = C(Data[m][i])

    return Data
