import json
import nltk
import numpy as np
from gensim.models import Word2Vec
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.util import ngrams
import matplotlib
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
from collections import Iterable,Counter



class Data_Processor:

    def __init__(self, start_month='2010-06', end_month='2020-06', template=["/Users/ethan_bao/Wealth_Management"],
                 tokenizer=TweetTokenizer(strip_handles=True, reduce_len=True), stemmer=nltk.stem.porter.PorterStemmer(),
                 lemma=nltk.wordnet.WordNetLemmatizer(),model_name='D'):
        self._S_m = start_month
        self._E_m = end_month
        self._template=template
        self._Dir = template
        self.data = []
        self._sentwords=['yes','sorry','thank']
        self._text = {}
        self._Lens = []
        self._tokenizer = tokenizer
        self._stemmer = stemmer
        self._topicmodel=gensim.models.Word2Vec.load(model_name) if model_name else None
        self._lemma = lemma
        self._stopwords = (set(nltk.corpus.stopwords.words('english'))).union(
            set(['http', 'via', 'ha', 'We', 'I', 'make', 'today', 'A', 'the', 'http', 'one', 'This', 'LLC', 'Inc']))
        self._unigrams = []
        self._raw_data=[]
        self._users={}
        self._unigrams=None
        self.dlist = self.datelist()
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
                        'Allstocknews','sheilaballarano','beastsaver','Isabel1170','Vanguard_Group','NoMoreBenjamins','IBKR','goinhouse','AskRobinhood','TopTickrs',
                        'RobinhoodApp','TowelieTrades','RobinHoodNation','RobinhoodPromo','robinhoodportf1','RobinhoodClass','ClassRobinhood','RobinhoodDT',
                        'RobinHoodPlay','robintrack','EzekialBone','cofoundertron','TradeStation','GreekGodTrading','ClockworkAlerts','RandomFour','Ellevest',
                        'StyleSalute_com','EllevateNtwk','EllevateCHS','EllevateSAN','EllevatePIT','EllevateLA','netguru','Berry_Blooom','brennan_bloom',
                        'UlyssesReader','Bloom_Nobly','ArtyPetals','Sara_Bloom_s_','bloom_brenda','marvellebrooks','Milly_Tshivhase','peach_blooom',
                        'SONIC_BLOOM','Gawgeous_bloom','Lia_in_bloom','Real_Tess_Bloom']



    def weeklydatelist(self):
        start_year = int(self._S_m[:4])
        start_month = int(self._S_m[-2:])
        end_year = int(self._E_m[:4])
        end_month = int(self._E_m[-2:])
        if start_year == end_year:
            month_range = range(start_month, end_month + 1)
            date_list = ["{year}-{month:0=2d}-{week}".format(year=str(start_year), month=M,week=w) for M in month_range for w in (1,2,3,4)]
            return date_list
        year_range = range(start_year + 1, end_year)
        start_year_month_range = range(start_month, 13)
        end_year_month_range = range(1, end_month + 1)
        date_list = ["{year}-{month:0=2d}-{week}".format(year=str(start_year), month=M,week=w) for M in start_year_month_range for w in (1,2,3,4)]
        date_list += ["{year}-{month:0=2d}-{week}".format(year=str(Y), month=M,week=w) for Y in year_range for M in range(1, 13) for w in (1,2,3,4)]
        date_list += ["{year}-{month:0=2d}-{week}".format(year=str(end_year), month=M,week=w) for M in end_year_month_range for w in (1,2,3,4)]
        return date_list

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
        date_list = ["{year}-{month:0=2d}".format(year=str(start_year), month=M) for M in start_year_month_range ]
        date_list += ["{year}-{month:0=2d}".format(year=str(Y), month=M) for Y in year_range for M in range(1, 13)]
        date_list += ["{year}-{month:0=2d}".format(year=str(end_year), month=M) for M in end_year_month_range]
        return date_list

    def gettopic(self,key,counts=1,threshold=20):
        keys = [_[0] for _ in self._topicmodel.wv.most_similar(key, topn=threshold)] + self._sentwords
        unigrams=self.getngrams(num=1,lemma=True)
        topic_data = []
        for m,m_data in enumerate(unigrams):
            current_topic_data = []
            for i,i_data in enumerate(m_data):
                count = 0
                for token in unigrams[m][i]:
                    if token in keys:
                        if count >= counts:
                            current_topic_data.append(self.data[m][i])
                            break
                        else:
                            count += 1
            topic_data.append(current_topic_data)
        return topic_data

    def sentperct(self,month):
        idx = self.dlist.index(month)
        count = Counter([(item['positive'], item['negative']) for item in self.data[idx]])
        for i in sorted(count.items(), key=lambda x: -x[1]):
            print(i[0],i[1]/len(self.data[idx]),i[1])



    def trend_analysis(self,keyword_list,polarity=0,count=1,ma=None,simple=False,title=None,ynorm=False,weighted=False,reverse=False):
        if not isinstance(polarity,Iterable):
            polarity=[polarity]
        data=self._M_data
        all_months=self.dlist
        app = self._subanalyze_sentiment(keyword_list, count,weighted=weighted)
        for pos in polarity:
            if pos==2:
                line=(np.array(app[0])+np.array(app[1]))/2
            elif pos==3:
                line=(np.array(app[0])-np.array(app[1]))/2
            else:
                line=np.array(app[pos])
                if pos==1:
                    line-=1
                if pos==0:
                    line+=2
            if ynorm:
                line=(line-np.min(line))/(np.max(line)-np.min(line))
            if reverse:
                line=1-line
            if ma:
                supline=np.zeros(len(all_months))
                subline=np.zeros(len(all_months))
                average = np.zeros(len(all_months))
                for i, m in enumerate(all_months):
                    if i == 0:
                        average[i] =line[i]
                        supline[i]=average[i]
                        subline[i]=average[i]
                    elif i<ma:
                        average[i]=np.mean(line[:i+1])
                        supline[i]=average[i]+np.std(line[:i+1])
                        subline[i]=average[i]-np.std(line[:i+1])
                    else:
                        average[i]=np.mean(line[i-ma+1:i+1])
                        supline[i]=average[i]+np.std(line[i-ma+1:i+1],ddof=1)
                        subline[i]=average[i]-np.std(line[i-ma+1:i+1],ddof=1)

            matplotlib.rcParams['figure.dpi'] = 100
            fig, ax1 = plt.subplots(figsize=(7, 4))
            if not simple:
                ax1.plot(np.array(all_months), line, color='C' + str(pos + 1))
            if ma:
                ax1.plot(np.array(all_months), average, 'plum')
                if not simple:
                    ax1.plot(np.array(all_months), supline, 'b--')
                    ax1.plot(np.array(all_months), subline, 'r--')
            ax1.set_xticks(np.arange(36))
            ax1.set_xticklabels([i if x % 3 == 0 else '' for x, i in enumerate(all_months)], rotation=45)
            ax1.set_ylabel('Sentiment Polarity')
            for t in ax1.xaxis.get_ticklines():
                t.set_visible(False)
            for t in ax1.xaxis.get_ticklines()[::2][::3]:
                t.set_visible(True)
            ax2 = ax1.twinx()
            for t in ax2.xaxis.get_ticklines():
                t.set_visible(False)
            for t in ax2.xaxis.get_ticklines()[::2][::3]:
                t.set_visible(True)
            ax2.set_xticks(np.arange(36))
            ax2.set_xticklabels([i if x % 3 == 0 else '' for x, i in enumerate(all_months)], rotation=45)
            ax2.bar(all_months, app[2], alpha=0.3)
            ax2.set_ylabel('Sentiment Frequency')
            extra = '...' if len(keyword_list) > 5 else ''
            if title:
                ax1.set_title(title)
            else:
                ax1.set_title('Keywords: ' + ', '.join(keyword_list[:5]) + extra)
            plt.show()

    def show_tweets(self,month_list, keywords, count=1, threshold=2, unique=False):
        if not isinstance(threshold,Iterable):
            threshold=[threshold,threshold]
        result = []
        for m in month_list:
            idx = self.dlist.index(m)
            for item in self._M_data[idx]:
                if sum([int(k in item['text'].lower()) for k in keywords]) >= count and (
                        item['positive'] > threshold[0] or item['negative'] < -threshold[1]):
                    result.append((item['text'], item['positive'], item['negative'],item['tweet_url']))

        return sorted(list(set(result)), key=lambda x: x[0]) if unique else sorted(result, key=lambda x: x[0])

    def readdata(self,weekly=False):
        self.dlist = self.datelist() if not weekly else self.weeklydatelist()
        self._M_data=[]
        for date in self.dlist:
            month = []
            for key in self._Dir:
                with open("{dir}{D}.json".format(dir=key, D=date), "r") as read_file:
                    temp = json.load(read_file)[:-1]
                    month += temp
            for i in month:
                if 'positive' not in i.keys():
                    break
                elif i['positive']>=-i['negative']*1.5:
                    i['sentiment']=1
                elif i['negative']<=-i['positive']*1.5:
                    i['sentiment']=-1
                else:i['sentiment']=0
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
        self._stopwords = self._stopwords.union(set(addon))

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

    def getkeyworddata(self,key_word,text=False):
        data=[]
        for m in self._M_data:
            month_data=[]
            for i in m:
                if key_word in i['text'].lower():
                    if text:month_data.append(i['text'])
                    else:month_data.append(i)
            data.append(month_data)

        return data

    def tokenizetext(self, Lemma=True):
        for i in range(len(self._M_data)):
            month = self.dlist[i]
            data = self._M_data[i]
            for j in data:
                j['text'] = ' '.join(self.getngrams(data=j['text'], num=1, lemma=Lemma))
            self._M_data[i] = data
        self._recalc()

    def _subanalyze_sentiment(self,keywords, count,weighted=False):  # data hierarchy: months*(text,score)*piece
        data=self._M_data
        positivity = []
        negativity = []
        freq = []
        for month_data in data:
            pos, neg, num = 0, 0, 0

            for item in month_data:
                if sum([int(k.lower() in item['text'].lower()) for k in keywords]) >= count:
                    if weighted:
                        pos += item['positive']*(np.log(2+item['replies']))
                        neg += item['negative']*(np.log(2+item['replies']))
                        num += (np.log(2+item['replies']))
                    else:
                        pos += item['positive']
                        neg += item['negative']
                        num += 1  # save number of tweets containing the filtered words with at least #count times

            positivity.append(pos / max(1, num))
            negativity.append(neg / max(1, num))
            freq.append(num)

        return negativity, positivity, freq

    def _remove_similar(self):
        new_data=[]
        current_text =set()
        for m in self._M_data:
            current_data=[]
            for i in m:
                if i['text'] not in current_text:
                    current_text.union(i['text'])
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
            self._text[self.dlist[i]] = [_['text'] for _ in self._M_data[i]]
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

def process(dirlists,start_month='2017-06',end_month='2020-05'):
    result={}
    for i,dirlist in enumerate(dirlists):
        result[i]=Data_Processor(start_month=start_month,end_month=end_month,
                  template=dirlist)
        result[i].readdata()
        result[i].readdata()
        result[i].specifylang()
        result[i].removenoise()
        result[i].clean()
    return result

def percentage(tweetresult):
    dict={}
    l=len(tweetresult)
    for item in tweetresult:
        if (item[1],item[2]) not in dict.keys():
            dict[(item[1],item[2])] = 1
        else:
            dict[(item[1], item[2])] += 1
    items=dict.items()
    items=sorted(items,key=lambda x:x[1],reverse=True)
    res=[list(i)+[i[1]/l] for i in items]
    return res