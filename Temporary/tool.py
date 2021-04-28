#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import re
import matplotlib

from nltk.stem.snowball import SnowballStemmer
import nltk
import gensim
import spacy

import matplotlib.cm as cm
import os

stemmer = SnowballStemmer("english")
nlp = spacy.load("en_core_web_sm")


# In[ ]:

def unique(x):
    d={}
    for i in x:
        d[i]=d.get(i,0)+1
    total=len(x)
    result=[]
    for i in d.items():
        result.append(i+(round(i[1]/max(total,1)*100,2),))
    return sorted(result,key=lambda x: -x[1])

def find_kol(data,rank):
    user=[]
    for i in data:
        user.append(i['screen_name'])
    user_d={}
    for i in user:
        user_d[i]=user_d.get(i,0)+1
    user_frequency=unique(user)
    kol_id=[i[0] for i in user_frequency[:rank]]
    kol_name=[]
    for j in kol_id:
        for i in data:
            if i['screen_name']==j:
                kol_name.append(i['screen_name'])
                break
    return kol_name

def unique_time(data):
    time=[]
    for i in data:
        time.append(i['timestamp'][:10])
    return sorted(unique(time),key=lambda x: x[0])

def exforeign(data):
    eng_data=[]
    for i in data:
        pos=i['text_html'].find('>')
        if 'lang="en"' in i['text_html'][:pos]:
            eng_data.append(i)
    return eng_data

def duplicate(data,key):
    result=[]
    id_=[]
    for i in data:
        if i[key] not in id_:
            result.append(i)
            id_.append(i[key])
    return result

def remove_link(data):
    for x,i in enumerate(data):
        result = i['text']
        for j in i['img_urls']:
            result = result.replace(j,'')
        for j in i['links']:
            result = result.replace(j,'')
        #print(result)
        if '/' in i['text']:
            result = re.sub(r"(https|http)\S+", "", i['text'])
            #result=re.sub(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b','',i['text'])
            result = re.sub(r"\S+.com\S+", "", result)
            #data[x]['text']=result
        data[x]['text']=result
    return data

def remove_sign(data):
    for x,i in enumerate(data):
        # hash tag
        result=re.sub(r'#\S+', '', i['text'])
        # user mention
        result=re.sub(r'@\S+', '', result)
        # emoji
        result=re.sub(r'[^\x00-\x7F]+', '', result)
        # html tags
        result=re.sub(r'<.*?>', '', result)
        # punctuation
        from string import punctuation as punc
        result=re.sub('[{}]'.format(punc), '', result)
        # extra spaces
        result=re.sub(r'\s+', ' ', result)
        data[x]['text']=result
    return data

def remove_rubbish(data):
    rubbish=set(['GoldmanSachBOT','UBS','UBScareers','vandaplas1','ubscenter','UBSf1','UBSglobalart','UBS_France','UBSschweiz','UBSvisionaries',
                        'UBSathletics','ubs_digital','roboadvisorpros','AskRobinhood','UBSOffice','PersonalCapital','WellsFargoGolf','IN2ecosystem','WFAssetMgmt'
                        ,'WellsFargoJobs','WFInvesting','WellsFargoCtr','WFB_Fraud','Ask_WellsFargo','WellsFargo','MorganStanley','GoldmanSachs','Shareworks','ArnoldRKellyms',
                        'kmac_onjohn','stephfinebot','optimaoptionstw','infoguy411','keithcarron','TrumpIdeasBot','Wealthfront','wltheng','WealthfrontJobs',
                        'arachleff','TDAmeritrade','CharlesSchwab','truthsearch1957','VGaykin','Noalpha_allbeta','Noalpha_allbeta','TDAmeritradePR','stdojo',
                        'LJames_TDA','_RobTrader_','bankedits','SIFMAFoundation','NoMoreBenjamins'])
    rubbish=rubbish.union({'BColwell_Invest','No1isHomeless',})
    result=[]
    for i in data:
        if i['screen_name'] not in rubbish:
            result.append(i)
    return result

def interaction(data):
    result=[]
    for i in data:
        if i['likes'] or i['retweets'] or i['is_replied'] or i['is_reply_to'] or (not i['links']):
            result.append(i)
    return result
    
def total_clean(indata):
    data=interaction(indata)
    data=exforeign(data)
    data=duplicate(data,'tweet_id')
    data=remove_link(data)
    data=remove_sign(data)
    data=remove_rubbish(data)
    data=duplicate(data,'text')
    return data
        
def toword(data,stem=False,lemma=False,stop=True,otherwords={}):
    import gensim
    result=[]
    t=nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
    f=(stemmer.stem) if (stem) else (lambda x: x)
    STOPWORDS= set(nltk.corpus.stopwords.words('english')).union(otherwords) if stop else set()
    l=nltk.wordnet.WordNetLemmatizer() if lemma else (lambda x: x)
    for x,i in enumerate(data):
        temp=[f(j).lower() for j in t.tokenize(i['text']) if j.isalnum() and (f(j).lower() not in STOPWORDS)]
        temp=[l.lemmatize(i) for i in temp]
        result.append(temp)
    return result#stemmer.stem(j)

def month(start,end):
    temp=[]
    for i in range(int(start[:4]),int(end[:4])+1):
        for j in range(1,13):
            if i == int(start[:4]) and j < int(start[5:]):
                continue
            elif i == int(end[:4]) and j > int(end[5:]):
                break
            else:
                temp.append('{}-{:02d}'.format(i,j))
    return temp

def search(data,keywords):
    result=[]
    for i in data:
        for j in keywords:
            if j in i['text'].lower():
                result.append(i)
                break
    return result

def text(data):
    a=sorted([i['text'] for i in data])
    print(len(a))
    return a

def extreme(data,threshold=4):
    result=[]
    for i in data:
        if i['positive']>=threshold or i['negative']<=-threshold:
            result.append(i)
    return result

def LDA(data, topics, words,otherwords):
    import gensim
    from gensim import corpora
    from gensim.models.ldamodel import LdaModel
    texts=toword(data,stem=False,lemma=True,stop=True,otherwords=otherwords)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    lda = LdaModel(corpus,id2word=dictionary,num_topics=topics,passes=20)
    temp=lda.print_topics(num_words=words)
    return [i[1] for i in temp]

def NMF(data,topics,words,otherwords):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    string=[' '.join(i) for i in toword(data,stem=False,lemma=True,stop=True,otherwords=otherwords)]
    import gensim
    STOPWORDS= set(nltk.corpus.stopwords.words('english')).union(otherwords)
    tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words=STOPWORDS)
    doc_term_matrix = tfidf_vect.fit_transform(string)
    nmf = NMF(n_components=topics)
    nmf.fit(doc_term_matrix)
    
    return [' + '.join(["\"{}\"".format(tfidf_vect.get_feature_names()[i]) for i in topic.argsort()[-words:]]) for topic in nmf.components_]
# In[ ]:

'''
def get_words(url):
    import requests
    words = requests.get(url).content.decode('latin-1')
    word_list = words.split('\n')
    index = 0
    while index < len(word_list):
        word = word_list[index]
        if ';' in word or not word:
            word_list.pop(index)
        else:
            index+=1
    return word_list
p_url = 'http://ptrckprry.com/course/ssd/data/positive-words.txt'
n_url = 'http://ptrckprry.com/course/ssd/data/negative-words.txt'
pos=set(get_words(p_url))
neg=set(get_words(n_url))
opinion=pos.union(neg)
'''

######
def subanalyze(data,keywords=['']): # data hierarchy: months*(text,score)*piece
    pospolarity=[]
    negpolarity=[]
    freq=[]
    for k in data:
        pos=0
        neg=0
        like=0
        count=0
        for i in k:
            for j in keywords:
                if j in i['text'].lower():
                    #pos+=i['positive']*(1+i['likes'])
                    #neg+=i['negative']*(1+i['likes'])
                    pos+=i['positive']
                    neg+=i['negative']
                    like+=1+i['likes']
                    count+=1
                    break
        pospolarity.append(pos/max(1,count))
        negpolarity.append(neg/max(1,count))
        freq.append(count)
    return pospolarity,negpolarity,freq

def double(data,keyword_list=['']):
    app=subanalyze(data,keyword_list)
    matplotlib.rcParams['figure.dpi'] = 100
    fig, ax1 = plt.subplots()
    ax1.plot(np.array(m),app[0],color='C1')
    ax1.set_xticks(np.arange(36))
    ax1.set_xticklabels([i if x%3==0 else '' for x,i in enumerate(m)],rotation=45)
    ax1.set_ylabel('Positive')
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
    ax2.set_xticklabels([i if x%3==0 else '' for x,i in enumerate(m)],rotation=45)
    ax2.plot(m,app[1],color='C2')
    ax2.set_ylabel('Negative')
    ax1.set_title('Keywords: '+', '.join(keyword_list[:3]))
    plt.show()
    
def trend(data,hint,keyword_list=['']): # hint=0 for pos, 1 for neg
    app=subanalyze(data,keyword_list)
    matplotlib.rcParams['figure.dpi'] = 100
    fig, ax1 = plt.subplots()
    ax1.plot(np.array(m),app[hint],color='C'+str(hint+1))
    ax1.set_xticks(np.arange(36))
    ax1.set_xticklabels([i if x%3==0 else '' for x,i in enumerate(m)],rotation=45)
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
    ax2.set_xticklabels([i if x%3==0 else '' for x,i in enumerate(m)],rotation=45)
    ax2.bar(m,app[2],alpha=0.3)
    ax2.set_ylabel('Sentiment Frequency')
    ax1.set_title('Keywords: '+', '.join(keyword_list[:3]))
    plt.show()

    
    
root_path=os.getcwd()
category=os.listdir(root_path+'\\SentiStrength\\final')

def all_trend(data,hint,keyword_list=['']):
    app=[subanalyze(i,keyword_list) for i in data]
    matplotlib.rcParams['figure.dpi'] = 200
    fig, ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(12,8))
    for i in range(2):
        for j in range(2):
            temp=2*i+j
            
            ax[i][j].plot(np.array(m),app[temp][hint],color='C'+str(hint+1))
            ax[i][j].set_xticks(np.arange(36))
            ax[i][j].set_xticklabels([i if x%3==0 else '' for x,i in enumerate(m)],rotation=45)
            if not j:
                ax[i][j].set_ylabel('Sentiment Polarity')
            for t in ax[i][j].xaxis.get_ticklines():
                t.set_visible(False)
            for t in ax[i][j].xaxis.get_ticklines()[::2][::3]:
                t.set_visible(True)
            ax2 = ax[i][j].twinx()
            for t in ax2.xaxis.get_ticklines():
                t.set_visible(False)
            for t in ax2.xaxis.get_ticklines()[::2][::3]:
                t.set_visible(True)
            ax2.set_xticks(np.arange(36))
            ax2.set_xticklabels([i if x%3==0 else '' for x,i in enumerate(m)],rotation=45)
            ax2.bar(m,app[temp][2],alpha=0.3)
            if j:
                ax2.set_ylabel('Sentiment Frequency')
            ax[i][j].set_title(category[temp])
    plt.suptitle('Keywords: '+', '.join(keyword_list[:3]))
    plt.show()
    
def show(entity,month,keyword):
    a,b=0,0
    result=[]
    for i in metadata[entity][month]:
        s=0
        for j in keyword:
            if j in i['text'].lower():
                s=1
        if s:
            #print(i,data[entity][month][1][x])
            result.append(i)
    return sorted(result,key=lambda x : x['text'])

def classify(monthdata):
    pos=[]
    neg=[]
    for i in monthdata:
        poscnt=0
        negcnt=0
        for k in i:
            if k['positive']>=-1.5*k['negative']:
                poscnt+=1
            elif -k['negative']>=1.5*k['positive']:
                negcnt+=1
        pos.append(poscnt/max(len(i),1))
        neg.append(negcnt/max(len(i),1))
    return pos,neg



def ma(data,keyword_list,polarity=0,count=1,ma=None):
    # polarity=0 for pos, 1 for neg
    if not isinstance(polarity,list):
        polarity=[polarity]
    all_months=month('2017-06','2020-05')
    app=subanalyze(data,keyword_list)
    for pos in polarity:
        if ma:
            supline=np.zeros(len(all_months))
            subline=np.zeros(len(all_months))
            average = np.zeros(len(all_months))
            for i, m in enumerate(all_months):
                if i == 0:
                    average[i] =app[pos][i]
                    supline[i]=average[i]
                    subline[i]=average[i]
                elif i<ma:
                    average[i]=np.mean(app[pos][:i+1])
                    supline[i]=average[i]+np.std(app[pos][:i+1])
                    subline[i]=average[i]-np.std(app[pos][:i+1])
                else:
                    average[i]=np.mean(app[pos][i-ma+1:i+1])
                    supline[i]=average[i]+np.std(app[pos][i-ma+1:i+1],ddof=1)
                    subline[i]=average[i]-np.std(app[pos][i-ma+1:i+1],ddof=1)

        matplotlib.rcParams['figure.dpi'] = 100
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(np.array(all_months), app[pos], color='C' + str(pos + 1))
        if ma:
            ax1.plot(np.array(all_months), average, 'plum')
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
        ax1.set_title('Keywords: ' + ', '.join(keyword_list[:5]) + extra)
        plt.show()


# In[ ]:

'''
# create a list of globally defined positive and negative words to identify sentiment
# sentiment score based on the laxicon neg, pos words
def debug2(sentence, pos=pos, neg=neg):

    #input: dictionary and sentence
    #function: appends dictionary with new features if the feature
    #          did not exist previously,then updates sentiment to
    #          each of the new or existing features
    #output: updated dictionary
    
    sent_dict = dict()
    noun_mark=[]
    sentence = nlp(sentence)
    opinion_words = neg.union(pos)
    op=0
    for token in sentence:
        # check if the word is an opinion word, then assign sentiment
        if token.text in opinion_words:
            op=1
            sentiment = 1 if token.text in pos else -1
            # if target is an adverb modifier (i.e. pretty, highly, etc.)
            # but happens to be an opinion word, ignore and pass
            if (token.dep_ == "advmod"):
                continue
            else:
                for child in token.children:
                    # if there's a adj modifier (i.e. very, pretty, etc.) add more weight to sentiment
                    # This could be better updated for modifiers that either positively or negatively emphasize
                    if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in opinion_words):
                        sentiment *= 2
                    # check for negation words and flip the sign of sentiment
                    if child.dep_ == "neg":
                        sentiment *= -1
                if (token.pos_) == "VERB":
                    for child in token.children:
                        # if verb, check if there's a direct object
                        if (child.dep_ == "dobj") or (child.dep_=="nsubj"):                        
                            sent_dict[child] = sentiment
                            if child not in noun_mark: noun_mark.append(child)
                            # check for conjugates (a AND b), then add both to dictionary
                            subchildren = []
                            conj = 0
                            for subchild in child.children:
                                if (subchild.text == "and"):
                                    conj=1
                                if (conj == 1) and (subchild.text != "and"):
                                    subchildren.append(subchild)
                                    conj = 0
                            for subchild in subchildren:
                                sent_dict[subchild] = sentiment
                                if subchild not in noun_mark: noun_mark.append(subchild)

                if (token.dep_ == "amod"):
                    #print('a')
                    sent_dict[token.head] = sentiment
                    noun_mark.append(token.head)
                
                    subchildren = []
                    conj = 0
                    for subchild in token.head.children:
                        if (subchild.text == "and"):
                            conj=1
                        if (conj == 1) and (subchild.text != "and"):
                            subchildren.append(subchild)
                            conj = 0
                    for subchild in subchildren:
                        sent_dict[subchild] = sentiment
                        if subchild not in noun_mark: noun_mark.append(subchild)
                
                for child in token.head.children:
                    if (child.pos_ == "NOUN") and (child.text not in sent_dict)                     and (child.dep_ not in ["npadvmod"]):
                        sent_dict[child] = sentiment
                        if child not in noun_mark: noun_mark.append(child)
                        #print(token,sent_dict)
                        #print(token.text+':','compund noun',sent_dict)

    temp=[]  
    for noun in set(noun_mark):
        #print(noun,noun_mark,sent_dict)
        #print('1:',temp)
        sentiment = sent_dict[noun]
        if (noun.pos_ == "NOUN" or noun.pos_ == "PROPN") and (noun not in temp):
            temp.append(noun)
            s=noun.text
            current=noun
            state=1
            while current.children and state:
                temp.append(current)
                state=0
                for child in list(current.children)[::-1]:
                    if ((child.pos_ == "NOUN") or (child.pos_ == "PROPN")) and (child.dep_ == "compound"):
                        s = child.text + " " + s
                        current=child
                        temp.append(child)
                        state=1
            del sent_dict[noun]
            sent_dict[s.lower()] = sentiment
    #print(sent_dict)
    sent_dict=dict([i for i in sent_dict.items() if type(i[0])==str])
    #print(sent_dict)

    # check for nouns
    #for child in token.head.children:
    #    noun = ""
    #    if (child.pos_ == "NOUN") and (child.text not in sent_dict):
    #        noun = child.text
    #        # Check for compound nouns
    #        for subchild in child.children:
    #            if subchild.dep_ == "compound":
    #                noun = subchild.text + " " + noun
    #        sent_dict[noun] = sentiment

    if (not sent_dict) and op:
        for token in sentence:
            if token.pos_=="NOUN":
                sent_dict[token.text.lower()]=sentiment
    #print(sent_dict)
    return sent_dict
'''

# global variables
m=month('2017-06','2020-05')

