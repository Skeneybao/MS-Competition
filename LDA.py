def LDA(texts,topics=20,num_words=15):
    from gensim import corpora
    from gensim.models.ldamodel import LdaModel
    import pprint
    from operator import itemgetter
    dictionary = corpora.Dictionary(texts) # texts: list of list of words
    corpus = [dictionary.doc2bow(text) for text in texts]
    num_topics = topics #The number of topics that should be generated
    passes = 20
    lda = LdaModel(corpus,
              id2word=dictionary,
              num_topics=num_topics,
              passes=passes)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(lda.print_topics(num_words=num_words))
    
    lda.get_document_topics(corpus[0],minimum_probability=0.05,per_word_topics=False) 
    # correlation between a tweet and each topic
    pp.pprint(sorted(lda.get_document_topics(corpus[0],minimum_probability=0,per_word_topics=False),key=itemgetter(1),reverse=True))
