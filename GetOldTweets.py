import GetOldTweets3 as got
import csv

tweetCriteria = got.manager.TweetCriteria().setQuerySearch('Morgan Stanley')\
                                           .setSince("2015-05-01")\
                                           .setUntil("2020-05-01")\
                                           .setMaxTweets(10000)\
                                            #.setUsername(str or iterable): An optional specific username(s) from a twitter account (with or without "@").
                                            #.setTopTweets (bool): If True only the Top Tweets will be retrieved.
                                            #.setNear(str): A reference location area from where tweets were generated.
                                            #.setWithin (str): A distance radius from "near" location (e.g. 15mi).

tweets = got.manager.TweetManager.getTweets(tweetCriteria)

def writeCsvFile(fname, data, *args, **kwargs):
    mycsv = csv.writer(open(fname, 'wt'), *args, **kwargs)
    mycsv.writerow(['Id',"permalink","username","to","date","retweets","favorites","mentions","hashtags","geo","text"])
    for row in data:
        mycsv.writerow([row.id,row.permalink,row.username,row.to,row.date,row.retweets,row.favorites,row.mentions,row.hashtags,row.geo,row.text])

writeCsvFile('Data.csv',tweets,dialect='excel')