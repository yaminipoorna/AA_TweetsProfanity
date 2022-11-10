################################### Importing packages
pip install tweepy
import configparser
import pandas as pd
import tweepy
import re
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet                   #for parts of speech
from wordcloud import STOPWORDS


######## racial_slur_words
#Took a reference from the below link for the rational slur words
#https://ora.ox.ac.uk/objects/uuid:9232010f-72c5-44d7-aaa6-9e7896e13ea8/download_file?file_format=application%2Fpdf&safe_filename=Racial%2Bslurs%2Bin%2Bdictionaries&type_of_work=Conference+item

with open("C:/Users/yamini/Desktop/AAtwitter/racial_slurs_words.txt","r") as rsw:
  rswords = rsw.read().split("\n") 
rswords


################################ Read config.ini file
config_obj = configparser.ConfigParser()
config_obj.read("C:/Users/yamini/Desktop/AAtwitter/configfile.ini")

api_key=config_obj['twitter']['api_key']
api_key_secret=config_obj['twitter']['api_key_secret']

access_token=config_obj['twitter']['access_token']
access_token_secret=config_obj['twitter']['access_token_secret']

print(api_key)

################################### authentication
auth=tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token,access_token_secret)

api=tweepy.API(auth)

public_tweets=api.home_timeline()
print(public_tweets)
print(public_tweets[0].text)
print(public_tweets[0].created_at)
print(public_tweets[0].user.screen_name)
for i in public_tweets:
    print(i.text)

columns=['Time','User','Tweet']
data=[]
for i in public_tweets:
    data.append([i.created_at,i.user.screen_name,i.text])
print(data)

df=pd.DataFrame(data,columns=columns)
print(df)

df.to_csv('tweets.csv')

################################# Taking the saved tweets
data=pd.read_csv("C:/Users/yamini/Desktop/AAtwitter/tweets.csv")
data
data.columns
data1=data["Tweet"]
data1
data2=" ".join(data1)
data2

with open("twitter_tweets.txt",mode="w",encoding="utf8") as output:
    output.write((str(data2)))

def main():
    ################################### Data cleaning
    data3=data2.lower()
    data3
    
    #removing numbers from a string
    order=r'[0-9]'
    data4=re.sub(order,'',data3)
    data4
    
    #removing special characters from a string
    data5=re.sub(r'[^a-zA-Z0-9 ]',"",data4)
    data5                          #now the text was clean
    
    data6=re.sub(r'http\S+', '', data5)
    data6
    
    #lemmatization to remove prefix and suffix
    lemmatizer=WordNetLemmatizer()
    
    #lemmatized_output = ''.join([lemmatizer.lemmatize(w) for w in data5])
    #print(lemmatized_output)           #still not getting the correct output.
    
    #parts of speech tagging
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    data7=[lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(data6)]
    data7
    
    #stopword removal
    stopwords_wc = set(STOPWORDS) 
    data8 = [word for word in data7 if word not in stopwords_wc]
    data8
    total_words=len(data8)
    
    
    ######################################## number of slur words in total #######################################
    df = pd.DataFrame(data8)      #created dataframe
    print(df)
    df.columns = ['preprocess_txt']           # created column
    print(df)
    num_rsw = [w for w in df["preprocess_txt"] if w in rswords]
    num_rsw
    total_rsw_score=len(num_rsw)                             #10
    total_rsw_score
    
    ################################### Total Number of Sentences ###############################
    sentences = data4.split(".") #split the text into a list of sentences.
    total_sen=len(sentences)
    total_sen                                          #45
    
    ############################### degree of profanity for each sentence ###########################
    degree_of_profanity=total_rsw_score/total_sen
    print(degree_of_profanity)

if __name__ == '__main__':
    main()










