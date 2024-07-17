from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


import pandas as pd
import numpy as np

from datasetsHF import load_dataset
import datasetsHF as ds

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download(
    ["punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "universal_tagset"]
)
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

import re

import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go


from PIL import Image

from ntscraper import Nitter


from deep_translator import GoogleTranslator


scraper = Nitter()

source_lang = "tl"
translated_to = "en"

def get_latest_tweet_df(username, number_of_tweets):    
    tweets = scraper.get_tweets(username, mode = "user", number = number_of_tweets)
    final_tweets = []
    
    for tweet in tweets['tweets']:
        data = [tweet['is-retweet'], tweet['user']['name'], tweet['date'], tweet['text']]
        final_tweets.append(data)

    tweet_df = pd.DataFrame(
        final_tweets, columns = ['Retweet?', 'User Name', 'Date', 'Tweet']
    )

    return tweet_df

def text_translate(textTweet):
    translated_text = GoogleTranslator(source='auto', target='en').translate(textTweet)
    return translated_text

def text_preprocessing(text):
    stopwords = set()
    with open("static/en_stopwords.txt", "r") as file:
        for word in file:
            stopwords.add(word.rstrip("\n"))
    lemmatizer = WordNetLemmatizer()
    try:
        url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
        user_pattern = r"@[^\s]+"
        entity_pattern = r"&.*;"
        neg_contraction = r"n't\W"
        non_alpha = "[^a-z]"
        cleaned_text = text.lower()
        cleaned_text = re.sub(neg_contraction, " not ", cleaned_text)
        cleaned_text = re.sub(url_pattern, " ", cleaned_text)
        cleaned_text = re.sub(user_pattern, " ", cleaned_text)
        cleaned_text = re.sub(entity_pattern, " ", cleaned_text)
        cleaned_text = re.sub(non_alpha, " ", cleaned_text)
        tokens = word_tokenize(cleaned_text)
        # provide POS tag for lemmatization to yield better result
        word_tag_tuples = pos_tag(tokens, tagset="universal")
        tag_dict = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
        final_tokens = []
        for word, tag in word_tag_tuples:
            if len(word) > 1 and word not in stopwords:
                if tag in tag_dict:
                    final_tokens.append(lemmatizer.lemmatize(word, tag_dict[tag]))
                else:
                    final_tokens.append(lemmatizer.lemmatize(word))
        return " ".join(final_tokens)
    except:
        return np.nan


def predict_sentiment(tweet_df):

    temp_df = tweet_df.copy()
    temp_df["Translated Tweet"] = temp_df["Tweet"].apply(text_translate)
    temp_df["Cleaned Tweet"] = temp_df["Translated Tweet"].apply(text_preprocessing)
    temp_df = temp_df[(temp_df["Cleaned Tweet"].notna()) & (temp_df["Cleaned Tweet"] != "")]

    dtSet = ds.load_dataset('vibhorag101/suicide_prediction_dataset_phr', split='train')
    df = dtSet.to_pandas()
    df = df.dropna()

    stopset = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=list(stopset))

    y = df.values[:, 1]
    X = vectorizer.fit_transform(df.values[:, 0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1000)

    clf = MultinomialNB()
    clf.fit(X_train, y_train) 

    test_predict_array = temp_df["Cleaned Tweet"].to_numpy()
    test_predict_vector = vectorizer.transform(test_predict_array)
    sentiment = clf.predict(test_predict_vector)    
    proba = clf.predict_proba(test_predict_vector)
    proba_df = pd.DataFrame(proba, columns=["proba_non_suicide", "proba_suicide"])
    # proba_df_converted = (proba_df * 100).round(2)

    temp_df["Sentiment"] = pd.Series(sentiment)
    new_df = pd.concat([temp_df, proba_df], axis=1)
    new_df = new_df[(new_df["Sentiment"].notna())]
    new_df["Row No."] = np.arange(new_df.shape[0])
    new_df["Sentiment"] = new_df["Sentiment"].str.capitalize()
    # new_df = new_df.reset_index()

    print(new_df)
    return new_df

def plot_sentiment(tweet_df):
    # count the number tweets based on the sentiment
    sentiment_count = tweet_df["Sentiment"].value_counts()

    # plot the sentiment distribution in a pie chart
    fig = px.pie(
        values=sentiment_count.values,
        names=sentiment_count.index,
        hole=0.3,
        title="<b>Sentiment Distribution</b>",
        color=sentiment_count.index,
        # set the color of positive to blue and negative to orange
        color_discrete_map={"Suicide": "#EE4B2B", "Non-suicide": "#33AAFF"},
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
        # texttemplate="<b>%{percent}%</b>",
        # texttemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
    )
    fig.update_layout(showlegend=True)    

    return fig

def plot_probability(tweet_df):
    fig = go.Figure(data=[
        go.Bar(name='Non-Suicide', x=tweet_df['Row No.'], y=tweet_df['proba_non_suicide'], marker_color="#33AAFF", text=tweet_df['proba_non_suicide'], textposition='auto', texttemplate='%{y:.1%}'),
        go.Bar(name='Suicide', x=tweet_df['Row No.'], y=tweet_df['proba_suicide'], marker_color="#EE4B2B", text=tweet_df['proba_suicide'], textposition='auto', texttemplate='%{y:.1%}')
    ])

    fig.update_layout(barmode='group', title_text="<b>{}</b>".format('Sentiment Probability Result'))
    return fig

 
def plot_table(tweet_df):
    fig = go.Figure(data=[
        go.Table(columnwidth = [300, 250, 300, 1000],
                 header=dict(values=['<b>Sentiment</b>', '<b>Date</b>', '<b>Retweet?</b>', '<b>Tweet</b>'],
                             height=40,
                             font_size=20),
                 cells=dict(values=[tweet_df['Sentiment'], tweet_df['Date'], tweet_df['Retweet?'], tweet_df['Tweet']],
                            height=30,
                            align=['center', 'left', 'center', 'left'],
                            font_size=15,
                            fill_color=[["rgb(238, 75, 43)" if x == "Suicide" else "rgb(51, 170, 255)" if x == "Non-suicide" else "rgb(255, 255, 255)" for x in list(tweet_df['Sentiment'])],
                                        'rgb(255, 255, 255)', 'rgb(255, 255, 255)', 'rgb(255, 255, 255)'])
                 
                 )
        ])
    
    return fig

# dataa = get_latest_tweet_df('annecurtissmith', 10)

# predict_sentiment(dataa)