from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import string
import nltk
import csv
import re
import os

warnings.simplefilter("ignore", DeprecationWarning)
sns.set_style('whitegrid')

wordnet_lemmatizer = WordNetLemmatizer()
default_stopwords = stopwords.words('english') # or any other list of your choice

DATASET_PATH = 'DataSample/S140SampleED.csv'
CLEANED_DATASET_PATH = 'DataSample/S140SampleEDCleaned.csv'
TOPIC_FILE = 'DataSample/S140SampleEDTopics.csv'

def clean_text(text, stop_words=default_stopwords, extra_words = []):

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text):
        tokens = tokenize_text(text)
        return ' '.join(re.sub('[^a-z]+', '', x) for x in tokens)

    def lemma_text(text, lemmatizer=wordnet_lemmatizer):
        tokens = tokenize_text(text)
        return ' '.join([lemmatizer.lemmatize(t) for t in tokens])

    def remove_stopwords(text, stop_words= (stop_words + extra_words)):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    text = str(text).strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = remove_special_characters(text) # remove punctuation and symbols
    text = lemma_text(text) # stemming
    text = remove_stopwords(text) # remove stopwords
    #text.strip(' ') # strip whitespaces again?

    return text

def readData():
    data = None

    try:
        data = pd.read_csv(CLEANED_DATASET_PATH, engine='python', encoding='utf-8', error_bad_lines=False)
    except:
        data = pd.read_csv(DATASET_PATH, engine='python', encoding='utf-8', error_bad_lines=False)
        data['text'] = data['text'].map(lambda x: clean_text(x))

        data.to_csv(CLEANED_DATASET_PATH, sep=',', encoding='utf-8', index=False)

    return data


def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))

    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    figure = plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')

    figure.savefig('[OLDA]Sentiment140SampleCommon.png')

def saveTopics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    data_labels = []

    for _, topic in enumerate(model.components_):
        data_labels.append(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

    topics = pd.DataFrame(data=data_labels, columns=['topic'])
    topics.to_csv(TOPIC_FILE, sep=',', encoding='utf-8', index=False)
    
    return topics

def trainOLDA():
    papers = readData()

    # Join the different processed titles together.
    long_string = ','.join(list(str(x) for x in papers['text'].values))

    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    cloud = wordcloud.generate(long_string)
    cloud.to_file('[OLDA]CLOUD_Sentiment140Sample.png')
    print('Generated WordCloud')

    # Initialise the count vectorizer
    count_vectorizer = CountVectorizer()

    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(papers['text'].values.astype('str'))

    # Visualise the 10 most common words
    plot_10_most_common_words(count_data, count_vectorizer)
    print('Plotted Most Common Words')

    # Tweak the two parameters below
    number_topics = 50
    number_words = 10

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics)
    lda.fit(count_data)

    print('OLDA is done')
    # Print the topics found by the LDA model
    return saveTopics(lda, count_vectorizer, number_words), papers

def main():
    topics = None
    papers = None

    try:
        topics = pd.read_csv(TOPIC_FILE, engine='python', encoding='utf-8', error_bad_lines=False)
        papers = readData()
    except:
        topics, papers = trainOLDA()

    topics['set'] = topics['topic'].map(lambda x: set(x.split()))

    print('Loaded Topics and Dataset')

    magnitude = {}
    event_start = {}
    event_end = {}

    for _, row in papers.iterrows():
        try:
            rowSet = set(row['text'].split())
        except:
            continue
            
        for _, topicRow in topics.iterrows():
            topicSet = topicRow['set']
            original_topic = topicRow['topic']

            if len(topicSet.intersection(rowSet)) > 1:

                if original_topic not in magnitude:
                    magnitude[original_topic] = 0
                
                magnitude[original_topic] = magnitude[original_topic] + 1

                given_date = row['date']

                if original_topic not in event_start:
                    event_start[original_topic] = given_date

                if given_date < event_start[original_topic]:
                    event_start[original_topic] = given_date

                if original_topic not in event_end:
                    event_end[original_topic] = given_date

                if given_date > event_end[original_topic]:
                    event_end[original_topic] = given_date

    print('Writing Output')

    with open('[OLDA]TopicResults.csv', mode='w') as csv_out_file:
        csv_writer = csv.writer(csv_out_file, delimiter=',')
        
        for _, row in topics.iterrows():
            topic = row['topic']

            csv_writer.writerow([magnitude.get(topic, 0),  
            '\'' + str(event_start.get(topic, 'NULL')) +'\'',
            '\'' + str(event_end.get(topic, 'NULL')) +'\'',
            '\'' + str(topic) +'\''])

if __name__ == "__main__":
    main()