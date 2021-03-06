from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.tokenize import word_tokenize, sent_tokenize
from datetime import datetime, timedelta
from nltk.stem import WordNetLemmatizer
from scipy.signal import find_peaks
from nltk.corpus import stopwords
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import string
import nltk
import math
import csv
import re
import os

warnings.simplefilter("ignore", DeprecationWarning)
sns.set_style('whitegrid')

wordnet_lemmatizer = WordNetLemmatizer()
default_stopwords = stopwords.words('english') # or any other list of your choice

SAMPLE_DATASET_PATH = os.path.join('DataSample',  'S140SampleED.csv')
SAMPLE_CLEANED_DATASET_PATH = os.path.join('DataSample',  'S140SampleEDCleaned.csv')
SAMPLE_TOPIC_FILE = os.path.join('DataSample',  'S140SampleEDTopics.csv')

DATASET_PATH = os.path.join('DataSample',  'S140ED.csv')
CLEANED_DATASET_PATH = os.path.join('DataSample',  'S140EDCleaned.csv')
S140_ED_FILE = os.path.join('Datasets', 'S140ED.csv')
S140_CO_ED_FILE = os.path.join('Datasets', 'S140EDC0.csv')

class EDMethod(object):    
    def __init__(self, 
                methodName,
                numberOfTopics = 10, 
                numberOfWords = 5, 
                datasetPath = DATASET_PATH, 
                cleanedDatasetPath = CLEANED_DATASET_PATH):
        self.methodName = methodName
        self.TopicFile = '[' + methodName + ']S140EDTopics.csv'

        self.dataset = datasetPath
        self.cleanedDataset = cleanedDatasetPath

        self.numberOfTopics = numberOfTopics
        self.numberOfWords = numberOfWords
        
    def cleanText(self, text, stop_words=default_stopwords, extra_words = []):
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

    def readData(self):
        data = None

        try:
            data = pd.read_csv(self.dataset, engine='python', encoding='utf-8', error_bad_lines=False)
        except:
            data = pd.read_csv(self.dataset, engine='python', encoding='utf-8', error_bad_lines=False)
            data['text'] = data['text'].map(lambda x: self.cleanText(x))

            data.to_csv(self.cleanedDataset, sep=',', encoding='utf-8', index=False)

        return data

    def getStats(self, papers, topics):
        papers.columns = papers.columns.str.lower()
        topics['set'] = topics['topic'].map(lambda x: set(x.split()))

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

        print(self.methodName + ': Writing Output')

        with open('[Stats]' + self.TopicFile, mode='w') as csv_out_file:
            csv_writer = csv.writer(csv_out_file, delimiter=',')
            
            for _, row in topics.iterrows():
                topic = row['topic']

                csv_writer.writerow([magnitude.get(topic, 0),  
                '\'' + str(event_start.get(topic, 'NULL')) +'\'',
                '\'' + str(event_end.get(topic, 'NULL')) +'\'',
                '\'' + str(topic) +'\''])

    def train(self):
        None

    def run(self):
        topics = None
        papers = None

        try:
            topics = pd.read_csv(self.TopicFile, engine='python', encoding='utf-8', error_bad_lines=False)
            papers = self.readData()

            print(self.methodName + ': Loaded Topics and Dataset')
        except:
            print(self.methodName + ': Training...')
            topics, papers = self.train()

            print(self.methodName + ': Trained')

        self.getStats(papers, topics)

class OLDA(EDMethod):
    def __init__(self, 
                numberOfTopics = 10, 
                numberOfWords = 5, 
                datasetPath = SAMPLE_DATASET_PATH, 
                cleanedDatasetPath = SAMPLE_CLEANED_DATASET_PATH):
        EDMethod.__init__(self, 'OLDA', numberOfTopics, numberOfWords,datasetPath,  cleanedDatasetPath)

    def plotMost10CommonWords(self, count_data, count_vectorizer):
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

    def saveTopics(self, model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        data_labels = []

        for _, topic in enumerate(model.components_):
            data_labels.append(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

        topics = pd.DataFrame(data=data_labels, columns=['topic'])
        topics.to_csv(self.TopicFile, sep=',', encoding='utf-8', index=False)
        
        return topics

    def train(self):
        print('Gathering Data...')
        papers = self.readData()

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
        self.plotMost10CommonWords(count_data, count_vectorizer)
        print('Plotted Most Common Words')

        # Create and fit the LDA model
        lda = LDA(n_components=self.numberOfTopics)
        lda.fit(count_data)

        print('OLDA is done')
        # Print the topics found by the LDA model
        return self.saveTopics(lda, count_vectorizer, self.numberOfWords), papers

class MABED(EDMethod):
    def __init__(self, 
                numberOfTopics = 10, 
                numberOfWords = 5, 
                datasetPath = SAMPLE_DATASET_PATH, 
                cleanedDatasetPath = SAMPLE_CLEANED_DATASET_PATH):
        EDMethod.__init__(self, 'MABED', numberOfTopics, numberOfWords,datasetPath,  cleanedDatasetPath)

    def run(self):
        print('Gathering Data...')
        self.readData()
        currentDir = os.getcwd()

        os.chdir(os.path.join(currentDir, 'ED', 'pyMABED'))

        print('Starting MABED')
        os.system('python3 detect_events.py ' + self.cleanedDataset + ' ' + str(self.numberOfTopics) + ' --o mabed.out --sep ,')

        print('Showing Results')
        os.system('python3 build_event_browser.py mabed.out')

        os.chdir(currentDir)
        
class PeakDetection(EDMethod):    
    def __init__(self, 
                numberOfTopics = 10, 
                numberOfWords = 5, 
                datasetPath = SAMPLE_DATASET_PATH, 
                cleanedDatasetPath = SAMPLE_CLEANED_DATASET_PATH):
        EDMethod.__init__(self, 'PeakDetection', numberOfTopics, numberOfWords,datasetPath,  cleanedDatasetPath)
        self.timeSplit = 100

    def getHoursBetween(self, diff):
        days, seconds = diff.days, diff.seconds
        hours = days * 24 + seconds // 3600

        return hours

    def getMinutesBetween(self, diff):
        days, seconds = diff.days, diff.seconds
        minutes = days * 24 * 60 + seconds/60

        return minutes

    def generateBins(self, binCount):
        tweets = self.dataset

        minDate = tweets['Date'].min()
        maxDate = tweets['Date'].max()

        diff = maxDate - minDate
        binCount += 1

        split = int(math.ceil(self.getHoursBetween(diff) / binCount))
        bins = {}

        if (split == 0):
            currentBin = []

            for _, row in tweets.iterrows():
                currentBin.append((row['Date'], row['text']))

            bins[(minDate, maxDate)] = currentBin
        else:
            maxBinIndex = binCount - 1
            for i in range(maxBinIndex):
                lDate = minDate + timedelta(hours=(i * split))
                uDate = minDate + timedelta(hours=((i + 1) * split))

                bins[(lDate, uDate)] = []

            lastBinLeft = minDate + timedelta(hours=((maxBinIndex) * split))
            bins[(lastBinLeft, maxDate)] = []

            for _, row in tweets.iterrows():
                currentDate = row['Date']
                diff = currentDate - minDate

                binIndex = 0 if (split == 0) else min([int(math.floor(self.getHoursBetween(diff)/split)), maxBinIndex])

                lDate = lastBinLeft if (binIndex == maxBinIndex) else (minDate + timedelta(hours=(binIndex * split)))
                uDate = maxDate if (binIndex == maxBinIndex) else (minDate + timedelta(hours=((binIndex + 1) * split)))

                bins[(lDate, uDate)].append((currentDate, row['text']))

        return bins

    def saveTopics(self, popularTopics): 
        topics = pd.DataFrame(data=popularTopics, columns=['topic'])
        topics.to_csv(self.TopicFile, sep=',', encoding='utf-8', index=False)
        
        print(self.methodName + ': Saved Topics...')

        return topics

    def train(self):
        print('Running PeakDetection...')
        print('Gathering Data...')
        self.dataset = self.readData()

        self.dataset['Date'] = self.dataset['Date'].map(lambda x: datetime.strptime(x, "%Y-%m-%d-%H-%M-%S"))
        self.dataset = self.dataset.sort_values('Date')
        print('Gathered Data...')
        
        bins = self.generateBins(self.numberOfTopics)
        print('Generated Bins...')

        print('Getting Topics...')
        topics = []
        for (interval, listOfTweets) in  bins.items():
            maxIndex = len(listOfTweets)
            if maxIndex == 0:
                continue

            (lowD, highD) = interval
            lowS = lowD.strftime("%Y-%m-%d-%H-%M-%S")
            highS = highD.strftime("%Y-%m-%d-%H-%M-%S")
            print('Bin for: ' + lowS + ' to ' + highS)
            
            diff = highD - lowD
            minutesBetween = self.getMinutesBetween(diff)
            increment = float(minutesBetween / self.timeSplit)
            index = 0

            left = lowD
            right = lowD + timedelta(minutes=increment)

            stats = {}

            for i in range(self.timeSplit):
                if index >= maxIndex:
                    break
                while index < maxIndex:
                    (currentDate, _) = listOfTweets[index]
                    index += 1

                    if (left <= currentDate) and (currentDate <= right):
                        stats[i] = stats.get(i, 0) + 1
                    else:
                        left = right
                        right += timedelta(minutes=increment)
                        break

            lastIndex = maxIndex - 1
            while index < maxIndex:
                (currentDate, _) = listOfTweets[index]
                index += 1

                stats[lastIndex] = stats.get(lastIndex, 0) + 1
             
            distribution = list(stats.values())
            peaks, _ = find_peaks(distribution)

            if len(peaks) == 0:
                continue

            popularTweets = []
            for peak in peaks:
                initialDate = lowD + timedelta(minutes=(increment * peak))

                left = peak - 1
                right = peak + 1

                while left > 0:
                    (currentDate, tweet) = listOfTweets[left]
                    left -= 1

                    diff = initialDate - currentDate
                    minutesBetween = self.getMinutesBetween(diff)

                    if minutesBetween > increment:
                        break

                    popularTweets.append(str(tweet))

                while right < maxIndex:
                    (currentDate, tweet) = listOfTweets[right]
                    right += 1

                    diff = initialDate - currentDate
                    minutesBetween = self.getMinutesBetween(diff)

                    if minutesBetween > increment:
                        break

                    popularTweets.append(str(tweet))

            if len(popularTweets) > 0:
                try:
                    tfidf = TfidfVectorizer(stop_words=default_stopwords, max_features=50)
                    response = tfidf.fit_transform(popularTweets)

                    feature_array = np.array(tfidf.get_feature_names())
                    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

                    top_n = feature_array[tfidf_sorting[:self.numberOfWords]]

                    topics.append(' '.join(top_n.tolist()))
                except:
                    None

        return self.saveTopics(topics), self.dataset

def sampleRun():
    MABEDSample = MABED()
    OLDASample = OLDA()
    PDSample = MABED()

    for method in [MABEDSample, OLDASample, PDSample]:
        method.run()

def trueRun():
    peakDetection = PeakDetection(datasetPath=S140_ED_FILE,
                                cleanedDatasetPath=S140_ED_FILE)

    for method in [peakDetection]:
        method.run()


if __name__ == '__main__':
    trueRun()