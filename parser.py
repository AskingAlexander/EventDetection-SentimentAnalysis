from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import re

from datetime import datetime
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

wordnet_lemmatizer = WordNetLemmatizer()
default_stopwords = stopwords.words('english') # or any other list of your choice

def initial_setup():
    df = pd.read_csv('Sentiment140.csv')
    print(df.head())

    columns = ['ID', 'Date', 'Query', 'User']
    df.drop(columns, inplace=True, axis=1)

    print(df.head())

    df.rename(columns={'Sentiment': 'polarity', 'Tweet': 'text'}, inplace=True)

    print(df.head())

    df['polarity'] = df['polarity'].map({0: 'neg', 2: 'neu', 4: 'pos'})
    print(df.head())

    df.to_csv('S140.csv', encoding='utf-8', index=False)

def cleanText(text, stop_words=default_stopwords, extra_words=[], useFE=False):
    def applyFE(text):
        """This method will combine the negation with the words
        Will result in a bigger vocabulary but with less bias
        """
        final_text = text.replace('cannot', 'can not')
        final_text = text.replace('can’t', 'can not')
        final_text = final_text.replace('won’t', 'will not')
        final_text = final_text.replace('n\'t', ' not')
        final_text = final_text.replace('n\'t', 'not')
        final_text = final_text.replace(' not ', ' not')

        return final_text

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text):
        tokens = tokenize_text(text)
        return ' '.join(re.sub('[^a-z]+', '', x) for x in tokens)

    def lemma_text(text, lemmatizer=wordnet_lemmatizer):
        tokens = tokenize_text(text)
        return ' '.join([lemmatizer.lemmatize(t) for t in tokens])

    def remove_stopwords(text, stop_words=(stop_words + extra_words)):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    text = str(text).strip(' ')  # strip whitespaces
    text = text.lower()  # lowercase
    # remove punctuation and symbols
    text = remove_special_characters(text)
    text = lemma_text(text)  # stemming
    text = remove_stopwords(text)  # remove stopwords
    text = text.strip(' ') # strip whitespaces again?
    if useFE:
        text = applyFE(text)

    return text

def split(df, extra_tags=''):
    df_c1 = df.head(10000).append(df.tail(10000), ignore_index=True)
    df_c2 = df.head(250000).append(df.tail(250000), ignore_index=True)

    train, test = train_test_split(df_c1, test_size=0.2)

    train.to_csv('C1Train' + extra_tags +'.csv', encoding='utf-8', index=False)
    test.to_csv('C1Test' + extra_tags +'.csv', encoding='utf-8', index=False)

    train, test = train_test_split(df_c2, test_size=0.2)

    train.to_csv('C2Train' + extra_tags +'.csv', encoding='utf-8', index=False)
    test.to_csv('C2Test' + extra_tags +'.csv', encoding='utf-8', index=False)

    train, test = train_test_split(df, test_size=0.2)

    train.to_csv('C3Train' + extra_tags +'.csv', encoding='utf-8', index=False)
    test.to_csv('C3Test' + extra_tags +'.csv', encoding='utf-8', index=False)

def clean_and_split():
    df = pd.read_csv('S140.csv')
    df['text'] = df['text'].map(lambda x: cleanText(x))
    df.to_csv('S140FE.csv', encoding='utf-8', index=False)

    df = pd.read_csv('S140C1.csv')
    df['text'] = df['text'].map(lambda x: cleanText(x))
    df.to_csv('S140C1FE.csv', encoding='utf-8', index=False)

    df = pd.read_csv('S140C2.csv')
    df['text'] = df['text'].map(lambda x: cleanText(x))
    df.to_csv('S140C2FE.csv', encoding='utf-8', index=False)
    
    df = pd.read_csv('C1Train.csv')
    df['text'] = df['text'].map(lambda x: cleanText(x))
    df.to_csv('C1FETrain.csv', encoding='utf-8', index=False)

    df = pd.read_csv('C2Train.csv')
    df['text'] = df['text'].map(lambda x: cleanText(x))
    df.to_csv('C2FETrain.csv', encoding='utf-8', index=False)

    df = pd.read_csv('C3Train.csv')
    df['text'] = df['text'].map(lambda x: cleanText(x))
    df.to_csv('C3FETrain.csv', encoding='utf-8', index=False)

    df = pd.read_csv('C1Test.csv')
    df['text'] = df['text'].map(lambda x: cleanText(x))
    df.to_csv('C1FETest.csv', encoding='utf-8', index=False)

    df = pd.read_csv('C2Test.csv')
    df['text'] = df['text'].map(lambda x: cleanText(x))
    df.to_csv('C2FETest.csv', encoding='utf-8', index=False)

    df = pd.read_csv('C3Test.csv')
    df['text'] = df['text'].map(lambda x: cleanText(x))
    df.to_csv('C3FETest.csv', encoding='utf-8', index=False)

def ED_setup():
    df = pd.read_csv('Sentiment140.csv')
    print(df.head())

    columns = ['ID', 'Sentiment', 'Query', 'User']
    df.drop(columns, inplace=True, axis=1)

    print(df.head())
    df.rename(columns={'Tweet': 'text'}, inplace=True)
    print(df.head())
    df['Date'] = df['Date'].map(lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S PDT %Y").strftime("%Y-%m-%d-%H-%M-%S"))
    df['text'] = df['text'].map(lambda x: cleanText(x, useFE=False))

    df.to_csv('S140ED.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    ED_setup()