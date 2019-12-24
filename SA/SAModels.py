from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import TweetTokenizer
from time import gmtime, strftime
from inspect import signature
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import joblib
import csv

class SAModel(object):
    def __init__(self, modelName = '', outputFileName = ''):
        self.name = modelName
        self.outputName = outputFileName

        self.model = None
        self.vocabulary = None
        self.vectorizer = None
    
    def applyFE(self, text):
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

    def loadOrTrain(self):
        None

    def textToFeatures(self, text):
        return self.vectorizer.transform(text)

    def predictData(self, toPredict):
        """This method will predict the given input
        toPredict: the data that the model will predict, as plain text
        vocabulary: the vocabulary used in the training step
        """
        return self.model.predict(self.textToFeatures(toPredict))

    def scoreModel(self, predicted, trueLabels):
        """This method will score the model with the given data
        predicted: the predicted data, preferably self.predictData(text)
        trueLabels: the actual labels for text
        """
        print('Computing: AUC, Precision, Accuracy, Recall and AvgPrecision')
        precision, recall, _ = precision_recall_curve(trueLabels, predicted)
        average_precision = average_precision_score(trueLabels, predicted)
        acc = accuracy_score(trueLabels, predicted)
        auc = roc_auc_score(trueLabels, predicted)
        
        scoreData = [['AUC', auc], ['Precision', precision], ['Acc', acc], ['Recall', recall], ['AvgPrecision', average_precision]]
        self.saveAsCSV(self.outputName + '.csv', scoreData)

        print('Plotting the PrecisionRecall Curve')
        self.plotPrecisionRecall(recall, precision, average_precision)

        print('Plotting the ROC Curve')
        self.plotROCCurve(predicted, trueLabels)
        
        
    def plotPrecisionRecall(self, recall, precision, average_precision):
        figure = plt.figure()
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        
        figure.savefig('[PRCurve]' + self.outputName + '.png')
    

    def plotROCCurve(self, predicted, trueLabels):
        fpr, tpr, _ = roc_curve(trueLabels, predicted)
        
        figure = plt.figure(figsize=(14,8))
        plt.plot(fpr, tpr, color="red")
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc curve')
        
        figure.savefig('[ROC]' + self.outputName + '.png')
        
    def saveAsCSV(self, fileName, data):
        with open(fileName, mode='w') as csv_out_file:
            csv_writer = csv.writer(csv_out_file, delimiter=',')
            
            for to_write in data:
                csv_writer.writerow(to_write)
                
    def readFromCSV(self, filePath,  encoding='ISO-8859-1'):
        """This method will read from a CSV file and return a dataframe
        Only the 'text' and 'label' columns will be taken in consideration
        """
        data_frame = pd.read_csv(filePath, engine='python', encoding=encoding, error_bad_lines=False)
        data_frame['label'] = data_frame['label'].map(lambda x: 1 if x == 'pos' else 0)

        return data_frame

class MultinomialNBModel(SAModel):
    def loadOrTrain(self, trainPath = ''):
        None

class RidgeClassifierModel(SAModel):
    def loadOrTrain(self, trainPath = ''):
        None

class LRModel(SAModel):
    def loadOrTrain(self, trainPath = ''):
        savedModel = None
        vocabulary = None

        try:
            vocabulary = pickle.load(open(self.name + '.vocabulary', 'rb'))
            print('LR: Loaded Vocabulary')

            savedModel = pickle.load(open(self.name, 'rb'))
            print('LR: Loaded Model')

            self.model = savedModel
            self.vocabulary = vocabulary            
            self.vectorizer = CountVectorizer(
                vocabulary = vocabulary,
                analyzer = 'word',
                lowercase = True)
        except:
            vocabulary = None
            savedModel = None

            if (trainPath == ''):
                raise Exception('LR: Could not load model or vocabulary and no training path is specified') 

        if vocabulary is None:
            dataset = self.readFromCSV(trainPath)
            print('LR: Preparing Vocabulary')

            self.vectorizer = CountVectorizer(analyzer = 'word', lowercase = True)
            features = self.vectorizer.fit_transform(dataset['text'].values)
            self.vocabulary = self.vectorizer.get_feature_names()
            pickle.dump(self.vocabulary, open(self.name + '.vocabulary', 'wb'))
            print('LR: Saved Vocabulary')
            
            print('LR: Starting to Train...')

            savedModel = LogisticRegression()
            savedModel = savedModel.fit(X=features, y=dataset['label'])
            
            pickle.dump(savedModel, open(self.name, 'wb'))
            print('LR: Saved Model')
            self.model = savedModel

class SVMModel(SAModel):    
    def tokenize(self, text): 
        tknzr = TweetTokenizer()
        return tknzr.tokenize(text)

    def loadVocabulary(self):
        model_data = joblib.load(self.name)

        print('Could not load ' + self.name)
        
        savedModel = model_data._final_estimator
        vocabulary = model_data.named_steps['countvectorizer'].vocabulary_
        
        print('SVM: Loaded Model and Vocabulary')
        
        self.model = savedModel
        self.vocabulary = vocabulary
        self.vectorizer = CountVectorizer(
            vocabulary = vocabulary,
            analyzer = 'word',
            tokenizer = self.tokenize,
            lowercase = True,
            ngram_range=(1, 1))

    def loadOrTrain(self, trainPath = ''):
        savedModel = None
        vocabulary = None

        try:
            self.loadVocabulary()
            savedModel = self.model
            vocabulary = self.vocabulary
        except:
            vocabulary = None
            savedModel = None

            if (trainPath == ''):
                raise Exception('SVM: Could not load model or vocabulary and no training path is specified')

        if (vocabulary is None) or (savedModel is None):
            dataset = self.readFromCSV(trainPath)
            print('SVM: Preparing Setup')

            self.vectorizer = CountVectorizer(
                analyzer = 'word',
                tokenizer = self.tokenize,
                lowercase = True,
                ngram_range=(1, 1))

            np.random.seed(1)

            pipeline_svm = make_pipeline(self.vectorizer, SVC(probability=True, kernel="linear", class_weight="balanced"))

            grid_svm = GridSearchCV(pipeline_svm,
                                    param_grid = {'svc__C': [0.1]},
                                    scoring="roc_auc",
                                    verbose=1000,
                                    n_jobs=-1) 

            print('SVM: Train Starts')
            grid_svm.fit(dataset['text'], dataset['label'])
            print('SVM: Train Ended')

            joblib.dump(grid_svm.best_estimator_, self.name, compress = 1)
            self.loadVocabulary()


def sampleRun():
    trainSamplePath = 'DataSample\\S140SampleTrain.csv'
    testSamplePath = 'DataSample\\S140SampleTest.csv'

    lr = LRModel('SampleLR', 'SampleLROutput') 
    svm = SVMModel('SampleSVM', 'SampleSVMOutput')
    
    lr.loadOrTrain(trainSamplePath)
    svm.loadOrTrain(trainSamplePath)
    
    testDataset = SAModel().readFromCSV(testSamplePath)

    lr_predicted = lr.predictData(testDataset['text'].values)
    svm_predicted = svm.predictData(testDataset['text'].values)

    lr.scoreModel(lr_predicted, testDataset['label'].values)
    svm.scoreModel(svm_predicted, testDataset['label'].values)