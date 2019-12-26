from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
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
import os

class SAModel(object):
    def __init__(self, modelName = '', outputFileName = '', labelColumn = 'label'):
        self.name = modelName
        self.outputName = outputFileName

        self.labelColumn = labelColumn

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

    def tokenize(self, text): 
        tknzr = TweetTokenizer()
        return tknzr.tokenize(text)

    def loadVocabulary(self):
        model_data = joblib.load(self.name)

        print('Could not load ' + self.name)
        
        savedModel = model_data._final_estimator
        vectorizer = model_data.named_steps['vect'] \
            if ('vect' in model_data.named_steps) \
            else model_data.named_steps['countvectorizer']
        
        print(self.name + ': Loaded Model and Vocabulary')
        
        self.model = savedModel
        self.vectorizer = vectorizer
        self.vocabulary = vectorizer.vocabulary_

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
                
    def readFromCSV(self, filePath, encoding='ISO-8859-1', alternateEncoding = 'utf-8'):
        """This method will read from a CSV file and return a dataframe
        Only the 'text' and 'label' columns will be taken in consideration
        """
        data_frame = None
        try:
            data_frame = pd.read_csv(filePath, engine='python', encoding=encoding, error_bad_lines=False)
        except:
            data_frame = pd.read_csv(filePath, engine='python', encoding=alternateEncoding, error_bad_lines=False)

        data_frame[self.labelColumn] = data_frame[self.labelColumn].map(lambda x: 1 if x == 'pos' else 0)
        data_frame['text'] = data_frame['text'].map(lambda x : str(x))

        return data_frame

class PipeLineModel(SAModel):
    def __init__(self, modelName = '', outputFileName = '', labelColumn = 'label', chosenModel = None, params = None):
        SAModel.__init__(self, modelName, outputFileName, labelColumn=labelColumn)

        self.ModelToTrain = chosenModel
        self.PipelineParameters = params

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
                raise Exception(self.name + ': Could not load model or vocabulary and no training path is specified') 

        if vocabulary == None:
            dataset = self.readFromCSV(trainPath)

            trainPipeline = Pipeline([
                ('vect', CountVectorizer()), 
                ('tfidf', TfidfTransformer()), 
                ('clf', self.ModelToTrain)
                ])

            print(self.name + ': Preparing to train')
            with np.errstate(divide='ignore'):
                gridModel = GridSearchCV(trainPipeline, 
                    self.PipelineParameters, 
                    cv=4, 
                    scoring='roc_auc',
                    n_jobs = -1,
                    verbose = 1)
                gridModel.fit(dataset['text'], dataset[self.labelColumn])

                print(self.name + ': Training Done')
                joblib.dump(gridModel.best_estimator_, self.name, compress = 1)
                self.loadVocabulary()

class MultinomialNBModel(PipeLineModel):
    def __init__(self, modelName = '', outputFileName = '', labelColumn = 'label'):
        PipeLineModel.__init__(self, modelName, outputFileName,
            labelColumn=labelColumn,
            chosenModel = MultinomialNB(),
            params = {
                'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
                'tfidf__use_idf': (True, False),
                'tfidf__norm': ('l1', 'l2'),
                'clf__alpha': [1, 1e-1, 1e-2]
            })

class RidgeClassifierModel(PipeLineModel):
    def __init__(self, modelName = '', outputFileName = '', labelColumn = 'label'):
        PipeLineModel.__init__(self, modelName, outputFileName,
            labelColumn=labelColumn, 
            chosenModel = RidgeClassifier(),
            params = {
                'vect__ngram_range': [(1, 1), (1, 2)],
                'tfidf__use_idf': [False],
                'tfidf__norm': ('l1', 'l2'),
                'clf__alpha': [10,1,0.1,0.01,0.001]
            })

class LRModel(PipeLineModel):
    def __init__(self, modelName = '', outputFileName = '', labelColumn = 'label'):
        PipeLineModel.__init__(self, modelName, outputFileName,
            labelColumn=labelColumn, 
            chosenModel = LogisticRegression(),
            params = {
                'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
                'tfidf__use_idf': (True, False),
                'tfidf__norm': ('l1', 'l2'),
                'clf__penalty' : ['l1', 'l2'],
                'clf__C' : np.logspace(-4, 4, 20),
                'clf__solver' : ['liblinear']
            })

class SVMModel(SAModel):
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
            grid_svm.fit(dataset['text'], dataset[self.labelColumn])
            print('SVM: Train Ended')

            joblib.dump(grid_svm.best_estimator_, self.name, compress = 1)
            self.loadVocabulary()

def sampleRun():
    trainSamplePath = os.path.join('DataSample', 'S140SampleTrain.csv')
    testSamplePath = os.path.join('DataSample', 'S140SampleTest.csv')

    lr = LRModel('SampleLR', 'SampleLROutput') 
    svm = SVMModel('SampleSVM', 'SampleSVMOutput')
    nb = MultinomialNBModel('SampleNB', 'SampleNBOutput')
    rc = RidgeClassifierModel('SampleRC', 'SampleRCOutput')

    testDataset = SAModel().readFromCSV(testSamplePath)

    for model in [lr, nb, rc, svm]:
        model.loadOrTrain(trainSamplePath)

        predicted = model.predictData(testDataset['text'].values)

        model.scoreModel(predicted, testDataset['label'].values)

def trueRun():
    labelColumn = 'polarity'
    for dataset in ['C1FE', 'C2FE', 'C3FE']:
        trainPath = os.path.join('Datasets', 'S140FE', dataset + 'Train.csv')
        testPath = os.path.join('Datasets', 'S140FE', dataset + 'Test.csv')

        lr = LRModel(dataset + 'LR', dataset + 'LROutput', labelColumn=labelColumn)
        nb = MultinomialNBModel(dataset + 'NB', dataset + 'NBOutput', labelColumn=labelColumn)
        rc = RidgeClassifierModel(dataset + 'RC', dataset + 'RCOutput', labelColumn=labelColumn)

        testDataset = SAModel(labelColumn=labelColumn).readFromCSV(testPath)

        for model in [lr, nb, rc]:
            model.loadOrTrain(trainPath)

            predicted = model.predictData(testDataset['text'].values)

            model.scoreModel(predicted, testDataset[labelColumn].values)

if __name__ == '__main__':
    trueRun()