import argparse
from time import gmtime, strftime

parser = argparse.ArgumentParser(description='Process Selector')
parser.add_argument('--taskType', type=str, help='The type of task to be executed: SA or ED. Default is None.', default=None)
parser.add_argument('--trainFileSA', type=str, help='The path to the SA training file. Only \'text\' and \'label\' columns will be considered. Default is the sample.', default='DataSample/S140SampleTrain.csv')
parser.add_argument('--testFileSA', type=str, help='The path to the SA test file. Only \'text\' and \'label\' columns will be considered. Default is the sample.', default='DataSample/S140SampleTest.csv')
parser.add_argument('--modelTypeSA', type=str, help='Model type for SA: SVM or LR. Default is BOTH.', default='BOTH')
parser.add_argument('--methodTypeED', type=str, help='ED task Type: MABED or OLDA. Default is None.', default=None)
parser.add_argument('--numberOfTopics', type=int, help='The number of topics desired for the ED Task. Default is 10.', default=10)
parser.add_argument('--numberOfWords', type=int, help='The maximum number of words desired for a Topic. Default is 5.', default=5)

args = parser.parse_args()

if args.taskType != None:
    if args.taskType == 'SA':
        from SA.SAModels import LRModel, SVMModel, SAModel

        now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        lr = LRModel('LR' + now, 'LROutput' + now)
        svm = SVMModel('SVM' + now, 'Output' + now)

        modelsToUse = None

        if args.modelTypeSA == 'BOTH':
            modelsToUse = [lr, svm]
        else:
            if args.modelTypeSA == 'LR':
                modelsToUse = [lr]
            else:
                if args.modelTypeSA == 'SVM':
                    modelsToUse = [svm]
                else:
                    print('No valid model type was provided. Use --help to see the proper models.')

        if modelsToUse != None:
            for modelType in modelsToUse:
                modelType.loadOrTrain(args.trainFileSA)
            
            testDataset = SAModel().readFromCSV(args.testFileSA)

            for modelType in modelsToUse:
                modelType.scoreModel(modelType.predictData(testDataset['text'].values))
    else:
        if args.taskType == 'ED':
            from ED.EDMethods import OLDA
        else:
            print('\'' + args.taskType + '\'' + ' is not valid method. Chose either \'SA\' or \'ED\'')
else:
    print('There is no task argument provided. Nothing will be done...\nCall --help to see the arguments.')