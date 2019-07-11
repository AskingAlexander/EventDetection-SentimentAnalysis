import argparse
from time import gmtime, strftime
from SA.SAModels import LRModel, SVMModel, SAModel
from ED.EDMethods import OLDA, MABED, DATASET_PATH, CLEANED_DATASET_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Selector')
    parser.add_argument('--taskType', type=str, help='The type of task to be executed: SA or ED. Default is None.', default=None)
    parser.add_argument('--trainFileSA', type=str, help='The path to the SA training file. Only \'text\' and \'label\' columns will be considered. Default is the sample.', default='DataSample/S140SampleTrain.csv')
    parser.add_argument('--testFileSA', type=str, help='The path to the SA test file. Only \'text\' and \'label\' columns will be considered. Default is the sample.', default='DataSample/S140SampleTest.csv')
    parser.add_argument('--modelTypeSA', type=str, help='Model type for SA: SVM or LR. Default is BOTH.', default='BOTH')
    parser.add_argument('--methodTypeED', type=str, help='ED task Type: MABED or OLDA. Default is BOTH.', default='BOTH')
    parser.add_argument('--numberOfTopics', type=int, help='The number of topics desired for the ED Task. Default is 10.', default=10)
    parser.add_argument('--numberOfWords', type=int, help='The maximum number of words desired for a Topic. Default is 5.', default=5)
    parser.add_argument('--datasedED', type=str, help='The path to the ED initial file. Only \'text\' and \'date\' columns will be considered. Default is the sample.', default=DATASET_PATH)
    parser.add_argument('--datasedCleannedED', type=str, help='The path to the ED final cleanned file. Only \'text\' and \'date\' columns will be considered. Default is the sample that will be created after the first run.', default=CLEANED_DATASET_PATH)

    args = parser.parse_args()

    if args.taskType != None:
        if args.taskType == 'SA':
            now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
            lr = LRModel('LR', 'LROutput' + now)
            svm = SVMModel('SVM', 'SVMOutput' + now)

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
                    modelType.scoreModel(modelType.predictData(testDataset['text'].values), testDataset['label'].values)
        else:
            if args.taskType == 'ED':

                mabed = MABED(numberOfTopics=args.numberOfTopics, numberOfWords=args.numberOfWords, datasetPath=args.datasedED, cleanedDatasetPath=args.datasedCleannedED)

                olda = OLDA(numberOfTopics=args.numberOfTopics, numberOfWords=args.numberOfWords, datasetPath=args.datasedED, cleanedDatasetPath=args.datasedCleannedED)

                modelsToUse = None

                if args.modelTypeSA == 'BOTH':
                    modelsToUse = [olda, mabed]
                else:
                    if args.modelTypeSA == 'MABED':
                        modelsToUse = [mabed]
                    else:
                        if args.modelTypeSA == 'OLDA':
                            modelsToUse = [olda]
                        else:
                            print('No valid model type was provided. Use --help to see the proper models.')

                if modelsToUse != None:
                    for modelType in modelsToUse:
                        modelType.run()
            else:
                print('\'' + args.taskType + '\'' + ' is not valid method. Chose either \'SA\' or \'ED\'')
    else:
        print('There is no task argument provided. Nothing will be done...\nCall --help to see the arguments.')