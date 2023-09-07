from userPackage.PackagePredict import TOP_TTCA_Predict

dataset = 'DS1'
if dataset == 'DS1':
    model_use = '1'
elif dataset == 'DS2':
    model_use = '2'

pathDict = {'paramPath': f'../data/param/{dataset}/',
            'saveCsvPath': '../data/mlData/new_data/',
            'modelPath': f'../data/finalModel/{dataset}/',
            'scorePath': '../data/mlScore/'}

testNegFastaPath = '../data/mlData/DS1/test_neg.FASTA'
testPosFastaPath = '../data/mlData/DS1/test_pos.FASTA'

topObj = TOP_TTCA_Predict(model_use=model_use, pathDict=pathDict, modelNameList=['catboost', 'et', 'gbc'])
topObj.loadData(testNegFastaPath=testNegFastaPath, testPosFastaPath=testPosFastaPath)
topObj.featureEncode()
topObj.doPredict()
scoreDf = topObj.doScoring()
