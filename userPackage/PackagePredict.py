from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
import pandas as pd
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Scoring import Scoring
from MLProcess.Predict import Predict


class TOP_TTCA_Predict:

    def __init__(self, model_use, pathDict, modelNameList):
        self.model_use = model_use
        self.pathDict = pathDict
        self.modelNameList = modelNameList
        self.dataDict = {}
        self.predVectorDf = None
        self.probVectorDf = None
        self.dataTest_y = None
        self.predVectorListIndp = None
        self.probVectorListIndp = None
        if self.model_use == '1':
            self.featureNum = 150
            self.featureTypeDictPkl = 'DS1_featureTypeDict.pkl'
            self.nmlzPkl = 'DS1_robust.pkl'
            self.featureRankCsv = '../data/mlData/DS1/featureRank_DS1.csv'
        elif self.model_use == '2':
            self.featureNum = 210
            self.featureTypeDictPkl = 'DS2_featureTypeDict.pkl'
            self.nmlzPkl = 'DS2_robust.pkl'
            self.featureRankCsv = '../data/mlData/DS2/featureRank_DS2.csv'
        else:
            raise NameError('model_use should input 1 or 2, 1 = DS1, 2 = DS2')

    def loadData(self, testNegFastaPath, testPosFastaPath):
        ldObj = LoadDataset()
        testNegSeqDict = ldObj.readFasta(testNegFastaPath, minSeqLength=5)
        testPosSeqPath = ldObj.readFasta(testPosFastaPath, minSeqLength=5)
        self.dataDict = {'testNegSeqDict': testNegSeqDict,
                         'testPosSeqDict': testPosSeqPath}

    def featureEncode(self):
        encodeObj = EncodeAllFeatures(dataDict=self.dataDict)
        encodeObj.dataEncode(loadPlkPath=f'{self.pathDict["paramPath"]}/{self.featureTypeDictPkl}')
        testDf = encodeObj.dataNormalization(loadNmlzScalerPklPath=f'{self.pathDict["paramPath"]}/{self.nmlzPkl}')
        featureDf = pd.read_csv(self.featureRankCsv)
        featureList = featureDf['feature'].to_list() + ['y']
        testDf = testDf[featureList]
        testDf.to_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv')

    def doPredict(self):
        pycObj = PycaretWrapper()
        modelList = pycObj.doLoadModel(path=self.pathDict['modelPath'], fileNameList=self.modelNameList)
        dataTestDf = pd.read_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv', index_col=[0])
        dataTest_X = dataTestDf.drop(["y"], axis=1)
        self.dataTest_y = dataTestDf[["y"]]
        predObjIndp = Predict(dataX=dataTest_X, modelList=modelList)
        self.predVectorListIndp, self.probVectorListIndp = predObjIndp.doPredict()
        self.predVectorDf = pd.DataFrame(self.predVectorListIndp, index=self.modelNameList, columns=dataTestDf.index).T
        self.probVectorDf = pd.DataFrame(self.probVectorListIndp, index=self.modelNameList, columns=dataTestDf.index).T
        self.predVectorDf.to_csv(f'{self.pathDict["scorePath"]}/binary_vector.csv')
        self.probVectorDf.to_csv(f'{self.pathDict["scorePath"]}/probability_vector.csv')

    def doScoring(self):
        scoreObj = Scoring(predVectorList=self.predVectorListIndp, probVectorList=self.probVectorListIndp,
                           answerDf=self.dataTest_y, modelNameList=self.modelNameList)
        scoreObj.optimizeMcc(cutOffList=[0.2, 0.4], method='mcc')
        scoreDfIndp = scoreObj.doScoring(b_optimizedMcc=True, sortColumn='mcc', path=f'{self.pathDict["scorePath"]}/test_score.csv')

        return scoreDfIndp
