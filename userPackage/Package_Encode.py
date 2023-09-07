from devPackage.PackageModelAmp import EncodeModelAmp
from devPackage.PackageiFeature import iFeature
from devPackage.PackagePFeature import PFeature
from devPackage.OVP import OVP
from devPackage.MotifBitVec import MotifBitVec
from devPackage.Normalization import Normalization
import pandas as pd
import pickle


class EncodeAllFeatures:
    def __init__(self, dataDict):
        self.indpDf = None
        self.dataDictValueList = list(dataDict.values())
        self.answerList = [0, 1, 0, 1]

    def dataEncode(self, loadPlkPath=None):
        encodedDfList = []
        path = loadPlkPath
        with open(path, 'rb') as f:
            loadFeatureDict = pickle.load(f)
        for (dataDictValue, answer) in zip(self.dataDictValueList, self.answerList):
            if dataDictValue is None:
                encodedDfList.append(None)
            else:
                eifObj = iFeature(dataDictValue, loadFeatureDict['iFeature'])
                epfObj = PFeature(dataDictValue, loadFeatureDict['pFeature'])
                emaObj = EncodeModelAmp(dataDictValue, loadFeatureDict['ampFeature'])  # windows 拉出去dict
                eovpObj = OVP(dataDictValue, loadFeatureDict['ovpFeature'])
                embvObj = MotifBitVec(dataDictValue, loadFeatureDict['motifBitVecFeature'])
                a = eifObj.getOutputDf()
                b = epfObj.getOutputDf()
                c = emaObj.getOutputDf()
                d = eovpObj.getOutputDf()
                e = embvObj.getOutputDf()
                encodedDf = pd.concat([a, b], axis=1)
                encodedDf = pd.concat([encodedDf, c], axis=1)
                encodedDf = pd.concat([encodedDf, d], axis=1)
                encodedDf = pd.concat([encodedDf, e], axis=1)
                encodedDf.insert(encodedDf.shape[1], 'y', answer)
                encodedDfList.append(encodedDf)
        if len(encodedDfList) == 2:
            indpDf = pd.concat([encodedDfList[0], encodedDfList[1]])
        self.indpDf = indpDf

    def dataNormalization(self, loadNmlzScalerPklPath='./data/'):
        nmlzObj = Normalization(testDf=self.indpDf)
        indpNmlzDf = nmlzObj.robustTest(loadNmlzParamsPklPath=loadNmlzScalerPklPath)
        self.indpDf = indpNmlzDf

        return self.indpDf
