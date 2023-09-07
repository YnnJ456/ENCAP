import pickle
import pandas as pd


class Normalization:
    def __init__(self, testDf):
        self.testDfIndex = testDf.index.to_list()
        self.testDfAnswer = testDf[['y']]
        testDfFeature = testDf.drop('y', axis=1)
        self.testDfFeatureCol = testDfFeature.columns.to_list()
        self.testArray = testDfFeature.values

    def robustTest(self, loadNmlzParamsPklPath=None):
        with open(loadNmlzParamsPklPath, 'rb') as f:
            robustSca = pickle.load(f)
            scalerDf = robustSca.fit_transform(self.testArray)
        scalerDf = pd.DataFrame(scalerDf, index=self.testDfIndex, columns=self.testDfFeatureCol)
        scalerDf.insert(scalerDf.shape[1], "y", self.testDfAnswer)
        return scalerDf
