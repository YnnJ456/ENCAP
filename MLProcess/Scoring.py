import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, matthews_corrcoef, confusion_matrix
import pandas as pd


class Scoring:
    def __init__(self, predVectorList, probVectorList, answerDf, modelNameList=None):
        self.predVectorList = predVectorList
        self.probVectorList = probVectorList
        self.predVectorList_cutOff = None
        self.answerDf = answerDf
        self.modelNameList = []
        self.bestCutOffList = None
        if modelNameList is None:
            modelNameList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm', 'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        for modelName in modelNameList:
            self.modelNameList.append(modelName.lower())
        if len(self.predVectorList) != len(self.modelNameList) or len(self.probVectorList) != len(self.modelNameList):
            print('predVectorList length not equal to modelNameList length')

    def doScoring(self, b_optimizedMcc=False, path=None, sortColumn='mcc'):
        scoreList = []
        if b_optimizedMcc:
            predVectorList = self.predVectorList_cutOff
            bestCutOffList = self.bestCutOffList
        else:
            predVectorList = self.predVectorList
            bestCutOffList = [0.5] * len(self.modelNameList)
        for (predVector, probVector, bestCutOff) in zip(predVectorList, self.probVectorList, bestCutOffList):
            fpr, tpr, threshold = roc_curve(self.answerDf, probVector)
            auc1 = auc(fpr, tpr)
            if len(confusion_matrix(self.answerDf, predVector).ravel()) < 4:
                specificity = None
            else:
                tn, fp, fn, tp = confusion_matrix(self.answerDf, predVector).ravel()
                specificity = tn / (tn + fp)
            scoreDict = {"accuracy": accuracy_score(self.answerDf, predVector),
                         "precision": precision_score(self.answerDf, predVector),
                         "recall": recall_score(self.answerDf, predVector),
                         "f1_score": f1_score(self.answerDf, predVector),
                         "auc": auc1,
                         "specificity": specificity,
                         "BACC": (specificity + recall_score(self.answerDf, predVector))/2,
                         "mcc": matthews_corrcoef(self.answerDf, predVector),
                         "bestCutoff": bestCutOff}
            scoreList.append(scoreDict)
        scoreDf = pd.DataFrame(scoreList, index=[self.modelNameList])
        if sortColumn is not None:
            scoreDf = scoreDf.sort_values(by=sortColumn, ascending=False)
        if path is not None:
            scoreDf.to_csv(path)
        return scoreDf

    def optimizeMcc(self, cutOffList=None, method='mcc'):
        if cutOffList is None:
            cutOffList = [0.5]
        probDf = pd.DataFrame(self.probVectorList).T
        probDf.columns = self.modelNameList
        scoreDf = pd.DataFrame(index=cutOffList)
        predArrDf = pd.DataFrame(index=cutOffList)
        for modelName in self.modelNameList:
            scoreList = []
            predArrList = []
            for cutOff in cutOffList:
                probDf.loc[probDf[modelName] > cutOff, 'predVector_' + str(modelName)] = 1
                probDf.loc[probDf[modelName] <= cutOff, 'predVector_' + str(modelName)] = 0
                predArr = np.array(probDf['predVector_' + str(modelName)].values.tolist())
                predArrList.append(predArr)
                if method == 'mcc':
                    mcc = matthews_corrcoef(self.answerDf, predArr)
                    scoreList.append(mcc)
                elif method == 'acc':
                    acc = accuracy_score(self.answerDf, predArr)
                    scoreList.append(acc)
                elif method == 'auc':
                    fpr, tpr, threshold = roc_curve(self.answerDf, predArr)
                    auc1 = auc(fpr, tpr)
                    scoreList.append(auc1)
            scoreDf[modelName] = scoreList
            predArrDf[modelName] = predArrList
        mccMaxSeries = scoreDf.idxmax(axis=0)
        mccMaxList = mccMaxSeries.tolist()
        bestPredArrList = []
        for (modelName, mccMax) in zip(self.modelNameList, mccMaxList):
            bestPredArr = predArrDf[modelName].loc[mccMax]
            bestPredArrList.append(bestPredArr)
        self.bestCutOffList = mccMaxList
        self.predVectorList_cutOff = bestPredArrList
