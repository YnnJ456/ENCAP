class Predict:
    def __init__(self, dataX, modelList):
        self.dataX = dataX
        self.predVectorList = []
        self.probVectorList = []
        self.modelList = modelList

    def doPredict(self):
        for model in self.modelList:
            predVector = model.predict(self.dataX)
            self.predVectorList.append(predVector)
            probVector = model.predict_proba(self.dataX)
            self.probVectorList.append(probVector[:, 1])

        return self.predVectorList, self.probVectorList
