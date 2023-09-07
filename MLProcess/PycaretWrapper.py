from pycaret.classification import *
import os


class PycaretWrapper:
    def __init__(self):
        pass

    def doLoadModel(self, path, fileNameList=None):
        modelList = []
        for fileName in fileNameList:
            loadPath = os.path.join(path, fileName+'_final')
            loadedModel = load_model(loadPath)
            resultModel = loadedModel.named_steps.trained_model
            modelList.append(resultModel)

        return modelList
