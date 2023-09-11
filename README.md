# TOP-TTCA
TOP-TTCA: Prediction of Tumor T Cell Antigens Using Two-stage Optimization of Machine Learning Models 

# Description
This is the source code of TOP-TTCA, a machine learning predictor for tumor T cell antigen based on two-stage optimization (manuscript under review). The first stage is the optimization of feature vector, and the second stage is the optimization of machine learning models. The trained models are included in this package, facilitating prediction on a given data set.

# Install
Requiremenets:
* Python = 3.8, pycaret[full] = 2.3.10

Packages
* Install required packages using `pip install -r requirements.txt`

# Usage
Modify main_predict.py for your data set in fasta format
* Input file
  * Fasta file
  
* output file
  * binary_vector.csv -- The prediction output in binary format (1 for positive and 0 for negative)
    
    ![image](https://github.com/YnnJ456/TOP-TTCA/assets/95170485/3f50bd44-ff22-440e-b94d-b7a06ff75768)


  * probability.csv -- The prediction probability estimate
    
    ![image](https://github.com/YnnJ456/TOP-TTCA/assets/95170485/51b32ee7-80f6-4601-b47e-91486876fd4a)



When dataset = 'DS1', the program will use DS1's models, features and normalize scaler to process data and perform prediction.

When dataset = 'DS2' , the program will use DS2's models, features and normalize scaler to process data and perform prediction.
```py
# If you want to use different model, you can change dataset
dataset = 'DS1'
if dataset == 'DS1':
    model_use = '1'
elif dataset == 'DS2':
    model_use = '2'
```

```py
# Path setting
pathDict = {'paramPath': f'../data/param/{dataset}/',  # This path should have featureTypeDict.pkl and robust.pkl
            'saveCsvPath': '../data/mlData/new_data/',  # Your encoded data will save in this path
            'modelPath': f'../data/finalModel/{dataset}/',  # This path should have catboost, et, gbc models. ex: catboost_final.pkl
            'outputPath': '../data/output/'}  # Your prediction will save in this path
```

If you do not have positive data or negative data, you can input None

0 = Negative Data, 1 = Positive Data, -1 = No Label Data

```py
# Input your FASTA file, the example file can find in data/mlData/DS1/test_neg.FASTA
inputDataDict = {0: '../data/mlData/DS1/test_neg.FASTA',
                 1: '../data/mlData/DS1/test_pos.FASTA',
                 -1: None}
```

Here is the code in main_predict.py of which parameters are set and the program is ready to be excecuted.

If your data have label, please set True in havelabel, else set False

```py
# If your data have label, please set True in havelabel, else set False
topObj = TOP_TTCA_Predict(model_use=model_use, pathDict=pathDict, haveLabel=True)
topObj.loadData(inputDataDict=inputDataDict)
topObj.featureEncode()
topObj.doPredict()
```
