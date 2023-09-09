# TOP-TTCA
TOP-TTCA: Prediction of Tumor T Cell Antigens Using Two-stage Optimization of Machine Learning Models 

# Abstract
Cancer immunotherapy enhances the body’s natural immune system to combat cancer, offering the advantage of lowered side effects compared to traditional treatments because of its high selectivity and efficacy. Utilizing computational methods to identify tumor T cell antigens (TTCAs) is valuable in unraveling biological mechanisms and enhancing the effectiveness of immunotherapy. In this study, we present TOP-TTCA, a machine learning predictor based on two-stage optimization. Sequences are encoded as a feature vector of 4349 entries based on 57 feature types, followed by optimization of feature subset and machine learning models in the first and the second stages of the proposed pipeline, respectively. Benchmark evaluations show that TOP-TTCA generates 4.8% and 13.5% improvements in Matthew’s correlation coefficient (MCC) over the state-of-the-art methods on two popular TTCA data sets, respectively. For the third test data set of 71 experimentally validated TTCAs from the literature, our best model yields prediction accuracy of 0.873, achieving improvements ranging from 12% to 25.7% compared to three state-of-the-art methods. The optimized feature subsets are primarily composed of physicochemical properties, with several features specifically related to hydrophobicity and amphiphilicity.

# Install
Requiremenets:
* Python = 3.8, pycaret[full] = 2.3.10

Packages
* Install required packages using `pip install -r requirements.txt`

# Usage
The only main program is main_predict.py
* Input file
  * Fasta file
  
* output file
  * binary_vector.csv -- The 0 or 1 predictions.
  * probability.csv -- the probability predictions.


When dataset = 'DS1' will use DS1's models, features and normalize scaler.

When dataset = 'DS2' will use DS2's models, features and normalize scaler.
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
            'scorePath': '../data/mlScore/'}  # Your score will save in this path
```

If you don't have positive data or negative data, you can input None
```py
# Input your FASTA file, the example file can find in data/mlData/DS1/test_neg.FASTA
testNegFastaPath = '../data/mlData/DS1/test_neg.FASTA'
testPosFastaPath = '../data/mlData/DS1/test_pos.FASTA'
```

```py
topObj = TOP_TTCA_Predict(model_use=model_use, pathDict=pathDict, modelNameList=['catboost', 'et', 'gbc'])
topObj.loadData(testNegFastaPath=testNegFastaPath, testPosFastaPath=testPosFastaPath)
topObj.featureEncode()
topObj.doPredict()
```
