# TOP-TTCA
TOP-TTCA: Prediction of Tumor T Cell Antigens Using Two-stage Optimization of Machine Learning Models 
# Abstract
Cancer immunotherapy enhances the body’s natural immune system to combat cancer, offering the advantage of lowered side effects compared to traditional treatments because of its high selectivity and efficacy. Utilizing computational methods to identify tumor T cell antigens (TTCAs) is valuable in unraveling biological mechanisms and enhancing the effectiveness of immunotherapy. In this study, we present TOP-TTCA, a machine learning predictor based on two-stage optimization. Sequences are encoded as a feature vector of 4349 entries based on 57 feature types, followed by optimization of feature subset and machine learning models in the first and the second stages of the proposed pipeline, respectively. Benchmark evaluations show that TOP-TTCA generates 4.8% and 13.5% improvements in Matthew’s correlation coefficient (MCC) over the state-of-the-art methods on two popular TTCA data sets, respectively. For the third test data set of 71 experimentally validated TTCAs from the literature, our best model yields prediction accuracy of 0.873, achieving improvements ranging from 12% to 25.7% compared to three state-of-the-art methods. The optimized feature subsets are primarily composed of physicochemical properties, with several features specifically related to hydrophobicity and amphiphilicity.
# Install
Requiremenets:
* Python3.8, pycaret[full]2.3.10
Packages
* Install required packages using `pip install -r requirements.txt`
