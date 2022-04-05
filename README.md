# plasticCCS
Prediction of collision cross section values for extractables and leachables from plastic products

This repository mainly shows the R code for predicting the collision cross section (CCS) values for extractables and lachables from plastic products.

We present here two machine learning algorithms for building prediction models: Support Vector Machine (SVM) and Extreme Gradient Boosting (XGBoost). 
However, We recommend to use SVM algorithms, due to its easy configuration with few hyperparameters, its ability to provide more 
accurate predictions without much tuning effort, as well as its ability to provide repeatable prediction results.


What should be mentioned:
The molecular weight (MW) is calculated by Monoisotopic Mass + mass of adducts (1.0073 for [M+H]+, 22.9892 for [M+Na]+). For example, when predicting the 
CCS of protonated Tris(2,4-ditert-butylphenyl)phosphate (Monoisotopic Mass: 662.4464 Da), its MW in model is 663.4547, not 662.4464.



