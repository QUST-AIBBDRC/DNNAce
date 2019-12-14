##DNNAce

DNNAce: prediction of prokaryote lysine acetylation sites through deep neural networks with multi-information fusion.


###DNNAce uses the following dependencies:
* MATLAB2014a
* python 3.6 
* numpy
* scipy
* scikit-learn
* TensorFlow 
* keras


###Guiding principles:

**The dataset file contains nine categories lysine acetylation sites datasets, which contain training dataset and independent test dataset.

**feature extraction methods:
1) Sequence-based features: 
   BE_DATA.m is the implementation of BE.
   PAAC.py is the implementation of PseAAC.
2) Physicochemical property-based features: 
   AAindex.py is the implementation of AAindex.
   EBGW_DATA.m is the implementation of EBGW.
   MMI_DATA.m is the implementation of MMI.
3) Evolutionary-derived features:
   BLOSUM62.py is the implementation of BLOSUM62.
   KNN.py is the implementation of KNN.
   
** feature selection methods:
   Elastic_net .py represents the elastic.
   ET.py represents the Extra-Trees.
   Group_Lasso.py represents the Group Lasso.
   IG.py represents the information gain.
   LR.py represents the logistic regression.
   MI.py represents the mutual information.
   SVD.py represents the singular value decomposition.

** Classifier:
   AdaBoost.py is the implementation of AdaBoost.
   DNN.py is the implementation of DNN.
   KNN.py is the implementation of KNN.
   NB.py is the implementation of NB.
   RF.py is the implementation of RF.
   SVM.py is the implementation of SVM.
   XGBoost.py is the implementation of XGBoost.
   CNN.py is the implementation of CNN.
   LSTM.py is the implementation of LSTM.

