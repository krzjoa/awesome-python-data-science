# Awesome Python Data Science
Curated list of data science software in Python

[skl]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/skl.png "scikit-learn logo" 
[th]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/th.png "Theano logo" 
[tf]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/tf.png "TensorFlow logo" 
[pt]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/pytorch.png "PyTorch logo" 
[gpu]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/gpu.png "GPU badge"

###### Legend:
![alt text][skl] - [scikit-learn](http://scikit-learn.org/stable/) compatible API <br/>
![alt text][th] - [Theano](http://scikit-learn.org/stable/) based project <br/>
![alt text][tf] - [TensorFlow](http://scikit-learn.org/stable/) based project <br/>
![alt text][pt] - [PyTorch](http://scikit-learn.org/stable/) based project <br/>
![alt text][gpu] - GPU-accelerated computations


## General purpouse Machine Learning
* [scikit-learn](http://scikit-learn.org/stable/) ![alt text][skl] - machine learning in Python
* [Shogun](http://www.shogun-toolbox.org/) - machine learning toolbox
* [MLxtend](https://github.com/rasbt/mlxtend) ![alt text][skl] - extension and helper modules for Python's data analysis and machine learning libraries
* [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) ![alt text][skl] - multi-label classification for python
* [tslearn](https://github.com/rtavenar/tslearn) ![alt text][skl] - machine learning toolkit dedicated to time-series data

## Ensemble methods
* [ML-Ensemble](http://ml-ensemble.com/) ![alt text][skl] -  high performance ensemble learning 
* [brew](https://github.com/viisar/brew) ![alt text][skl] - Python Ensemble Learning API
* [Stacking](https://github.com/ikki407/stacking) ![alt text][skl] - Simple and useful stacking library, written in Python.
## Feature engineering
* [Featuretools](https://github.com/Featuretools/featuretools) - automated feature engineering
* [scikit-feature](https://github.com/jundongl/scikit-feature) -  feature selection repository in python
* [skl-groups](https://github.com/dougalsutherland/skl-groups) ![alt text][skl] - scikit-learn addon to operate on set/"group"-based features
* [Feature Forge](https://github.com/machinalis/featureforge) ![alt text][skl] - a set of tools for creating and testing machine learning feature
* [boruta_py](https://github.com/scikit-learn-contrib/boruta_py) ![alt text][skl] -  implementations of the Boruta all-relevant feature selection method
* [BoostARoota](https://github.com/chasedehan/BoostARoota) ![alt text][skl] - a fast xgboost feature selection algorithm

## Gradient boosting
* [XGBoost](https://github.com/dmlc/xgboost) ![alt text][skl] ![alt text][gpu]  - Scalable, Portable and Distributed Gradient Boosting 
* [LightGBM](https://github.com/Microsoft/LightGBM) ![alt text][skl] ![alt text][gpu] - a fast, distributed, high performance gradient boosting by Microsoft
* [CatBoost](https://github.com/catboost/catboost) ![alt text][skl] ![alt text][gpu] - an open-source gradient boosting on decision trees library by Yandex
* [TGBoost](https://github.com/wepe/tgboost) ![alt text][skl] - Tiny Gradient Boosting Tree

## Resampling & augmentations
* [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) ![alt text][skl]  - module to perform under sampling and over sampling with various techniques
* [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms) ![alt text][skl] ![alt text][tf]  - Python-based implementations of algorithms for learning on imbalanced data.

## Data manipulation
* [pandas](https://pandas.pydata.org/pandas-docs/stable/) - powerful Python data analysis toolkit
* [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) ![alt text][skl]  - Pandas integration with sklearn
* [alexander](https://github.com/annoys-parrot/alexander) ![alt text][skl] - wrapper that aims to make scikit-learn fully compatible with pandas

## Deep Learning

### Tensorflow
* [TensorFlow](https://github.com/tensorflow/tensorflow) ![alt text][tf] - omputation using data flow graphs for scalable machine learning by Google
* [TensorLayer](https://github.com/zsdonghao/tensorlayer) ![alt text][tf] - Deep Learning and Reinforcement Learning Library for Researcher and Engineer.
* [TFLearn](https://github.com/tflearn/tflearn) ![alt text][tf] - Deep learning library featuring a higher-level API for TensorFlow

### Theano
**WARNING: Theano development has been stopped**
* [Theano](https://github.com/Theano/Theano)![alt text][th] - is a Python library that allows you to define, optimize, and evaluate mathematical expressions
* [Lasagne](https://github.com/Lasagne/Lasagne) ![alt text][th] - Lightweight library to build and train neural networks in Theano
* [nolearn](https://github.com/dnouri/nolearn) ![alt text][th] ![alt text][skl] - scikit-learn compatible neural network library (mainly for Lasagne)

### PyTorch
* [PyTorch](https://github.com/pytorch/pytorch) ![alt text][pt]  - Tensors and Dynamic neural networks in Python with strong GPU acceleration 
* [skorch](https://github.com/dnouri/skorch) ![alt text][skl] ![alt text][pt]  - a scikit-learn compatible neural network library that wraps pytorch
* [PyTorchNet](https://github.com/pytorch/tnt) ![alt text][pt]  - an abstraction to train neural networks

## Experiments tools
* [Xcessiv](https://github.com/reiinakano/xcessiv) - a web-based application for quick, scalable, and automated hyperparameter tuning and stacked ensembling

## Automated machine learning

## Probabilistic methods
* [skggm](https://github.com/skggm/skggm) ![alt text][skl] - estimation of general graphical models 
* [bayesloop](https://github.com/christophmark/bayesloop) - Probabilistic programming framework that facilitates objective model selection for time-varying parameter models
* [pyro](https://github.com/uber/pyro) ![alt text][pt] - Pyro is a flexible, scalable deep probabilistic programming library built on PyTorch.
* [ZhuSuan](http://zhusuan.readthedocs.io/en/latest/) ![alt text][tf] - Bayesian Deep Learning
* [pomegranate](https://github.com/jmschrei/pomegranate) ![alt text][gpu] - probabilistic and graphical models for Python]
* [pyMC3](http://docs.pymc.io/) ![alt text][th] - Python package for Bayesian statistical modeling and Probabilistic Machine Learning
* [Edward](http://edwardlib.org/) ![alt text][tf] - A library for probabilistic modeling, inference, and criticism.
* [GPflow - Gaussian processes in TensorFlow](http://gpflow.readthedocs.io/en/latest/?badge=latest)
* [Stan](https://github.com/stan-dev/stan)
* [gelato](https://github.com/ferrine/gelato) ![alt text][th] - Bayesian dessert for Lasagne

## Natural Language Processing
* [NLTK](https://github.com/nltk/nltk) -  modules, data sets, and tutorials supporting research and development in Natural Language Processing
* [CLTK](https://github.com/cltk/cltk) - The Classical Language Toolkik
* [PSI-Toolkit](http://psi-toolkit.amu.edu.pl/) - a natural language processing toolkit by Adam Mickiewicz University in Poznań
