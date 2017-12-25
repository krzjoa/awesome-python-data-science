# Awesome Python Data Science ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
Curated list of data science software in Python

[skl]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/skl.png "scikit-learn logo" 
[th]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/th.png "Theano logo" 
[tf]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/tf.png "TensorFlow logo" 
[pt]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/pytorch.png "PyTorch logo" 
[cp]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/cupy.png "CuPy badge"
[gpu]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/gpu.png "GPU badge"

###### Legend:
![alt text][skl] - [scikit-learn](http://scikit-learn.org/stable/) compatible (or inspired) API <br/>
![alt text][th] - [Theano](http://deeplearning.net/software/theano/) based project <br/>
![alt text][tf] - [TensorFlow](https://www.tensorflow.org/) based project <br/>
![alt text][pt] - [PyTorch](http://pytorch.org/) based project <br/>
![alt text][cp] - [CuPy](https://github.com/cupy/cupy/) based project <br/>
![alt text][gpu] - GPU-accelerated computations (if not based on Theano, Tensorflow, PyTorch or CuPy)


## General purpouse Machine Learning
* [scikit-learn](http://scikit-learn.org/stable/) ![alt text][skl] - machine learning in Python
* [Shogun](http://www.shogun-toolbox.org/) - machine learning toolbox
* [MLxtend](https://github.com/rasbt/mlxtend) ![alt text][skl] - extension and helper modules for Python's data analysis and machine learning libraries
* [sklearn-extensions](https://github.com/wdm0006/sklearn-extensions) ![alt text][skl] - a consolidated package of small extensions to scikit-learn 
* [civisml-extensions](https://github.com/civisanalytics/civisml-extensions) ![alt text][skl]  - scikit-learn-compatible estimators from Civis Analytics
* [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) ![alt text][skl] - multi-label classification for python
* [tslearn](https://github.com/rtavenar/tslearn) ![alt text][skl] - machine learning toolkit dedicated to time-series data
* [seqlearn](https://github.com/larsmans/seqlearn) ![alt text][skl] - seqlearn is a sequence classification toolkit for Python
* [pystruct](https://github.com/pystruct/pystruct) ![alt text][skl] - Simple structured learning framework for python
* [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys) ![alt text][skl] - Highly interpretable classifiers for scikit learn, producing easily understood decision rules instead of black box models
* [skutil](https://github.com/tgsmith61591/skutil) ![alt text][skl] - A set of scikit-learn and h2o extension classes (as well as caret classes for python)
* [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite) ![alt text][skl] - scikit-learn inspired API for CRFsuite
* [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm) ![alt text][skl] - Relevance Vector Machine implementation using the scikit-learn API
* [RuleFit](https://github.com/christophM/rulefit) ![alt text][skl] - implementation of the rulefit 
* [Python-ELM](https://github.com/dclambert/Python-ELM) ![alt text][skl]  - Extreme Learning Machine implementation in Python
* [Python Extreme Learning Machine (ELM)](https://github.com/acba/elm) - a machine learning technique used for classification/regression tasks
* [hpelm](https://github.com/akusok/hpelm) ![alt text][gpu]  - High performance implementation of Extreme Learning Machines (fast randomized neural networks).
* [pyFM](https://github.com/coreylynch/pyFM) ![alt text][skl] - Factorization machines in python
* [fastFM](https://github.com/ibayer/fastFM) ![alt text][skl] - a library for Factorization Machines
* [tffm](https://github.com/geffy/tffm) ![alt text][skl] ![alt text][tf] - TensorFlow implementation of an arbitrary order Factorization Machine


## Ensemble methods
* [ML-Ensemble](http://ml-ensemble.com/) ![alt text][skl] -  high performance ensemble learning 
* [brew](https://github.com/viisar/brew) ![alt text][skl] - Python Ensemble Learning API
* [Stacking](https://github.com/ikki407/stacking) ![alt text][skl] - Simple and useful stacking library, written in Python.
* [stacked_generalization](https://github.com/fukatani/stacked_generalization) ![alt text][skl] - library for machine learning stacking generalization.

## Feature engineering
* [Featuretools](https://github.com/Featuretools/featuretools) - automated feature engineering
* [scikit-feature](https://github.com/jundongl/scikit-feature) -  feature selection repository in python
* [skl-groups](https://github.com/dougalsutherland/skl-groups) ![alt text][skl] - scikit-learn addon to operate on set/"group"-based features
* [Feature Forge](https://github.com/machinalis/featureforge) ![alt text][skl] - a set of tools for creating and testing machine learning feature
* [boruta_py](https://github.com/scikit-learn-contrib/boruta_py) ![alt text][skl] -  implementations of the Boruta all-relevant feature selection method
* [BoostARoota](https://github.com/chasedehan/BoostARoota) ![alt text][skl] - a fast xgboost feature selection algorithm

## Gradient boosting
* [XGBoost](https://github.com/dmlc/xgboost) ![alt text][skl] ![alt text][gpu]  - Scalable, Portable and Distributed Gradient Boosting 
* [LightGBM](https://github.com/Microsoft/LightGBM) ![alt text][skl] ![alt text][gpu] - a fast, distributed, high performance gradient boosting by [Microsoft](https://www.microsoft.com)
* [CatBoost](https://github.com/catboost/catboost) ![alt text][skl] ![alt text][gpu] - an open-source gradient boosting on decision trees library by [Yandex](https://www.yandex.com/)
* [TGBoost](https://github.com/wepe/tgboost) ![alt text][skl] - Tiny Gradient Boosting Tree

## Resampling & augmentations
* [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) ![alt text][skl]  - module to perform under sampling and over sampling with various techniques
* [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms) ![alt text][skl] ![alt text][tf]  - Python-based implementations of algorithms for learning on imbalanced data.

## Data manipulation & pipelines
* [pandas](https://pandas.pydata.org/pandas-docs/stable/) - powerful Python data analysis toolkit
* [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) ![alt text][skl]  - Pandas integration with sklearn
* [alexander](https://github.com/annoys-parrot/alexander) ![alt text][skl] - wrapper that aims to make scikit-learn fully compatible with pandas
* [blaze](https://github.com/blaze/blaze) - NumPy and Pandas interface to Big Data
* [pandasql](https://github.com/yhat/pandasql) -  allows you to query pandas DataFrames using SQL syntax
* [pandas-gbq](https://github.com/pydata/pandas-gbq) - Pandas Google Big Query
* [xpandas](https://github.com/alan-turing-institute/xpandas) - universal 1d/2d data containers with Transformers functionality for data analysis by [The Alan Turing Institute](https://www.turing.ac.uk/)
* [Fuel](https://github.com/mila-udem/fuel) - data pipeline framework for machine learning

## Deep Learning

### Keras
* [Keras](https://keras.io) - a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano
* [Hyperas](https://github.com/maxpumperla/hyperas) - Keras + Hyperopt: A very simple wrapper for convenient hyperparameter 
* [Elephas](https://github.com/maxpumperla/elephas) - Distributed Deep learning with Keras & Spark 
* [Hera](https://github.com/keplr-io/hera) - Train/evaluate a Keras model, get metrics streamed to a dashboard in your browser.

### Tensorflow
* [TensorFlow](https://github.com/tensorflow/tensorflow) ![alt text][tf] - omputation using data flow graphs for scalable machine learning by Google
* [TensorLayer](https://github.com/zsdonghao/tensorlayer) ![alt text][tf] - Deep Learning and Reinforcement Learning Library for Researcher and Engineer.
* [TFLearn](https://github.com/tflearn/tflearn) ![alt text][tf] - Deep learning library featuring a higher-level API for TensorFlow

### Theano
**WARNING: Theano development has been stopped**
* [Theano](https://github.com/Theano/Theano)![alt text][th] - is a Python library that allows you to define, optimize, and evaluate mathematical expressions
* [Lasagne](https://github.com/Lasagne/Lasagne) ![alt text][th] - Lightweight library to build and train neural networks in Theano
* [nolearn](https://github.com/dnouri/nolearn) ![alt text][th] ![alt text][skl] - scikit-learn compatible neural network library (mainly for Lasagne)
* [Blocks](https://github.com/mila-udem/blocks) ![alt text][th] - a Theano framework for building and training neural networks
* [platoon](https://github.com/mila-udem/platoon) ![alt text][th] - Multi-GPU mini-framework for Theano
* [NeuPy](https://github.com/itdxer/neupy) ![alt text][th] - NeuPy is a Python library for Artificial Neural Networks and Deep Learning

### PyTorch
* [PyTorch](https://github.com/pytorch/pytorch) ![alt text][pt]  - Tensors and Dynamic neural networks in Python with strong GPU acceleration 
* [skorch](https://github.com/dnouri/skorch) ![alt text][skl] ![alt text][pt]  - a scikit-learn compatible neural network library that wraps pytorch
* [PyTorchNet](https://github.com/pytorch/tnt) ![alt text][pt]  - an abstraction to train neural networks

### MXNet
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler
* [Gluon](https://github.com/gluon-api/gluon-api) - a clear, concise, simple yet powerful and efficient API for deep learning (now included in MXNet)
* [MXbox](https://github.com/Lyken17/mxbox) - simple, efficient and flexible vision toolbox for mxnet framework.

### CNTK
* [CNTK](https://github.com/Microsoft/CNTK) - Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit 

### Chainer
* [Chainer](https://github.com/chainer/chainer) - a flexible framework for neural networks
* [ChainerMN](https://github.com/chainer/chainermn) - scalable distributed deep learning with Chainer
* [scikit-chainer](https://github.com/lucidfrontier45/scikit-chainer) ![alt text][skl] - scikit-learn like interface to chainer
* [chainer_sklearn](https://github.com/corochann/chainer_sklearn) ![alt text][skl] - Sklearn (Scikit-learn) like interface for Chainer 

### Other
* [Neon](https://github.com/NervanaSystems/neon) - Intel® Nervana™ reference deep learning framework committed to best performance on all hardware
* [scikit-neuralnetwork](https://github.com/aigamedev/scikit-neuralnetwork) ![alt text][skl]  ![alt text][th] - Deep neural networks without the learning cliff

## Experiments tools
* [Sacred](https://github.com/IDSIA/sacred) - a tool to help you configure, organize, log and reproduce experiments by [IDSIA](http://www.idsia.ch/)
* [Xcessiv](https://github.com/reiinakano/xcessiv) - a web-based application for quick, scalable, and automated hyperparameter tuning and stacked ensembling
* [Persimmon](https://github.com/AlvarBer/Persimmon)  ![alt text][skl] - A visual dataflow programming language for sklearn

## Automated machine learning
* [TPOT](https://github.com/rhiever/tpot) ![alt text][skl] -  Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming
* [auto-sklearn](https://github.com/automl/auto-sklearn) ![alt text][skl] - is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator
* [MLBox](https://github.com/AxeldeRomblay/MLBox) - a powerful Automated Machine Learning python library.

## Genetic Programming
* [gplearn](https://github.com/trevorstephens/gplearn) ![alt text][skl] - Genetic Programming in Python
* [karoo_gp](https://github.com/kstaats/karoo_gp) ![alt text][tf] - A Genetic Programming platform for Python with GPU support

## Optimization
* [Spearmint](https://github.com/HIPS/Spearmint) - Bayesian optimization 
* [SMAC3](https://github.com/automl/SMAC3) - Sequential Model-based Algorithm Configuration 
* [Optunity](https://github.com/claesenm/optunity) - is a library containing various optimizers for hyperparameter tuning. 
* [htperopt](https://github.com/hyperopt/hyperopt) - Distributed Asynchronous Hyperparameter Optimization in Python
* [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn) ![alt text][skl]  - hyper-parameter optimization for sklearn 
* [sklearn-deap](https://github.com/rsteca/sklearn-deap) ![alt text][th] - use evolutionary algorithms instead of gridsearch in scikit-learn
* [sigopt_sklearn](https://github.com/sigopt/sigopt_sklearn) ![alt text][skl] - SigOpt wrappers for scikit-learn methods
* [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization) - A Python implementation of global optimization with gaussian processes.
* [SafeOpt](https://github.com/befelix/SafeOpt) - Safe Bayesian Optimization
* [scikit-optimize]() - 

## Probabilistic methods
* [skggm](https://github.com/skggm/skggm) ![alt text][skl] - estimation of general graphical models 
* [bayesloop](https://github.com/christophmark/bayesloop) - Probabilistic programming framework that facilitates objective model selection for time-varying parameter models
* [pyro](https://github.com/uber/pyro) ![alt text][pt] - Pyro is a flexible, scalable deep probabilistic programming library built on PyTorch.
* [ZhuSuan](http://zhusuan.readthedocs.io/en/latest/) ![alt text][tf] - Bayesian Deep Learning
* [pomegranate](https://github.com/jmschrei/pomegranate) ![alt text][cp] - probabilistic and graphical models for Python
* [pyMC3](http://docs.pymc.io/) ![alt text][th] - Python package for Bayesian statistical modeling and Probabilistic Machine Learning
* [Edward](http://edwardlib.org/) ![alt text][tf] - A library for probabilistic modeling, inference, and criticism.
* [GPflow](http://gpflow.readthedocs.io/en/latest/?badge=latest) ![alt text][tf]  - Gaussian processes in TensorFlow
* [Stan](https://github.com/stan-dev/pystan) - Bayesian inference using the No-U-Turn sampler (Python interface)
* [gelato](https://github.com/ferrine/gelato) ![alt text][th] - Bayesian dessert for Lasagne
* [sklearn-bayes](https://github.com/AmazaspShumik/sklearn-bayes) ![alt text][skl]  - Python package for Bayesian Machine Learning with scikit-learn API
* [pgmpy](https://github.com/pgmpy/pgmpy) - a python library for working with Probabilistic Graphical Models.
* [skpro](https://github.com/alan-turing-institute/skpro) ![alt text][skl] - supervised domain-agnostic prediction framework for probabilistic modelling by [The Alan Turing Institute](https://www.turing.ac.uk/)

## Natural Language Processing
* [NLTK](https://github.com/nltk/nltk) -  modules, data sets, and tutorials supporting research and development in Natural Language Processing
* [CLTK](https://github.com/cltk/cltk) - The Classical Language Toolkik
* [gensim](https://radimrehurek.com/gensim/) - Topic Modelling for Humans
* [PSI-Toolkit](http://psi-toolkit.amu.edu.pl/) - a natural language processing toolkit by [Adam Mickiewicz University](https://zpjn.wmi.amu.edu.pl/en/) in Poznań

## Statistics
* [statsmodels](https://github.com/statsmodels/statsmodels) - statistical modeling and econometrics in Python
* [stockstats](https://github.com/jealous/stockstats) - Supply a wrapper ``StockDataFrame`` based on the ``pandas.DataFrame`` with inline stock statistics/indicators support.

## Visualization
* [Matplotlib](https://github.com/matplotlib/matplotlib) - plotting with Python
* [seaborn](https://github.com/mwaskom/seaborn) - statistical data visualization using matplotlib
* [Bokeh](https://github.com/bokeh/bokeh) - Interactive Web Plotting for Python
* [Alphalens](https://github.com/quantopian/alphalens) - performance analysis of predictive (alpha) stock factors by [Quantopian](https://www.quantopian.com/)
* [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) ![alt text][skl]- visual analysis and diagnostic tools to facilitate machine learning model selection
* [scikit-plot](https://github.com/reiinakano/scikit-plot) ![alt text][skl] - an intuitive library to add plotting functionality to scikit-learn objects

## Evaluation
* [kaggle-metrics](https://github.com/krzjoa/kaggle-metrics) - Metrics for Kaggle competitions
* [Metrics](https://github.com/benhamner/Metrics) - machine learning evaluation metric

## Computations
* [numpy](http://www.numpy.org/) - the fundamental package needed for scientific computing with Python.
* [bottleneck](https://github.com/kwgoodman/bottleneck) - Fast NumPy array functions written in C
* [minpy](https://github.com/dmlc/minpy) - NumPy interface with mixed backend execution
* [CuPy](https://github.com/cupy/cupy) - NumPy-like API accelerated with CUDA 
* [scikit-tensor](https://github.com/mnick/scikit-tensor) - Python library for multilinear algebra and tensor factorizations
