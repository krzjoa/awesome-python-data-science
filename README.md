<div align="center">
	<a href="https://krzjoa.github.io/awesome-python-data-science/"><img width="250" height="250" src="img/py-datascience.png" alt="pyds"></a>
	<br>
	<br>
	<br>
</div>

<h1 align="center">
	Awesome Python Data Science
</h1>
<div align="center"><a href="https://github.com/sindresorhus/awesome">
<img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg" alt="Awesome" border="0">
</a>
</div>
</br>

> Probably the best curated list of data science software in Python

## Contents
* [Machine Learning](#machine-learning)
* [Deep Learning](#deep-learning)
* [Data Manipulation](#data-manipulation)
* [Feature Engineering](#feature-engineering)
* [Visualization](#visualization)
* [Model Explanation](#model-explanation)
* [Reinforcement Learning](#reinforcement-learning)
* [Probabilistic Methods](#probabilistic-methods)
* [Genetic Programming](#genetic-programming)
* [Optimization](#optimization)
* [Natural Language Processing](#natural-language-processing)
* [Computer Audition](#computer-audition)
* [Computer Vision](#computer-vision)
* [Statistics](#statistics)
* [Distributed Computing](#distributed-computing)
* [Experimentation](#experimentation)
* [Evaluation](#evaluation)
* [Computations](#computations)
* [Spatial Analysis](#spatial-analysis)
* [Quantum Computing](#quantum-computing)
* [Conversion](#conversion)

## Machine Learning

### General Purpouse Machine Learning
* [scikit-learn](http://scikit-learn.org/stable/) - Machine learning in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Shogun](http://www.shogun-toolbox.org/) - Machine learning toolbox.
* [xLearn](https://github.com/aksnzhy/xlearn) - High Performance, Easy-to-use, and Scalable Machine Learning Package.
* [cuML](https://github.com/rapidsai/cuml) - RAPIDS Machine Learning Library. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [modAL](https://github.com/cosmic-cortex/modAL) - Modular active learning framework for Python3. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Sparkit-learn](https://github.com/lensacom/sparkit-learn) - PySpark + scikit-learn = Sparkit-learn. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/spark_big.png" alt="Apache Spark based">
* [mlpack](https://github.com/mlpack/mlpack) - A scalable C++ machine learning library (Python bindings).
* [dlib](https://github.com/davisking/dlib) - Toolkit for making real world machine learning and data analysis applications in C++ (Python bindings).
* [MLxtend](https://github.com/rasbt/mlxtend) - Extension and helper modules for Python's data analysis and machine learning libraries. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [hyperlearn](https://github.com/danielhanchen/hyperlearn) - 50%+ Faster, 50%+ less RAM usage, GPU support re-written Sklearn, Statsmodels. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Reproducible Experiment Platform (REP)](https://github.com/yandex/rep) - Machine Learning toolbox for Humans. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) - Multi-label classification for python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [seqlearn](https://github.com/larsmans/seqlearn) - Sequence classification toolkit for Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [pystruct](https://github.com/pystruct/pystruct) - Simple structured learning framework for Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys) - Highly interpretable classifiers for scikit learn, producing easily understood decision rules instead of black box models. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [RuleFit](https://github.com/christophM/rulefit) - Implementation of the rulefit. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [metric-learn](https://github.com/all-umass/metric-learn) - Metric learning algorithms in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [pyGAM](https://github.com/dswah/pyGAM) - Generalized Additive Models in Python.
* [Karate Club](https://github.com/benedekrozemberczki/karateclub) - An unsupervised machine learning library for graph structured data.

### Time Series
* [tslearn](https://github.com/rtavenar/tslearn) - Machine learning toolkit dedicated to time-series data. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [tick](https://github.com/X-DataInitiative/tick) - Module for statistical learning, with a particular emphasis on time-dependent modelling.  <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Prophet](https://github.com/facebook/prophet) - Automatic Forecasting Procedure.
* [PyFlux](https://github.com/RJT1990/pyflux) - Open source time series library for Python.
* [bayesloop](https://github.com/christophmark/bayesloop) - Probabilistic programming framework that facilitates objective model selection for time-varying parameter models.
* [luminol](https://github.com/linkedin/luminol) - Anomaly Detection and Correlation library.

### Automated Machine Learning
* [TPOT](https://github.com/rhiever/tpot) - Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [auto-sklearn](https://github.com/automl/auto-sklearn) - An automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [MLBox](https://github.com/AxeldeRomblay/MLBox) - A powerful Automated Machine Learning python library.

### Ensemble Methods
* [ML-Ensemble](http://ml-ensemble.com/) - High performance ensemble learning. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Stacking](https://github.com/ikki407/stacking) - Simple and useful stacking library, written in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [stacked_generalization](https://github.com/fukatani/stacked_generalization) - Library for machine learning stacking generalization. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [vecstack](https://github.com/vecxoz/vecstack) - Python package for stacking (machine learning technique). <img height="20" src="img/sklearn_big.png" alt="sklearn">

### Imbalanced Datasets
* [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) - Module to perform under sampling and over sampling with various techniques. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms) - Python-based implementations of algorithms for learning on imbalanced data. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/tf_big2.png" alt="sklearn">

### Random Forests
* [rpforest](https://github.com/lyst/rpforest) - A forest of random projection trees. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sklearn-random-bits-forest](https://github.com/tmadl/sklearn-random-bits-forest) - Wrapper of the Random Bits Forest program written by (Wang et al., 2016).<img height="20" src="img/sklearn_big.png" alt="sklearn">
* [rgf_python](https://github.com/fukatani/rgf_python) - Python Wrapper of Regularized Greedy Forest. <img height="20" src="img/sklearn_big.png" alt="sklearn">

### Extreme Learning Machine
* [Python-ELM](https://github.com/dclambert/Python-ELM) - Extreme Learning Machine implementation in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Python Extreme Learning Machine (ELM)](https://github.com/acba/elm) - A machine learning technique used for classification/regression tasks.
* [hpelm](https://github.com/akusok/hpelm) - High performance implementation of Extreme Learning Machines (fast randomized neural networks). <img height="20" src="img/gpu_big.png" alt="GPU accelerated">

### Kernel Methods
* [pyFM](https://github.com/coreylynch/pyFM) - Factorization machines in python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [fastFM](https://github.com/ibayer/fastFM) - A library for Factorization Machines. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [tffm](https://github.com/geffy/tffm) - TensorFlow implementation of an arbitrary order Factorization Machine. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/tf_big2.png" alt="sklearn">
* [liquidSVM](https://github.com/liquidSVM/liquidSVM) - An implementation of SVMs.
* [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm) - Relevance Vector Machine implementation using the scikit-learn API. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) - A fast SVM Library on GPUs and CPUs. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">

### Gradient Boosting
* [XGBoost](https://github.com/dmlc/xgboost) - Scalable, Portable and Distributed Gradient Boosting. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [LightGBM](https://github.com/Microsoft/LightGBM) - A fast, distributed, high performance gradient boosting. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [CatBoost](https://github.com/catboost/catboost) - An open-source gradient boosting on decision trees library. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) - Fast GBDTs and Random Forests on GPUs. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">

## Deep Learning

### PyTorch
* [PyTorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python with strong GPU acceleration. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [torchvision](https://github.com/pytorch/vision) - Datasets, Transforms and Models specific to Computer Vision. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [torchtext](https://github.com/pytorch/text) - Data loaders and abstractions for text and NLP. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [torchaudio](https://github.com/pytorch/audio) - An audio library for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [ignite](https://github.com/pytorch/ignite) - High-level library to help with training neural networks in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [PyToune](https://github.com/GRAAL-Research/pytoune) - A Keras-like framework and utilities for PyTorch.
* [skorch](https://github.com/dnouri/skorch) - A scikit-learn compatible neural network library that wraps pytorch. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [PyTorchNet](https://github.com/pytorch/tnt) - An abstraction to train neural networks. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) - Geometric Deep Learning Extension Library for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [Catalyst](https://github.com/catalyst-team/catalyst) - High-level utils for PyTorch DL & RL research. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">

### TensorFlow
* [TensorFlow](https://github.com/tensorflow/tensorflow) - Computation using data flow graphs for scalable machine learning by Google. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TensorLayer](https://github.com/zsdonghao/tensorlayer) - Deep Learning and Reinforcement Learning Library for Researcher and Engineer. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TFLearn](https://github.com/tflearn/tflearn) - Deep learning library featuring a higher-level API for TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Sonnet](https://github.com/deepmind/sonnet) - TensorFlow-based neural network library. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [tensorpack](https://github.com/ppwwyyxx/tensorpack) - A Neural Net Training Interface on TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Polyaxon](https://github.com/polyaxon/polyaxon) - A platform that helps you build, manage and monitor deep learning models. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [NeuPy](https://github.com/itdxer/neupy) - NeuPy is a Python library for Artificial Neural Networks and Deep Learning (previously: <img height="20" src="img/theano_big.png" alt="Theano compatible">). <img height="20" src="img/tf_big2.png" alt="sklearn">
* [tfdeploy](https://github.com/riga/tfdeploy) - Deploy tensorflow graphs for fast evaluation and export to tensorflow-less environments running numpy. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [tensorflow-upstream](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream) - TensorFlow ROCm port. <img height="20" src="img/tf_big2.png" alt="sklearn"> <img height="20" src="img/amd_big.png" alt="Possible to run on AMD GPU">
* [TensorFlow Fold](https://github.com/tensorflow/fold) - Deep learning with dynamic computation graphs in TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [tensorlm](https://github.com/batzner/tensorlm) - Wrapper library for text generation / language models at char and word level with RNN. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TensorLight](https://github.com/bsautermeister/tensorlight) - A high-level framework for TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Mesh TensorFlow](https://github.com/tensorflow/mesh) - Model Parallelism Made Easier. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Ludwig](https://github.com/uber/ludwig) - A toolbox, that allows to train and test deep learning models without the need to write code. <img height="20" src="img/tf_big2.png" alt="sklearn">

### Keras
* [Keras](https://keras.io) - A high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.  <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [keras-contrib](https://github.com/keras-team/keras-contrib) - Keras community contributions. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Hyperas](https://github.com/maxpumperla/hyperas) - Keras + Hyperopt: A very simple wrapper for convenient hyperparameter. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Elephas](https://github.com/maxpumperla/elephas) - Distributed Deep learning with Keras & Spark. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Hera](https://github.com/keplr-io/hera) - Train/evaluate a Keras model, get metrics streamed to a dashboard in your browser. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [Spektral](https://github.com/danielegrattarola/spektral) - Deep learning on graphs. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [qkeras](https://github.com/google/qkeras) - A quantization deep learning library. <img height="20" src="img/keras_big.png" alt="Keras compatible">

### MXNet
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler. <img height="20" src="img/mxnet_big.png" alt="MXNet based">
* [Gluon](https://github.com/gluon-api/gluon-api) - A clear, concise, simple yet powerful and efficient API for deep learning (now included in MXNet). <img height="20" src="img/mxnet_big.png" alt="MXNet based">
* [MXbox](https://github.com/Lyken17/mxbox) - Simple, efficient and flexible vision toolbox for mxnet framework. <img height="20" src="img/mxnet_big.png" alt="MXNet based">
* [gluon-cv](https://github.com/dmlc/gluon-cv) - Provides implementations of the state-of-the-art  deep learning models in computer vision. <img height="20" src="img/mxnet_big.png" alt="MXNet based">
* [gluon-nlp](https://github.com/dmlc/gluon-nlp) - NLP made easy. <img height="20" src="img/mxnet_big.png" alt="MXNet based">
* [Xfer](https://github.com/amzn/xfer) - Transfer Learning library for Deep Neural Networks. <img height="20" src="img/mxnet_big.png" alt="MXNet based">
* [MXNet](https://github.com/ROCmSoftwarePlatform/mxnet) - HIP Port of MXNet. <img height="20" src="img/mxnet_big.png" alt="MXNet based"> <img height="20" src="img/amd_big.png" alt="Possible to run on AMD GPU">

### Chainer
* [Chainer](https://github.com/chainer/chainer) - A flexible framework for neural networks.
* [ChainerCV](https://github.com/chainer/chainercv) - A Library for Deep Learning in Computer Vision.
* [ChainerMN](https://github.com/chainer/chainermn) - Scalable distributed deep learning with Chainer.

### Theano
**WARNING: Theano development has been stopped**
* [Theano](https://github.com/Theano/Theano) - A Python library that allows you to define, optimize, and evaluate mathematical expressions.<img height="20" src="img/theano_big.png" alt="Theano compatible">
* [Lasagne](https://github.com/Lasagne/Lasagne) - Lightweight library to build and train neural networks in Theano. <img height="20" src="img/theano_big.png" alt="Theano compatible">
* [nolearn](https://github.com/dnouri/nolearn) - A scikit-learn compatible neural network library (mainly for Lasagne). <img height="20" src="img/theano_big.png" alt="Theano compatible"> <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Blocks](https://github.com/mila-udem/blocks) - A Theano framework for building and training neural networks. <img height="20" src="img/theano_big.png" alt="Theano compatible">
* [scikit-neuralnetwork](https://github.com/aigamedev/scikit-neuralnetwork) - Deep neural networks without the learning cliff. <img height="20" src="img/sklearn_big.png" alt="sklearn">  <img height="20" src="img/theano_big.png" alt="Theano compatible">
* [platoon](https://github.com/mila-udem/platoon) - Multi-GPU mini-framework for Theano. <img height="20" src="img/theano_big.png" alt="Theano compatible">
* [Theano-MPI](https://github.com/uoguelph-mlrg/Theano-MPI) - MPI Parallel framework for training deep learning models built in Theano. <img height="20" src="img/theano_big.png" alt="Theano compatible">

### Others
* [CNTK](https://github.com/Microsoft/CNTK) - Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit.
* [Neon](https://github.com/NervanaSystems/neon) - Intel® Nervana™ reference deep learning framework committed to best performance on all hardware.
* [Tangent](https://github.com/google/tangent) - Source-to-Source Debuggable Derivatives in Pure Python.
* [autograd](https://github.com/HIPS/autograd) - Efficiently computes derivatives of numpy code.
* [Myia](https://github.com/mila-udem/myia) - Deep Learning framework (pre-alpha).
* [nnabla](https://github.com/sony/nnabla) - Neural Network Libraries by Sony.
* [Caffe](https://github.com/BVLC/caffe) - A fast open framework for deep learning.
* [Caffe2](https://github.com/pytorch/pytorch/tree/master/caffe2) -  A lightweight, modular, and scalable deep learning framework (now a part of PyTorch).
* [hipCaffe](https://github.com/ROCmSoftwarePlatform/hipCaffe) - The HIP port of Caffe. <img height="20" src="img/amd_big.png" alt="Possible to run on AMD GPU">

## Data Manipulation

### Data Containers
* [pandas](https://pandas.pydata.org/pandas-docs/stable/) - Powerful Python data analysis toolkit.
* [cuDF](https://github.com/rapidsai/cudf) - GPU DataFrame Library. <img height="20" src="img/pandas_big.png" alt="pandas compatible"> <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [blaze](https://github.com/blaze/blaze) - NumPy and pandas interface to Big Data. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [pandasql](https://github.com/yhat/pandasql) -  Allows you to query pandas DataFrames using SQL syntax. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [pandas-gbq](https://github.com/pydata/pandas-gbq) - pandas Google Big Query. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [xpandas](https://github.com/alan-turing-institute/xpandas) - Universal 1d/2d data containers with Transformers .functionality for data analysis by [The Alan Turing Institute](https://www.turing.ac.uk/).
* [pysparkling](https://github.com/svenkreiss/pysparkling) - A pure Python implementation of Apache Spark's RDD and DStream interfaces. <img height="20" src="img/spark_big.png" alt="Apache Spark based">
* [Arctic](https://github.com/manahl/arctic) - High performance datastore for time series and tick data.
* [datatable](https://github.com/h2oai/datatable) - Data.table for Python. <img height="20" src="img/R_big.png" alt="R inspired/ported lib">
* [koalas](https://github.com/databricks/koalas) - pandas API on Apache Spark. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [modin](https://github.com/modin-project/modin) - Speed up your pandas workflows by changing a single line of code. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [swifter](https://github.com/jmcarpenter2/swifter) - A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner.
* [pandas_flavor](https://github.com/Zsailer/pandas_flavor) - A package which allow to write your own flavor of Pandas easily.
* [pandas-log](https://github.com/eyaltrabelsi/pandas-log) - A package which allow to provide feedback about basic pandas operations and find both buisness logic and performance issues.

### Pipelines
* [pdpipe](https://github.com/shaypal5/pdpipe) - Sasy pipelines for pandas DataFrames.
* [SSPipe](https://sspipe.github.io/) - Python pipe (|) operator with support for DataFrames and Numpy and Pytorch.
* [pandas-ply](https://github.com/coursera/pandas-ply) - Functional data manipulation for pandas. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Dplython](https://github.com/dodger487/dplython) - Dplyr for Python. <img height="20" src="img/R_big.png" alt="R inspired/ported lib">
* [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) - pandas integration with sklearn. <img height="20" src="img/sklearn_big.png" alt="sklearn"> <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Dataset](https://github.com/analysiscenter/dataset) - Helps you conveniently work with random or sequential batches of your data and define data processing.
* [pyjanitor](https://github.com/ericmjl/pyjanitor) - Clean APIs for data cleaning. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [meza](https://github.com/reubano/meza) - A Python toolkit for processing tabular data.
* [Prodmodel](https://github.com/prodmodel/prodmodel) - Build system for data science pipelines.
* [dopanda](https://github.com/dovpanda-dev/dovpanda) -  Hints and tips for using pandas in an analysis environment. <img height="20" src="img/pandas_big.png" alt="pandas compatible">

## Feature Engineering

### General
* [Featuretools](https://github.com/Featuretools/featuretools) - Automated feature engineering.
* [skl-groups](https://github.com/dougalsutherland/skl-groups) - A scikit-learn addon to operate on set/"group"-based features. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Feature Forge](https://github.com/machinalis/featureforge) - A set of tools for creating and testing machine learning feature. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [few](https://github.com/lacava/few) - A feature engineering wrapper for sklearn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [scikit-mdr](https://github.com/EpistasisLab/scikit-mdr) - A sklearn-compatible Python implementation of Multifactor Dimensionality Reduction (MDR) for feature construction. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [tsfresh](https://github.com/blue-yonder/tsfresh) - Automatic extraction of relevant features from time series. <img height="20" src="img/sklearn_big.png" alt="sklearn">

### Feature Selection
* [scikit-feature](https://github.com/jundongl/scikit-feature) - Feature selection repository in python.
* [boruta_py](https://github.com/scikit-learn-contrib/boruta_py) - Implementations of the Boruta all-relevant feature selection method. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [BoostARoota](https://github.com/chasedehan/BoostARoota) - A fast xgboost feature selection algorithm. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [scikit-rebate](https://github.com/EpistasisLab/scikit-rebate) - A scikit-learn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms for Machine Learning. <img height="20" src="img/sklearn_big.png" alt="sklearn">

## Visualization
* [Matplotlib](https://github.com/matplotlib/matplotlib) - Plotting with Python.
* [seaborn](https://github.com/mwaskom/seaborn) - Statistical data visualization using matplotlib.
* [Bokeh](https://github.com/bokeh/bokeh) - Interactive Web Plotting for Python.
* [HoloViews](https://github.com/ioam/holoviews) - Stop plotting your data - annotate your data and let it visualize itself.
* [prettyplotlib](https://github.com/olgabot/prettyplotlib) - Painlessly create beautiful matplotlib plots.
* [python-ternary](https://github.com/marcharper/python-ternary) - Ternary plotting library for python with matplotlib.
* [missingno](https://github.com/ResidentMario/missingno) - Missing data visualization module for Python.
* [chartify](https://github.com/spotify/chartify/) - Python library that makes it easy for data scientists to create charts.
* [physt](https://github.com/janpipek/physt) - Improved histograms.
* [animatplot](https://github.com/t-makaro/animatplot) - A python package for animating plots build on matplotlib.
* [plotly](https://plot.ly/python/) - A Python library that makes interactive and publication-quality graphs.

## Model Explanation
* [Alibi](https://github.com/SeldonIO/alibi) - Algorithms for monitoring and explaining machine learning models.
* [anchor](https://github.com/marcotcr/anchor) - Code for "High-Precision Model-Agnostic Explanations" paper.
* [aequitas](https://github.com/dssg/aequitas) - Bias and Fairness Audit Toolkit.
* [Contrastive Explanation](https://github.com/MarcelRobeer/ContrastiveExplanation) - Contrastive Explanation (Foil Trees). <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) - Visual analysis and diagnostic tools to facilitate machine learning model selection. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [scikit-plot](https://github.com/reiinakano/scikit-plot) - An intuitive library to add plotting functionality to scikit-learn objects. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [shap](https://github.com/slundberg/shap) - A unified approach to explain the output of any machine learning model. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [ELI5](https://github.com/TeamHG-Memex/eli5) - A library for debugging/inspecting machine learning classifiers and explaining their predictions.
* [Lime](https://github.com/marcotcr/lime) - Explaining the predictions of any machine learning classifier. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [FairML](https://github.com/adebayoj/fairml) - FairML is a python toolbox auditing the machine learning models for bias. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [L2X](https://github.com/Jianbo-Lab/L2X) - Code for replicating the experiments in the paper *Learning to Explain: An Information-Theoretic Perspective on Model Interpretation*.
* [PDPbox](https://github.com/SauceCat/PDPbox) - Partial dependence plot toolbox.
* [pyBreakDown](https://github.com/MI2DataLab/pyBreakDown) - Python implementation of R package breakDown. <img height="20" src="img/sklearn_big.png" alt="sklearn"><img height="20" src="img/R_big.png" alt="R inspired/ported lib">
* [PyCEbox](https://github.com/AustinRochford/PyCEbox) - Python Individual Conditional Expectation Plot Toolbox.
* [Skater](https://github.com/datascienceinc/Skater) - Python Library for Model Interpretation.
* [model-analysis](https://github.com/tensorflow/model-analysis) - Model analysis tools for TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [themis-ml](https://github.com/cosmicBboy/themis-ml) - A library that implements fairness-aware machine learning algorithms. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [treeinterpreter](https://github.com/andosa/treeinterpreter) - Interpreting scikit-learn's decision tree and random forest predictions. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [AI Explainability 360](https://github.com/IBM/AIX360) - Interpretability and explainability of data and machine learning models.
* [Auralisation](https://github.com/keunwoochoi/Auralisation) - Auralisation of learned features in CNN (for audio).
* [CapsNet-Visualization](https://github.com/bourdakos1/CapsNet-Visualization) - A visualization of the CapsNet layers to better understand how it works.
* [lucid](https://github.com/tensorflow/lucid) - A collection of infrastructure and tools for research in neural network interpretability.
* [Netron](https://github.com/lutzroeder/Netron) - Visualizer for deep learning and machine learning models (no Python code, but visualizes models from most Python Deep Learning frameworks).
* [FlashLight](https://github.com/dlguys/flashlight) - Visualization Tool for your NeuralNetwork.
* [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) - Tensorboard for pytorch (and chainer, mxnet, numpy, ...).
* [mxboard](https://github.com/awslabs/mxboard) - Logging MXNet data for visualization in TensorBoard. <img height="20" src="img/mxnet_big.png" alt="MXNet based">

## Reinforcement Learning
* [OpenAI Gym](https://github.com/openai/gym) - A toolkit for developing and comparing reinforcement learning algorithms.
* [Coach](https://github.com/NervanaSystems/coach) - Easy experimentation with state of the art Reinforcement Learning algorithms.
* [garage](https://github.com/rlworkgroup/garage) - A toolkit for reproducible reinforcement learning research.
* [OpenAI Baselines](https://github.com/openai/baselines) - High-quality implementations of reinforcement learning algorithms.
* [Stable Baselines](https://github.com/hill-a/stable-baselines) - A set of improved implementations of reinforcement learning algorithms based on OpenAI Baselines.
* [RLlib](https://ray.readthedocs.io/en/latest/rllib.html) - Scalable Reinforcement Learning.
* [Horizon](https://github.com/facebookresearch/Horizon) - A platform for Applied Reinforcement Learning.
* [TF-Agents](https://github.com/tensorflow/agents) - A library for Reinforcement Learning in TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">  
* [TensorForce](https://github.com/reinforceio/tensorforce) - A TensorFlow library for applied reinforcement learning. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [TRFL](https://github.com/deepmind/trfl) - TensorFlow Reinforcement Learning. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [Dopamine](https://github.com/google/dopamine) - A research framework for fast prototyping of reinforcement learning algorithms.
* [keras-rl](https://github.com/keras-rl/keras-rl) - Deep Reinforcement Learning for Keras. <img height="20" src="img/keras_big.png" alt="Keras compatible">
* [ChainerRL](https://github.com/chainer/chainerrl) - A deep reinforcement learning library built on top of Chainer.

## Distributed Computing
* [Horovod](https://github.com/uber/horovod) - Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html) - Exposes the Spark programming model to Python. <img height="20" src="img/spark_big.png" alt="Apache Spark based">
* [Veles](https://github.com/Samsung/veles) - Distributed machine learning platform.
* [Jubatus](https://github.com/jubatus/jubatus) - Framework and Library for Distributed Online Machine Learning.
* [DMTK](https://github.com/Microsoft/DMTK) - Microsoft Distributed Machine Learning Toolkit.
* [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - PArallel Distributed Deep LEarning.
* [dask-ml](https://github.com/dask/dask-ml) - Distributed and parallel machine learning. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Distributed](https://github.com/dask/distributed) - Distributed computation in Python.

## Probabilistic Methods
* [pomegranate](https://github.com/jmschrei/pomegranate) - Probabilistic and graphical models for Python. <img height="20" src="img/gpu_big.png" alt="GPU accelerated">
* [pyro](https://github.com/uber/pyro) - A flexible, scalable deep probabilistic programming library built on PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [ZhuSuan](http://zhusuan.readthedocs.io/en/latest/) - Bayesian Deep Learning. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [PyMC](https://github.com/pymc-devs/pymc) - Bayesian Stochastic Modelling in Python.
* [PyMC3](http://docs.pymc.io/) - Python package for Bayesian statistical modeling and Probabilistic Machine Learning. <img height="20" src="img/theano_big.png" alt="Theano compatible">
* [sampled](https://github.com/ColCarroll/sampled) - Decorator for reusable models in PyMC3.
* [Edward](http://edwardlib.org/) - A library for probabilistic modeling, inference, and criticism. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [InferPy](https://github.com/PGM-Lab/InferPy) - Deep Probabilistic Modelling Made Easy.  <img height="20" src="img/tf_big2.png" alt="sklearn">
* [GPflow](http://gpflow.readthedocs.io/en/latest/?badge=latest) - Gaussian processes in TensorFlow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [PyStan](https://github.com/stan-dev/pystan) - Bayesian inference using the No-U-Turn sampler (Python interface).
* [gelato](https://github.com/ferrine/gelato) - Bayesian dessert for Lasagne. <img height="20" src="img/theano_big.png" alt="Theano compatible">
* [sklearn-bayes](https://github.com/AmazaspShumik/sklearn-bayes) - Python package for Bayesian Machine Learning with scikit-learn API. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [skggm](https://github.com/skggm/skggm) - Estimation of general graphical models. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [pgmpy](https://github.com/pgmpy/pgmpy) - A python library for working with Probabilistic Graphical Models.
* [skpro](https://github.com/alan-turing-institute/skpro) - Supervised domain-agnostic prediction framework for probabilistic modelling by [The Alan Turing Institute](https://www.turing.ac.uk/). <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Aboleth](https://github.com/data61/aboleth) - A bare-bones TensorFlow framework for Bayesian deep learning and Gaussian process approximation. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [PtStat](https://github.com/stepelu/ptstat) - Probabilistic Programming and Statistical Inference in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [PyVarInf](https://github.com/ctallec/pyvarinf) - Bayesian Deep Learning methods with Variational Inference for PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [emcee](https://github.com/dfm/emcee) - The Python ensemble sampling toolkit for affine-invariant MCMC.
* [hsmmlearn](https://github.com/jvkersch/hsmmlearn) - A library for hidden semi-Markov models with explicit durations.
* [pyhsmm](https://github.com/mattjj/pyhsmm) - Bayesian inference in HSMMs and HMMs.
* [GPyTorch](https://github.com/cornellius-gp/gpytorch) - A highly efficient and modular implementation of Gaussian Processes in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [MXFusion](https://github.com/amzn/MXFusion) - Modular Probabilistic Programming on MXNet. <img height="20" src="img/mxnet_big.png" alt="MXNet based">
* [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite) - A scikit-learn inspired API for CRFsuite. <img height="20" src="img/sklearn_big.png" alt="sklearn">

## Genetic Programming
* [gplearn](https://github.com/trevorstephens/gplearn) - Genetic Programming in Python. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [DEAP](https://github.com/DEAP/deap) - Distributed Evolutionary Algorithms in Python.
* [karoo_gp](https://github.com/kstaats/karoo_gp) - A Genetic Programming platform for Python with GPU support. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [monkeys](https://github.com/hchasestevens/monkeys) - A strongly-typed genetic programming framework for Python.
* [sklearn-genetic](https://github.com/manuel-calzolari/sklearn-genetic) - Genetic feature selection module for scikit-learn. <img height="20" src="img/sklearn_big.png" alt="sklearn">

<a name="opt"></a>
## Optimization
* [Spearmint](https://github.com/HIPS/Spearmint) - Bayesian optimization.
* [BoTorch](https://github.com/pytorch/botorch) - Bayesian optimization in PyTorch. <img height="20" src="img/pytorch_big2.png" alt="PyTorch based/compatible">
* [scikit-opt](https://github.com/guofei9987/scikit-opt) - Heuristic Algorithms for optimization.
* [SMAC3](https://github.com/automl/SMAC3) - Sequential Model-based Algorithm Configuration.
* [Optunity](https://github.com/claesenm/optunity) - Is a library containing various optimizers for hyperparameter tuning.
* [hyperopt](https://github.com/hyperopt/hyperopt) - Distributed Asynchronous Hyperparameter Optimization in Python.
* [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn) - Hyper-parameter optimization for sklearn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sklearn-deap](https://github.com/rsteca/sklearn-deap) - Use evolutionary algorithms instead of gridsearch in scikit-learn. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [sigopt_sklearn](https://github.com/sigopt/sigopt_sklearn) - SigOpt wrappers for scikit-learn methods. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization) - A Python implementation of global optimization with gaussian processes.
* [SafeOpt](https://github.com/befelix/SafeOpt) - Safe Bayesian Optimization.
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) - Sequential model-based optimization with a `scipy.optimize` interface.
* [Solid](https://github.com/100/Solid) - A comprehensive gradient-free optimization framework written in Python.
* [PySwarms](https://github.com/ljvmiranda921/pyswarms) - A research toolkit for particle swarm optimization in Python.
* [Platypus](https://github.com/Project-Platypus/Platypus) - A Free and Open Source Python Library for Multiobjective Optimization.
* [GPflowOpt](https://github.com/GPflow/GPflowOpt) - Bayesian Optimization using GPflow. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [POT](https://github.com/rflamary/POT) - Python Optimal Transport library.
* [Talos](https://github.com/autonomio/talos) - Hyperparameter Optimization for Keras Models.
* [nlopt](https://github.com/stevengj/nlopt) - Library for nonlinear optimization (global and local, constrained or unconstrained).

## Natural Language Processing
* [NLTK](https://github.com/nltk/nltk) -  Modules, data sets, and tutorials supporting research and development in Natural Language Processing.
* [CLTK](https://github.com/cltk/cltk) - The Classical Language Toolkik.
* [gensim](https://radimrehurek.com/gensim/) - Topic Modelling for Humans.
* [PSI-Toolkit](http://psi-toolkit.amu.edu.pl/) - A natural language processing toolkit.
* [pyMorfologik](https://github.com/dmirecki/pyMorfologik) - Python binding for <a href="https://github.com/morfologik/morfologik-stemming">Morfologik</a>.
* [skift](https://github.com/shaypal5/skift) - Scikit-learn wrappers for Python fastText. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Phonemizer](https://github.com/bootphon/phonemizer) - Simple text to phonemes converter for multiple languages.
* [flair](https://github.com/zalandoresearch/flair) - Very simple framework for state-of-the-art NLP.
* [spaCy](https://spacy.io/) - Industrial-Strength Natural Language Processing.

## Computer Audition
* [librosa](https://github.com/librosa/librosa) - Python library for audio and music analysis.
* [Yaafe](https://github.com/Yaafe/Yaafe) - Audio features extraction.
* [aubio](https://github.com/aubio/aubio) - A library for audio and music analysis.
* [Essentia](https://github.com/MTG/essentia) - Library for audio and music analysis, description and synthesis.
* [LibXtract](https://github.com/jamiebullock/LibXtract) - A simple, portable, lightweight library of audio feature extraction functions.
* [Marsyas](https://github.com/marsyas/marsyas) - Music Analysis, Retrieval and Synthesis for Audio Signals.
* [muda](https://github.com/bmcfee/muda) - A library for augmenting annotated audio data.
* [madmom](https://github.com/CPJKU/madmom) - Python audio and music signal processing library.

## Computer Vision
* [OpenCV](https://github.com/opencv/opencv) - Open Source Computer Vision Library.
* [scikit-image](https://github.com/scikit-image/scikit-image) - Image Processing SciKit (Toolbox for SciPy).
* [imgaug](https://github.com/aleju/imgaug) - Image augmentation for machine learning experiments.
* [imgaug_extension](https://github.com/cadenai/imgaug_extension) - Additional augmentations for imgaug.
* [Augmentor](https://github.com/mdbloice/Augmentor) - Image augmentation library in Python for machine learning.
* [albumentations](https://github.com/albu/albumentations) - Fast image augmentation library and easy to use wrapper around other libraries.

## Statistics
* [pandas_summary](https://github.com/mouradmourafiq/pandas-summary) - Extension to pandas dataframes describe function. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) - Create HTML profiling reports from pandas DataFrame objects. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [statsmodels](https://github.com/statsmodels/statsmodels) - Statistical modeling and econometrics in Python.
* [stockstats](https://github.com/jealous/stockstats) - Supply a wrapper ``StockDataFrame`` based on the ``pandas.DataFrame`` with inline stock statistics/indicators support.
* [weightedcalcs](https://github.com/jsvine/weightedcalcs) - A pandas-based utility to calculate weighted means, medians, distributions, standard deviations, and more.
* [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) - Pairwise Multiple Comparisons Post-hoc Tests.
* [Alphalens](https://github.com/quantopian/alphalens) - Performance analysis of predictive (alpha) stock factors.

## Distributed Computing
* [Horovod](https://github.com/uber/horovod) - Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. <img height="20" src="img/tf_big2.png" alt="sklearn">
* [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html) - Exposes the Spark programming model to Python. <img height="20" src="img/spark_big.png" alt="Apache Spark based">
* [Veles](https://github.com/Samsung/veles) - Distributed machine learning platform.
* [Jubatus](https://github.com/jubatus/jubatus) - Framework and Library for Distributed Online Machine Learning.
* [DMTK](https://github.com/Microsoft/DMTK) - Microsoft Distributed Machine Learning Toolkit.
* [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - PArallel Distributed Deep LEarning.
* [dask-ml](https://github.com/dask/dask-ml) - Distributed and parallel machine learning. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Distributed](https://github.com/dask/distributed) - Distributed computation in Python.

## Experimentation
* [Sacred](https://github.com/IDSIA/sacred) - A tool to help you configure, organize, log and reproduce experiments.
* [Xcessiv](https://github.com/reiinakano/xcessiv) - A web-based application for quick, scalable, and automated hyperparameter tuning and stacked ensembling.
* [Persimmon](https://github.com/AlvarBer/Persimmon) - A visual dataflow programming language for sklearn.
* [Ax](https://github.com/facebook/Ax) - Adaptive Experimentation Platform. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [Neptune](https://neptune.ml) - A lightweight ML experiment tracking, results visualization and management tool.

## Evaluation
* [recmetrics](https://github.com/statisticianinstilettos/recmetrics) - Library of useful metrics and plots for evaluating recommender systems.
* [Metrics](https://github.com/benhamner/Metrics) - Machine learning evaluation metric.
* [sklearn-evaluation](https://github.com/edublancas/sklearn-evaluation) - Model evaluation made easy: plots, tables and markdown reports. <img height="20" src="img/sklearn_big.png" alt="sklearn">
* [AI Fairness 360](https://github.com/IBM/AIF360) - Fairness metrics for datasets and ML models, explanations and algorithms to mitigate bias in datasets and models.

## Computations
* [numpy](http://www.numpy.org/) - The fundamental package needed for scientific computing with Python.
* [Dask](https://github.com/dask/dask) - Parallel computing with task scheduling. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [bottleneck](https://github.com/kwgoodman/bottleneck) - Fast NumPy array functions written in C.
* [CuPy](https://github.com/cupy/cupy) - NumPy-like API accelerated with CUDA.
* [scikit-tensor](https://github.com/mnick/scikit-tensor) - Python library for multilinear algebra and tensor factorizations.
* [numdifftools](https://github.com/pbrod/numdifftools) - Solve automatic numerical differentiation problems in one or more variables.
* [quaternion](https://github.com/moble/quaternion) - Add built-in support for quaternions to numpy.
* [adaptive](https://github.com/python-adaptive/adaptive) - Tools for adaptive and parallel samping of mathematical functions.

## Spatial Analysis
* [GeoPandas](https://github.com/geopandas/geopandas) - Python tools for geographic data. <img height="20" src="img/pandas_big.png" alt="pandas compatible">
* [PySal](https://github.com/pysal/pysal) - Python Spatial Analysis Library.

## Quantum Computing
* [PennyLane](https://github.com/XanaduAI/pennylane) - Quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations.
* [QML](https://github.com/qmlcode/qml) - A Python Toolkit for Quantum Machine Learning.

## Conversion
* [sklearn-porter](https://github.com/nok/sklearn-porter) - Transpile trained scikit-learn estimators to C, Java, JavaScript and others.
* [ONNX](https://github.com/onnx/onnx) - Open Neural Network Exchange.
* [MMdnn](https://github.com/Microsoft/MMdnn) -  A set of tools to help users inter-operate among different deep learning frameworks.

## Contributing
Contributions are welcome! :sunglasses: </br>
Read the <a href=https://github.com/krzjoa/awesome-python-datascience/blob/master/CONTRIBUTING.md>contribution guideline</a>.

## License
This work is licensed under the Creative Commons Attribution 4.0 International License - [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

<div align="center">
	<a href="other/deprecated.md">Deprecated Libs</a>&nbsp;&nbsp;&nbsp;
	<a href="other/waiting-room.md">Waiting Room</a>&nbsp;&nbsp;&nbsp;
<div>
