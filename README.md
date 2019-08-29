<div align="center">
	<img width="250" height="250" src="img/py-datascience.png" alt="pyds">
	<br>
	<br>
	<br>
</div>

<h1 align="center">
	Awesome Python Data Science
</h1>

> Probably the best curated list of data science software in Python

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

[skl]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/skl.png "scikit-learn compatible/inspired"
[th]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/th.png "Theano based"
[tf]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/tf2.png "TensorFlow based"
[pt]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/pt2.png "PyTorch based"
[cp]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/cupy.png "CuPy based"
[mx]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/mxnet.png "MXNet based"
[R]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/R.png "R inspired/ported lib"
[gpu]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/gpu.png "GPU accelerated"
[sp]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/spark.png "Apache Spark based"
[amd]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/amd.png "possible to run on AMD"
[pd]: https://raw.githubusercontent.com/krzjoa/awesome-python-datascience/master/img/pandas.png "pandas based"

## Contents
* [Machine Learning](#ml)
  * [General Purpouse ML](#ml-gen)
  * [Time Series](#ml-ts)
  * [Automated Machine Learning](#ml-automl)
  * [Ensemble Methods](#ml-ens)
  * [Imbalanced Datasets](#imb)
  * [Random Forests](#ml-rf)
  * [Extreme Learning Machine](#ml-elm)
  * [Kernel Methods](#ml-fm)
  * [Gradient Boosting](#ml-gbt)
* [Deep Learning](#dl)
  * [PyTorch](#dl-pytorch)
  * [Keras](#dl-keras)
  * [TensorFlow](#dl-tf)
  * [MXnet](#dl-mxnet)
  * [Chainer](#dl-chainer)
  * [Theano](#dl-theano)
  * [Others](#dl-others)
* [Data Manipulation](#data-man)
  * [Data Containers](#dm-cont)
  * [Pipelines](#dm-pipe)
* [Feature Engineering](#feat-eng)
  * [General](#fe-general)
  * [Feature Selection](#fe-selection)
* [Visualization](#vis)
* [Model Explanation](#expl)
* [Reinforcement Learning](#rl)
* [Distributed Computing](#dist)
* [Probabilistic Methods](#bayes)
* [Genetic Programming](#gp)
* [Optimization](#opt)
* [Natural Language Processing](#nlp)
* [Computer Audition](#ca)
* [Computer Vision](#cv)
* [Statistics](#stat)
* [Experimentation](#tools)
* [Evaluation](#eval)
* [Computations](#compt)
* [Spatial Analysis](#spatial)
* [Quantum Computing](#quant)
* [Conversion](#conv)

<a name="ml"></a>
## Machine Learning

<a name="ml-gen"></a>
### General Purpouse Machine Learning
* [scikit-learn](http://scikit-learn.org/stable/) - Machine learning in Python. ![alt text][skl]
* [Shogun](http://www.shogun-toolbox.org/) - Machine learning toolbox.
* [xLearn](https://github.com/aksnzhy/xlearn) - High Performance, Easy-to-use, and Scalable Machine Learning Package.
* [cuML](https://github.com/rapidsai/cuml) - RAPIDS Machine Learning Library. ![alt text][skl] ![alt text][gpu] 
* [Reproducible Experiment Platform (REP)](https://github.com/yandex/rep) - Machine Learning toolbox for Humans. ![alt text][skl]
* [modAL](https://github.com/cosmic-cortex/modAL) -  Modular active learning framework for Python3. ![alt text][skl]
* [Sparkit-learn](https://github.com/lensacom/sparkit-learn) - PySpark + Scikit-learn = Sparkit-learn. ![alt text][skl] ![alt text][sp]
* [mlpack](https://github.com/mlpack/mlpack) - A scalable C++ machine learning library (Python bindings).
* [dlib](https://github.com/davisking/dlib) - Toolkit for making real world machine learning and data analysis applications in C++ (Python bindings).
* [MLxtend](https://github.com/rasbt/mlxtend) - Extension and helper modules for Python's data analysis and machine learning libraries. ![alt text][skl]
* [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) - Multi-label classification for python. ![alt text][skl]
* [seqlearn](https://github.com/larsmans/seqlearn) - Sequence classification toolkit for Python. ![alt text][skl]
* [pystruct](https://github.com/pystruct/pystruct) - Simple structured learning framework for Python. ![alt text][skl]
* [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys) - Highly interpretable classifiers for scikit learn, producing easily understood decision rules instead of black box models. ![alt text][skl]
* [RuleFit](https://github.com/christophM/rulefit) - Implementation of the rulefit. ![alt text][skl]
* [metric-learn](https://github.com/all-umass/metric-learn)  - Metric learning algorithms in Python. ![alt text][skl]
* [pyGAM](https://github.com/dswah/pyGAM) - Generalized Additive Models in Python.
* [Other...](https://github.com/krzjoa/awesome-python-datascience/blob/master/other/general-ml.md)

<a name="ml-ts"></a>
### Time Series
* [tslearn](https://github.com/rtavenar/tslearn) - Machine learning toolkit dedicated to time-series data. ![alt text][skl]
* [tick](https://github.com/X-DataInitiative/tick) - Module for statistical learning, with a particular emphasis on time-dependent modelling.  ![alt text][skl]
* [Prophet](https://github.com/facebook/prophet) - Automatic Forecasting Procedure.
* [PyFlux](https://github.com/RJT1990/pyflux) - Open source time series library for Python.
* [bayesloop](https://github.com/christophmark/bayesloop) - Probabilistic programming framework that facilitates objective model selection for time-varying parameter models.
* [luminol](https://github.com/linkedin/luminol) - Anomaly Detection and Correlation library.

<a name="ml-automl"></a>
### Automated Machine Learning
* [TPOT](https://github.com/rhiever/tpot) -  Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming. ![alt text][skl]
* [auto-sklearn](https://github.com/automl/auto-sklearn) - An automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator. ![alt text][skl]
* [MLBox](https://github.com/AxeldeRomblay/MLBox) - A powerful Automated Machine Learning python library.

<a name="ml-ens"></a>
### Ensemble Methods
* [ML-Ensemble](http://ml-ensemble.com/) -  High performance ensemble learning. ![alt text][skl]
* [Stacking](https://github.com/ikki407/stacking) - Simple and useful stacking library, written in Python. ![alt text][skl]
* [stacked_generalization](https://github.com/fukatani/stacked_generalization) - Library for machine learning stacking generalization. ![alt text][skl]
* [vecstack](https://github.com/vecxoz/vecstack) - Python package for stacking (machine learning technique). ![alt text][skl]

<a name="imb"></a>
### Imbalanced Datasets
* [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)  - Module to perform under sampling and over sampling with various techniques. ![alt text][skl]
* [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms)  - Python-based implementations of algorithms for learning on imbalanced data. ![alt text][skl] ![alt text][tf]

<a name="ml-rf"></a>
### Random Forests
* [rpforest](https://github.com/lyst/rpforest)  - A forest of random projection trees. ![alt text][skl]
* [Random Forest Clustering](https://github.com/joshloyal/RandomForestClustering) - Unsupervised Clustering using Random Forests.![alt text][skl]
* [sklearn-random-bits-forest](https://github.com/tmadl/sklearn-random-bits-forest) - Wrapper of the Random Bits Forest program written by (Wang et al., 2016).![alt text][skl]
* [rgf_python](https://github.com/fukatani/rgf_python) - Python Wrapper of Regularized Greedy Forest. ![alt text][skl]

<a name="ml-elm"></a>
### Extreme Learning Machine
* [Python-ELM](https://github.com/dclambert/Python-ELM)  - Extreme Learning Machine implementation in Python. ![alt text][skl]
* [Python Extreme Learning Machine (ELM)](https://github.com/acba/elm) - a machine learning technique used for classification/regression tasks.
* [hpelm](https://github.com/akusok/hpelm)  - High performance implementation of Extreme Learning Machines (fast randomized neural networks). ![alt text][gpu]

<a name="ml-fm"></a>
### Kernel Methods
* [pyFM](https://github.com/coreylynch/pyFM) - Factorization machines in python. ![alt text][skl]
* [fastFM](https://github.com/ibayer/fastFM) - A library for Factorization Machines. ![alt text][skl]
* [tffm](https://github.com/geffy/tffm) - TensorFlow implementation of an arbitrary order Factorization Machine. ![alt text][skl] ![alt text][tf] 
* [liquidSVM](https://github.com/liquidSVM/liquidSVM) - An implementation of SVMs.
* [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm) - Relevance Vector Machine implementation using the scikit-learn API. ![alt text][skl]
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) - A fast SVM Library on GPUs and CPUs. ![alt text][skl] ![alt text][gpu]

<a name="ml-gbt"></a>
### Gradient Boosting
* [XGBoost](https://github.com/dmlc/xgboost) - Scalable, Portable and Distributed Gradient Boosting. ![alt text][skl] ![alt text][gpu]
* [LightGBM](https://github.com/Microsoft/LightGBM)- A fast, distributed, high performance gradient boosting by [Microsoft](https://www.microsoft.com). ![alt text][skl] ![alt text][gpu] 
* [CatBoost](https://github.com/catboost/catboost) - An open-source gradient boosting on decision trees library by [Yandex](https://www.yandex.com/). ![alt text][skl] ![alt text][gpu]
* [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) - Fast GBDTs and Random Forests on GPUs. ![alt text][skl] ![alt text][gpu]
* [Other...](https://github.com/krzjoa/awesome-python-datascience/blob/master/other/gbm.md)

<a name="dl"></a>
## Deep Learning

<a name="dl-pytorch"></a>
### PyTorch
* [PyTorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python with strong GPU acceleration. ![alt text][pt] 
* [torchvision](https://github.com/pytorch/vision) - Datasets, Transforms and Models specific to Computer Vision. ![alt text][pt]
* [torchtext](https://github.com/pytorch/text) - Data loaders and abstractions for text and NLP. ![alt text][pt]
* [torchaudio](https://github.com/pytorch/audio) - An audio library for PyTorch. ![alt text][pt]
* [ignite](https://github.com/pytorch/ignite)  - High-level library to help with training neural networks in PyTorch. ![alt text][pt]
* [PyToune](https://github.com/GRAAL-Research/pytoune) - A Keras-like framework and utilities for PyTorch.
* [skorch](https://github.com/dnouri/skorch)  - A scikit-learn compatible neural network library that wraps pytorch. ![alt text][skl] ![alt text][pt]
* [PyTorchNet](https://github.com/pytorch/tnt)  - An abstraction to train neural networks ![alt text][pt]
* [Aorun](https://github.com/ramon-oliveira/aorun) - Intend to implement an API similar to Keras with PyTorch as backend. ![alt text][pt]
* [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) - Geometric Deep Learning Extension Library for PyTorch. ![alt text][pt]

<a name="dl-keras"></a>
### Keras
* [Keras](https://keras.io) - A high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
* [keras-contrib](https://github.com/keras-team/keras-contrib) - Keras community contributions.
* [Hyperas](https://github.com/maxpumperla/hyperas) - Keras + Hyperopt: A very simple wrapper for convenient hyperparameter.
* [Elephas](https://github.com/maxpumperla/elephas) - Distributed Deep learning with Keras & Spark.
* [Hera](https://github.com/keplr-io/hera) - Train/evaluate a Keras model, get metrics streamed to a dashboard in your browser.
* [dist-keras](https://github.com/cerndb/dist-keras) ![alt text][sp] - Distributed Deep Learning, with a focus on distributed training.
* [Spektral](https://github.com/danielegrattarola/spektral) - Deep learning on graphs.
* [qkeras](https://github.com/google/qkeras) - A quantization deep learning library.
* [Keras add-ons...](https://github.com/krzjoa/awesome-python-datascience/blob/master/addons/keras_addons.md)

<a name="dl-tf"></a>
### TensorFlow
* [TensorFlow](https://github.com/tensorflow/tensorflow) - Computation using data flow graphs for scalable machine learning by Google. ![alt text][tf]
* [TensorLayer](https://github.com/zsdonghao/tensorlayer) - Deep Learning and Reinforcement Learning Library for Researcher and Engineer. ![alt text][tf]
* [TFLearn](https://github.com/tflearn/tflearn) - Deep learning library featuring a higher-level API for TensorFlow. ![alt text][tf]
* [Sonnet](https://github.com/deepmind/sonnet) - TensorFlow-based neural network library by [DeepMind](https://deepmind.com/). ![alt text][tf]
* [TensorForce](https://github.com/reinforceio/tensorforce) - A TensorFlow library for applied reinforcement learning. ![alt text][tf]
* [tensorpack](https://github.com/ppwwyyxx/tensorpack) - A Neural Net Training Interface on TensorFlow ![alt text][tf]
* [Polyaxon](https://github.com/polyaxon/polyaxon) - A platform that helps you build, manage and monitor deep learning models. ![alt text][tf]
* [NeuPy](https://github.com/itdxer/neupy) - NeuPy is a Python library for Artificial Neural Networks and Deep Learning (previously: ![alt text][th]). ![alt text][tf]
* [tfdeploy](https://github.com/riga/tfdeploy) - Deploy tensorflow graphs for fast evaluation and export to tensorflow-less environments running numpy. ![alt text][tf]
* [tensorflow-upstream](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream) - TensorFlow ROCm port. ![alt text][tf] ![alt text][amd]
* [TensorFlow Fold](https://github.com/tensorflow/fold) - Deep learning with dynamic computation graphs in TensorFlow. ![alt text][tf]
* [tensorlm](https://github.com/batzner/tensorlm) - Wrapper library for text generation / language models at char and word level with RNN. ![alt text][tf]
* [TensorLight](https://github.com/bsautermeister/tensorlight)  - A high-level framework for TensorFlow. ![alt text][tf]
* [Mesh TensorFlow](https://github.com/tensorflow/mesh) - Model Parallelism Made Easier. ![alt text][tf]
* [Ludwig](https://github.com/uber/ludwig) - A toolbox, that allows to train and test deep learning models without the need to write code. ![alt text][tf]

<a name="dl-mxnet"></a>
### MXNet
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler. ![alt text][mx]
* [Gluon](https://github.com/gluon-api/gluon-api) - A clear, concise, simple yet powerful and efficient API for deep learning (now included in MXNet). ![alt text][mx]
* [MXbox](https://github.com/Lyken17/mxbox) - Simple, efficient and flexible vision toolbox for mxnet framework. ![alt text][mx]
* [gluon-cv](https://github.com/dmlc/gluon-cv) - Provides implementations of the state-of-the-art  deep learning models in computer vision. ![alt text][mx]
* [gluon-nlp](https://github.com/dmlc/gluon-nlp) - NLP made easy. ![alt text][mx]
* [Xfer](https://github.com/amzn/xfer) - Transfer Learning library for Deep Neural Networks. ![alt text][mx]
* [MXNet](https://github.com/ROCmSoftwarePlatform/mxnet)  - HIP Port of MXNet. ![alt text][mx] ![alt text][amd]

<!--a name="dl-cntk"></a-->
<a name="dl-chainer"></a>
### Chainer
* [ChainerRL](https://github.com/chainer/chainerrl) - A deep reinforcement learning library built on top of Chainer.
* [Chainer](https://github.com/chainer/chainer) - A flexible framework for neural networks.
* [ChainerCV](https://github.com/chainer/chainercv) - A Library for Deep Learning in Computer Vision.
* [ChainerMN](https://github.com/chainer/chainermn) - Scalable distributed deep learning with Chainer.
* [scikit-chainer](https://github.com/lucidfrontier45/scikit-chainer) - Scikit-learn like interface to chainer. ![alt text][skl]
* [chainer_sklearn](https://github.com/corochann/chainer_sklearn) - Sklearn (Scikit-learn) like interface for Chainer. ![alt text][skl]

<a name="dl-theano"></a>
### Theano
**WARNING: Theano development has been stopped**
* [Theano](https://github.com/Theano/Theano) - A Python library that allows you to define, optimize, and evaluate mathematical expressions.![alt text][th]
* [Lasagne](https://github.com/Lasagne/Lasagne) - Lightweight library to build and train neural networks in Theano [Lasagne add-ons...](https://github.com/krzjoa/awesome-python-datascience/blob/master/addons/lasagne_addons.md) ![alt text][th]
* [nolearn](https://github.com/dnouri/nolearn) - A scikit-learn compatible neural network library (mainly for Lasagne). ![alt text][th] ![alt text][skl]
* [Blocks](https://github.com/mila-udem/blocks) - A Theano framework for building and training neural networks. ![alt text][th]
* [platoon](https://github.com/mila-udem/platoon) - Multi-GPU mini-framework for Theano. ![alt text][th]
* [scikit-neuralnetwork](https://github.com/aigamedev/scikit-neuralnetwork) - Deep neural networks without the learning cliff. ![alt text][skl]  ![alt text][th]
* [Theano-MPI](https://github.com/uoguelph-mlrg/Theano-MPI) - MPI Parallel framework for training deep learning models built in Theano. ![alt text][th]

<a name="dl-others"></a>
### Others
* [CNTK](https://github.com/Microsoft/CNTK) - Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit.
* [Neon](https://github.com/NervanaSystems/neon) - Intel® Nervana™ reference deep learning framework committed to best performance on all hardware.
* [Tangent](https://github.com/google/tangent) - Source-to-Source Debuggable Derivatives in Pure Python.
* [autograd](https://github.com/HIPS/autograd) - Efficiently computes derivatives of numpy code.
* [Myia](https://github.com/mila-udem/myia) - Deep Learning framework (pre-alpha).
* [nnabla](https://github.com/sony/nnabla) - Neural Network Libraries by Sony.
* [Caffe](https://github.com/BVLC/caffe) - a fast open framework for deep learning.
* [Caffe2](https://github.com/pytorch/pytorch/tree/master/caffe2) -  A lightweight, modular, and scalable deep learning framework (now a part of PyTorch).
* [hipCaffe](https://github.com/ROCmSoftwarePlatform/hipCaffe) - The HIP port of Caffe. ![alt text][amd]

<a name="data-man"></a>
## Data Manipulation

<a name="dm-cont"></a>
### Data Containers
* [pandas](https://pandas.pydata.org/pandas-docs/stable/) - Powerful Python data analysis toolkit.
* [cuDF](https://github.com/rapidsai/cudf) - GPU DataFrame Library. ![alt text][pd] ![alt text][gpu]
* [blaze](https://github.com/blaze/blaze) - NumPy and Pandas interface to Big Data. ![alt text][pd]
* [pandasql](https://github.com/yhat/pandasql) -  Allows you to query pandas DataFrames using SQL syntax. ![alt text][pd]
* [pandas-gbq](https://github.com/pydata/pandas-gbq) - Pandas Google Big Query. ![alt text][pd]
* [xpandas](https://github.com/alan-turing-institute/xpandas) - Universal 1d/2d data containers with Transformers .functionality for data analysis by [The Alan Turing Institute](https://www.turing.ac.uk/).
* [pysparkling](https://github.com/svenkreiss/pysparkling) - A pure Python implementation of Apache Spark's RDD and DStream interfaces. ![alt text][sp]
* [Arctic](https://github.com/manahl/arctic) - High performance datastore for time series and tick data.
* [datatable](https://github.com/h2oai/datatable) - Data.table for Python. ![alt text][R]
* [koalas](https://github.com/databricks/koalas) - Pandas API on Apache Spark. ![alt text][pd]

<a name="dm-pipe"></a>
### Pipelines
* [Fuel](https://github.com/mila-udem/fuel) - Data pipeline framework for machine learning.
* [pdpipe](https://github.com/shaypal5/pdpipe) - Sasy pipelines for pandas DataFrames.
* [SSPipe](https://sspipe.github.io/) - Python pipe (|) operator with support for DataFrames and Numpy and Pytorch.
* [meza](https://github.com/reubano/meza) - A Python toolkit for processing tabular data.
* [pandas-ply](https://github.com/coursera/pandas-ply) - Functional data manipulation for pandas. ![alt text][pd]
* [Dplython](https://github.com/dodger487/dplython) - Dplyr for Python. ![alt text][R]
* [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas)  - Pandas integration with sklearn. ![alt text][skl] ![alt text][pd]
* [quinn](https://github.com/MrPowers/quinn)  - pyspark methods to enhance developer productivity. ![alt text][sp]
* [Dataset](https://github.com/analysiscenter/dataset) - Helps you conveniently work with random or sequential batches of your data and define data processing.
* [swifter](https://github.com/jmcarpenter2/swifter) - A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner.
* [pyjanitor](https://github.com/ericmjl/pyjanitor) - Clean APIs for data cleaning. ![alt text][pd]
* [modin](https://github.com/modin-project/modin) - Speed up your Pandas workflows by changing a single line of code. ![alt text][pd]
* [Prodmodel](https://github.com/prodmodel/prodmodel) - Build system for data science pipelines.


<a name="feat-eng"></a>
## Feature Engineering

<a name="fe-general"></a>
### General
* [Featuretools](https://github.com/Featuretools/featuretools) - Automated feature engineering.
* [skl-groups](https://github.com/dougalsutherland/skl-groups) - Scikit-learn addon to operate on set/"group"-based features. ![alt text][skl]
* [Feature Forge](https://github.com/machinalis/featureforge) - A set of tools for creating and testing machine learning feature. ![alt text][skl]
* [few](https://github.com/lacava/few) - A feature engineering wrapper for sklearn. ![alt text][skl]
* [scikit-mdr](https://github.com/EpistasisLab/scikit-mdr) - A sklearn-compatible Python implementation of Multifactor Dimensionality Reduction (MDR) for feature construction. ![alt text][skl]
* [tsfresh](https://github.com/blue-yonder/tsfresh) - Automatic extraction of relevant features from time series. ![alt text][skl]

<a name="fe-selection"></a>
### Feature Selection
* [scikit-feature](https://github.com/jundongl/scikit-feature) -  Feature selection repository in python.
* [boruta_py](https://github.com/scikit-learn-contrib/boruta_py) -  Implementations of the Boruta all-relevant feature selection method. ![alt text][skl] 
* [BoostARoota](https://github.com/chasedehan/BoostARoota) - A fast xgboost feature selection algorithm. ![alt text][skl]
* [scikit-rebate](https://github.com/EpistasisLab/scikit-rebate)- A scikit-learn-compatible Python  ![alt text][skl] implementation of ReBATE, a suite of Relief-based feature selection algorithms for Machine Learning.

<a name="vis"></a>
## Visualization
* [Matplotlib](https://github.com/matplotlib/matplotlib) - Plotting with Python.
* [seaborn](https://github.com/mwaskom/seaborn) - Statistical data visualization using matplotlib.
* [Bokeh](https://github.com/bokeh/bokeh) - Interactive Web Plotting for Python.
* [HoloViews](https://github.com/ioam/holoviews) - Stop plotting your data - annotate your data and let it visualize itself.
* [Alphalens](https://github.com/quantopian/alphalens) - Performance analysis of predictive (alpha) stock factors by [Quantopian](https://www.quantopian.com/).
* [prettyplotlib](https://github.com/olgabot/prettyplotlib) - Painlessly create beautiful matplotlib plots.
* [python-ternary](https://github.com/marcharper/python-ternary) - Ternary plotting library for python with matplotlib.
* [missingno](https://github.com/ResidentMario/missingno) - Missing data visualization module for Python.


<a name="expl"></a>
## Model Explanation
* [Alibi](https://github.com/SeldonIO/alibi) - Algorithms for monitoring and explaining machine learning models.
* [Auralisation](https://github.com/keunwoochoi/Auralisation) - Auralisation of learned features in CNN (for audio).
* [CapsNet-Visualization](https://github.com/bourdakos1/CapsNet-Visualization) - A visualization of the CapsNet layers to better understand how it works.
* [lucid](https://github.com/tensorflow/lucid) - A collection of infrastructure and tools for research in neural network interpretability.
* [Netron](https://github.com/lutzroeder/Netron) - Visualizer for deep learning and machine learning models (no Python code, but visualizes models from most Python Deep Learning frameworks).
* [FlashLight](https://github.com/dlguys/flashlight) - Visualization Tool for your NeuralNetwork.
* [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) - Tensorboard for pytorch (and chainer, mxnet, numpy, ...).
* [anchor](https://github.com/marcotcr/anchor) - Code for "High-Precision Model-Agnostic Explanations" paper.
* [aequitas](https://github.com/dssg/aequitas) - Bias and Fairness Audit Toolkit.
* [Contrastive Explanation](https://github.com/MarcelRobeer/ContrastiveExplanation) - Contrastive Explanation (Foil Trees). ![alt text][skl]
* [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick)- Visual analysis and diagnostic tools to facilitate machine learning model selection. ![alt text][skl]
* [scikit-plot](https://github.com/reiinakano/scikit-plot) - An intuitive library to add plotting functionality to scikit-learn objects. ![alt text][skl]
* [shap](https://github.com/slundberg/shap) - A unified approach to explain the output of any machine learning model. ![alt text][skl]
* [ELI5](https://github.com/TeamHG-Memex/eli5) - A library for debugging/inspecting machine learning classifiers and explaining their predictions.
* [Lime](https://github.com/marcotcr/lime)- Explaining the predictions of any machine learning classifier. ![alt text][skl] 
* [FairML](https://github.com/adebayoj/fairml)- FairML is a python toolbox auditing the machine learning models for bias. ![alt text][skl] 
* [L2X](https://github.com/Jianbo-Lab/L2X) - Code for replicating the experiments in the paper *Learning to Explain: An Information-Theoretic Perspective on Model Interpretation*.
* [PDPbox](https://github.com/SauceCat/PDPbox) - Partial dependence plot toolbox.
* [pyBreakDown](https://github.com/MI2DataLab/pyBreakDown) - Python implementation of R package breakDown. ![alt text][skl]
* [PyCEbox](https://github.com/AustinRochford/PyCEbox) - Python Individual Conditional Expectation Plot Toolbox.
* [Skater](https://github.com/datascienceinc/Skater) - Python Library for Model Interpretation.
* [model-analysis](https://github.com/tensorflow/model-analysis)- Model analysis tools for TensorFlow. ![alt text][tf] 
* [themis-ml](https://github.com/cosmicBboy/themis-ml) - A library that implements fairness-aware machine learning algorithms. ![alt text][skl]
* [treeinterpreter](https://github.com/andosa/treeinterpreter) - Interpreting scikit-learn's decision tree and random forest predictions. ![alt text][skl]
* [mxboard](https://github.com/awslabs/mxboard) - Logging MXNet data for visualization in TensorBoard. ![alt text][mx]
* [AI Explainability 360](https://github.com/IBM/AIX360) - Interpretability and explainability of data and machine learning models.

<a name="rl"></a>
## Reinforcement Learning
* [OpenAI Gym](https://github.com/openai/gym) - A toolkit for developing and comparing reinforcement learning algorithms.

<a name="dist"></a>
## Distributed Computing
* [Horovod](https://github.com/uber/horovod)- Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. ![alt text][tf] 
* [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html) - Exposes the Spark programming model to Python. ![alt text][sp]
* [Veles](https://github.com/Samsung/veles) - Distributed machine learning platform by [Samsung](https://github.com/Samsung).
* [Jubatus](https://github.com/jubatus/jubatus) - Framework and Library for Distributed Online Machine Learning.
* [DMTK](https://github.com/Microsoft/DMTK) - Microsoft Distributed Machine Learning Toolkit.
* [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - PArallel Distributed Deep LEarning by [Baidu](https://www.baidu.com/).
* [dask-ml](https://github.com/dask/dask-ml)- Distributed and parallel machine learning. ![alt text][skl] 
* [Distributed](https://github.com/dask/distributed) - Distributed computation in Python.

<a name="bayes"></a>
## Probabilistic Methods
* [pomegranate](https://github.com/jmschrei/pomegranate)- Probabilistic and graphical models for Python. ![alt text][cp] 
* [pyro](https://github.com/uber/pyro) - A flexible, scalable deep probabilistic programming library built on PyTorch. ![alt text][pt]
* [ZhuSuan](http://zhusuan.readthedocs.io/en/latest/)- Bayesian Deep Learning. ![alt text][tf] 
* [PyMC](https://github.com/pymc-devs/pymc) - Bayesian Stochastic Modelling in Python.
* [PyMC3](http://docs.pymc.io/)- Python package for Bayesian statistical modeling and Probabilistic Machine Learning. ![alt text][th] 
* [sampled](https://github.com/ColCarroll/sampled) - Decorator for reusable models in PyMC3.
* [Edward](http://edwardlib.org/) - A library for probabilistic modeling, inference, and criticism. ![alt text][tf]
* [InferPy](https://github.com/PGM-Lab/InferPy) - Deep Probabilistic Modelling Made Easy.  ![alt text][tf] 
* [GPflow](http://gpflow.readthedocs.io/en/latest/?badge=latest) - Gaussian processes in TensorFlow. ![alt text][tf]
* [PyStan](https://github.com/stan-dev/pystan) - Bayesian inference using the No-U-Turn sampler (Python interface).
* [gelato](https://github.com/ferrine/gelato) - Bayesian dessert for Lasagne. ![alt text][th]
* [sklearn-bayes](https://github.com/AmazaspShumik/sklearn-bayes) - Python package for Bayesian Machine Learning with scikit-learn API. ![alt text][skl]
* [skggm](https://github.com/skggm/skggm) - Estimation of general graphical models. ![alt text][skl]
* [pgmpy](https://github.com/pgmpy/pgmpy) - A python library for working with Probabilistic Graphical Models.
* [skpro](https://github.com/alan-turing-institute/skpro) - Supervised domain-agnostic prediction framework for probabilistic modelling by [The Alan Turing Institute](https://www.turing.ac.uk/). ![alt text][skl]
* [Aboleth](https://github.com/data61/aboleth) - A bare-bones TensorFlow framework for Bayesian deep learning and Gaussian process approximation. ![alt text][tf]
* [PtStat](https://github.com/stepelu/ptstat) - Probabilistic Programming and Statistical Inference in PyTorch. ![alt text][pt]
* [PyVarInf](https://github.com/ctallec/pyvarinf) - Bayesian Deep Learning methods with Variational Inference for PyTorch. ![alt text][pt]
* [emcee](https://github.com/dfm/emcee) - The Python ensemble sampling toolkit for affine-invariant MCMC.
* [hsmmlearn](https://github.com/jvkersch/hsmmlearn) - A library for hidden semi-Markov models with explicit durations.
* [pyhsmm](https://github.com/mattjj/pyhsmm) - Bayesian inference in HSMMs and HMMs.
* [GPyTorch](https://github.com/cornellius-gp/gpytorch) - A highly efficient and modular implementation of Gaussian Processes in PyTorch. ![alt text][pt]
* [MXFusion](https://github.com/amzn/MXFusion) - Modular Probabilistic Programming on MXNet ![alt text][mx]
* [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite) - Scikit-learn inspire.d API for CRFsuite. ![alt text][skl]

<a name="gp"></a>
## Genetic Programming
* [gplearn](https://github.com/trevorstephens/gplearn) - Genetic Programming in Python. ![alt text][skl] 
* [DEAP](https://github.com/DEAP/deap) - Distributed Evolutionary Algorithms in Python.
* [karoo_gp](https://github.com/kstaats/karoo_gp)  - A Genetic Programming platform for Python with GPU support. ![alt text][tf]
* [monkeys](https://github.com/hchasestevens/monkeys) - A strongly-typed genetic programming framework for Python.
* [sklearn-genetic](https://github.com/manuel-calzolari/sklearn-genetic)  - Genetic feature selection module for scikit-learn. ![alt text][skl]

<a name="opt"></a>
## Optimization
* [Spearmint](https://github.com/HIPS/Spearmint) - Bayesian optimization.
* [BoTorch](https://github.com/pytorch/botorch)  - Bayesian optimization in PyTorch. ![alt text][pt]
* [SMAC3](https://github.com/automl/SMAC3) - Sequential Model-based Algorithm Configuration.
* [Optunity](https://github.com/claesenm/optunity) - Is a library containing various optimizers for hyperparameter tuning.
* [hyperopt](https://github.com/hyperopt/hyperopt) - Distributed Asynchronous Hyperparameter Optimization in Python.
* [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)   - Hyper-parameter optimization for sklearn. ![alt text][skl]
* [sklearn-deap](https://github.com/rsteca/sklearn-deap) - Use evolutionary algorithms instead of gridsearch in scikit-learn. ![alt text][skl] 
* [sigopt_sklearn](https://github.com/sigopt/sigopt_sklearn)  - SigOpt wrappers for scikit-learn methods. ![alt text][skl]
* [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization) - A Python implementation of global optimization with gaussian processes.
* [SafeOpt](https://github.com/befelix/SafeOpt) - Safe Bayesian Optimization.
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) - Sequential model-based optimization with a `scipy.optimize` interface.
* [Solid](https://github.com/100/Solid) - A comprehensive gradient-free optimization framework written in Python.
* [PySwarms](https://github.com/ljvmiranda921/pyswarms) - A research toolkit for particle swarm optimization in Python.
* [Platypus](https://github.com/Project-Platypus/Platypus) - A Free and Open Source Python Library for Multiobjective Optimization.
* [GPflowOpt](https://github.com/GPflow/GPflowOpt) - Bayesian Optimization using GPflow. ![alt text][tf] 
* [POT](https://github.com/rflamary/POT) - Python Optimal Transport library.
* [Talos](https://github.com/autonomio/talos) - Hyperparameter Optimization for Keras Models.
* [nlopt](https://github.com/stevengj/nlopt) - Library for nonlinear optimization (global and local, constrained or unconstrained).

<a name="nlp"></a>
## Natural Language Processing
* [NLTK](https://github.com/nltk/nltk) -  Modules, data sets, and tutorials supporting research and development in Natural Language Processing.
* [CLTK](https://github.com/cltk/cltk) - The Classical Language Toolkik.
* [gensim](https://radimrehurek.com/gensim/) - Topic Modelling for Humans.
* [PSI-Toolkit](http://psi-toolkit.amu.edu.pl/) - A natural language processing toolkit by [Adam Mickiewicz University](https://zpjn.wmi.amu.edu.pl) in Poznań.
* [pyMorfologik](https://github.com/dmirecki/pyMorfologik) - Python binding for [Morfologik](https://github.com/morfologik/morfologik-stemming) (Polish morphological analyzer).
* [skift](https://github.com/shaypal5/skift)- Scikit-learn wrappers for Python fastText. ![alt text][skl] 
* [Phonemizer](https://github.com/bootphon/phonemizer) - Simple text to phonemes converter for multiple languages.
* [flair](https://github.com/zalandoresearch/flair) - Very simple framework for state-of-the-art NLP by [Zalando Research](https://research.zalando.com/).

<a name="ca"></a>
## Computer Audition
* [librosa](https://github.com/librosa/librosa) - Python library for audio and music analysis.
* [Yaafe](https://github.com/Yaafe/Yaafe) - Audio features extraction.
* [aubio](https://github.com/aubio/aubio) - A library for audio and music analysis.
* [Essentia](https://github.com/MTG/essentia) - Library for audio and music analysis, description and synthesis.
* [LibXtract](https://github.com/jamiebullock/LibXtract) - A simple, portable, lightweight library of audio feature extraction functions.
* [Marsyas](https://github.com/marsyas/marsyas) - Music Analysis, Retrieval and Synthesis for Audio Signals.
* [muda](https://github.com/bmcfee/muda) - A library for augmenting annotated audio data.
* [madmom](https://github.com/CPJKU/madmom) - Python audio and music signal processing library.
* [more: Python for Scientific Audio](https://github.com/faroit/awesome-python-scientific-audio)

<a name="cv"></a>
## Computer Vision
* [OpenCV](https://github.com/opencv/opencv) - Open Source Computer Vision Library.
* [scikit-image](https://github.com/scikit-image/scikit-image) - Image Processing SciKit (Toolbox for SciPy).
* [imgaug](https://github.com/aleju/imgaug) - Image augmentation for machine learning experiments.
* [imgaug_extension](https://github.com/cadenai/imgaug_extension) - Additional augmentations for imgaug.
* [Augmentor](https://github.com/mdbloice/Augmentor) - Image augmentation library in Python for machine learning.
* [albumentations](https://github.com/albu/albumentations) - Fast image augmentation library and easy to use wrapper around other libraries.

<a name="stat"></a>

## Statistics
* [pandas_summary](https://github.com/mouradmourafiq/pandas-summary) - Extension to pandas dataframes describe function. ![alt text][pd] 
* [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) - Create HTML profiling reports from pandas DataFrame objects. ![alt text][pd] 
* [statsmodels](https://github.com/statsmodels/statsmodels) - Statistical modeling and econometrics in Python
* [stockstats](https://github.com/jealous/stockstats) - Supply a wrapper ``StockDataFrame`` based on the ``pandas.DataFrame`` with inline stock statistics/indicators support.
* [simplestatistics](https://github.com/sheriferson/simplestatistics) - Simple statistical functions implemented in readable Python.
* [weightedcalcs](https://github.com/jsvine/weightedcalcs) - pandas-based utility to calculate weighted means, medians, distributions, standard deviations, and more.
* [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) - Pairwise Multiple Comparisons Post-hoc Tests.
* [pysie](https://github.com/chen0040/pysie) - Provides python implementation of statistical inference engine.

<a name="tools"></a>
## Experimentation
* [Sacred](https://github.com/IDSIA/sacred) - A tool to help you configure, organize, log and reproduce experiments by [IDSIA](http://www.idsia.ch/).
* [Xcessiv](https://github.com/reiinakano/xcessiv) - A web-based application for quick, scalable, and automated hyperparameter tuning and stacked ensembling.
* [Persimmon](https://github.com/AlvarBer/Persimmon) - A visual dataflow programming language for sklearn.
* [Ax](https://github.com/facebook/Ax) - Adaptive Experimentation Platform. ![alt text][skl]

<a name="eval"></a>
## Evaluation
* [recmetrics](https://github.com/statisticianinstilettos/recmetrics) - Library of useful metrics and plots for evaluating recommender systems.
* [kaggle-metrics](https://github.com/krzjoa/kaggle-metrics) - Metrics for Kaggle competitions.
* [Metrics](https://github.com/benhamner/Metrics) - Machine learning evaluation metric.
* [sklearn-evaluation](https://github.com/edublancas/sklearn-evaluation) - Scikit-learn model evaluation made easy: plots, tables and markdown reports.
* [AI Fairness 360](https://github.com/IBM/AIF360) - Fairness metrics for datasets and ML models, explanations and algorithms to mitigate bias in datasets and models.

<a name="compt"></a>
## Computations
* [numpy](http://www.numpy.org/) - The fundamental package needed for scientific computing with Python.
* [Dask](https://github.com/dask/dask) - Parallel computing with task scheduling. ![alt text][pd]
* [bottleneck](https://github.com/kwgoodman/bottleneck) - Fast NumPy array functions written in C.
* [minpy](https://github.com/dmlc/minpy) - NumPy interface with mixed backend execution.
* [CuPy](https://github.com/cupy/cupy) - NumPy-like API accelerated with CUDA.
* [scikit-tensor](https://github.com/mnick/scikit-tensor) - Python library for multilinear algebra and tensor factorizations.
* [numdifftools](https://github.com/pbrod/numdifftools) - Solve automatic numerical differentiation problems in one or more variables.
* [quaternion](https://github.com/moble/quaternion) - Add built-in support for quaternions to numpy.
* [adaptive](https://github.com/python-adaptive/adaptive) - Tools for adaptive and parallel samping of mathematical functions.

<a name="spatial"></a>
## Spatial Analysis
* [GeoPandas](https://github.com/geopandas/geopandas) - Python tools for geographic data. ![alt text][pd]
* [PySal](https://github.com/pysal/pysal) - Python Spatial Analysis Library.

<a name="quant"></a>
## Quantum Computing
* [QML](https://github.com/qmlcode/qml) - A Python Toolkit for Quantum Machine Learning.

<a name="conv"></a>
## Conversion
* [sklearn-porter](https://github.com/nok/sklearn-porter) - Transpile trained scikit-learn estimators to C, Java, JavaScript and others.
* [ONNX](https://github.com/onnx/onnx) - Open Neural Network Exchange.
* [MMdnn](https://github.com/Microsoft/MMdnn) -  A set of tools to help users inter-operate among different deep learning frameworks.

## Contributing
Contributions are welcome! :sunglasses: 
Read the <a href=https://github.com/krzjoa/awesome-python-datascience/blob/master/CONTRIBUTING.md>contribution guideline</a>.

## License
This work is licensed under the Creative Commons Attribution 4.0 International License - [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

<div align="center">
	<a href="other/deprecated.md">Deprecated Libs</a>&nbsp;&nbsp;&nbsp;
	<a href="other/waiting-room.md">Waiting Room</a>&nbsp;&nbsp;&nbsp;
<div>
