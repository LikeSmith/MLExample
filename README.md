# MLExample
Example code and tutorials for Machine Learning applications

This repo is organized based on what libraries the tutorials are written to 
support.  Currently we are only writing tutorials for tensorflow, but we plan to
add pyTorch tutorials in the future.  Each of the following tutorials will be
implimented in all the libraries supported for this tutorial.  It is recommended
to go through the tutorials in the order they appear here.  We recommend going
through all the tutorials for one librarie before moving on to the others.

Each tutorial conatins a Jupyter Notebook that goes through the code and
explains all the steps.  The notebook also has a list of suggested exercises for
expanding and embedding the ideas of each tutorial.  In addition to the
notebook, each tutorial has a python script file associated with it that
contains all the code for the tutorial.  It is recommended to make modifications
to this file as modifying the Jupyter notebook may be tedious and lead to new
and interesting errors if you do not restart the kernel.

## Libraries
### Tensorflow
Tensorflow is an open source ``high performance numerical computation'' library 
originally developed at Google Brain.  It is highly taylored for Machine
Learning and other data science applications.  Tensorflow works by specifying a
compute graph that is then optimized.  The graph can then be run to take
advantage of different hardware that is available such as multiple CPUs, and
Cuda enabled GPUs. Tensorflow is a very powerfull tool and is currently the
industry standard.  You can learn more about it at their
[website](https://www.tensorflow.org/).  [Here](https://github.com/LikeSmith/MLExample/blob/master/TensorFlowExamples/TensorFlowInstallationGuide.md)
is the installation guide for tensorflow for the purposes of these tutorials.

### PyTorch
PyTorch is another open source library designed to allow easy implimentation of 
Neural Networks and other Deep Learning Tools.  It a python implimentation of
the Torch library in Lua, and is primarially developed by Facebook. The key way
in which PyTorch differs from Tensorflow is that it allows for dynamic compute
graphs.  Where Tensorflow requires the compute graph to be fully defined before
any actual computation takes place, PyTorch allows the compute graph to be
defined and changed more dynamically.  While not quite as pervasive in the
research literature as Tensorflow, PyTorch may offer some slightly speedier
ad-hoc implimentations that may be worth considering for prototyping and other
uses.  Learn more about PyTorch at their [website](https://pytorch.org/)

## Tutorials
### Installation
Before doing anything else, make sure you have all the relevant libraries
installed for these tutorials.  See in install directions for your preferred
library.  In addition to either Tensorflow or PyTorch, you will also need:
- MatPlotLib
- Jupyter
- ipywidgets
- TQDM

We also recommend an installation of Spyder for modifying python script files
directly.  Note that spyder is not installed as part of the installation guids.
To install it, execute the following command in the Anaconda Prompt:
```
(env)$ conda install spyder
```

### Optimization By Stochastic Gradient Descent
This tutorial introduces Optimization and Stochastic Gradient Descent, the
numerical method most commonly used to train neural networks.  This tutorial
goes through an example case fitting a curve to a nonlinear dataset.

### MNIST Integer Classification
This tutorial will build a simple MultiLayered Perceptron to classify a set of
images of handwritten numerals.  This is a classic Machine Learning Problem and
an interesting demonstration.

## Authors
The following contribute to these tutorials, feel free to email them with
questions

- Kyle Crandall (crandallk@gwu.edu)
