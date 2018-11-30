# TensorFlow Installation Guide:

This guide will instruct you on how to install TensorFlow on your machine using 
Anaconda.  The install done here is a basic install for the purposes of the 
tutorials in this repo.  This install will not include TensorFlow’s GPU 
compatibilities.  For more information, please see the TensorFlow installation 
page on their [website](https://www.tensorflow.org/install/).

## Step 1: Install Anaconda

The first step is to install Anaconda.  Anaconda is a python platform that works
on Windows, Linux, and Mac.  It includes a python build and a wide assortment of
python packages and tools.  Anaconda manages packages and makes sure the
versions you have installed are all compatible with each other.  It also allows
you to create virtual environments that are good for separating your libraries
and preventing them from interfering with each other, allowing different
versions to be installed on your machine next to each other for different
purposes.  The installer for Windows and Mac can be downloaded from their
[website](https://www.anaconda.com/download/), if you are using Linux, a quick
google search can tell you what the preferred method of installation is for your
favored distribution.  The python3 version is recommended as that is what we
will be using in these tutorials, however it does not really matter as we will
be setting up our own virtual environment.

## Step 2: Setup New Anaconda Environment

In this step, you will set up a new environment that we will install TensorFlow
into.  To setup the environment, open the anaconda prompt and run the following
command (in Linux and Mac, this can also just be run in the terminal):

```
(base)$ conda -n tf pip conda_nb python=3.6
(base)$ source activate tf
(tf)$ conda install -c conda-forge matplotlib tqdm
```

This command will create a new environment called “tf” that has pip and all the
Jpyter Notebook packages installed for python version 3.6.  If you are familiar
with Anaconda and wish to include other packages, feel free to do so.  The next
line switches the current environment from “base” to the new “tf” environment we
just created.  The last line installs matplotlib, a library of MatLab like
plotting tools, and TQDM, a progress bar tool that is nice to have.

## Step 3: Install TensorFlow

It is now time!  We will use the recommended method of installing TensorFlow,
that is through pip rather than conda.  Make sure you are in the “tf”
environment and type the following comman:

```
(tf)$ pip install --ignore-installed --upgrade tensorflow
```

This command will install TensorFlow in the environment we created in the
previous step.  After this finishes, you can proceed to open the notebooks in
this tutorial set.  Make sure you open them with the Jupyter version that is
installed in the same environment as TensorFlow (in windows, it will have (tf)
in front of it in the start menu, you can also launch it from the CLI or form
the anaconda navigator in the correct environment.
