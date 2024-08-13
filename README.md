# Gaussian Process emulation of GlaDS subglacial drainage model

Tim Hill, 2024 (tim_hill_2@sfu.ca)

This project emulates Glacier Drainage System (GlaDS) model ([Werder et al., 2013](https://doi.org/10.1002/jgrf.20146)) outputs.

## Description

The project structure is:

 * `src/`: shared code for setting up experiments and analyzing outputs
 * `experiments/`: individual directories for each model experiment
 * `examples/`: example notebooks for GP emulation of univariate and multivariate simulation outputs

Each directory has a README file to describe the contents.

## Installation

The analysis source code has been tested against python 3.11 (the specified versions explicitly do not work with python 3.12). Package requirements are listed in `requirements.txt`, and it is recommended to use a virtual environment to manage versions. For example

```
virtualenv --python 3.11 pyenv/
source pyenv/bin/activate
pip install -r requirements.txt
```

To install the code for this project on your python path, navigate into the `src` directory and install in editable (`-e`) mode with pip:

```
cd src
pip install -e .
```

This code also depends on a fork of the SEPIA package ([timghill/SEPIA](https://github.com/timghill/SEPIA)) that that can be installed using `pip install -e .`.
