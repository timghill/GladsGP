# Gaussian Process emulation of GlaDS subglacial drainage model

Tim Hill, 2024 (tim_hill_2@sfu.ca)
https://github.com/timghill/GladsGP

Code corresponding to "Computationally efficient subglacial drainage modeling using Gaussian Process emulators".

This project emulates Glacier Drainage System (GlaDS) model ([Werder et al., 2013](https://doi.org/10.1002/jgrf.20146)) outputs.

## Description

The project structure is:

 * `src/`: shared code for setting up experiments and analyzing outputs
 * `experiments/`: individual directories for model experiments
 * `examples/`: example notebooks for GP emulation of univariate and multivariate simulation outputs

Each directory has a README file to describe the contents.

## Installation

The analysis source code has been tested against python 3.11. Package requirements are listed in `requirements.txt`, and it is recommended to use a virtual environment to manage versions. For example

```
virtualenv --python 3.11 pyenv/
source pyenv/bin/activate
pip install -r requirements.txt
```

To install the code for this project on your python path, install in editable (`-e`) mode with pip:

```
pip install -e .
```

This code also depends on a fork of the SEPIA package ([timghill/SEPIA](https://github.com/timghill/SEPIA)) that that can be installed using `pip install -e .`, and simulations are run with the Ice-sheet and Sea-level System Model ([ISSM](https://github.com/ISSMteam/ISSM)).

Your python environment and installation can be verified by running `test_install.sh`. This script should run with no errors and should update several figures in `experiments/synthetic/analysis/figures/`.
