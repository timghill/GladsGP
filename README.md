# GP emulation of GlaDS subglacial drainage model

## Installation

The analysis source code has been tested against python 3.11. Package requirements are listed in `requirements.txt`, and it is recommended to use a virtual environment to manage versions. For example

```
virtualenv --python 3.12 pyenv/
source pyenv/bin/activate
pip install -r requirements.txt
```

Note that the specified versions explicitly do not work with python 3.12, and versions <3.11 have not been tested.

This code also depends on [timghill/GPmodule](https://github.com/timghill/GPmodule) and [timghill/SEPIA](https://github.com/timghill/SEPIA) packages that can be installed using `pip install -e GPmodule`.
