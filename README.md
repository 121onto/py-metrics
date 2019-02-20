PyMetrics
=========

Use python for econometrics.  The API generally follows Stata, is built around
the Pandas project, and supports modern econometric best practices.

Most of the decisions regarding output and functionality come from Bruce Hansen's
Econometrics textbook ([link](https://www.ssc.wisc.edu/~bhansen/econometrics/)).
I primarily used his January 2019 version during the development of this package.

Installation
------------

Note: tested with Python 3.6 only.

1. Install from source

    ```bash
    pip install git+git://github.com/121onto/py-metrics.git
    ```

TODO (121onto): release `v0.0.1` and update these instructions to install from
  that release.


Documentation
-------------

See the project wiki for official docs [link](https://github.com/121onto/py-metrics/wiki).
Additional examples are available in the examples directory.


Developer Installation
----------------------

Note: tested using Python 3.6 only.

1. Clone this repository:

    ```bash
    git clone git@github.com:121onto/py-metrics.git
    cd py-metrics
    ```

2. Create a new virtualenv (assumes you're using virtualenvwrapper):

    ``` bash
    mkv py-metrics --python=python3
    actv
    ```

3. Install the repository in editable mode:

    ```bash
    pip install -e .
    ```
