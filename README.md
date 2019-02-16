PyMetrics
=========

Use python for econometrics.  The API generally follows Stata, is built around
the Pandas project, and supports modern econometric best practices.

Most of the decisions regarding output and functionality come from Bruce Hansen's
Econometrics textbook ([link](https://www.ssc.wisc.edu/~bhansen/econometrics/)).
I primarily used his January 2019 version during the development of this package.

Getting Started
---------------

For more detailed examples, see the examples directory.

1. Run a regression:

    ``` python
    from __future__ import print_function
    from __future__ import absolute_import
    from __future__ import division

    import pandas as pd
    import numpy as np

    from py_metrics import caches
    from py_metrics import reg

    # Load data
    frame = pd.read_csv(caches.data_path('cps09mar.txt'))
    frame['wage'] = frame['earnings'] / (frame['week'] * frame['hours'])
    frame['log(wage)'] = np.log(frame['wage'])
    frame['experience'] = np.maximum(frame['age'] - frame['education'] - 6, 0)

    # Setup regression
    x = ['intercept', 'female', 'experience']
    y = 'log(wage)'
    reg = Reg(x, y)

    # Fit regression
    reg.fit(frame)
    reg.print()
    ```

2. Estimate a covariance matrix: ...


Developer Installation
----------------------

Note: this repo is compatible with python 3.6 only.

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
