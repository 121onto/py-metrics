"""A setuptools based setup module.
"""
import os

from setuptools import setup, find_packages
from setuptools.command.develop import develop

__version__ = 'v0.0.1'
__appname__ = 'py_metrics'

class Cache(develop):
    # Add custom build step to setup.py
    def run(self):
        develop.run(self)
        import appdirs # Import here to ensure install_requires runs
        vsrc = appdirs.user_cache_dir(
            appname=self.distribution.metadata.name,
            version=__version__)
        src = appdirs.user_cache_dir(
            appname=self.distribution.metadata.name)
        dst = os.path.join(os.path.dirname(__file__), 'cache')
        if not os.path.exists(vsrc):
            os.makedirs(vsrc)
        if not os.path.exists(dst):
            os.symlink(src, dst)


setup(
    name=__appname__,
    version=__version__,
    python_requires='>=3.6',
    description='Use python for econometrics',
    author='121onto',
    author_email='121onto@gmail.com',

    cmdclass={'develop': Cache},
    packages=find_packages(exclude=['output', 'tests', 'examples']),
    include_package_data=True,

    install_requires=[
        'scipy',
        'numpy',
        'pandas',
        'appdirs',
    ],

    extras_require={
        'dev': [
            'ipython',
            'sklearn',
            'matplotlib',
        ],
    },

    setup_requires=[
        'appdirs',
    ]
)
