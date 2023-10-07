from distutils.core import setup
from os import path
this_directory = path.abspath(path.dirname(__file__))

setup(
      name='ADSWildfireDixie',
      version='1.0',
      description='Wildfire Prediction',
      author='ADS Wildfire',
      url='https://github.com/ese-msc-2022/',
      packages=['dixie']
      )