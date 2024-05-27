import os, subprocess
from setuptools.command.build_ext import build_ext
from distutils.errors import DistutilsSetupError
from distutils import log as distutils_logger


from setuptools import setup, Extension

import numpy as np
setup(
   name='egotaxis',
   version='0.1.0',
   author='Julian Rode, based on earlier version by Andrea Auconi',
   author_email='benjamin.m.friedrich@tu-dresden.de', # email of corresponding author of accompanying publication
   ext_modules = [
    Extension('pyegotaxis._cpredict', 
        sources = ['cpp/cnpy.cpp','cpp/cpredict.cpp','cpp/cpredict.i'],
        swig_opts = ['-c++'],
        language ="c++" ,
        include_dirs=[np.get_include()],
        extra_compile_args =["-std=c++17",'-O3','-march=native'],
        libraries=['z' ] ),
    Extension('pyegotaxis._cfunctions', 
        sources = ['cpp/cfunctions.cpp','cpp/cfunctions.i'],
        swig_opts = ['-c++'],
        language ="c++" ,
        include_dirs=[np.get_include()],
        extra_compile_args =["-std=c++17",'-O3','-march=native'] ),
    Extension('pyegotaxis._cfunctionsExp', 
        sources = ['cpp/cfunctionsExp.cpp','cpp/cfunctionsExp.i'],
        swig_opts = ['-c++'],
        language ="c++" ,
        include_dirs=[np.get_include()],
        extra_compile_args =["-std=c++17",'-O3','-march=native'] )],
   packages=['pyegotaxis'],
   scripts=[],
   url='',
   license='LICENSE.txt',
   description='Python code to simulate chemotactic agents performing infotaxis',
   long_description=open('README.md').read(),
   install_requires=[
       "numpy","pandas","scipy"
   ],
)
