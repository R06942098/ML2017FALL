#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:12:08 2017

@author: cengbowei
"""


#!/bin/bash 
wget 'https://www.dropbox.com/s/6k7dreg96nd6qad/82556.h5?dl=1' -O'model.h5'
wget 'https://www.dropbox.com/s/utp9ziqajqap2g9/tokenizer.pickle?dl=1' -O'tokenizer.pickle'
python hw4_test.py $1 $2