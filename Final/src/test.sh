#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:41:30 2017

@author: cengbowei
"""

#!/bin/bash 
wget 'https://www.dropbox.com/s/rralut7q0m7q5vy/pre_4900.h5?dl=1' -O'pre_4900.h5'
wget 'https://www.dropbox.com/s/aaq7ol5a9ul08zu/pre_bes.h5?dl=1' -O'pre_bes.h5'
wget 'https://www.dropbox.com/s/obupi455th9ry2y/donknow.h5?dl=1' -O'donknow.h5'
wget 'https://www.dropbox.com/s/opyr5gbzkiniw83/563.h5?dl=1' -O'563.h5'
wget 'https://www.dropbox.com/s/w8lrlisbkyvsd99/pre_0.0090.h5?dl=1' -O'pre_0.0090.h5'
wget 'https://www.dropbox.com/s/wi80o3t8foeq4xg/pre_541.h5?dl=1' -O'pre_541.h5'
wget 'https://www.dropbox.com/s/ksxnwtl40qketm1/pre_527.h5?dl=1' -O'pre_527.h5'
wget 'https://www.dropbox.com/s/opyr5gbzkiniw83/563.h5?dl=1' -O'563.h5'

python test.py $1 $2 $3