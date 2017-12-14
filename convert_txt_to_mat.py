#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import re
import os

import numpy as np
import scipy.io

def main(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    files.sort()

    for input_filename in files:
        output_filename = re.sub(r"\.txt$", r".mat", input_filename)
        print( "converting %s --> %s" % (input_filename, output_filename) )
        a = np.genfromtxt(input_filename, delimiter="\t")
        data = { "A" : a }
        scipy.io.savemat(output_filename, data, format='5', oned_as='row')

if __name__ == '__main__':
    main(directory=".")

