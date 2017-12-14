#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Denoise measured data curves.

JJ 2017-08-07"""

import re

import numpy as np

import scipy.io

import util
import filter_H
import filter_B
import filter_pol
import filter_magnetostriction


def main(path):
    # get data files in specified directory
    #
    data_items = util.listfiles(path, verbose=False)

    for sigma,input_filename in data_items:
        output_filename = re.sub( r'\.mat$', r'_denoised.mat', input_filename )

        print( "Scrubbing '%s' --> '%s'..." % (input_filename, output_filename) )

        H   = filter_H.scrub(sigma, path)
        B   = filter_B.scrub(sigma, path)
        pol = filter_pol.scrub(sigma, path)
        lam = filter_magnetostriction.scrub(sigma, path)

        assert H.shape == B.shape == pol.shape == lam.shape

        A = np.empty( (H.shape[0],4), dtype=np.float64 )
        A[:,0] = H[:]
        A[:,1] = B[:]
        A[:,2] = pol[:]
        A[:,3] = lam[:]

        scipy.io.savemat( output_filename, mdict={ 'A' : A } )

    print( "All done." )

if __name__ == '__main__':
    main( path="." )

