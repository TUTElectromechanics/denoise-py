#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import re
import operator

def listfiles(path, verbose=True):
    """List measurement data files.

    Can be used to get valid sigma values for scrub().

    Parameters:
        path: str
            where to look for the data files

    Returns:
        list of tuples (sigma, filename), where:
            sigma: int
                stress level (MPa)
            filename: str
                filename of data corresponding to stress level sigma
"""
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    just_files = [ x for x in os.listdir(path) if os.path.isfile( os.path.join(path, x) ) ]

    matching_files = [ x for x in just_files if x.endswith("MPa.mat") ]
    n_data_files = len(matching_files)

    if n_data_files < 1:
        print( "No data files found, exiting", file=sys.stderr )
        sys.exit(1)

    data_items = []
    for filename in matching_files:
        tmp = re.findall(r"\d+", filename)
        sigma = int(tmp[0])
        data_items.append( (sigma, filename) )

    data_items.sort( key=operator.itemgetter(0) )

    if verbose:
        print( "%d measurement data files detected:" % (n_data_files) )
        for sigma, filename in data_items:
            print( "    sigma = %d MPa, filename = '%s'" % (sigma, filename) )

    return data_items

