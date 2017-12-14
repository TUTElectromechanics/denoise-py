#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Denoise measured flux density (B) curves.

JJ 2017-08-07"""

import sys
import os

import numpy as np

import skimage.restoration  # sudo pip3 install scikit-image

import scipy.signal
import scipy.io

import util

__version__ = '0.1.1'


def scrub(sigma, path, show=False, verbose=False):
    """Denoise a flux density (B) curve.

    The data is assumed to be periodic (to wrap around at the end).

    Parameters:
        sigma: int
            stress level in MPa; chooses the data file to be loaded.

        path: str
            directory in which to look for measurement data files.

        show: bool
            if True, results from various stages of processing will be plotted and shown (requires Matplotlib).

        verbose: bool
            if True, progress messages will be printed to stdout.

    Returns:
        rank-1 np.array, denoised signal.
"""
    datafile_basename = "%dMPa.mat" % (sigma)
    filename = os.path.join(path, datafile_basename)

    if verbose:
        print( "Loading and processing file '%s'" % (filename) )

    try:
        data = scipy.io.loadmat(filename)
    except FileNotFoundError:
        print( "Data file named '%s' not found, exiting (use --list to see available data files)" % (filename), file=sys.stderr )
        sys.exit(1)

    A    = data["A"]

    H    = A[:,0]  # Field strength H (A/m)
    B    = A[:,1]  # Flux density B (T)
    pol  = A[:,2]  # magnetic polarization J = B - mu0*H
    lam  = A[:,3]  # Magnetostriction lambda (ppm)

    # choose signal to filter
    yy_raw  = B

    if verbose:
        print( "    loaded signal with %d samples" % (yy_raw.shape[0]) )

    # -------------------------------------------------------------------
    # periodize
    # -------------------------------------------------------------------

    # the data is known to be periodic - copy half a period to both edges to make the fitters/detectors see the periodicity
    #
    # (it doesn't matter what the processing does in the padding region, as the padding is discarded at the end;
    #  we just need enough surrounding data to e.g. see transients located near the wrap-around)
    #
    datalen = yy_raw.shape[0]
    padding = datalen // 2
    yy_padded  = np.empty( (2*datalen,), dtype=np.float64 )
    yy_padded[:padding]                  = yy_raw[-padding:]
    yy_padded[padding:(padding+datalen)] = yy_raw
    yy_padded[(padding+datalen):]        = yy_raw[:(datalen-padding)]
    yy_raw = yy_padded

    # -------------------------------------------------------------------
    # filtering
    # -------------------------------------------------------------------

    # start with the raw signal
    #
    yy_filt = yy_raw

    yy_filt = skimage.restoration.denoise_tv_chambolle(yy_filt, weight=100)
    yy_tvdenoise_only = yy_filt

    # postprocess by a lowpass filter to smooth out some of the wiggling without damaging the data too much
    #
    b,a     = scipy.signal.butter(4, 0.10)  # Butterworth filter;  order, cutoff [fraction of Nyquist]
    yy_filt = scipy.signal.filtfilt(b, a, yy_filt)
    yy_tv_and_lp = yy_filt

    yy_final = yy_filt

    # -------------------------------------------------------------------
    # de-periodize
    # -------------------------------------------------------------------

    # remove periodicity-enforcing padding
    #
    yy_raw            = yy_raw[padding:(padding+datalen)]             # measured signal
    yy_tvdenoise_only = yy_tvdenoise_only[padding:(padding+datalen)]  # after TV denoising
    yy_tv_and_lp      = yy_tv_and_lp[padding:(padding+datalen)]       # after LP filtering
    yy_final          = yy_final[padding:(padding+datalen)]           # final processed signal

    # -------------------------------------------------------------------
    # plot results
    # -------------------------------------------------------------------

    if show:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        plt.plot(yy_raw,  color='#a0a0a0', linestyle='solid')
        plt.plot(yy_tvdenoise_only, color='#606060', linestyle='solid')   # TV denoise only
        plt.plot(yy_tv_and_lp, color='k', linestyle='solid')   # TV denoise + lowpass filter

        # spectrogram

        titles = (r'raw', r'$\rightarrow$ TV denoising $\rightarrow$', r'$\rightarrow$ Butterworth LP')
        n_psd = len(titles)
        psds = []

        nps = 128
        nov = 96
        for i,y in enumerate( (yy_raw, yy_tvdenoise_only, yy_tv_and_lp) ):
            sample_f = (y.shape[0] - 1)  # for convenience, we define the data span in "time" as  t = [0,1]
            f, t, Sxx = scipy.signal.spectrogram(y/np.max(y), fs=sample_f, window=('chebwin', 100), nperseg=nps, noverlap=nov)
            Sxx = np.log10(Sxx)
            f = 2.*f/sample_f  # convert raw frequency to fraction of Nyquist
            psds.append( (f, t, Sxx) )

        # find min/max of log10(S), to use the same color scale in each plot
        # 
        vmin = min( [np.min(item[2]) for item in psds] )
        vmax = max( [np.max(item[2]) for item in psds] )

        # plot...
        #
        plt.figure(2, figsize=(13.5,9))

        # ...the same data twice:
        #
        # 1) using per-plot individual color scales (to maximize detail visibility on the color axis)
        #
        for i,psd in enumerate(psds):
            f,t,Sxx = psd
            plt.subplot(2,n_psd, i+1)
            plt.pcolormesh(t, f, Sxx, shading='gouraud')
            if i == 0:
                plt.ylabel('f [Nyquist]')
            mytitle = "%s%s" % ( ("log10(PSD) " if i == 0 else ""), titles[i] )
            plt.title(mytitle)
            plt.colorbar()

        # 2) using a single color scale locked to global min/max across subplots (to facilitate comparison)
        #
        for i,psd in enumerate(psds):
            f,t,Sxx = psd
            plt.subplot(2,n_psd, n_psd+(i+1))
            plt.pcolormesh(t, f, Sxx, vmin=vmin, vmax=vmax, shading='gouraud')
            if i == 0:
                plt.ylabel('f [Nyquist]')
            plt.xlabel('t [data length]')
            plt.colorbar()

        # annotate
        #
        # https://matplotlib.org/examples/pylab_examples/text_rotation.html
        align  = {'ha': 'center', 'va': 'center'}
        rotate = {'rotation' : 90}
        props1 = dict(align)
        props1.update(rotate)
        props2 = dict(align)
        plt.figtext( 0.02, 0.71, "individual color scales", **props1 )
        plt.figtext( 0.02, 0.29, "common color scale", **props1 )
        plt.figtext( 0.50, 0.98, r"$\rightarrow$ $\rightarrow$ $\rightarrow$ direction of processing $\rightarrow$ $\rightarrow$ $\rightarrow$", **props2 )

        plt.show()

    return yy_final


def main():
    import argparse
    parser = argparse.ArgumentParser(description="""Denoise measured flux density (B) curves.""", formatter_class=argparse.RawDescriptionHelpFormatter)

    # -------------------------------------------------
    # ungrouped meta options

    parser.add_argument( '-v', '--version', action='version', version=('%(prog)s ' + __version__) )

    parser.add_argument( '-l', '--list', dest='listfiles', action='store_true',
                         help='List available stress levels (measurement data files) and exit.' )
    parser.set_defaults(listfiles=False)

    # -------------------------------------------------
    # data options

    group_data = parser.add_argument_group('data', 'Data file options.')

    group_data.add_argument( '-s', '--sigma',
                             dest='sigma',
                             default=0,
                             type=int,
                             metavar='x',
                             help='Stress level, selects the corresponding data file. (This expects only the number in MPa, leaving out the unit.) (default: %(default)s).' )

    group_data.add_argument( '-p', '--path',
                             dest='path',
                             default=".",
                             type=str,
                             metavar='my/directory/path',
                             help='Path where to look for measurement data files (default: current working directory).' )

    # -------------------------------------------------

    # http://parezcoydigo.wordpress.com/2012/08/04/from-argparse-to-dictionary-in-python-2-7/
    kwargs = vars( parser.parse_args() )

    if kwargs["listfiles"]:
        util.listfiles(kwargs["path"])
        sys.exit(0)
    else:
        lam = scrub(kwargs["sigma"], kwargs["path"], show=True, verbose=True)


if __name__ == '__main__':
    main()

