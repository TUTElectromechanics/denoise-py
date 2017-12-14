#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Denoise measured magnetostriction (lambda) curves.

JJ 2017-08-07"""

# TODO:
# - improvement: maybe better for avoiding wiggling at valley edges:
#   - look at where the curvature changes sign, pick that if closer to the valley than a local maximum

import sys
import os

import numpy as np

import skimage.restoration  # sudo pip3 install scikit-image

import scipy.signal
import scipy.interpolate
import scipy.io

import util

__version__ = '0.1.1'


def scrub(sigma, path, show=False, verbose=False):
    """Denoise a measured magnetostriction (lambda) curve by signal filtering and model-based reconstruction.

    The data is assumed to be periodic (to wrap around at the end).

    For the reconstruction, this function assumes that the curve consists of deep localized valleys
    surrounded by nearly-flat saturated regions.

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
    yy_raw  = lam

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

#    # scale data to [0,1] (TEST/DEBUG)
#    m1 = np.min(yy_raw)
#    m2 = np.max(yy_raw)
#    yy_raw = (yy_raw - m1) / (m2 - m1)

    # -------------------------------------------------------------------
    # filtering
    # -------------------------------------------------------------------

    # start with the raw signal
    #
    yy_filt = yy_raw

    # now we operate only on yy_filt, preserving yy_raw for comparison
    #
    #  - yy_filt always refers to the output of the previous stage (latest processed signal)
    #  - each processing stage follows the "contract" that the contents of yy_filt must not be modified in-place (without making a copy first)
    #  - snapshots from various stages will be preserved, to show comparison at the end

    # preliminary processing - denoise and filter the raw signal

    # Total Variation denoising, for an example see
    #
    #     http://www.scipy-lectures.org/advanced/image_processing/auto_examples/plot_lena_tv_denoise.html
    #
    # and, from "help skimage.restoration.denoise_tv_chambolle":
    #
    #    The principle of total variation denoising is explained in
    #    http://en.wikipedia.org/wiki/Total_variation_denoising
    #
    #    The principle of total variation denoising is to minimize the
    #    total variation of the image, which can be roughly described as
    #    the integral of the norm of the image gradient. Total variation
    #    denoising tends to produce "cartoon-like" images, that is,
    #    piecewise-constant images.
    #
    #    This code is an implementation of the algorithm of Rudin, Fatemi and Osher
    #    that was proposed by Chambolle in [1]_.
    #
    #    References
    #    ----------
    #    .. [1] A. Chambolle, An algorithm for total variation minimization and
    #           applications, Journal of Mathematical Imaging and Vision,
    #           Springer, 2004, 20, 89-97.
    #
    yy_filt = skimage.restoration.denoise_tv_chambolle(yy_filt, weight=100)
    yy_tvdenoise_only = yy_filt

    # postprocess by a lowpass filter to smooth out some of the wiggling without damaging the data too much
    #
    b,a     = scipy.signal.butter(4, 0.1)  # Butterworth filter;  order, cutoff [fraction of Nyquist]
    yy_filt = scipy.signal.filtfilt(b, a, yy_filt)
    yy_tv_and_lp = yy_filt

    # -------------------------------------------------------------------
    # transient detection
    # -------------------------------------------------------------------

    # we use this to locate the valleys

    # use a spectrogram with high time resolution (low frequency resolution, we don't need it as the transients span the whole frequency axis)
    nps = 8
    nov = 0  # here we don't need to overlap the segments
    y = yy_filt
    sample_f = (y.shape[0] - 1)  # for convenience, we define the data span in "time" as  t = [0,1]
    f, t, Sxx = scipy.signal.spectrogram(y/np.max(y), fs=sample_f, window=('chebwin', 100), nperseg=nps, noverlap=nov)
    Sxx = np.log10(Sxx)
    f = 2.*f/sample_f  # convert raw frequency to fraction of Nyquist

    # denoise the spectrogram
    Sxx = skimage.restoration.denoise_tv_chambolle(Sxx, weight=100)

    # scale the result to [0,1]
    m1 = np.min(Sxx)
    m2 = np.max(Sxx)
    Sxx = (Sxx - m1) / (m2 - m1)

    # pick one row (i.e. one frequency) for use in detection
    row = 0

    # At the chosen frequency, find local maxima in the denoised power spectral density, with relative data value > 0.5
    # This should give the locations of the transients in the spectrogram data.
    #
    m = np.squeeze(scipy.signal.argrelextrema(Sxx[row,:], np.greater))
    m = m[Sxx[row,m] > 0.5]

    # convert to index of original sampled data
    #
    sample_idx = nps*m  # no overlap (nov=0), so we can just multiply by nperseg to get the sample index in the original data

#    # DEBUG
#    plt.figure(3)
#    plt.clf()
#    plt.plot(Sxx[row,:])
#    plt.plot(m,Sxx[row,m],'ko')
#    print(sample_idx)

    # -------------------------------------------------------------------
    # reconstruction, valleys
    # -------------------------------------------------------------------

    # smooth the curve inside the valley regions (to eliminate spurious wiggles there)
    #
    #   - the valleys look transcendental, so we need to be selective about applying a polynomial fit

    # we will next modify the data in-place, hence copy first
    yy_filt = yy_filt.copy()

    # detect valley regions
    #
    # - we use the result of the transient detector
    # - we find "small" data values between pairs of transients
    #
    m1 = np.min(yy_filt)
    m2 = np.max(yy_filt)
    yy_rel = (yy_filt - m1) / (m2 - m1)
    data_small_mask = (yy_rel < 0.04)  # <-- manually tuned threshold, using the measured data for galfenol
    data_small_idx  = np.nonzero(data_small_mask)[0]
    for start,end in zip(sample_idx[:-1], sample_idx[1:]):
        # not a valley if too large a fraction of data length (likely a region between valleys)
        if end - start > int(0.15*datalen):  # <-- manually tuned threshold, using the measured data for galfenol
            continue

        # find region to be filtered (data is "small" --> inside the valley)

        # indices to data_small_idx
        i1 = np.searchsorted(data_small_idx, start)
        i2 = np.searchsorted(data_small_idx, end)

        # corresponding indices to yy_filt
        k1 = data_small_idx[ max(0, i1) ]
        k2 = data_small_idx[ min(i2-1, data_small_idx.shape[0]-1) ]

        # limit any changes to the data to be applied only in the region between the detected pair of transients
        k1 = max(start, k1)
        k2 = min(k2, end)

        # first pass - replace the deepest region of the valley with a cubic polynomial approximation (PCHIP)
        #
        kmid = (k1+k2)//2
        ks_tuple = (k1-1, k1, k1+1, kmid, k2-1, k2, k2+1)
        ks_float = np.array( ks_tuple, dtype=np.float64 )
        ks_idx   = np.array( ks_tuple, dtype=int )
        f = scipy.interpolate.PchipInterpolator( ks_float, yy_filt[ks_idx] )
        r = np.arange(ks_float[0],ks_float[-1]+1,dtype=np.float64)
        yy_filt[ks_idx[0]:(ks_idx[-1]+1)] = f(r)

        # second pass - smooth the seams using a simple blur operator
        #
        center_ks = (k1, k2)  # smoothing will be applied around these points
        for center_k in center_ks:
            ks = np.arange(center_k-5, center_k+5+1, dtype=int)
            tmp = yy_filt[ks].copy()
            for i in range(10):  # blurring iterations; 10 seems good
                tmp[1:-1] = 0.25*tmp[:-2] + 0.5*tmp[1:-1] + 0.25*tmp[2:]
            yy_filt[ks] = tmp

#        print(k1,k2)  # DEBUG

    yy_valley_reco = yy_filt

    # -------------------------------------------------------------------
    # reconstruction, flat regions
    # -------------------------------------------------------------------

    # use a piecewise cubic fit (PCHIP) in the "flat" parts of the curve
    #
    # This is effectively equivalent to a lowpass filter at a very low cutoff frequency,
    # applied to only those parts of the curve where we know that any
    # transients are nonphysical (caused by measurement noise).

    # find the flat regions
    #
    idxs = find_important_points_pchip(yy_filt)

    # extract unique indices for plotting points used in the curve fitting
    plot_idxs = []
    for kk in idxs:
        plot_idxs.extend(kk)
    plot_idxs = np.unique(plot_idxs)

    # find approximate "average curve" in each region
    #
    # This disregards boundary conditions on purpose. The idea is to remove the wiggle
    # at the ends of the measured data, and near the center of the flat regions between the valleys.
    #
    results_spline = []
    for kk in idxs:
        # from the filtered curve, take all data in the specified interval
        xx = np.arange( kk[0], kk[-1]+1, dtype=np.float64 )
        xx_idx = np.array(xx, dtype=int)

        # fit a spline (with no internal knots) to find the "average curve"
        f = scipy.interpolate.LSQUnivariateSpline( xx, yy_filt[xx_idx], t=[], k=3 )

        xx_spline = np.linspace( kk[0], kk[-1], 10001 )
        yy_spline = f(xx_spline)
        results_spline.append( (f, xx_spline, yy_spline) )

    # fit the interpolating polynomial
    #
    results_pchip = []

    # region before the first valley
    #
    # at points where we do not need to follow the wiggling data exactly,
    # make the fit follow the average curve instead
    #
    kk         = idxs[0]
    spline_fit = results_spline[0]
    f_spline   = spline_fit[0]

    my_yy    = yy_filt[kk].copy()
    my_yy[0] = f_spline(float(kk[0]))  # start point
    my_yy[1] = f_spline(float(kk[1]))  # midpoint

    f = scipy.interpolate.PchipInterpolator(kk, my_yy)
    xx_pchip = np.linspace( kk[0], kk[-1], 10001 )
    yy_pchip = f(xx_pchip)
    results_pchip.append( (f, xx_pchip, yy_pchip) )

    # regions between the valleys
    #
    for kk,spline_fit in zip(idxs[1:-1], results_spline[1:-1]):
        f_spline = spline_fit[0]

        my_yy = yy_filt[kk].copy()
        my_yy[3] = f_spline(float(kk[3]))  # midpoint

        f = scipy.interpolate.PchipInterpolator(kk, my_yy)
        xx_pchip = np.linspace( kk[0], kk[-1], 10001 )
        yy_pchip = f(xx_pchip)
        results_pchip.append( (f, xx_pchip, yy_pchip) )

    # region after the last valley
    #
    kk         = idxs[-1]
    spline_fit = results_spline[-1]
    f_spline   = spline_fit[0]

    my_yy    = yy_filt[kk].copy()
    my_yy[-2] = f_spline(float(kk[-2]))  # midpoint
    my_yy[-1] = f_spline(float(kk[-1]))  # endpoint

    f = scipy.interpolate.PchipInterpolator(kk, my_yy)
    xx_pchip = np.linspace( kk[0], kk[-1], 10001 )
    yy_pchip = f(xx_pchip)
    results_pchip.append( (f, xx_pchip, yy_pchip) )

#    # original version (just one loop for all regions), no spline helper (inaccurate at ends and at the midpoints between valleys)
#    results_pchip = []
#    for kk in find_important_points_pchip(yy_filt):
#        f = scipy.interpolate.PchipInterpolator(kk, yy_filt[kk])
#        xx_pchip = np.linspace( kk[0], kk[-1], 10001 )
#        yy_pchip = f(xx_pchip)
#        results_pchip.append( (xx_pchip, yy_pchip) )

    # apply the reconstruction, updating the filtered signal
    #
    # - make a copy of the previous output
    # - replace the flat regions by their pchip fits (which have been constructed to coincide with a few data points at the edges of each valley)
    #
    yy_filt = yy_filt.copy()
    for kk,item in zip(idxs, results_pchip):
        xx = np.arange(kk[0], kk[-1]+1, dtype=np.float64)
        xx_idx = np.array(xx, dtype=int)
        f = item[0]
        yy_filt[xx_idx] = f( xx )

    # this was the last stage of processing, so now we have the final output signal
    #
    yy_final = yy_filt

    # -------------------------------------------------------------------
    # de-periodize
    # -------------------------------------------------------------------

    # remove periodicity-enforcing padding
    #
    yy_raw            = yy_raw[padding:(padding+datalen)]             # measured signal
    yy_tvdenoise_only = yy_tvdenoise_only[padding:(padding+datalen)]  # after TV denoising
    yy_tv_and_lp      = yy_tv_and_lp[padding:(padding+datalen)]       # after LP filtering, but no reconstruction yet
    yy_valley_reco    = yy_valley_reco[padding:(padding+datalen)]     # after reconstruction applied to valleys only
    yy_final          = yy_final[padding:(padding+datalen)]           # final, reconstruction applied also to flat regions

    plot_idxs[:] -= padding
    plot_idxs = plot_idxs[ (plot_idxs >= 0) * (plot_idxs < datalen) ]

    sample_idx[:] -= padding
    sample_idx = sample_idx[ (sample_idx >= 0) * (sample_idx < datalen) ]

    def depad(data):
        tmp = []
        for f,x,y in data:
            x[:] -= padding
            mask = (x >= 0) * (x < datalen)
            x = x[mask]
            y = y[mask]

            # fix the padding offset in the argument to f() by wrapping it with an adaptor
            def correct_x_offset(f, padding):
                return lambda x: f(x - padding)

            if len(x):
                tmp.append( (correct_x_offset(f,padding),x,y) )
        return tmp
    results_pchip  = depad(results_pchip)
    results_spline = depad(results_spline)

    # -------------------------------------------------------------------
    # plot results
    # -------------------------------------------------------------------

    if show:
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.clf()

        # detected transients, used in pairs (if applicable) in reconstruction of the valleys
        #
        for x in sample_idx:
            plt.axvline( x, color='#c0c0c0', linestyle='dashed' ) 

        # data at various stages of filtering
        #
        plt.plot(yy_raw,  color='#a0a0a0', linestyle='solid')
        plt.plot(yy_tvdenoise_only, color='#606060', linestyle='solid')   # TV denoise only
        plt.plot(yy_tv_and_lp, color='k', linestyle='dashed')   # TV denoise + lowpass filter
#        plt.plot(yy_valley_reco, color='k', linestyle='dashdot')       # TV denoise + lowpass filter + valley reconstruction (no point in drawing, gets drawn over by the final fit)
#        plt.plot(xx_spline, yy_spline, color='orange', linestyle='solid')

        # average curve helpers in the flat parts
        #
        for dummy,xx_spline,yy_spline in results_spline:
            plt.plot(xx_spline, yy_spline, color='blue', linestyle='dashed')

        # final piecewise polynomial fits in the flat parts
        #
        # also this is drawn over by the final fit, so make it bolder
        for dummy,xx_pchip,yy_pchip in results_pchip:
            plt.plot(xx_pchip, yy_pchip, color='orange', linestyle='dashed', linewidth=2.0)

        # final reconstructed signal
        #
        plt.plot(yy_final, color='orange', linestyle='solid')

        # points of interest, used in the reconstruction of the flat parts
        #
        plt.plot(plot_idxs, yy_final[plot_idxs], 'ko', markersize=5.0)


        # plot the power spectral density (to visually judge the quality of the denoising)
        #
        titles = (r'raw', r'$\rightarrow$ TV denoising $\rightarrow$', r'$\rightarrow$ Butterworth LP $\rightarrow$', r'$\rightarrow$ reconstruction')
        n_psd = len(titles)
        psds = []

        # compute the power spectra
        #
        # We use a smoothly-decaying window type (Dolph-Chebyshev with 100dB attenuation) with very high overlap (75% of window width).
        #
        # As always, this is a tradeoff:
        #  + clear-looking visualization, readable at first glance
        #  + result is highly insensitive to the placement of segment start points in the signal...
        #    + ...which avoids inconsistent treatment of features in the resulting pictures
        #      (e.g., without overlap, a double transient may show as two, or as just one,
        #       if both transients happen to fall into the same segment)
        #  - blurry overall look
        #  - loss of statistical independence of the segments (not needed here)
        #
        nps = 128
        nov = 96
        for i,y in enumerate( (yy_raw, yy_tvdenoise_only, yy_tv_and_lp, yy_final) ):
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
        plt.figure(2, figsize=(18,9))

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

    # return the processed signal
    return yy_final


# Find indices (in yy) of points important for constructing a piecewise polynomial interpolant,
# which will (hopefully) eliminate the rest of the spurious wiggles.
#
# This is an internal function that finds the deep localized valleys. Each valley is identified by the index triple (i1, k, i2),
# where i1 is the previous local maximum, k the local minimum, and i2 the next local maximum.
#
def find_important_points(yy):
    # find all local extrema
    #
    local_minima = np.squeeze(scipy.signal.argrelextrema(yy, np.less))  # note arg...: the result array contains indices to yy
    local_maxima = np.squeeze(scipy.signal.argrelextrema(yy, np.greater))

    # special cases: detect possible minima at edges of data
    if yy[0] < yy[1]:
        tmp = [0]
        tmp.extend( local_minima.tolist() )
        local_minima = np.array(tmp, dtype=int)

    end = yy.shape[0] - 1
    if yy[-1] < yy[-2]:
        tmp = local_minima.tolist()
        tmp.append( end )
        local_minima = np.array(tmp, dtype=int)

    # remove spurious minima caused by wiggles:
    #
    # accept only those minima that are at <30% of relative data value; these are likely the two valleys (i.e. real physical minima) in the lambda curve.
    #
    yy_max = np.max(yy)
    yy_min = np.min(yy)
    yy_rel = (yy - yy_min) / (yy_max - yy_min)
    global_minima = local_minima[ yy_rel[local_minima] < 0.3 ]

    # build the index triples
    #
    idxs = []
    for k in global_minima:
        # get the local maxima immediately preceding and following the valley (if they exist)
        lm = local_maxima[ local_maxima < k ]
        i1 = lm[-1]  if len(lm) > 0 and k > 0    else None

        lm = local_maxima[ local_maxima > k ]
        i2 = lm[0]   if len(lm) > 0 and k < end  else None

        idxs.append( (i1, k, i2) )  # (preceding local maximum, local minimum, next local maximum)

    return idxs


# *** Spline fitting does not work well in practice for this problem - see pchip version further below. ***
#
def find_important_points_spline(yy):
    knots = []
    end = yy.shape[0] - 1

    for i in range(4):  # cubic spline begins here, so multiplicity 4 is required
        knots.append( 0 )  # first data point

    # points required for handling each valley:
    #
    for i1,k,i2 in find_important_points(yy):
        if i1 is not None:
            for i in range(2):  # at the local maxima surrounding each valley, make the cubic spline continuous up to 1st derivative only
                knots.append( i1 )

        if k > 0  and  k < end:  # exclude start/end points of the spline (can happen if i1 or i2 is None)
            knots.append(k)  # the valley itself can have full C3 continuity

        if i2 is not None:
            for i in range(2):
                knots.append( i2 )

    for i in range(4):  # cubic spline ends here
        knots.append( end )  # last data point

    return np.array(knots, dtype=int)


# Find points for a piecewise cubic (Hermite) fit.
#
def find_important_points_pchip(yy):
    idxs   = find_important_points(yy)
    result = []

    # region before the first valley
    #
    # - take the start point as given
    # - at the start of the valley, sample a few points to make the fitted curve follow the data closely
    # - insert a point at the midpoint of the region to account for possible curvature
    #
    i1,k,i2 = idxs[0]
    if i1 is not None:
        tmp = []

        tmp.append( 0 )  # first data point

        tmp.append( i1 // 2 )

        tmp.append( i1 )
        tmp.append( i1+1 )
        tmp.append( i1+2 )

        tmp = np.unique(tmp)  # very short valleys may cause duplicate indices to appear
        result.append( tmp )


    # regions between each pair of valleys
    #
    for item1,item2 in zip(idxs[:-1], idxs[1:]):   # 0,1; 1,2; ...
        i1 = item1[2]  # end of current valley
        i2 = item2[0]  # start of next valley

        if i1 is not None and i2 is not None:
            tmp = []

            tmp.append( i1-2 )
            tmp.append( i1-1 )
            tmp.append( i1 )

            tmp.append( (i1 + i2) // 2 )

            tmp.append( i2 )
            tmp.append( i2+1 )
            tmp.append( i2+2 )

            tmp = np.unique(tmp)
            result.append( tmp )


    # region after the last valley
    #
    i1,k,i2 = idxs[-1]
    if i2 is not None:
        tmp = []

        tmp.append( i2-2 )
        tmp.append( i2-1 )
        tmp.append( i2 )

        end = yy.shape[0]-1
        tmp.append( (i2 + end) // 2 )

        tmp.append( end )  # last data point

        tmp = np.unique(tmp)
        result.append( tmp )

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="""Denoise measured magnetostriction (lambda) curves.""", formatter_class=argparse.RawDescriptionHelpFormatter)

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

