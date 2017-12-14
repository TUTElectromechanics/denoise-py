#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Postprocess denoised curves to make them single-valued, for fitting a hysteresis-free model.

JJ 2017-08-23"""

import re
import math

import numpy as np

import scipy.spatial
import scipy.io

import matplotlib.pyplot as plt

# pip install wlsqm
#import wlsqm

# pip install bspline
import bspline
import bspline.splinelab as splinelab

import util


# from various scripts, e.g. miniprojects/misc/tworods/main2.py
def axis_marginize(ax, epsx, epsy):
    a = ax.axis()
    w = a[1] - a[0]
    h = a[3] - a[2]
    ax.axis( [ a[0] - w*epsx, a[1] + w*epsx, a[2] - h*epsy, a[3] + h*epsy] )


# FREYA / numutil.py
#
def mirspace(xx):
    """Mirror the vector xx and concatenate with the original.

    That is:
        x0, x1, ..., xN-2, xN-1
            =>   x0, x1, ..., xN-2, xN-1, xN-2, ..., x1, x0         (if xN-1 = 1)
            =>   x0, x1, ..., xN-2, xN-1, xN-1, xN-2, ..., x1, x0   (if xN-1 < 1)
    where for all xj,  0 <= xj <= 1  and  xj+1 > xj.

    If xN-1 = 1, this last point is not mirrored.
    If xN-1 < 1, it will also be mirrored.

    Example:
        This can be used in construction of symmetric spacings, which become denser
        toward both ends, and have one point exactly at the center:

        xx = linspace(0,1,11) ** 2
        xx = mirspace(xx)

    Parameters:
        xx = rank-1 np.array. Arbitrary partition of the interval [0, 1].

    """
    # Sanity checks.
    #
    if not isinstance(xx, np.ndarray):
        raise ValueError("mirspace(): xx must be an np.array, but is '%s'." % type(xx))
    if xx.ndim != 1:
        raise ValueError("mirspace(): xx must be a rank-1 np.array, but has rank %d." % xx.ndim)
    if np.min(xx) < 0:
        raise ValueError("mirspace(): xx must be a partition of [0,1]; however, got min(xx) = %g." % np.min(xx))
    if np.max(xx) > 1:
        raise ValueError("mirspace(): xx must be a partition of [0,1]; however, got max(xx) = %g." % np.max(xx))
    xm = xx[:-1]  # "x minus"
    xp = xx[1:]   # "x plus"
    diff = xp -xm
    if (diff <= 0).any():
        raise ValueError("mirspace(): xx must be a strictly increasing sequence.")

    # Do it.
    #
    xx_firsthalf  = xx.tolist()
    if xx[-1] == 1:
        xx_secondhalf = list(xx_firsthalf[:-1])  # skip last point
    else:
        xx_secondhalf = list(xx_firsthalf)       # mirror last point, too
    xx_secondhalf.reverse()
    return np.array( xx_firsthalf + list(1.0 + 1.0-np.array(xx_secondhalf)) ) / 2


def fit_1d_wlsqm(x, y):
    fit_order = 2  # order of the piecewise polynomials to fit
#    nk = 100       # how many neighbors
    nvis = 10001    # how many points on the output grid


    npoints  = x.shape[0]

    minx = np.min(x)
    maxx = np.max(x)
    r    = 0.015 * (maxx - minx)

    x_rank2 = np.atleast_2d(x).T
    tree = scipy.spatial.cKDTree( data=x_rank2 )


#    # Create neighborhoods: take nk closest neighbors of each point.
#    #
#    dd,ii = tree.query( x_rank2, 1 + nk )
#
#    # Take only the neighbors of points[i], excluding the point itself.
#    ii = ii[:,1:]  # points[ ii[i,k] ] is the kth nearest neighbor of points[i]. Shape of ii is (npoints, nk).
#
#    # neighbor point indices (pointing to rows in x[]); typecast to int32
#    #
#    hoods = np.array( ii, dtype=np.int32 )
#    nk_array = nk * np.ones( (npoints,), dtype=np.int32 )  # number of neighbors, i.e. nk_array[i] is the number of actually used columns in hoods[i,:]


    # Create neighborhoods: take neighbors within distance r (on the x axis) from each point. (maybe better)

    # For each point in x, find all points within radius r.
    #
    # From the docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_tree.html#scipy.spatial.cKDTree.query_ball_tree
    #     For each element self.data[i] of this tree, results[i] is a list of the indices of its neighbors in other.data.
    #
    ii       = tree.query_ball_tree( other=tree, r=r )
    nk_array = np.array( [ len(L) for L in ii], dtype=np.int32 )  # number of neighbors
    nk_array -= 1  # for each point, exclude the point itself

    hoods = np.zeros( (npoints, np.max(nk_array)), dtype=np.int32 )  # hoods[m,:] = model indices (pointing to Case numbers) for x[m]
    for m,L in enumerate(ii):
        L = [i for i in L if i != m]  # filter out the point itself
        hoods[m,:len(L)] = L

#    print( [f(nk_array) for f in [np.min, np.mean, np.max]] )  # DEBUG


    # Construct the model by least-squares fitting
    #
    import wlsqm
    fit_order_array = fit_order            * np.ones( (npoints,), dtype=np.int32 )
    knowns_array    = wlsqm.b2_F           * np.ones( (npoints,), dtype=np.int64 )  # bitmask! wlsqm.b*
    wm_array        = wlsqm.WEIGHT_UNIFORM * np.ones( (npoints,), dtype=np.int32 )
    solver = wlsqm.ExpertSolver( dimension=1,
                                 nk=nk_array,
                                 order=fit_order_array,
                                 knowns=knowns_array,
                                 weighting_method=wm_array,
                                 algorithm=wlsqm.ALGO_BASIC,
                                 do_sens=False,
                                 max_iter=10,  # must be an int even though this parameter is not used in ALGO_BASIC mode
                                 ntasks=8,
                                 debug=False )

    no = wlsqm.number_of_dofs( dimension=2, order=fit_order )
    fi = np.empty( (npoints,no), dtype=np.float64 )
    fi[:,0] = y  # fi[i,0] contains the function value at point x[i,:]

    solver.prepare( xi=x, xk=x[hoods] )  # generate problem matrices from the geometry of the point cloud
    solver.solve( fk=fi[hoods,0], fi=fi, sens=None )  # compute least-squares fit to data

    # Using the model, interpolate onto a regular grid
    #
    xx   = np.linspace( minx, maxx, nvis )
    solver.prep_interpolate()  # prepare global model
    yy,dummy = solver.interpolate( xx, mode='continuous', r=2.*r )  # use a larger r here to smooth using the surrounding models

    return (xx, yy)


# nvis = how many points on the output grid
def fit_1d_weighted_average_globalr(x, y, nvis=10001):
#    npoints  = x.shape[0]

    minx = np.min(x)
    maxx = np.max(x)
    r    = 0.015 * (maxx - minx)

    x_rank2 = np.atleast_2d(x).T
    tree_in = scipy.spatial.cKDTree( data=x_rank2 )

    xx = np.linspace( minx, maxx, nvis )  # output
    xx_rank2 = np.atleast_2d(xx).T
    tree_out = scipy.spatial.cKDTree( data=xx_rank2 )

    # For each point in xx, find all data points within radius r.
    #
    # From the docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_tree.html#scipy.spatial.cKDTree.query_ball_tree
    #     For each element self.data[i] of this tree, results[i] is a list of the indices of its neighbors in other.data.
    #
    ii = tree_out.query_ball_tree( other=tree_in, r=r )

#    # DEBUG: number of points taking part in each averaging
#    nk_array = np.array( [ len(L) for L in ii], dtype=np.int32 )
#    print( [f(nk_array) for f in [np.min, np.mean, np.max]] )

    # compute the weighted average

#    # distance squared, flipped on the distance axis (fast falloff near origin)
#    alpha  = 0.          # weight remaining at maximum distance
#    beta   = 1. - alpha
#    max_d2 = r*r
#    def weight(d2):
#        tmp = 1. - math.sqrt(d2 / max_d2)
#        return alpha + beta * tmp*tmp

    def weight(d):
        s   = d/r * np.pi/2.
        tmp = math.cos(s)
        return tmp*tmp

    yy = np.empty_like(xx)
    for m,L in enumerate(ii):  # m = output index (to xx)
        acc   = 0.
        sum_w = 0.
        for i in L:  # i = input index (to x)
            d  = abs(x[i] - xx[m])
#            d2 = d*d
#            w  = weight(d2)

            w = weight(d)

            acc   += w * y[i]
            sum_w += w

        yy[m] = acc / sum_w

    return (xx, yy)


# nvis = how many points on the output grid
def fit_1d_weighted_average_localr(x, y, symmetric_x=True, nvis=10001):
    minx = np.min(x)
    maxx = np.max(x)
    if symmetric_x:
        # The BH loop has been sampled for both positive and negative B (x data),
        # make sure the output B is symmetric
        #
        m1 = min(minx, -maxx)
        m2 = max(maxx, -minx)
        xx = np.linspace( m1, m2, nvis )  # output
    else:
        xx = np.linspace( minx, maxx, nvis )  # output

#    print( xx[0], xx[xx.shape[0]//2], xx[-1] )  # DEBUG

    x_rank2 = np.atleast_2d(x).T
    tree = scipy.spatial.cKDTree( data=x_rank2 )

#    # estimate neighbor distance (on x axis) locally for each input data point, using the fact that the (x,y) data forms a continuous curve
#    #
#    rs = np.empty_like(x)
#    rs[1:-1] = 0.5 * (abs(x[2:] - x[1:-1]) + abs(x[1:-1] - x[:-2]))
#    # no wraparound
##    rs[0]    = abs(x[1]  - x[0])
##    rs[-1]   = abs(x[-1] - x[-2])
#    # with wraparound (we know the data is a BH loop)
#    rs[0]    = 0.5 * (abs(x[1]  - x[0])  + abs(x[0] - x[-1]))
#    rs[-1]   = 0.5 * (abs(x[-1] - x[-2]) + abs(x[0] - x[-1]))

    # take the maximum distance from the 10 closest neighbors of each point (excluding the point itself)
    dd,ii = tree.query( x_rank2, 11 )
    rs = dd[:,-1]
#    rs = np.max( dd[:,1:], axis=-1 )

    # take local neighborhood size = some constant * local neighbor distance
    #
    rs *= 1.5

#    # DEBUG
#    tmp = r / (maxx - minx)
#    print( [f(tmp) for f in [np.min, np.mean, np.max]] )

    # compute the weighted average

    # find closest neighbor in x for each point in xx (we will use this to look up relevant r)
    xx_rank2 = np.atleast_2d(xx).T
    dd,ii = tree.query( xx_rank2, 1 )

    def weight(d, r):
        s   = d/r * np.pi/2.
        tmp = math.cos(s)
        return tmp*tmp

    yy = np.empty_like(xx)
    for m,nearest in enumerate(ii):  # m = output index (to xx)
        # neighbors inside local r
        r = rs[nearest]  # use the r value of the data point closest (along the x axis) to this output point
        L = tree.query_ball_point( np.atleast_1d(xx[m]), r )

        acc   = 0.
        sum_w = 0.
        for i in L:  # i = input index (to x)
            d  = abs(x[i] - xx[m])
            w = weight(d,r)

            acc   += w * y[i]
            sum_w += w

        yy[m] = acc / sum_w

    return (xx, yy)


# construct a least-squares optimal cubic spline interpolant to the data
#
# (TODO: much of this comes from fit_2d.py - maybe the bspline library needs some functions for spline fitting on n-D meshgrids.)
#
# knots: optional, preliminary knot vector (without repeated knots at ends)
#
def fit_1d_spline(x, y, knots=None, nvis=10001):
    spline_order = 3

    minx = np.min(x)
    maxx = np.max(x)

    # Preliminary placement of knots.
    #
    # Bump the last site slightly. The spline is nonzero only on the *half-open* interval [x1, x2),
    # so the value of the spline interpolant exactly at the end of the span is always 0.
    #
#    kk = np.linspace(minx, maxx + 1e-8*(maxx-minx), 21)  # better to adjust number and spacing of knots (maybe quadratic toward ends?)

    if knots is not None:
        kk = knots
    else:  # if no custom knot vector, make one now (emphasize ends -- good for BH curves)
        kk = np.linspace(0,1, 81)
        kk = kk**2
        kk = mirspace(kk)
        kk = minx + (1. + 1e-8)*(maxx - minx)*kk

    kk   = splinelab.aptknt(kk, order=spline_order)
    spl  = bspline.Bspline(order=spline_order, knot_vector=kk)

    nx  = x.shape[0]
    Au  = spl.collmat(x)

    # construct the overdetermined linear system for determining the optimal spline coefficients
    #
    nf = 1   # number of unknown fields
    nr = nx  # equation system rows per unknown field
    nxb = len( spl(0.) ) # get number of basis functions (perform dummy evaluation and count)
    A  = np.empty( (nf*nr, nxb), dtype=np.float64 )  # global matrix
    b  = np.empty( (nf*nr),      dtype=np.float64 )  # global RHS

    # loop only over rows of the equation system
    for i in range(nf*nr):
        A[nf*i,:] = Au[i,:]
    b[:] = y

    # solve the overdetermined linear system (in the least-squares sense)

#    # dense solver (LAPACK DGELSD)
#    ret = np.linalg.lstsq(A, b)  # c,residuals,rank,singvals
#    c = ret[0]

    # sparse solver (SciPy LSQR)
    S = scipy.sparse.coo_matrix(A)
    print( "    matrix shape %s = %d elements; %d nonzeros (%g%%)" % (S.shape, np.prod(S.shape), S.nnz, 100. * S.nnz / np.prod(S.shape) ) )
    ret = scipy.sparse.linalg.lsqr( S, b )
#    c,exit_reason,iters = ret[:3]
    c,exit_reason = ret[:2]
    if exit_reason != 2:  # 2 = least-squares solution found
        print( "WARNING: solver did not converge (exit_reason = %d)" % (exit_reason) )

    # evaluate the computed optimal b-spline
    #
    xx_spline = np.linspace(minx, maxx, nvis)
    Avis = spl.collmat(xx_spline)
    yy_spline = np.sum( Avis*c, axis=-1 )

    return (xx_spline, yy_spline)


# - attempt to remove y offset from (multivalued) raw data
#
def _deoffset_rawdata(x,y, x_rel_tol=1e-2):
#    yoffs = np.mean(y)

    minx = np.min(x)
    maxx = np.max(x)
    eps  = x_rel_tol * (maxx - minx)
    mask_near_zero_x = (np.abs(x) < eps)

#    print( np.nonzero(mask_near_zero_x)[0] )  # DEBUG

    yoffs = np.mean( y[mask_near_zero_x] )

    y2 = y - yoffs

    return (x, y2)


# find the self-crossing, take the y value there as the offset.
#
def _deoffset_lam_data(x,y):
    p = np.empty( (x.shape[0],2), dtype=np.float64 )
    p[:,0] = x
    p[:,1] = y
    tree = scipy.spatial.cKDTree(p)

    D,I = tree.query(p, k=2)

    # D[:,0] = distance to the point itself = 0
    # D[:,1] = distance to nearest neighbor
    #
    # The point that has the minimal distance to its nearest neighbor
    # is likely at the self-crossing of the curve.
    #
    # TODO: ...but apparently not. Due to the variable spacing of the points,
    # there are points which are closer to their neighbors in the continuous
    # parts of the curve. Need a new strategy to locate the self-intersection.
    #
    idx_in_I = np.argmin( D[:,1] )
    i = I[idx_in_I,1]

    yoffs = y[i]
    y2 = y - yoffs

    print( yoffs )  # DEBUG

    return (x, y2)


# - flip points where B < 0 (x < 0), to overlay both parts of the curve in the positive quadrant
#
def _overlay_antisymmetric(x, y):
    mask = (x < 0)

    x2 = x.copy()
    x2[mask] = -x[mask]

    y2 = y.copy()
    y2[mask] = -y2[mask]

    return (x2, y2)

def _overlay_symmetric(x, y):
    mask = (x < 0)

    x2 = x.copy()
    x2[mask] = -x[mask]

    y2 = y.copy()  # y doesn't change

    return (x2, y2)

# - remove y offset of a single-valued curve at x = 0 (defined for full range of x)
#
def _deoffset_output(x, y):
#    # find first element with x >= 0
#    mask = (x >= 0)
#    i = ( np.nonzero(mask)[0] )[0]
#    yoffs = y[i]

    # linearly interpolate the data to x = 0
    f = scipy.interpolate.interp1d(x,y)
    yoffs = f(0.0)

    y2 = y - yoffs

    return (x, y2)

# copy single-valued (and "positive", i.e. B>=0 only) curve (generated from overlaid data),
# to make an antisymmetric (skew-symmetric) curve that allows both B>0 and B<0.
#
# x must be in increasing order, with the sample for B=0 at index 0.
# The sample y[j] must correspond to x[j].
#
def _antisymmetrize(x, y):
    nx    = x.shape[0] - 1  # -1: do not duplicate the sample for B=0
    lenx2 = 2*nx + 1
    imid  = lenx2//2

    x2 = np.empty( (lenx2,), dtype=x.dtype )
    x2[imid]       =  x[0]
    x2[imid+1:]    =  x[1:]
    x2[imid-1::-1] = -x[1:]

    y2 = np.empty_like(x2)
    y2[imid]       =  y[0]
    y2[imid+1:]    =  y[1:]
    y2[imid-1::-1] = -y[1:]

    return (x2, y2)

def _symmetrize(x, y):
    nx    = x.shape[0] - 1  # -1: do not duplicate the sample for B=0
    lenx2 = 2*nx + 1
    imid  = lenx2//2

    x2 = np.empty( (lenx2,), dtype=x.dtype )
    x2[imid]       =  x[0]
    x2[imid+1:]    =  x[1:]
    x2[imid-1::-1] = -x[1:]

    y2 = np.empty_like(x2)
    y2[imid]       =  y[0]
    y2[imid+1:]    =  y[1:]
    y2[imid-1::-1] =  y[1:]  # no sign flip

    return (x2, y2)


def fit_1d_doeverything_old(x, y):
    x2,y2 = _deoffset_rawdata(x,y)
    x2,y2 = _overlay_antisymmetric(x2,y2)

    # this is the single-valuization step
    #
#    xx,yy = fit_1d_weighted_average_globalr(x2, y2, nvis=5001)
#    xx,yy = fit_1d_weighted_average_localr(x2, y2, symmetric_x=False, nvis=5001)
    xx,yy = fit_1d_spline(x2,y2, nvis=5001)

    xx2,yy2 = _antisymmetrize(xx,yy)
    xx2,yy2 = _deoffset_output(xx2,yy2)

    return (xx2,yy2)


def fit_1d_doeverything(x, y):
    x2,y2 = _deoffset_rawdata(x,y)
    x2,y2 = _overlay_antisymmetric(x2,y2)
    xx,yy = _antisymmetrize(x2,y2)
    xx2,yy2 = fit_1d_spline(xx,yy, nvis=10001)
    xx2,yy2 = _deoffset_output(xx2,yy2)
    return (xx2,yy2)

def fit_1d_doeverything_for_lam(x, y):
    x2,y2 = _deoffset_lam_data(x,y)
    x2,y2 = _overlay_symmetric(x2,y2)
    xx,yy = _symmetrize(x2,y2)

    # here we must emphasize the center (data changes faster there)
    minx = np.min(x)
    maxx = np.max(x)
    kk = np.linspace(0,1, 81)
    kk = 1 - (1 - kk)**2
    kk = mirspace(kk)
    kk = minx + (1. + 1e-8)*(maxx - minx)*kk

    xx2,yy2 = fit_1d_spline(xx,yy, knots=kk, nvis=10001)
    xx2,yy2 = _deoffset_output(xx2,yy2)
    return (xx2,yy2)


def take_positive_half(x, y, tol=1e-8):  # w.r.t x; tol detects origin
    mask = (x >= 0)
    x   = x[mask]
    y   = y[mask]
    if x[0] > tol:  # no point "numerically exactly" at zero
        tmpx = np.empty( (x.shape[0]+1,), dtype=x.dtype )
        tmpy = np.empty( (y.shape[0]+1,), dtype=y.dtype )
        tmpx[0]  = 0.0
        tmpy[0]  = 0.0
        tmpx[1:] = x
        tmpy[1:] = y
        x = tmpx
        y = tmpy
    return (x,y)


def main(path):
    # get list of raw data files in specified directory
    #
    data_items = util.listfiles(path, verbose=False)

    for sigma,input_filename in data_items:
#    for dummy,input_filename in data_items:
        if sigma != 0:  # XXX DEBUG
            continue

        input_filename  = re.sub( r'\.mat$', r'_denoised.mat', input_filename )      # denoised data files
        output_filename = re.sub( r'\.mat$', r'_singlevalued.mat', input_filename )

        print( "Single-valuizing '%s' --> '%s'..." % (input_filename, output_filename) )

        try:
            data = scipy.io.loadmat(input_filename)
        except FileNotFoundError:
            import sys
            print( "Data file named '%s' not found, exiting (use --list to see available data files)" % (input_filename), file=sys.stderr )
            sys.exit(1)

        A    = data["A"]

        H    = A[:,0]  # Field strength H (A/m)
        B    = A[:,1]  # Flux density B (T)
        pol  = A[:,2]  # magnetic polarization J = B - mu0*H
        lam  = A[:,3]  # Magnetostriction lambda (ppm)

        assert H.shape == B.shape == pol.shape == lam.shape

#        # de-hysterize
#        xx,yy = fit_1d_weighted_average_localr(B,H)
#
#        # symmetrize w.r.t. B = 0
#        #
#        # we average the positive and negative parts.
#        #
#        imid = yy.shape[0]//2
#        ymid = yy[imid]
##        tmp = 0.5 * ((ymid - yy[imid-1::-1]) + (yy[imid+1:] - ymid))
#        tmp = 0.5 * (yy[imid+1:] - yy[imid-1::-1])  # equivalent
#        yy2 = np.empty_like(yy)
#        yy2[imid-1::-1] = ymid - tmp
#        yy2[imid+1:]    = ymid + tmp
#        yy2[imid]       = ymid
#
#        # DEBUG TEST - swap pos/neg parts
#        tmp_p = yy[imid+1:] - ymid
#        tmp_n = ymid - yy[imid-1::-1]
#        yy3 = np.empty_like(yy)
#        yy3[imid-1::-1] = ymid - tmp_p
#        yy3[imid+1:]    = ymid + tmp_n
#        yy3[imid]       = ymid

#        xx,yy = fit_1d_doeverything(B,H)
#        xx,yy = take_positive_half(xx, yy)

        xout = []
        yout = []
        xdata = [B, pol, H]
        ydata = [H, H, lam]
        xlabels = [r"$B$", r"$pol$", r"$H$"]
        ylabels = [r"$H$", r"$H$",   r"$\lambda$"]
        deoffsetters = [_deoffset_rawdata, _deoffset_rawdata, _deoffset_lam_data]
        fs      = [fit_1d_doeverything, fit_1d_doeverything, fit_1d_doeverything_for_lam]
        for xx,yy,f in zip(xdata,ydata,fs):
            xx2,yy2 = f(xx,yy)
            xx2,yy2 = take_positive_half(xx2,yy2)
            xout.append(xx2)
            yout.append(yy2)

#        plt.figure(1, figsize=(9,6))
#        plt.clf()
#        x,y = _deoffset_rawdata(B,H)
#        plt.plot(x,  y,   color='#d0d0d0', linestyle='solid')
#        plt.plot(xx, yy,  color='#909090', linestyle='solid')
#        plt.plot(xx, yy3, color='#909090', linestyle='dashed')  # DEBUG: pos/neg parts swapped
#        plt.plot(xx, yy2, color='k',       linestyle='solid')
#        plt.xlabel(r"$B$")
#        plt.ylabel(r"$H$")

        plt.figure(1, figsize=(14,6))
        plt.clf()
        nplots = len(ydata)
        for i,deof,xx_raw,xx_filt,xlabel,yy_raw,yy_filt,ylabel in \
                   zip(range(nplots), deoffsetters, xdata, xout, xlabels, ydata, yout, ylabels):
            ax = plt.subplot(1,nplots, i+1)
            x,y= deof(xx_raw,yy_raw)
            ax.plot(x,       y,        color='#d0d0d0', linestyle='solid')
            ax.plot(xx_filt, yy_filt,  color='#909090', linestyle='solid')
            ax.axis( [np.min(xx_filt), np.max(xx_filt), np.min(yy_filt), np.max(yy_filt)] )
            axis_marginize(ax, 0.02, 0.02)
            ax.grid(b=True, which='both')
            ax.set_title(r"%s(%s)" % (ylabel, xlabel))

        break

        # We have no guarantees that x starts from 0. Fix this.
        #
        tol = 1e-8
        if xout[0][0] < tol:
            xout[0][0] = 0
        if xout[1][0] < tol:
            xout[1][0] = 0

        # HACK: the lambda curve is fitted with the axes swapped.
        if yout[2][0] < tol:
            yout[2][0] = 0
        else:
            raise ValueError("something went wrong, lambda curve does not start from zero (got %g)" % (yout[2][0]))


        # Clip data to the smallest common max(H)
        #
        xs = [xout[0], xout[1], yout[2]]
        ys = [yout[0], yout[1], xout[2]]
        fs = []
        max_minx = -np.inf
        min_maxx = +np.inf
        for x,y in zip(xs,ys):
            min_maxx = min(np.max(x), min_maxx)
            max_minx = max(np.min(x), max_minx)
            fs.append( scipy.interpolate.interp1d(x,y) )

        # Interpolate to a common grid on the H axis
        #
        xx = np.linspace(0, min_maxx, 10001)
        yout2 = []
        for f in fs:
            yout2.append( f(xx) )


        # Save.
        #
        A = np.empty( (xx.shape[0],4), dtype=np.float64 )
        A[:,0] = xx
        A[:,1] = yout2[0]
        A[:,2] = yout2[1]
        A[:,3] = yout2[2]
        scipy.io.savemat( output_filename, mdict={ 'A' : A } )

    print( "All done." )

if __name__ == '__main__':
    main( path="." )
    plt.show()
