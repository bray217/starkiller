"""
PSF photometry routines for fitting grouped sources in 2-D images and data cubes.

This module provides the ``PSF_photom`` class for multi-source PSF fitting
using a pre-built PSF model, together with standalone helper functions used
by the parallel fitting back-end.
"""
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from starkiller.trail_psf import create_psf
from scipy import signal


class PSF_photom():
    def __init__(self,data,cat,psf=None,pad=20,psf_params=None,data_psf=True,cores=8):
        """
        Initialise the PSF photometry fitter.

        Parameters
        ----------
        data : numpy.ndarray
            2-D image or 3-D data cube to be fitted.
        cat : pandas.DataFrame
            Source catalogue with at least ``xint`` and ``yint`` integer pixel
            columns and an ``id`` column.
        psf : object, optional
            Pre-built PSF object (e.g. from ``starkiller.trail_psf``).  Either
            ``psf`` or ``psf_params`` must be provided.
        pad : int, optional
            Number of pixels to pad around the data array (default ``20``).
        psf_params : array-like, optional
            Parameters ``[alpha, beta, length, angle]`` used to construct a
            Moffat PSF when no pre-built ``psf`` is supplied.
        data_psf : bool, optional
            If ``True`` and a ``psf`` object is provided, replace ``longpsf``
            with the empirical data PSF (default ``True``).
        cores : int, optional
            Number of parallel cores for cube fitting (default ``8``).
        """
        self.pad = pad
        self._set_data(data)
        self.cores = cores

        #self._set_source_dims(source_dims)
        if psf is not None:
            self.psf = psf
            if data_psf:
                self.psf.longpsf = self.psf.data_psf
                self.psf.longPSF = self.psf.data_PSF
        elif psf_params is not None:
            self._set_psf(psf_params)
        else:
            raise ValueError('No psf info provided')
        self.cat = cat


    def _set_psf(self,psf_params):
        """Build and store a Moffat line PSF from the supplied parameter array."""
        self.psf_params = psf_params
        if self.cube:
            x=self.data.shape[2];y=self.data.shape[1]
        else:
            x=self.data.shape[1];y=self.data.shape[0]

        self.psf = create_psf(x=x,y=y,alpha=psf_params[0],
                              beta=psf_params[1],length=psf_params[2],angle=psf_params[3])
        self.psf.line()

    def _set_data(self,data):
        """Pad the input array and detect whether it is a cube or a single image."""
        self.data = np.pad(data,self.pad,constant_values=np.nan)
        if len(data.shape) == 2:
            self.cube = False
        else:
            self.cube = True
            self.data = self.data[self.pad:-self.pad]
            self.image = np.nanmedian(self.data,axis=0)
            self.image[self.image == 0] = np.nan
            

    def _set_source_dims(self,source_dims):
        """Parse and store x/y half-lengths for source cutout regions."""
        try:
            length = len(source_dims)
            if length == 2:
                self.x_length = source_dims[0]
                self.y_length = source_dims[1]
            else:
                self.x_length = source_dims; self.y_length = source_dims
        except:
            self.x_length = source_dims; self.y_length = source_dims

    def _make_masks(self,limit=70):
        """
        Construct per-source PSF footprint masks via convolution.

        Parameters
        ----------
        limit : float, optional
            Percentile threshold applied to the PSF to define the footprint
            kernel (default ``70``).
        """
        im = np.zeros((len(self.cat),self.image.shape[0],self.image.shape[1]))
        for i in range(len(self.cat)):
            im[i,self.cat.yint.values[i] + self.pad,self.cat.xint.values[i] + self.pad] = 1
        kernel = (self.psf.longpsf > np.percentile(self.psf.longpsf,85))*1#np.ones((2*self.y_length,2*self.x_length))
        im = signal.fftconvolve(im,kernel[np.newaxis,:,:],mode='same')
        im = im > 0.1
        self.source_masks = im

    def group_sources(self,limit=85):
        """
        Assign each source in the catalogue to a spatial group based on PSF overlap.

        Group labels are written into a new ``'group'`` column of ``self.cat``.

        Parameters
        ----------
        limit : float, optional
            Percentile threshold applied to the PSF to define the overlap
            kernel (default ``85``).
        """
        if self.cube:
            image = self.image
        else:
            image = self.data
        im = np.zeros_like(image)
        im[self.cat.yint.values+self.pad,self.cat.xint.values+self.pad] = 1
        kernel = (self.psf.longpsf > np.nanpercentile(self.psf.longpsf,limit))*1
        im = signal.fftconvolve(im,kernel,mode='same')

        im = im > 0.1
        labeled, nr_objects = label(im)
        self.cat['group'] = 0
        group_col = self.cat.columns.get_loc('group')
        for i in range(len(self.cat)):
            self.cat.iloc[i, group_col] = labeled[self.cat.yint.values[i]+self.pad,self.cat.xint.values[i]+self.pad]
            
    
    def _groups_mask(self,ind):
        """Return a combined footprint mask for the sources selected by ``ind``."""
        if np.sum(ind) > 1:
            mask = (np.nansum(self.source_masks[ind],axis=0) > 0) * 1.0
        else:
            mask = self.source_masks[ind] * 1.0
        mask[mask == 0] = np.nan

        return mask

    def _estimate_f0(self,data,ind):
        """Estimate initial flux guesses for sources selected by ``ind`` using their PSF footprints."""
        ind = np.where(ind)[0]
        f0 = []
        for i in ind:
            m = self.source_masks[i]
            f0 += [np.nansum(m * data)]
        f0 = np.array(f0)
        return f0

    def make_psf_image(self,x0,data,sources):
        """
        Render a model image by placing shifted PSF copies at each source position.

        Parameters
        ----------
        x0 : array-like
            Parameter vector: fluxes, then x sub-pixel offsets, then y
            sub-pixel offsets, with a trailing background constant.
        data : numpy.ndarray
            2-D reference array that sets the output image shape.
        sources : pandas.DataFrame
            Subset of the source catalogue rows to render.

        Returns
        -------
        f : numpy.ndarray
            Rendered model image with the same shape as ``data``.
        """
        f0 = x0[:len(sources)]; px = x0[len(sources):len(sources)*2] ; py = x0[2*len(sources):]
        bkg = x0[-1]
        tripys = []
        xs = sources.xint.values + self.pad
        ys = sources.yint.values + self.pad
        f = []
        if 'data' in self.psf.psf_profile:
            for i in range(len(f0)):
                seed = np.zeros_like(data)
                seed[int(ys[i]+0.5),int(xs[i]+0.5)] = f0[i]
                kernel = shift(self.psf.longpsf,[-py[i],-px[i]],mode='nearest')
                f += [signal.fftconvolve(seed,kernel,mode='same')]
        else:
            t = create_psf(x=data.shape[1],y=data.shape[0],alpha=self.psf_param[0],
                           beta=self.psf_param[1],length=self.psf_param[2],angle=self.psf_param[3])
            for i in range(len(f0)):
                t.line(shiftx=xs[i]+px[i],shifty=ys[i]+py[i])
                f += [t.linepsf * f0[i]]
        f = np.array(f)
        f = np.nansum(f,axis=0) + bkg
        return f 

    def _group_minimizer(self,x0,data,sources):
        """Return the sum of squared residuals between the data and the PSF model for minimisation."""
        f = self.make_psf_image(x0,data,sources)
        res = np.nansum((data - f)**2)
        return res



    def group_fit(self,data,ind):
        """
        Fit fluxes, sub-pixel positions, and background for a group of sources.

        Parameters
        ----------
        data : numpy.ndarray
            2-D image slice to fit.
        ind : array-like of int
            Integer indices into ``self.cat`` identifying the source group.

        Returns
        -------
        flux : numpy.ndarray
            Best-fit flux for each source.
        xp : numpy.ndarray
            Best-fit x sub-pixel offset for each source.
        yp : numpy.ndarray
            Best-fit y sub-pixel offset for each source.
        res : numpy.ndarray
            Masked residual image after subtracting the model.
        bkg : float
            Best-fit background level.
        """
        sources = self.cat.iloc[ind]
        mask = self._groups_mask(ind)
        f0 = self._estimate_f0(data,ind)
        pos0 = np.append(sources['xint'].values,sources['yint'].values,axis=0)
        guess = np.append(f0,pos0,axis=0)
        f0_bounds = np.array([f0*0,f0*2]).T
        pos_bounds = np.zeros_like(f0_bounds)
        pos_bounds[:,0] = -1; pos_bounds[:,1] = 1
        bounds = np.append(f0_bounds,pos_bounds,axis=0)
        bounds = np.append(bounds,pos_bounds,axis=0)
        data_masked = data * mask
        # add background
        bkg = np.nanpercentile(data_masked,5)
        if bkg < 0:
            low = bkg - 0.2*bkg
            high = -bkg
        else:
            low = -bkg
            high = bkg + 0.2*bkg
        guess = np.append(guess,[bkg],axis=0)
        bounds = np.append(bounds,[np.array([low,high])],axis=0)
        print(bounds)
        
        res = minimize(self._group_minimizer,guess,args=(data_masked,sources),method='Nelder-Mead',bounds=bounds)

        flux = res.x[:len(sources)]; xp = res.x[len(sources):len(sources)*2]; yp = res.x[len(sources)*2:len(sources)*3]
        bkg = np.array(res.x[-1])

        res = (data - self.make_psf_image(res.x,data,sources)) * mask
        return flux, xp, yp, res, bkg 



    def group_psf(self):
        """
        Fit all source groups and accumulate fluxes and refined positions.

        For a 3-D cube the fitting is parallelised over spectral slices.
        Results are stored in ``self.fluxes`` (keyed by source id) and the
        ``'x'``/``'y'`` columns of ``self.cat`` are updated in-place.
        """
        groups = self.cat['group'].unique()
        fluxes = {}
        residual = np.zeros_like(self.data)
        for group in groups:
            ind = np.where(self.cat['group'].values == group)[0]
            sources = self.cat.iloc[ind]
            if len(ind) > 1:
                mask = (np.nansum(self.source_masks[ind],axis=0) > 0) * 1.0
            else:
                mask = self.source_masks[ind] * 1.0
            mask[mask == 0] = np.nan
            sources = self.cat.iloc[ind]
            groupmask = self._groups_mask(ind)
            sourcemasks = []
            for i in ind:
                sourcemasks += [self.source_masks[i]]
            
            if self.cube:
                f, xp, yp, res, bkg = zip(*Parallel(n_jobs=self.cores)(delayed(_group_fit)(sources,groupmask,sourcemasks,image,self.psf,self.pad) for image in self.data))
                flux = np.array(f).T; posx = np.array(xp); posy = np.array(yp); res = np.array(res)
                posx = np.nanmean(posx,axis=0); posy = np.nanmean(posy,axis=0)

            else:
                flux, posx, posy, res = _group_fit(sources,groupmask,sourcemasks,image,self.psf,self.pad)

            self.cat.iloc[ind, self.cat.columns.get_loc('x')] = posx
            self.cat.iloc[ind, self.cat.columns.get_loc('y')] = posy
            for i in range(len(sources)):
                fluxes[sources.iloc[i].id] = flux[i]
            #residual = residual + res

        self.fluxes = fluxes 
        #self.residual = residual




def _make_psf_image(x0,data,sources,psf,pad = 0):
    """Render a model image from a parameter vector; standalone version used by the parallel fitting back-end."""
    f0 = x0[:len(sources)]; px = x0[len(sources):len(sources)*2] ; py = x0[2*len(sources):]
    bkg = x0[-1]
    tripys = []
    xs = sources.xint.values + pad
    ys = sources.yint.values + pad
    f = []
    if 'data' in psf.psf_profile:
        for i in range(len(f0)):
            seed = np.zeros_like(data)
            seed[int(ys[i]+0.5),int(xs[i]+0.5)] = f0[i]
            kernel = shift(psf.longpsf,[-py[i],-px[i]],mode='nearest')
            f += [signal.fftconvolve(seed,kernel,mode='same')]
    else:
        t = create_psf(x=data.shape[1],y=data.shape[0],alpha=psf.psf_param[0],
                       beta=psf.psf_param[1],length=psf.psf_param[2],angle=psf.psf_param[3])
        for i in range(len(f0)):
            t.line(shiftx=xs[i]+px[i],shifty=ys[i]+py[i])
            f += [t.linepsf * f0[i]]
    f = np.array(f)
    f = np.nansum(f,axis=0) + bkg
    return f 

def _group_minimizer(x0,data,sources,psf,pad):
    """Return the sum of squared residuals for use as a scalar objective in ``scipy.optimize.minimize``."""
    f = _make_psf_image(x0,data,sources,psf,pad)
    res = np.nansum((data - f)**2)
    return res


def _estimate_f0(data,masks):
    """Return initial flux estimates by summing the masked data within each source footprint."""
    f0 = []
    for m in masks:
        f0 += [np.nansum(m * data)]
    f0 = np.array(f0)
    return f0

def _group_fit(sources,groupmask,sourcemasks,data,psf,pad=0,
               position_bound=2):
    """
    Fit fluxes, positions, and background for a group of sources in a single image slice.

    Parameters
    ----------
    sources : pandas.DataFrame
        Catalogue rows for the sources in the group.
    groupmask : numpy.ndarray
        Combined boolean footprint mask for the group.
    sourcemasks : list of numpy.ndarray
        Per-source footprint masks used for initial flux estimation.
    data : numpy.ndarray
        2-D image slice to fit.
    psf : object
        PSF model object with a ``longpsf`` attribute.
    pad : int, optional
        Pixel padding offset already applied to source coordinates (default
        ``0``).
    position_bound : float, optional
        Maximum allowed positional offset in pixels during optimisation
        (default ``2``).

    Returns
    -------
    flux : numpy.ndarray
        Best-fit flux for each source.
    xp : numpy.ndarray
        Best-fit x position for each source.
    yp : numpy.ndarray
        Best-fit y position for each source.
    res : numpy.ndarray
        Masked residual image after subtracting the model.
    bkg : float
        Best-fit background level.
    """
    f0 = _estimate_f0(data,sourcemasks)
    f0[f0 < 0] = 1
    pos0 = np.append(sources['xint'].values + pad,sources['yint'].values + pad,axis=0)
    guess = np.append(f0,pos0,axis=0)
    f0_bounds = np.array([f0*0,f0*2]).T
    pos_bounds = np.zeros_like(f0_bounds)
    pos_bounds[:,0] = -position_bound; pos_bounds[:,1] = position_bound
    bounds = np.append(f0_bounds,pos_bounds,axis=0)
    bounds = np.append(bounds,pos_bounds,axis=0)
    data_masked = data * groupmask
    # add background
    bkg = np.nanpercentile(data_masked,5)
    if bkg < 0:
        low = bkg - 0.2*bkg
        high = -bkg
    else:
        low = -bkg
        high = bkg + 0.2*bkg
    guess = np.append(guess,[bkg],axis=0)
    bounds = np.append(bounds,[np.array([low,high])],axis=0)
    print(bounds)

    res = minimize(_group_minimizer,guess,args=(data_masked,sources,psf,pad),method='Nelder-Mead',bounds=bounds)

    flux = res.x[:len(sources)]; xp = res.x[len(sources):len(sources)*2]; yp = res.x[len(sources)*2:len(sources)*3]
    bkg = res.x[-1]

    res = (data - _make_psf_image(res.x,data,sources,psf,pad)) * groupmask

    return flux, xp, yp, res, bkg 