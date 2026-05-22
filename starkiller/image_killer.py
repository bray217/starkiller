"""
Image-based star subtraction pipeline for 2-D astronomical images.

This module provides the ``starkiller_image`` class, which ingests a science
image together with its WCS, queries the Pan-STARRS catalogue, fits an
effective PSF, and removes field stars to produce a cleaned residual image.
A helper function ``affine_positions`` is also provided for refining source
positions via an affine transformation.
"""
from .helpers import *
import traceback
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from copy import deepcopy
from scipy import signal
from starkiller.trail_psf import create_psf
from scipy.ndimage import gaussian_filter, label
from photutils.psf import PSFPhotometry
from astropy.table import Table
from photutils.background import LocalBackground, MMMBackground, SExtractorBackground
from photutils.psf import EPSFBuilder, EPSFStar, EPSFStars
from astropy.table import QTable
from photutils.psf import SourceGrouper
from astropy.visualization import SqrtStretch,ImageNormalize,AsymmetricPercentileInterval




from astropy.stats import sigma_clipped_stats, sigma_clip
from starkiller.helpers import *

def affine_positions(phot,mask=None):
	"""
	Fit an affine transformation between initial and fitted source positions and
	apply it to correct outlier positions.

	Parameters
	----------
	phot : astropy.table.Table
		Photometry table containing ``x_init``, ``y_init``, ``x_fit``, and
		``y_fit`` columns (with units).
	mask : numpy.ndarray, optional
		Boolean array with the same shape as the image.  When provided, sources
		landing on ``False`` pixels are excluded from the affine fit in addition
		to sigma-clipping.

	Returns
	-------
	x_final : numpy.ndarray
		Corrected x pixel positions for all sources.
	y_final : numpy.ndarray
		Corrected y pixel positions for all sources.
	"""
	from scipy.optimize import least_squares
	def affine_model(params, x, y):
		"""Apply affine transform."""
		a, b, c, d, tx, ty = params
		x_new = a*x + b*y + tx
		y_new = c*x + d*y + ty
		return x_new, y_new

	def residuals(params, x, y, x_ref, y_ref):
		x_new, y_new = affine_model(params, x, y)
		return np.concatenate([(x_new - x_ref), (y_new - y_ref)])
	
	det_x = phot['x_init'].value
	det_y = phot['y_init'].value

	cat_x = phot['x_fit'].value
	cat_y = phot['y_fit'].value

	dist = np.sqrt((det_x-cat_x)**2 + (det_y-cat_y)**2)
	m,med,std  = sigma_clipped_stats(dist)
	if mask is None:
		good = (dist < med + 3*std)
	else:
		good = (dist < med + 3*std) & (mask[cat_y.astype(int),cat_x.astype(int)])


	# Initial guess = identity transform
	p0 = [1, 0, 0, 1, 0, 0]

	res = least_squares(residuals, p0, args=(det_x[good], det_y[good], cat_x[good], cat_y[good]))
	a, b, c, d, tx, ty = res.x

	x_warped, y_warped = affine_model(res.x,det_x, det_y)

	x_final = deepcopy(cat_x)
	x_final[~good] = x_warped[~good]

	y_final = deepcopy(cat_y)
	y_final[~good] = y_warped[~good]
	return x_final, y_final




class starkiller_image():
	def __init__(self,image,wcs,ref_filter='r',cal_maglim=[16,20],model_maglim=25,
				 psf_profile='gaussian',wcs_correction=True,trail=True,psf_align=True,
				 psf_preference='data',plot=True,run=True,numcores=5,rerun_cal=False,
				 calc_psf_only=False,fuzzy=True,bkg_sub=True,cal_gr_lims=[-0.5,0.8],
				 cutout_center=None,cutout_size=None,use_photutils=True,verbose=True):
		"""
		Initialise the starkiller_image pipeline and optionally run it immediately.

		Parameters
		----------
		image : numpy.ndarray
			2-D science image array.
		wcs : astropy.wcs.WCS
			World coordinate system for the image.
		ref_filter : str, optional
			Pan-STARRS filter to use for photometric calibration (default ``'r'``).
		cal_maglim : list of float, optional
			``[bright_limit, faint_limit]`` magnitude range for calibration sources
			(default ``[16, 20]``).
		model_maglim : float, optional
			Faint magnitude limit for sources included in the scene model
			(default ``25``).
		psf_profile : str, optional
			Functional form of the PSF model; ``'gaussian'`` or ``'moffat'``
			(default ``'gaussian'``).
		wcs_correction : bool, optional
			Whether to compute and apply a WCS offset correction (default
			``True``).
		trail : bool, optional
			Whether to estimate a satellite-trail angle for the PSF (default
			``True``).
		psf_align : bool, optional
			Whether to apply a fine PSF shift to the catalogue positions
			(default ``True``).
		psf_preference : str, optional
			Which PSF to prefer when both are available; ``'data'`` or
			``'model'`` (default ``'data'``).
		plot : bool, optional
			Whether to generate diagnostic plots during processing (default
			``True``).
		run : bool, optional
			Whether to execute the full pipeline immediately on construction
			(default ``True``).
		numcores : int, optional
			Number of parallel cores to use (default ``5``).
		rerun_cal : bool, optional
			Whether to rerun calibration source selection after the first PSF
			fit (default ``False``).
		calc_psf_only : bool, optional
			If ``True``, stop after computing the PSF without performing
			photometry (default ``False``).
		fuzzy : bool, optional
			Whether to detect and mask extended fuzzy sources (default
			``True``).
		bkg_sub : bool, optional
			Whether to subtract a 2-D background before processing (default
			``True``).
		cal_gr_lims : list of float, optional
			``[blue_limit, red_limit]`` colour range in ``g-r`` for selecting
			calibration stars (default ``[-0.5, 0.8]``).
		cutout_center : array-like, optional
			``[row, col]`` centre of a sub-image cutout.  When ``None`` the
			full image is used (default ``None``).
		cutout_size : array-like, optional
			``[nrows, ncols]`` size of the sub-image cutout (default ``None``).
		use_photutils : bool, optional
			Whether to use the photutils ePSF fitting pathway instead of the
			internal PSF fitter (default ``True``).
		verbose : bool, optional
			Whether to print progress messages (default ``True``).
		"""
		self.image = deepcopy(image)
		self._bkg_sub = bkg_sub
		self._clean_image()
		self.wcs = wcs
		self.ref_filter = ref_filter
		self.cal_maglim = cal_maglim
		self.cal_gr_lims = cal_gr_lims
		self.model_maglim = model_maglim
		self.fuzzy = fuzzy
		self.verbose = verbose
		self.numcores = numcores
		self.psf_profile = psf_profile.lower()
		self.psf_preference = psf_preference.lower()
		self._rerun_cal = rerun_cal
		self._calc_psf_only = calc_psf_only
		self._wcs_correction = wcs_correction
		self._psf_align = psf_align
		self.plot=plot
		self._use_photutils = use_photutils
		self.cutout_center = cutout_center
		self.cutout_size = cutout_size
		
		if run:
			self.run_image(trail)
	
	def _clean_image(self,sigma=3):
		rowsum = np.nanmean(self.image,axis=1)
		rowsum[rowsum==0] = np.nan
		m,med,std = sigma_clipped_stats(rowsum,maxiters=10)
		self.image[rowsum < med - sigma*std,:] = np.nan
		colsum = np.nanmean(self.image,axis=0)
		colsum[colsum==0] = np.nan
		m,med,std = sigma_clipped_stats(colsum,maxiters=10)
		self.image[:,colsum < med - sigma*std] = np.nan
		
		m,med,std = sigma_clipped_stats(self.image,maxiters=10)
		self.image[self.image < med - sigma*std]
		if self._bkg_sub:
			self._subtract_background()


	def _subtract_background(self):
		"""Estimate a 2-D median background and subtract it from the image in-place."""
		from astropy.stats import SigmaClip
		from photutils.background import Background2D, MedianBackground
		sigma_clip = SigmaClip(sigma=3.0)

		bkg_estimator = MedianBackground()
		self.bkg = Background2D(self.image, (50, 50), filter_size=(3, 3),
						   sigma_clip=sigma_clip, bkg_estimator=bkg_estimator).background
		self.image -= self.bkg

	
	
	def get_ps1(self):
		'''
		Test to see if it has updated...
		'''
		ra,dec = self.wcs.all_pix2world(self.image.shape[1]//2,self.image.shape[0]//2,0)
		coord = SkyCoord(ra,dec,unit='deg')
		foot = self.wcs.calc_footprint()
		diag = np.sqrt((foot[0,0] - foot[-2,0])**2 + (foot[0,1] - foot[-2,1])**2)
		cal_cat = query_ps1(coord.ra.deg,coord.dec.deg,diag/2,ref_filt=self.ref_filter,maglim=self.cal_maglim[1]+2)
		print('got cal_cat')
		if self.cutout_size is None:
			cat = query_ps1(coord.ra.deg,coord.dec.deg,diag/2,ref_filt=self.ref_filter,maglim=self.model_maglim)
			#cat = cat.loc[cat[self.ref_filter] > 0]

			x,y = self.wcs.all_world2pix(cat.ra.values,cat.dec.values,0)
			cat['x'] = x; cat['y'] = y
			cat['xint'] = np.round(x).astype(int)
			cat['yint'] = np.round(y).astype(int)
			# magic number for polar
			fx = [180,1860]
			fy = [0,970]
			ind = (x < np.max(fx)-50) & (x > np.min(fx) + 50) & (y < np.max(fy) - 50) & (y > np.min(fy) + 50)
			#cat = cat.iloc[ind]
			#ind = np.isfinite(self.image[cat['yint'].values,cat['xint'].values])
			#cat = cat.iloc[ind]
			cat = cat
		else:
			ra,dec = self.wcs.all_pix2world(self.cutout_center[1],self.cutout_center[0],0)
			coord = SkyCoord(ra,dec,unit='deg')
			rad_pix = np.max(self.cutout_size)
			c1 = self.wcs.all_pix2world(0,0,0)
			c2 = self.wcs.all_pix2world(rad_pix,rad_pix,0)
			diag = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
			cat = query_ps1(coord.ra.deg,coord.dec.deg,diag/2,ref_filt=self.ref_filter,maglim=self.model_maglim)
			cat = cat.loc[cat[self.ref_filter] > 0]
			print('got cat')

		x,y = self.wcs.all_world2pix(cat.ra.values,cat.dec.values,0)
		cat['x'] = x; cat['y'] = y
		cat['xint'] = np.round(x).astype(int)
		cat['yint'] = np.round(y).astype(int)

		x,y = self.wcs.all_world2pix(cal_cat.ra.values,cal_cat.dec.values,0)
		cal_cat['x'] = x; cal_cat['y'] = y
		cal_cat['xint'] = np.round(x).astype(int)
		cal_cat['yint'] = np.round(y).astype(int)

		gr = cal_cat['g'].values - cal_cat['r'].values
		ind = (gr > self.cal_gr_lims[0]) & (gr < self.cal_gr_lims[1]) & (cal_cat['g'].values > 0)

		self.cat = cat 
		self.cal_cat = cal_cat.iloc[ind]

	def _make_bright_mask(self):
		"""Build a boolean mask that is ``True`` for pixels significantly brighter than the background."""
		m,med,std = sigma_clipped_stats(self.image)
		self.bright_mask = self.image-med > 10 * std
	
	def _find_fuzzy_mask(self,sig_size=5):
		"""
		Identify extended, low-surface-brightness regions and store a dilated boolean mask.

		Parameters
		----------
		sig_size : float, optional
			Detection threshold in units of the background standard deviation
			(default ``5``).
		"""
		if self.fuzzy:
			m,med,std = sigma_clipped_stats(self.image)
			labeled, nr_objects = label(self.image > med + sig_size*std) 

			obj_size = []
			obj_counts = []
			for i in range(nr_objects):
				obj_size += [np.sum(labeled==i)]
				lab = (labeled==i) * 1.
				lab[lab==0] = np.nan
				obj_counts += [np.nanmean(lab * self.image)]
			obj_size = np.array(obj_size)
			obj_counts = np.array(obj_counts)

			m, med, std = sigma_clipped_stats(obj_size)
			ind = obj_size > med

			m, med, std = sigma_clipped_stats(obj_size[ind])

			fuzz_ind = np.where((obj_size > med + 10*std) & (obj_size < np.max(obj_size)))[0]
			fuzzy_mask = np.zeros_like(self.image)
			for i in range(len(fuzz_ind)):
				fuzzy_mask += labeled == fuzz_ind[i]
			fuzzy_mask = fuzzy_mask > 0
			fuzzy_mask = signal.fftconvolve(fuzzy_mask,np.ones((9,9)),mode='same')
			fuzzy_mask = fuzzy_mask > 0.5
		else:
			fuzzy_mask = np.zeros_like(self.image) > 1
		
		self.fuzzy_mask = fuzzy_mask

	def _fill_params(self):
		"""
		Fill the parameters for the cube simulator.
		"""
		self.y_length = 30 
		self.x_length = 30 
		self.angle = 0
		self.trail = 1
		self.wcs_shift = np.array([0,0,0])


	def _find_sources_cluster(self,trail):
		"""
		Find sources in the image using the cluster algorithm. 
		"""
		labeled, nr_objects = label(self.bright_mask > 0) 
		obj_size = []
		for i in range(nr_objects):
			obj_size += [np.sum(labeled==i)]
		obj_size = np.array(obj_size)
		image_size = self.image.shape[0] * self.image.shape[1]
		m,med, std = sigma_clipped_stats(obj_size)
		ind = obj_size > med
		m,med,std = sigma_clipped_stats(obj_size[ind])
		targs = np.where((obj_size > 10) & (obj_size<med + 3*std))[0]
		#good = []
		#for i in range(len(targs)):
		#	im = (labeled == targs[i]) * 1
		#	fuzz_overlap = np.nansum(im * self.fuzzy_mask)
		#	if fuzz_overlap == 0:
		#		good += [i]
		#good = np.array(good)
		#targs = targs[good]
		dy = np.zeros_like(targs)
		dx = np.zeros_like(targs)
		my = np.zeros_like(targs)
		mx = np.zeros_like(targs)
		sign = np.zeros_like(targs)
		for i in range(len(targs)):
			ty,tx = np.where(labeled == targs[i])
			#x,y = np.average(np.array([tx,ty]).T,weights=self.image[ty,tx],axis=0)
			my[i] = np.nanmedian(ty)
			mx[i] = np.nanmedian(tx)
			dy[i] = np.nanmax(ty) - np.nanmin(ty)
			dx[i] = np.nanmax(tx) - np.nanmin(tx)
			#if ty[np.argmin(tx)] > ty[np.argmax(tx)]:
			sign[i] = np.sign(ty[np.argmax(tx)] - ty[np.argmin(tx)])

		ind = ~sigma_clip(dx,sigma=2).mask & ~sigma_clip(dy,sigma=2).mask
		mx = mx[ind]; my = my[ind]; dx = dx[ind]; dy = dy[ind]; sign = sign[ind]
		targs = targs[ind]
		ind = ((mx < self.image.shape[1] - np.max(dx)/2) & (mx > np.max(dx)/2) &
				(my < self.image.shape[0] - np.max(dy)/2) & (my > np.max(dy)/2))
		mx = mx[ind]; my = my[ind]; dx = dx[ind]; dy = dy[ind]; sign = sign[ind]
		targs = targs[ind]
		if self.fuzzy_mask is not None:
			ind = self.fuzzy_mask[my,mx] == 0
			mx = mx[ind]; my = my[ind]
			targs = targs[ind]
		if len(mx) < 2:
			print(f'!!! Only {len(mx)} sources identified, coordinates will not fit!!!')
		self._dao_s = {'xcentroid':mx,'ycentroid':my}

		

		isox = np.sqrt(mx[:,np.newaxis] - mx[np.newaxis,:])
		isox[isox==0] = 900
		isox = np.nanmin(isox,axis=0)
		isoy = np.sqrt(my[:,np.newaxis] - my[np.newaxis,:])
		isoy[isoy==0] = 900
		isoy = np.nanmin(isoy,axis=0)
		iso = (isox > np.nanmedian(dx)) & (isoy > np.nanmedian(dy))
		#dx = dx[iso]; dy = dy[iso]; sign = sign[iso]
		buffer = np.nanmin([np.nanmedian(dy),np.nanmedian(dx)]) *0.6
		if buffer < 20:
			buffer = 20
		#print('buffered: ',buffer)

		self.y_length = int((np.nanmedian(dy)+buffer) / 2)
		self.x_length = int((np.nanmedian(dx)+buffer) / 2)
		self._trail_grad = np.nanmedian(dy/dx)
		self.angle = np.degrees(np.arctan2(self._trail_grad,1))
		if np.mean(sign) < 0:
			self.angle *= -1
		self.trail = np.nanmedian(np.sqrt(dy**2+dx**2))
		if not trail:
			self.trail = 1
			self.angle = 0
			
		ims = []
		for i in range(len(targs)):
			ims += [(labeled == targs[i])*1.0]
		ims = np.array(ims)
		im = (np.nansum(ims,axis=0) > 0) * 1.
		im[im==0] = np.nan
		self._lable_mask = im

		cims = []
		for i in range(len(ims)):
			myint = int(np.round(my[i],0))
			mxint = int(np.round(mx[i],0))
			cut = ims[i,myint - self.y_length//2:myint + self.y_length//2+1,mxint - self.x_length//2:mxint + self.x_length//2 +1]
			cims += [cut]
		cims = np.array(cims)

		s = (np.nanmedian(cims,axis=0) > 0.5) * 1.
		self._estimated_psf = (np.nanmedian(cims,axis=0) > 0.5) * 1.
	
	def _match_by_shift(self):
		"""
		Match the catalog to the sources in the image by shifting the catalog.
		"""
		#x, y, _ = self.wcs.all_world2pix(self.cat.ra.values,self.cat.dec.values,0,0)
		#sourcex = self._dao_s['xcentroid']; sourcey = self._dao_s['ycentroid']
		#x0 = [0,0,0]
		#bounds = [[-10,10],[-10,10],[0,np.pi/2]]
		#res = minimize(minimize_dist,x0,args=(x,y,sourcex,sourcey,self.image),method='Nelder-Mead',bounds=bounds)
		#if self.verbose:
		#	print('WCS shift: ',res.x)
		#self.wcs_shift = res.x
		s = self._estimated_psf
		im = self._lable_mask
		#if self.cal_maglim > 17:
		#	lim = 17
		#else:
		#	lim = self.cal_maglim
		#cat_ind = (self.cal_maglim[1] > self.cat[self.ref_filter].values) & (self.cal_maglim[0] < self.cat[self.ref_filter].values)
		cat_ind = self.cal_cat[self.ref_filter].values < 25 # hard code for now
		#x, y = self.wcs.all_world2pix(self.cat.ra.values[cat_ind],self.cat.dec.values[cat_ind],0)
		x, y = self.wcs.all_world2pix(self.cal_cat.ra.values,self.cal_cat.dec.values,0)
		# brute force it
		X,Y = np.meshgrid(np.arange(-50,50,5),np.arange(-50,50,5))
		positions = np.vstack([X.ravel(), Y.ravel(),X.ravel()*0]).T
		res = []
		for p in positions:
			guess = basic_image(p,x,y,im,s)
			res += [np.nansum(im - guess)]
		res = np.array(res)

		ind = np.argmin(res)
		x0 = positions[ind]
		bounds = [[x0[0]-10,x0[0]+10],[x0[1]-10,x0[1]+10],[0,np.pi/2]]
		res2 = minimize(minimize_cats,x0,args=(x,y,im,s),method='Nelder-Mead',bounds=bounds)
		if self.verbose:
			print('wcs shift: ',res2.x)

		self.wcs_shift = res2.x

	def _transform_coords(self,cat,plot=False):
		"""
		Transform the coordinates of the catalog to match the image.

		Parameters:
		-----------
		plot : boolean
			Whether or not to plot the transformed coordinates.
		"""
		if plot is None:
			plot = self.plot

		x, y = self.wcs.all_world2pix(cat.ra.values,cat.dec.values,0)

		xx,yy = transform_coords(x,y,self.wcs_shift,self.image)

		#finite = signal.fftconvolve(np.isfinite(self.image),np.ones([self.x_length,self.y_length]),mode='same') > 0.
		finite = np.isfinite(self.image)
		#overlap = (yy+self.y_length > 0) & (yy+self.y_length < finite.shape[0]) & (xx+self.x_length > 0) & (xx+self.x_length < finite.shape[1])
		overlap = (yy > 0) & (yy < finite.shape[0]) & (xx > 0) & (xx < finite.shape[1])
		#ind = finite[(yy+self.y_length).astype(int)[overlap],(xx+self.x_length).astype(int)[overlap]] # precision not needed here
		ind = finite[(yy).astype(int)[overlap],(xx).astype(int)[overlap]] # precision not needed here
		overlap[overlap] = ind
		cat['x'] = xx
		cat['y'] = yy
		cat = cat.iloc[overlap]
		cat['xint'] = np.round(cat['x'].values,0).astype(int)
		cat['yint'] = np.round(cat['y'].values,0).astype(int)

		if plot:
			plt.figure()
			plt.title('Matching image sources with catalog')
			plt.imshow(self.image,vmin=np.nanpercentile(self.image,16),vmax=np.nanpercentile(self.image,85),cmap='gray',origin='lower')
			plt.plot(x,y,'C3x',label='Catalog')
			plt.plot(cat['x'],cat['y'],'C1*',label='Corrected')
			plt.legend()

		return cat
		
	def complex_isolation_cals(self,xdist=8,dmag=2):
		"""
		Isolate the calibration sources in the image. Creates a cals variable to track calibration sources.

		Parameters:
		-----------
		xdist : float
			Maximum distance in pixels between sources.
		dmag : float
			Difference in magnitude between sources to consider.

		"""
		mags = self.cal_cat[self.ref_filter].values
		mag_ind = self.cal_maglim[1] + dmag > mags
		mags = mags[mag_ind]
		cat = deepcopy(self.cal_cat).iloc[mag_ind]

		xx = self.cal_cat.xint.values[mag_ind]; yy = self.cal_cat.yint.values[mag_ind]
		ang = np.radians(self.angle)
		cx = self.image.shape[1]/2; cy = self.image.shape[0]/2
		xxx = cx + ((xx-cx)*np.cos(ang)-(yy-cy)*np.sin(ang))
		yyy = cy + ((xx-cx)*np.sin(ang)+(yy-cy)*np.cos(ang))

		
		dmags = mags[:,np.newaxis] - mags[np.newaxis,:]
		ind = dmags > dmag

		dy = abs(yyy[:,np.newaxis] - yyy[np.newaxis,:])
		dy[dy==0] = 1e3
		dy[ind] = 1e3

		dx = abs(xxx[:,np.newaxis] - xxx[np.newaxis,:])
		dx[dx==0] = 1e3
		dx[ind] = 1e3

		iy,ix = np.where(dx > xdist)
		dy[iy,ix] = 1e3
		iy,ix = np.where(dx < xdist)
		ii = dy[iy,ix] > self.trail*1.2
		dx[iy[ii],ix[ii]] = 1e3
		dy[iy[ii],ix[ii]] = 1e3
		
		if self.trail > 1:
			isoind = (np.nanmin(dy,axis=0) > self.trail*1.2) & (mags < self.cal_maglim[1]) & (mags > self.cal_maglim[0])
		else:
			isoind = (np.nanmin(dy,axis=0) > xdist) & (mags < self.cal_maglim[1]) & (mags > self.cal_maglim[0])

		ind = ((cat['x'].values.astype(int) < self.image.shape[1] - self.x_length) & 
				(cat['y'].values.astype(int) < self.image.shape[0] - self.y_length) & 
				(cat['x'].values.astype(int) > 0 + self.x_length) & 
				(cat['y'].values.astype(int) > 0 + self.y_length))
		indo = np.isfinite(self.image[cat['y'].values[ind].astype(int),cat['x'].values[ind].astype(int)])
		ind[ind] = indo
		mag_ind
		self.cat['fuzz'] = 0
		if np.nansum(self.fuzzy_mask) > 0:
			fuzz = (self.fuzzy_mask[cat['y'].values[ind].astype(int),cat['x'].values[ind].astype(int)] == 1)
			#cat['fuzz'].iloc[ind] = fuzz * 1
			#ind[ind] = ~fuzz

		index = deepcopy(mag_ind)
		index[index] = ind & isoind
		self.cal_cat['cal_source'] = 0 
		self.cal_cat.loc[index,'cal_source'] = 1
		self.cal_cat = self.cal_cat.iloc[index]
		
	def _isolate_cals(self):
		"""
		Isolate the calibration sources in the image. Creates a cal_cuts variable containing the cutouts and good_cals.

		"""
		cals = self.cal_cat
		star_cuts, good = get_star_cuts(self.x_length,self.y_length,self.image,cals)
		mags = cals[self.ref_filter].values
		ind = (mags < self.cal_maglim[1]) & (mags > self.cal_maglim[0])
		if np.sum(ind) < 1:
			m = f'{np.sum(ind)} targets above the mag lim, limit must be increased.\n Available mags: {cals[self.ref_filter].values}'
			raise ValueError(m)
		self.cal_cat.loc[(self.cal_cat['cal_source'].values == 1),'cal_source'] = good & ind
		self.cal_cuts = star_cuts[good & ind]
		
	def make_psf(self,fine_shift=None,data_containment_lim=0.95): 
		"""
		NOT CURRENTLY USED
		Make the psf for the image. Adds in the psf and psf_param variables.

		Parameters:
		-----------
		fine_shift : boolean
			Whether or not to do a fine shift using the psf offsets.
		data_containment_lim : float
			Containment limit of the data psf measured as percentage of total PSF flux.
		"""
		if fine_shift is None:
			fine_shift = self._psf_align
		self._isolate_cals()
		#good = self.good_cals
		ct = self.cal_cuts#[good]
		#ct -= np.nanmedian(ct,axis=(1,2))[:,np.newaxis,np.newaxis]

		psf = create_psf(self.x_length*2+1,self.y_length*2+1,self.angle,self.trail)

		params = []
		psfs = []
		shifts = []
		ind = len(ct)
		#if ind > 5:
		#	ind = 5
		'''for j in range(ind):
									psf.fit_psf(ct[j])
									psf.line()
									if self.psf_profile == 'moffat':
										params += [np.array([psf.alpha,psf.beta,psf.length,psf.angle])]
										shifts += [np.array([psf.source_x,psf.source_y])]
									elif self.psf_profile =='gaussian':
										params += [np.array([psf.stddev,psf.length,psf.angle])]
										shifts += [np.array([psf.source_x,psf.source_y])]
									psfs += [psf.longpsf]'''

		indo = np.arange(ind)
		if len(ct)>0:
			params, shifts = zip(*Parallel(n_jobs=self.numcores)(delayed(parallel_psf_fit)(ct[i],psf,self.psf_profile) for i in indo))
		elif len(ct) == 0:
			m = 'No suitable calibration sources!'
			raise ValueError(m)

		params = np.array(params)
		shifts = np.array(shifts)

		self.psf_param = np.nanmedian(params,axis=0)
		if 'moffat' in self.psf_profile:
			self.psf = create_psf(x=self.x_length*2+1,y=self.y_length*2+1,alpha=self.psf_param[0],
								  beta=self.psf_param[1],length=self.psf_param[2],angle=self.psf_param[3],
								  psf_profile=self.psf_profile)
		elif 'gaussian' in self.psf_profile:
			self.psf = create_psf(x=self.x_length*2+1,y=self.y_length*2+1,stddev=self.psf_param[0],length=self.psf_param[1],angle=self.psf_param[2],
								  psf_profile=self.psf_profile)
		self.psf.generate_line_psf()
		if fine_shift:
			self._fine_psf_shift(shifts)
		self.cat['containment'] = 1
		#self.complex_isolation_cals()
		#self._psf_isolation()
		#self._isolate_cals()

		ind = self.cat.containment.values[self.cat.cal_source.values > 0] > data_containment_lim
		if (sum(ind) >= 2):# | (self._force_flux_correction):
			#cuts = deepcopy(self.cal_cuts[ind])
			#cuts[cuts == 0] = np.nan
			self.psf.make_data_psf(self.cal_cuts[ind])
			self._check_psf_quality()
		else:
			m = f'!!! No sources within the data psf containment limit of {data_containment_lim} !!!\nData PSF can not be constructed.'
			print(m)


	def _check_psf_quality(self):
		"""
		Compare the data psf to the model psf. Prints a warning and updates the psf_profile if the difference is large.
		If the psf_preference is set to 'model' then the model psf is used regardless of the difference.
		"""
		diff = np.sum(abs(self.psf.data_psf-self.psf.longpsf))
		if (diff > 0.1) & (self.psf_preference=='data'):
			m = (f"!!! Large difference of {np.round(diff,2)} between model_psf and data_psf!!!\nUsing the data_psf, override by setting psf_preference='model'")
			print(m)
			self.psf_profile += ' data'
			self.psf.psf_profile += ' data'


	def _fine_psf_shift(self,shifts,plot=None):
		"""
		NOT CURRENTLY USED
		Perform a fine shift to the catalog positions based on the psf shifts.

		Parameters:
		-----------
		shifts : numpy array
			Shifts to apply to the psf.
		plot : boolean
			Whether or not to plot the new alignment.
		"""
		if plot is None:
			plot = self.plot
		if self.verbose:
			print('Calculating PSF coord shift')
		sources = deepcopy(self.cat.loc[self.cat['cal_source'] == 1])#.iloc[self.good_cals])
		catx = deepcopy(sources['xint'].values); caty = deepcopy(sources['yint'].values)
		sourcex = catx + shifts[:,0]; sourcey = caty + shifts[:,1]
		self.cat['x_fit'] = 0; self.cat['y_fit'] = 0
		self.cat['x_fit'].iloc[self.cat['cal_source'].values == 1] = sourcex
		self.cat['y_fit'].iloc[self.cat['cal_source'].values == 1] = sourcey

		bounds = [[-15,15],[-15,15],[0,0]]
		x0 = [0,0,0]
		res = minimize(minimize_dist,x0,args=(catx,caty,sourcex,sourcey,self.image),method='Nelder-Mead',bounds=bounds)
		if self.verbose:
			print('PSF shift: ',res.x)

		xx,yy = transform_coords(self.cat.x.values,self.cat.y.values,res.x,self.image)
		self.cat['x'] = xx
		self.cat['y'] = yy
		self.cat['xint'] = (self.cat['x'].values + 0.5).astype(int)
		self.cat['yint'] = (self.cat['y'].values + 0.5).astype(int)

		if plot:
			plt.figure()
			plt.title('Matched cube')
			plt.imshow(self.image,vmin=np.nanpercentile(self.image,16),vmax=np.nanpercentile(self.image,85),cmap='gray',origin='lower')
			plt.plot(self.cat['x'],self.cat['y'],'C1*',label='Catalog')

	def _psf_isolation(self,dmag = 2,containment=0.9,overlap_lim=0.2):
		"""
		NOT CURRENTLY USED
		Identifies the isolated sources based on the PSF. Creates a cal_cuts variable containing the cutouts and good_cals.

		Parameters:
		-----------
		dmag : float
			Difference in magnitude between sources to consider.
		containment : float
			Containment limit of the psf measured as percentage of total PSF flux.
		overlap_lim : float
			Overlap limit of the psf measured as percentage of total PSF flux.
		"""
		psf_mask = (self.psf.longpsf > 1e-10) * 1.
		y = deepcopy(self.cat.yint.values); x = deepcopy(self.cat.xint.values)
		ind = (y > 0) & (y < self.image.shape[0]) & (x > 0) & (x < self.image.shape[1])
		x= x[ind];y=y[ind]
		m = self.cat[self.ref_filter].values[ind]
		source_mask = np.zeros((len(y),self.image.shape[0],self.image.shape[1]))
		source_mask[np.arange(0,len(y)),y,x] = 1
		source_mask = signal.fftconvolve(source_mask,np.array([psf_mask]),mode='same')
		#source_mask[source_mask<1e-3] = 0
		source_mask = (source_mask > 1e-3).astype(int)

		sources = np.zeros((len(y),self.image.shape[0],self.image.shape[1]))
		sources[np.arange(0,len(y)),y,x] = 1
		sources = signal.fftconvolve(sources,np.array([self.psf.longpsf]),mode='same')

		isolated = []
		cont = []
		for i in range(len(x)):
			mask_overlap = (source_mask[0] + source_mask) == 2
			mask_overlap[0] = 0
			psf_overlap = np.nansum(sources * mask_overlap,axis=(1,2)) > overlap_lim

			mag_ind = m[i] - m > dmag

			im = deepcopy(self.image)
			im[np.isfinite(im)] = 1
			# Check if the source is mostly contained in the image
			c = np.nansum(im * sources[i])
			cont += [c]
			if (mag_ind & psf_overlap).any():
				isolated += [False]
			else:
				if c > containment:
					isolated += [True]
				else:
					isolated += [False]

		isolated = np.array(isolated)
		isolated[m > self.cal_maglim] = False
		indo = deepcopy(ind)
		indo[ind] = isolated

		indo = indo & (self.cat['fuzz'].values == 0) 
		self.cat['containment'] = 0
		self.cat.loc[ind,'containment'] = cont

		if sum(indo) > 0:
			self.cat.cal_source = 0
			self.cat.loc[indo,'cal_source'] = 1
		else:
			print('!!! No PSF isolated sources, using "complex sources" that are contained !!!')
			ind = self.cat.cal_source.values == 1
			c_ind = self.cat.containment.values > containment
			t_ind = c_ind & ind
			if sum(t_ind) > 0:
				self.cat.cal_source = 0
				self.cat.loc[t_ind,'cal_source'] = 1
			else:
				print('!!! No contained sources, just using complex sources ignoring containment !!!')
			if self.psf_preference == 'data':
				self.psf_preference = 'model'
				print('!!! Messy field, dissabling data psf !!!')
		self.cat = self.cat.loc[self.cat['containment'] > 0]



	def _psf_contained_check(self):
		"""
		Finds what sources are in the data cube bsed on the PSF
		"""
		pad = int(self.trail/2)
		psf_mask = (self.psf.longpsf > 1e-10) * 1.
		y = deepcopy(self.cat.yint.values)+pad; x = deepcopy(self.cat.xint.values)+pad
		indo = np.zeros_like(x)
		padded = np.pad(self.image,pad,constant_values=np.nan)
		ind = (y > 0) & (y < padded.shape[0]) & (x > 0) & (x < padded.shape[1])
		x= x[ind];y=y[ind]

		sources = np.zeros((len(y),padded.shape[0],padded.shape[1]))
		sources[np.arange(0,len(y)),y,x] = 1
		sources = signal.fftconvolve(sources,np.array([psf_mask]),mode='same')
		sources[sources<0.1] = 0
		masked = sources * padded[np.newaxis,:,:]
		contained = np.nansum(masked,axis=(1,2)) > 0
		indo[ind] = contained
		self.cat = self.cat.iloc[indo > 0]
		self._isolate_cals()	
		
	
	def all_phot(self,radius=5,plot=None):
		"""
		Extracts the spectra of all sources in the cube. Adds in the specs variable.
		"""
		if plot is None:
			plot = self.plot
		data_psf = None
		if 'data' in self.psf_profile:
			data_psf = self.psf.data_psf
		flux, residual, cat_off = get_phot(self.cat,self.image,self.x_length,self.y_length,
											 self.psf,num_cores=self.numcores,
											 data_psf=data_psf,pos_bound=radius)
		self.cat = cat_off
		self.flux = flux
		#self._sort_cat_filts_mags()

	
	def calc_zeropoint(self,plot=False):
		"""
		Compute the photometric zeropoint from isolated calibration sources.

		The result is stored in ``self.zp`` (median zeropoint) and
		``self.std_zp`` (scatter).

		Parameters
		----------
		plot : bool, optional
			Whether to produce a zeropoint diagnostic plot (default ``False``).
		"""
		cal = self.cat.loc[self.cat['cal_source'] == 1]
		sysmag = -2.5*np.log10(cal['psf_flux'])
		m,med,std = sigma_clipped_stats(cal[self.ref_filter]-sysmag,sigma=2,maxiters=10)
		self.zp = med
		self.std_zp = std
		if plot:
			plt.figure()
			plt.fill_between(cal['r'],med-std,med+std,color='C1',alpha=0.1)
			plt.plot(cal['r'],cal['r']-sysmag,'.')
			plt.axhline(med,color='C1')
			plt.xlabel(f'{self.ref_filter} mag')
			plt.ylabel(f'Zeropoint ({self.ref_filter} - sysmag)')

	
	def make_scene(self):
		"""Construct a simulated scene image from the fitted PSF and source catalogue."""
		if 'data' in self.psf_profile:
			data = True
		else:
			data = False
		scene = cube_simulator(self.image,psf=self.psf,catalog=self.cat,datapsf=data)
	
	def make_epsf(self,oversample=2,progress_bar=True,epsf_noise_lim=5e-4):
		"""
		Build an effective PSF (ePSF) from isolated calibration star cutouts.

		The resulting ePSF model is stored in ``self.epsf``.

		Parameters
		----------
		oversample : int, optional
			Oversampling factor for the ePSF grid (default ``2``).
		progress_bar : bool, optional
			Whether to display a progress bar during ePSF fitting (default
			``True``).
		epsf_noise_lim : float, optional
			Noise floor threshold below which ePSF pixels are set to zero
			(default ``5e-4``).
		"""
		cals = self.cal_cat
		cuts = self.cal_cuts
		cuts[np.isnan(cuts)] = 0
		init = Table()
		init['x_init'] = [cuts.shape[2]//2]
		init['y_init'] = [cuts.shape[1]//2]
		bkgstat = SExtractorBackground() #MMMBackground()
		localbkg_estimator = LocalBackground(5, 10, bkgstat)

		pos = np.array([(cals['x'] - cals['xint']).values + cuts.shape[2]//2,(cals['y'] - cals['yint']).values + cuts.shape[1]//2])

		stars = []
		for i in range(len(cuts)):
			stars += [EPSFStar(cuts[i],cutout_center=pos[:,i])]
		stars = EPSFStars(stars)
		if self.trail > 2:
			normrad = self.trail
		else:
			normrad = 5
		normrad = 10
		epsf_builder = EPSFBuilder(oversampling=oversample,norm_radius=normrad,recentering_boxsize=(5, 5), 
								   maxiters=40, progress_bar=progress_bar)
		epsf, fitted_stars = epsf_builder(stars)
		#epsf.data[self.epsf.data < epsf_noise_lim] = 0
		self.epsf = epsf
	
	
	def psf_phot(self,x=None,y=None,flux=None,image=None,x_bound=5,y_bound=5,group_sep=20,background=False):
		"""
		Run PSF photometry on the image using the fitted ePSF.

		Parameters
		----------
		x : array-like, optional
			Initial x pixel positions of sources.  Defaults to
			``self.cat['x']``.
		y : array-like, optional
			Initial y pixel positions of sources.  Defaults to
			``self.cat['y']``.
		flux : array-like, optional
			Initial flux guesses for sources.  If ``None``, photutils
			estimates the flux internally.
		image : numpy.ndarray, optional
			Image to perform photometry on.  Defaults to ``self.image``.
		x_bound : float, optional
			Maximum allowed positional offset in x during fitting (default
			``5``).
		y_bound : float, optional
			Maximum allowed positional offset in y during fitting (default
			``5``).
		group_sep : float, optional
			Minimum separation in pixels used to group nearby sources
			(default ``20``).
		background : bool, optional
			Whether to estimate and subtract a local background during fitting
			(default ``False``).

		Returns
		-------
		phot : astropy.table.Table
			Photometry results table from photutils.
		psfphot : photutils.psf.PSFPhotometry
			The fitted PSFPhotometry object.
		"""
		if (x is None) | (y is None):
			x = self.cat['x'].values
			y = self.cat['y'].values
		if image is None:
			image = self.image
			image[np.isnan(image)] = 0
		grouper = SourceGrouper(min_separation=20)

		fit_shape = (self.cal_cuts.shape[1],self.cal_cuts.shape[2])
		self._psf_fit_shape = fit_shape

		if background:
			bkgstat = MMMBackground()
			bkg = LocalBackground(5, 10, bkgstat)
		else:
			bkg = None

		psfphot = PSFPhotometry(self.epsf, fit_shape, finder=None,aperture_radius=10,
								 xy_bounds=(x_bound,y_bound),localbkg_estimator=bkg,grouper=grouper)

		init_params = QTable()
		init_params['x'] = x
		init_params['y'] = y
		if flux is not None:
			init_params['flux'] = flux

		phot = psfphot(image,init_params=init_params)
		return phot, psfphot


	def photutils_sequence(self,plot=None):
		"""
		Execute the full photutils-based calibration, zeropoint, and scene-subtraction sequence.

		Calibration sources are measured with ``psf_phot``, the photometric
		zeropoint is derived, all catalogue sources are fitted, and the
		simulated star field is subtracted to produce ``self.sim`` and
		``self.diff``.

		Parameters
		----------
		plot : bool, optional
			Whether to generate a zeropoint diagnostic plot.  Defaults to
			``self.plot``.
		"""
		if plot is None:
			plot = self.plot

		cal = self.cal_cat
		x = deepcopy(cal['x'].values)
		y = deepcopy(cal['y'].values)

		phot, psfphot = self.psf_phot(x=x,y=y)

		sysmag = -2.5*np.log10(phot['flux_fit'])
		m,med,std = sigma_clipped_stats(cal[self.ref_filter]-sysmag,maxiters=10)
		self.zp = med
		self.std_zp = std

		if plot:
			plt.figure()
			plt.fill_between([self.cal_maglim[0]-.1,self.cal_maglim[1]+.1],med-std,med+std,color='C1',alpha=0.1)
			plt.plot(cal[self.ref_filter],cal[self.ref_filter]-sysmag,'.')
			label = r'zp$_{\rm sys}= $' + str(np.round(self.zp,2)) + r'$\pm$' + str(np.round(self.std_zp,2))
			plt.axhline(med,color='C1',label=label)
			plt.xlabel(f'{self.ref_filter} mag')
			plt.ylabel(f'Zeropoint ({self.ref_filter} - sysmag)')
			plt.xlim(self.cal_maglim[0]-.1,self.cal_maglim[1]+.1)
			plt.legend()

		m,xmed,xstd = sigma_clipped_stats((phot['x_init']-phot['x_fit']))
		m,ymed,ystd = sigma_clipped_stats((phot['y_init']-phot['y_fit']))

		magind = self.cat[self.ref_filter].values < self.model_maglim
		x = (deepcopy(self.cat['x'].values) - xmed)[magind]
		y = (deepcopy(self.cat['y'].values) - ymed)[magind]
		flux = (10**((self.cat[self.ref_filter].values-self.zp)/-2.5))[magind]
		ind = np.ones_like(x) > 0
		if (self.cutout_center is not None): 
			if (self.cutout_size is not None):
				x = x - self.cutout_center[1] + self.cutout_size[1]//2
				y = y - self.cutout_center[0] + self.cutout_size[0]//2
				ind = (x > 1) & (x < self.cutout_size[1] - 1) & (y > 1) & (y < self.cutout_size[0] - 1)
				x = x[ind]; y = y[ind]; flux = flux[ind]

				xlow = int(self.cutout_center[1] - self.cutout_size[1]//2)
				xup = int(self.cutout_center[1] + self.cutout_size[1]//2 + 1)
				ylow = int(self.cutout_center[0] - self.cutout_size[0]//2)
				yup = int(self.cutout_center[0] + self.cutout_size[0]//2 + 1)
				image = self.image[ylow:yup,xlow:xup]
				fuzzy_mask = self.fuzzy_mask[ylow:yup,xlow:xup]
			else:
				m = 'A cutout size must be provided with a cutout center!'
				raise ValueError(m)
		else:
			image = self.image
			fuzzy_mask = self.fuzzy_mask

		phot2, psfphot2 = self.psf_phot(x=x,y=y,flux=flux,image=image,x_bound=xstd*5,y_bound=ystd*5)
		test = psfphot2.make_model_image(image.shape)
		self._psfphot2 = psfphot2
		self._phot2 = phot2

		x_final, y_final = affine_positions(phot2,~fuzzy_mask)
		ind_final = deepcopy(magind)
		ind_final[~ind] = False
		self.cat['x_final'] = np.nan#x_final
		self.cat['y_final'] = np.nan#y_final
		if (self.cutout_center is None):
			self.cat.loc[ind_final,'x_final'] = x_final
			self.cat.loc[ind_final,'y_final'] = y_final
		else:
			self.cat.loc[ind_final,'x_final'] = x_final + self.cutout_center[1] - self.cutout_size[1]//2
			self.cat.loc[ind_final,'y_final'] = y_final + self.cutout_center[0] - self.cutout_size[0]//2

		psfphot2._model_image_parameters[1]['flux'] = flux
		psfphot3 = deepcopy(psfphot2)
		psfphot3._model_image_parameters[1]['x_0'] = x_final
		psfphot3._model_image_parameters[1]['y_0'] = y_final
		self.sim = psfphot3.make_model_image(image.shape)
		self.diff = psfphot3.make_residual_image(image,psf_shape=self._psf_fit_shape)

	
	def plot_diff(self,xlim=None,ylim=None,percentile=[16,99.9],savename=None,include_sim=True):
		"""
		Plot the raw image, simulated scene, and star-subtracted residual side by side.

		Parameters
		----------
		xlim : list of int, optional
			``[xmin, xmax]`` pixel range to display.  Defaults to the full
			image width.
		ylim : list of int, optional
			``[ymin, ymax]`` pixel range to display.  Defaults to the full
			image height.
		percentile : list of float, optional
			``[low, high]`` percentiles used to set the colour stretch
			(default ``[16, 99.9]``).
		savename : str, optional
			File path at which to save the figure.  If ``None`` the figure is
			not saved (default ``None``).
		include_sim : bool, optional
			Whether to include the simulated image panel (default ``True``).
		"""
		image = self.image
		cx = self.cat['x_final'].values
		cy = self.cat['y_final'].values
		if self.cutout_center is not None:
			xlow = int(self.cutout_center[1] - self.cutout_size[1]//2)
			xup = int(self.cutout_center[1] + self.cutout_size[1]//2 + 1)
			ylow = int(self.cutout_center[0] - self.cutout_size[0]//2)
			yup = int(self.cutout_center[0] + self.cutout_size[0]//2 + 1)
			image = image[ylow:yup,xlow:xup]
			cx = cx - self.cutout_center[1] + self.cutout_size[1]//2
			cy = cy - self.cutout_center[0] + self.cutout_size[0]//2

		if xlim is None:
			xlim = [0,image.shape[1]]
		if ylim is None:
			ylim = [0,image.shape[0]]
		xlim = np.array(xlim); ylim = np.array(ylim)
		xlim[0] -= 0.5; xlim[1] += 0.5
		ylim[0] -= 0.5; ylim[1] += 0.5

		norm = ImageNormalize(self.diff[ylim[0]:ylim[1],xlim[0]:xlim[1]], 
							  interval=AsymmetricPercentileInterval(percentile[0],percentile[1]),
							  stretch=SqrtStretch())
		if include_sim:
			panels = 3
			plt.figure(figsize=(9,5))
		else:
			panels = 2
			plt.figure(figsize=(12,4))
		
		plt.subplot(1,panels,1)
		plt.title('Raw image')
		plt.imshow(image,norm=norm,cmap='grey')
		plt.scatter(cx,cy,marker='o',ec='C1',fc='None',s=50)
		plt.ylim(ylim[0],ylim[1])
		plt.xlim(xlim[0],xlim[1])
		if include_sim:
			plt.subplot(1,panels,2)
			plt.title('Simulated image')
			plt.imshow(self.sim,norm=norm,cmap='grey')
			plt.scatter(cx,cy,marker='o',ec='C1',fc='None',s=50)
			plt.ylim(ylim[0],ylim[1])
			plt.xlim(xlim[0],xlim[1])
			plt.subplot(1,panels,3)
		else:
			plt.subplot(1,panels,2)
		plt.title('Starkilled image')
		plt.imshow(self.diff,norm=norm,cmap='grey')
		plt.ylim(ylim[0],ylim[1])
		plt.xlim(xlim[0],xlim[1])
		plt.scatter(cx,cy,marker='o',ec='C1',fc='None',s=50)
		plt.tight_layout()
		if savename is not None:
			plt.savefig(savename,bbox_inches='tight',dpi=300)

		
	
	def run_image(self,trail=True):
		"""
		Run the cube reduction. This is the main function that calls all the other functions.

		Parameters:
		-----------
		trail : boolean
			Whether or not to estimate the trail angle. Turn off if this is a siderially tracked cube
		"""
		try:
			if self.verbose:
				print('making bright mask')
			self._make_bright_mask()
			if self.verbose:
				print('making fuzzy mask')
			self._find_fuzzy_mask()
			if self.verbose:
				print('getting ps1')
			self.get_ps1()

			if self._wcs_correction:
				if self.verbose:
					print('Finding WCS offset')
				self._find_sources_cluster(trail)
				self._match_by_shift()
			else:
				self._fill_params()

			if self.verbose:
				print('Coord transform')			
			self.cal_cat = self._transform_coords(self.cal_cat)
			self.cat = self._transform_coords(self.cat)

			#self._identify_cals()
			if self.verbose:
				print('Get calibration sources')
			self.complex_isolation_cals()
			self._isolate_cals()
			if self._use_photutils:
				self.make_epsf()
				self.photutils_sequence()
				
			else:
				self.make_psf()

				if self._rerun_cal:
					if self.verbose:
						print('Rerunning cal selection with psf sources')
					#self._psf_isolation()
					self.make_psf()#fine_shift=True)
					#self._psf_isolation()
				self._psf_contained_check()
				if self.verbose:
					print('Made PSF')
				if self._calc_psf_only:
					print('Exiting')
					return
		except Exception:
			print(traceback.format_exc())


				
