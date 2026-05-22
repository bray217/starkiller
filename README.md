[![DOI](https://zenodo.org/badge/616778733.svg)](https://doi.org/10.5281/zenodo.14189822)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/CheerfulUser/starkiller?style=social)](https://github.com/CheerfulUser/starkiller/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/CheerfulUser/starkiller)](https://github.com/CheerfulUser/starkiller)

# Starkiller: Removing stars and satellites from IFU data

Starkiller removes stellar contamination from integral field unit (IFU) spectroscopic datacubes. It works by:

1. Fetching a source catalogue (Gaia DR3 by default) for the observed field
2. Extracting PSF-photometry spectra for each star in the cube
3. Matching each spectrum to a stellar model (CK or ESO library) accounting for interstellar extinction
4. Normalising the best-matching model to catalogue photometry (Gaia G by default)
5. Injecting each modelled star into a synthetic scene cube using the measured PSF
6. Applying a secondary flux correction to calibration sources
7. Subtracting the synthetic scene from the input cube to produce a differenced FITS file

PSFs can be trailed (for non-sidereal tracking), Moffat or Gaussian, and can be constructed from the data itself. Satellite streaks can also be detected and removed.

---

## Installation

```bash
pip install git+https://github.com/CheerfulUser/starkiller.git
```

**Requirements:** Python ≥ 3.8, NumPy < 2.0 (the `pysynphot` and `extinction` dependencies are compiled against NumPy 1.x and will fail to import under NumPy 2.x).

---

## Usage

### Non-siderially tracked data (e.g. comets, asteroids)

```python
from starkiller import starkiller

ifu = starkiller(
    file='path/to/cube.fits',
    cal_maglim=20,    # brightest mag limit for PSF calibration stars
    run=True,
)
```

![plot](./figs/2I_MUSE_example.png)

### Siderially tracked data

```python
from starkiller import starkiller

ifu = starkiller(
    file='path/to/cube.fits',
    trail=False,           # no trailing — sidereal tracking
    numcores=7,
    savepath='output/',
    spec_catalog='ck+',    # extended CK model grid (recommended for faint stars)
    psf_profile='gaussian',
    cal_maglim=18,
    model_maglim=25,
    psf_preference='data', # build PSF from bright calibration stars in the cube
)
```

![plot](./figs/Nebula_subtraction_example_mean.png)

### Siderially tracked data with an extended source (e.g. nebula, galaxy)

Set `fuzzy=True` to detect and mask any extended emission before PSF fitting, preventing it from contaminating the stellar models.

```python
from starkiller import starkiller

ifu = starkiller(
    file='path/to/cube.fits',
    trail=False,
    fuzzy=True,            # mask extended emission (nebula, galaxy) before PSF fitting
    psf_preference='data',
    spec_catalog='ck+',
)
```

### Satellite streak removal

```python
from starkiller import starkiller

ifu = starkiller(
    file='path/to/cube.fits',
    spec_catalog='ck',
    savepath='output/',
    satellite=True,        # detect and model satellite streaks
)
```

![plot](./figs/satellite_subtraction_example.png)

The satellite spectrum is modelled as a highly-extincted solar spectrum and subtracted, recovering the underlying source.

![plot](./figs/Satellite_spec_subtraction_two.png)

---

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `trail` | `True` | Estimate and model trailing angle. Set `False` for sidereal data. |
| `cal_maglim` | `18` | Faint magnitude limit for PSF calibration stars. |
| `model_maglim` | `25` | Faint magnitude limit for stars included in the scene model. |
| `spec_catalog` | `'ck'` | Stellar spectral library: `'ck'` (Castelli-Kurucz), `'ck+'` (extended), `'eso'`. |
| `psf_profile` | `'gaussian'` | PSF profile shape: `'gaussian'` or `'moffat'`. |
| `psf_preference` | `'data'` | Build PSF from `'data'` (calibration stars in cube) or `'model'` (analytic). |
| `fuzzy` | `False` | Mask extended sources (nebulae, galaxies) before PSF fitting. |
| `satellite` | `False` | Detect and subtract satellite streaks. |
| `numcores` | `5` | Number of CPU cores for parallel spectral fitting. |
| `wavelength_sol` | `'air'` | Wavelength frame: `'air'` or `'vacuum'`. |
| `background` | `False` | If `True`, subtract a 2D sky background before modelling. |

---

## Accessing results

After a run the `starkiller` object exposes the key products as attributes:

```python
ifu = starkiller('cube.fits', run=True)

ifu.diff_cube    # differenced cube (stars removed)
ifu.scene        # synthetic star scene that was subtracted
ifu.ebvs         # E(B-V) extinction values per star
ifu.cat          # source catalogue used (pandas DataFrame)
ifu.psf          # fitted PSF object
```

Extinction values can be used directly for further analysis:

![plot](./figs/Extinction_nebula_ck+.png)

---

## Output FITS file

The differenced cube is saved as a FITS file with four extensions:

| Extension | Contents |
|---|---|
| 0 | Primary HDU (copy of input header) |
| 1 | Differenced cube (input − scene) |
| 2 | Error cube |
| 3 | Binary table of fitted PSF parameters |

The WCS from the input cube is preserved in all image extensions.

---

## 2D image reduction

For single broad-band images (rather than IFU cubes), use `starkiller_image`:

```python
from starkiller import starkiller_image

img = starkiller_image(
    image=data_array,
    wcs=wcs_object,
    ref_filter='r',
    cal_maglim=[16, 20],
    model_maglim=25,
    run=True,
)
```

---

## Papers

Starkiller was developed for the MUSE 2I/Borisov dataset to recover observations that would otherwise have been unusable due to stellar contamination. See **Deam et al. 2024** for details.

![plot](./figs/BeforeAfter_starkiller.png)

If you use starkiller, please cite the Zenodo release using the DOI badge at the top of this page.

---

## Current limitations

- Simultaneous PSF fitting for overlapping/crowded sources is not yet implemented.
- Very long trails that extend beyond the detector edges may cause the WCS matching and PSF fitting to struggle. Setting `fuzzy=True` can help in these cases.
