## Metal Optical Constants

The metallic presets use optical-constant data from the
[refractiveindex.info database](https://refractiveindex.info).

Source access date: October 2, 2026.

### Source Datasets

| Preset | Element | Local filename | Original dataset |
|---|---|---|---|
| Gold | Au | `Au-Johnson.yml` | [Johnson and Christy, 1972](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Au/nk/Johnson.yml) |
| Silver | Ag | `Ag-Johnson.yml` | [Johnson and Christy, 1972](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Ag/nk/Johnson.yml) |
| Copper | Cu | `Cu-Johnson.yml` | [Johnson and Christy, 1972](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Cu/nk/Johnson.yml) |
| Aluminum | Al | `Al-Rakic.yml` | [Rakić, 1995](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Al/nk/Rakic.yml) |
| Chrome | Cr | `Cr-Johnson.yml` | [Johnson and Christy, 1974](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Cr/nk/Johnson.yml) |

Each numerical row contains:

1. Wavelength in micrometers (µm).
2. `n`, stored as `eta`: the real part of the complex refractive index.
3. `k`: the extinction coefficient, representing its imaginary part.

Both `eta` and `k` are dimensionless optical coefficients and are not
restricted to the range `[0, 1]`.

### RGB Preset Generation

Each material's wavelength-dependent `eta` and `k` data is sampled using
piecewise linear interpolation at the following wavelengths:

| Output channel | Wavelength |
|---|---|
| Red | 0.650 µm / 650 nm |
| Green | 0.550 µm / 550 nm |
| Blue | 0.450 µm / 450 nm |

For a target wavelength between two tabulated wavelengths, the interpolation
weight is:

`t = (wavelength - wavelength0) / (wavelength1 - wavelength0)`

Each coefficient is then evaluated using:

`value = value0 + t * (value1 - value0)`

The resulting coefficients are stored in two `glm::vec3` values, in
**R, G, B order**:

- `etaT`: the material's real refractive-index coefficients.
- `k`: the material's extinction coefficients.

These coefficients are inputs to the conductor Fresnel calculation.

### Approximation

The choice of 650, 550, and 450 nm is a simplified approximation for this
RGB renderer. It is not prescribed by the source publications and is not
a full spectral-to-RGB conversion. RGB channels represent broad spectral
responses, so using one wavelength per channel can introduce color error.

### License and Attribution

The refractiveindex.info database is released under the
[CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
The database permits copying, modification, and redistribution, including
commercial use, without requesting permission.

See the database's [licensing and citation information](https://refractiveindex.info/about).

This license statement applies to the numerical database. Separately
distributed software may have different license terms.

### References

- M. N. Polyanskiy. “Refractiveindex.info database of optical constants.”
  *Scientific Data* **11**, 94 (2024).
  https://doi.org/10.1038/s41597-023-02898-2

- P. B. Johnson and R. W. Christy. “Optical constants of the noble metals.”
  *Physical Review B* **6**, 4370–4379 (1972).
  https://doi.org/10.1103/PhysRevB.6.4370

- P. B. Johnson and R. W. Christy. “Optical constants of transition metals:
  Ti, V, Cr, Mn, Fe, Co, Ni, and Pd.”
  *Physical Review B* **9**, 5056–5070 (1974).
  https://doi.org/10.1103/PhysRevB.9.5056

- A. D. Rakić. “Algorithm for the determination of intrinsic optical
  constants of metal films: application to aluminum.”
  *Applied Optics* **34**, 4755–4767 (1995).
  https://doi.org/10.1364/AO.34.004755