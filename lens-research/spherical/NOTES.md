# All-spherical lens prescriptions

Collected 2026-10-04 for the realistic camera (sequential trace through a lens table, PBRT-v3 style).

Conventions used in every table below unless a section says otherwise:

- Surfaces are listed from the object side. Radius sign is the modern one: positive when the center of curvature lies on the image side of the surface. Old patents (Taylor, Rudolph) use "plus = convex, minus = concave" per surface; those were converted and the raw values are quoted alongside.
- "Thickness" is the axial distance from this surface to the next one, which is the PBRT `.dat` column order (radius, thickness, n, aperture). The PBRT-v4 unscaled `*.dat` files use the other convention, see the PBRT section.
- `n_d` and `V_d` are the glass values as printed. Where a patent gives indices at several wavelengths but no Abbe number, the estimate is marked `~` and the method is stated.
- "Stop" rows have radius 0 in PBRT files; here they are marked in the surface column.
- Clear aperture is the full diameter where given. Patents from the 1890s to 1980s almost never print it; those cells say "n/a" and the entrance pupil diameter (f / N) is given in the text as the lower bound for the front group.

Folder layout:

    pbrt-v3/        the four .dat files from the pbrt-v3-scenes lenses/ folder
    pbrt-v4/        the eight .dat files from pbrt-v4-scenes lenses/
    cooke-triplet/  US540122.pdf (Taylor), Cooke_40_degree_field.zmx (Zemax sample)
    tessar/         US721240.pdf (Rudolph)
    double-gauss/   US1786916.pdf (Merte, Biotar), US583336.pdf (Rudolph, Planar, not transcribed)
    zoom/           US4380377.pdf (Canon 70-210, the chosen one), US4331389.pdf, US4380375.pdf, US4266860.pdf (alternates)

## 1. PBRT lens files

### Sources

- PBRT-v3: `https://pbrt.org/scenes-v3` says the lens files live in `lenses/` inside `https://pub-49ca6a23a58a46ef9cf5a5b34413a7ba.r2.dev/pbrt-v3-scenes.tar.gz` (4.02 GB). The tarball was not downloaded. The four files were taken from a GitHub mirror of that folder, `https://raw.githubusercontent.com/hpd/PBRTForMaya/master/maya/assets/pbrt-v3-scenes/lenses/`, and cross-checked: they are byte-identical to the `*.NNmm.dat` files in pbrt-v4-scenes, and identical ignoring whitespace to a second mirror (`zhengyao-lin/cs184-final-project`, `lenses/`).
- PBRT-v4: `https://github.com/mmp/pbrt-v4-scenes/tree/master/lenses` (raw: `https://raw.githubusercontent.com/mmp/pbrt-v4-scenes/master/lenses/<file>`). Eight files: the same four scaled files plus unscaled 100 mm originals `dgauss.dat`, `wide.dat`, `telephoto.dat`, `fisheye.dat`.

Format notes:

- `*.NNmm.dat` (both v3 and v4): `radius  thickness-to-next-surface  n  aperture`, mm. Stop row: radius 0, n 0. Last row thickness 0.
- `*.dat` 100 mm originals (v4 only): the second column is labeled `axpos` and is the gap **before** that surface (first row has 0). Shift by one row and scale to get the `NNmm` file. The `fisheye.dat` and `telephoto.dat` headers are the same, `wide.dat` too.
- The scaled files are the 100 mm tables times 0.5 (dgauss), 0.22 (wide), 2.5 (telephoto), 0.1 (fisheye). PBRT's dgauss stop sits 0.24 mm further forward than the patent's `b1`/`b2` split (5.705 + 4.5 = 10.205 vs patent 5.466 + 4.741 = 10.206, same total).

### dgauss.50mm.dat (Tronnier, US 2,673,491, f = 50 mm, f/2, 22 deg half field)

Glass data from the patent (Voigtlander, 1954; table on p. 3 of the PDF, `[f = 1.0, 1:2, P0' = 0.71436]`). The patent's radii times 50 match the file to the printed precision. `V_d` is printed in the patent (`nu` column), so every row is filled.

| # | radius (mm) | thickness (mm) | n_d | V_d | clear ap. (mm) | note |
|---|---|---|---|---|---|---|
| 1 | 29.475 | 3.76 | 1.67125 (file 1.67) | 47.1 | 25.2 | L1 |
| 2 | 84.83 | 0.12 | air | | 25.2 | |
| 3 | 19.275 | 4.025 | 1.67125 (file 1.67) | 47.1 | 23 | L2a |
| 4 | 40.77 | 3.275 | 1.69842 (file 1.699) | 30.1 | 23 | L2b, cemented |
| 5 | 12.75 | 5.705 | air | | 18 | |
| 6 | **stop** | 4.5 | | | 17.1 | |
| 7 | -14.495 | 1.18 | 1.60266 (file 1.603) | 38.4 | 17 | L3a |
| 8 | 40.77 | 6.065 | 1.65953 (file 1.658) | 57.0 | 20 | L3b, cemented |
| 9 | -20.385 | 0.19 | air | | 20 | |
| 10 | 437.065 | 3.22 | 1.71740 (file 1.717) | 48.1 | 20 | L4 |
| 11 | -39.73 | 0 | air | | 20 | |

Patent: f = 1, 1:2, back focus p0' = 0.71436 f (35.72 mm at 50 mm). The pbrt file comment "Modern Lens Design p.312" refers to W. J. Smith's book.

### wide.22mm.dat (Nakamura, "MLD p. 360", f = 22 mm, 38 deg)

| # | radius (mm) | thickness (mm) | n | V_d | clear ap. (mm) |
|---|---|---|---|---|---|
| 1 | 35.98738 | 1.21638 | 1.54 | not found | 23.716 |
| 2 | 11.69718 | 9.9957 | air | | 17.996 |
| 3 | 13.08714 | 5.12622 | 1.772 | not found | 12.364 |
| 4 | -22.63294 | 1.76924 | 1.617 | not found | 9.812 |
| 5 | 71.05802 | 0.8184 | air | | 9.152 |
| 6 | **stop** | 2.27766 | | | 8.756 |
| 7 | -9.58584 | 2.43254 | 1.617 | not found | 8.184 |
| 8 | -11.28864 | 0.11506 | air | | 9.152 |
| 9 | -166.7765 | 3.09606 | 1.713 | not found | 10.648 |
| 10 | -7.5911 | 1.32682 | 1.805 | not found | 11.44 |
| 11 | -16.7662 | 3.98068 | air | | 12.276 |
| 12 | -7.70286 | 1.21638 | 1.617 | not found | 13.42 |
| 13 | -11.97328 | 0 | air | | 17.996 |

Focal length, f-number and back focus are not in the file header beyond "22 mm" and "38-degree". No source patent was located for "Nakamura" (the MLD page reference is to Smith's Modern Lens Design). All seven `V_d` cells are unfilled.

### telephoto.250mm.dat (Sigler super achromat, "MLD p. 175", EFL 254 mm at 100 mm scale, f/5.6)

| # | radius (mm) | thickness (mm) | n | V_d | clear ap. (mm) |
|---|---|---|---|---|---|
| 1 | 54.6275 | 12.52 | 1.529 | not found | 47.5 |
| 2 | -86.365 | 3.755 | 1.599 | not found | 44.5 |
| 3 | 271.7625 | 2.8175 | air | | 41.5 |
| 4 | **stop** | 67.4125 | | | 40.5 |
| 5 | -32.13 | 3.755 | 1.613 | not found | 31.5 |
| 6 | 49.5325 | 12.52 | 1.603 | not found | 33.5 |
| 7 | -50.945 | 0 | air | | 37 |

Header says EFL = 254 mm, F/5.6 (for the 100 mm table; the scaled file is "250 mm from 100 mm", so the scaled EFL is about 635 mm, not 250; the file name is nominal). No Sigler patent or paper with this table was found; four `V_d` cells unfilled. A super achromat needs special glasses (fluor crown and KzFS types), so guessing from n alone is not safe.

### fisheye.10mm.dat (Muller, "MLD p. 164", 16 mm f/4, 155.9 deg at 100 mm scale)

| # | radius (mm) | thickness (mm) | n | V_d | clear ap. (mm) |
|---|---|---|---|---|---|
| 1 | 30.2249 | 0.8335 | 1.62 | not found | 30.34 |
| 2 | 11.3931 | 7.4136 | air | | 20.68 |
| 3 | 75.2019 | 1.0654 | 1.639 | not found | 17.8 |
| 4 | 8.3349 | 11.1549 | air | | 13.42 |
| 5 | 9.5882 | 2.0054 | 1.654 | not found | 9.02 |
| 6 | 43.8677 | 5.3895 | air | | 8.14 |
| 7 | **stop** | 1.4163 | | | 6.08 |
| 8 | 29.4541 | 2.1934 | 1.517 | not found | 5.96 |
| 9 | -5.2265 | 0.9714 | 1.805 | not found | 5.84 |
| 10 | -14.2884 | 0.0627 | air | | 5.96 |
| 11 | -22.3726 | 0.94 | 1.673 | not found | 5.96 |
| 12 | -15.0404 | 0 | air | | 6.52 |

Two Muller fisheye patents were checked and neither is this lens: US 4,525,038 (Rolf Muller, 1985, five elements, examples at f = 15.8 to 16.9 mm, 1:4, 153 deg) and US 3,737,214 (1973, f = 6.3 mm, 23 surfaces). Six `V_d` cells unfilled.

## 2. Cooke triplet

### Taylor, US 540,122 (1895), the US counterpart of GB 22,607/1893

- Source: `https://patents.google.com/patent/US540122A/en`, PDF `https://patentimages.storage.googleapis.com/7f/ff/52/ced9e8201eb312/US540122.pdf` (saved as `cooke-triplet/US540122.pdf`, 13 pages). Application filed February 5, 1894, patented May 28, 1895. Title "Lens", inventor Harold Dennis Taylor, York.
- GB 22,607 of 1893 ("A Simplified Form and Improved Type of Photographic Lens", filed 1893-11-25, published 1894-10-06): Google Patents has only the bibliographic record, no PDF or text (`https://patents.google.com/patent/GB189322607A/en`). Espacenet's PDF endpoint returns 403 to scripted downloads, so the GB document was not obtained. The US patent carries the numeric schedule.

The worked example (Fig. 9, "the only combination which I have had time to perfect") is a triplet of three **cemented doublets**, f/8, "moderately wide angle", covering a plate whose length equals the focal length. All lengths are fractions of the equivalent focal length (unity). The patent prints "plus = convex, minus = concave" per surface; the modern signed radius is in the second column. Cemented pairs share one surface (r2 = r3, r6 = r7, r10 = r11 in the patent's numbering).

| # | radius (f = 1) | patent value | thickness (f = 1) | n_d | V_d | clear ap. (f = 1) | element |
|---|---|---|---|---|---|---|---|
| 1 | +0.144 | r' = +.144 | 0.006 | 1.603 (silicate flint) | ~36 | 0.167 | L' (front, F) |
| 2 | +0.0816 | r2 = -.0816 = r3 | 0.054 | 1.5224 (light phosphate crown) | ~70 | 0.146 | L2, cemented to L' |
| 3 | -4.5 | r4 = +4.5 | 0.006 (see note) | air | | | |
| 4 | -0.421 | r5 = -.421 | 0.0175 | 1.650 (extra dense flint) | ~31 | 0.130 | L3 (negative component N) |
| 5 | -0.140 | r6 = +.140 = r7 | 0.003 | 1.5224 | ~70 | | L4, cemented to L3 |
| 6 | +0.156 | r8 = -.156 | 0 to stop | air | | about 0.110 | |
| 7 | **stop** | | ~0.05 (see note) | | | 0.098 at f/8 | "as closely as possible behind N" |
| 8 | +0.447 | r9 = +.447 | 0.042 | 1.5224 | ~70 | 0.170 | L5 (back, B) |
| 9 | -0.345 | r10 = +.345 = r11 | 0.006 | 1.603 | ~36 | | L6, cemented to L5 |
| 10 | -1.730 | r12 = +1.73 | | air | | 0.195 | |

Notes:

- Focal length 1 (unit), f/8 (stop 0.098 f), nodal point "just in front of the apex of the front lens". Back focus not stated.
- The two air gaps are **not** printed. The patent sets them by procedure: F and N together must give a parallel (or slightly divergent) exit beam, and B is moved toward N until distortion vanishes. A paraxial trace of the table above gives the afocal F-N gap as 0.0059 f (the elements nearly touch, which is what Fig. 9 shows). With F+N afocal the focal length is 0.949 f for any N-B gap; Taylor's "slightly divergent" lets it reach 1.0. The N-B gap is estimated from the to-scale Fig. 9 at about 0.05 f. Treat both gaps as derived, not quoted.
- Glass: the patent gives n_D and (n_D - n_G') per glass, plus a "reciprocal of dispersive power" (n_D - 1)/(n_D - n_G') = 29.1, 55.63, 25.2. The `~V_d` values use V_d = (n_D - 1) / ((n_G' - n_D) / 1.25), the 1.25 being the n_g-n_d to n_F-n_C ratio of BK7 (1.23) and F2 (1.29). Expect +/- 2.
- Component focal lengths from the trace: F +0.30, N -0.25, B +0.78 (f = 1).

### Textbook three-singlet triplet: Zemax sample "Cooke 40 degree field"

Saved as `cooke-triplet/Cooke_40_degree_field.zmx` from `https://raw.githubusercontent.com/xzos/PyZDDE/master/ZMXFILES/Cooke_40_degree_field.zmx` (the file shipped in the Zemax/OpticStudio samples folder, "A SIMPLE COOKE TRIPLET"). It is the usual three-singlet layout and is handy for validation because its glasses are catalog Schott types. Radii are 1/CURV from the file.

| # | radius (mm) | thickness (mm) | glass | n_d | V_d | semi-diameter (mm) |
|---|---|---|---|---|---|---|
| 1 | 22.01359 | 3.258956 | SK16 | 1.62041 | 60.32 | 9.5 |
| 2 | -435.7604 | 6.007551 | air | | | 9.5 |
| 3 | -22.21328 | 0.999975 | F2 | 1.62004 | 36.37 | 5.0 |
| 4 | 20.29192 (**stop** at this surface) | 4.750409 | air | | | 5.0 |
| 5 | 79.68366 | 2.952076 | SK16 | 1.62041 | 60.32 | 7.5 |
| 6 | -18.39533 | 42.20778 | air | | | 7.5 |
| 7 | image | | | | | 18.17 |

EFL 50.0 mm (paraxial trace of the table: 50.02), entrance pupil 10 mm so F/5, full field 40 deg (fields 0, 14, 20 deg). The 42.2078 mm image distance is the file's focus solve; the paraxial back focus from the table is 42.44 mm. The stop is on surface 4 (the rear of the flint), not a separate surface. Surface 6 carries a marginal-ray-angle solve (-0.1) in the file; the curvature value is what the solve produced and is listed as a plain sphere.

## 3. Zeiss Tessar, Rudolph, US 721,240 (1903) / DE 142,294

- Source: `https://patents.google.com/patent/US721240A/en`, PDF `https://patentimages.storage.googleapis.com/5b/da/59/4075b2ac2d340e/US721240.pdf` (saved as `tessar/US721240.pdf`, 4 pages). Filed July 15, 1902, patented February 24, 1903. Assigned to Carl Zeiss.
- DE 142,294 (filed 1902-04-25, published 1903-07-04): Google Patents page `https://patents.google.com/patent/DE142294C/de` exposes no PDF link and the DPMA/Espacenet endpoints refuse scripted downloads, so no DE PDF is saved. The German text read through Google Patents carries the same numbers as the US table except that `r5` came through as -1.3 (the US page image clearly reads -1.113; the OCR text layer of the US PDF also garbles `d1` as 0.038 and `r5` as -1.118; the table below is from the page image at 400 dpi).

Values are fractions of the focal length. Relative aperture 1:5.5, anastigmatic field about 60 deg. Light travels L1, L2, stop, L3, L4.

| # | radius (f = 1) | radius (f = 100 mm) | thickness (f = 1) | n_D | n_F | n_G' | ~V_d | clear ap. | element |
|---|---|---|---|---|---|---|---|---|---|
| 1 | +0.215 | 21.5 | d1 = 0.033 | 1.61132 | 1.61870 | 1.62462 | ~58 | n/a | L1 (positive) |
| 2 | infinity | flat | l = 0.019 | air | | | | n/a | |
| 3 | -0.742 | -74.2 | d2 = 0.011 | 1.60457 | 1.61436 | 1.62252 | ~43 | n/a | L2 (negative) |
| 4 | +0.208 | 20.8 | b1 = 0.030 | air | | | | n/a | |
| 5 | **stop** | | b2 = 0.030 | | | | | 0.18 f at f/5.5 | diaphragm B |
| 6 | -1.113 | -111.3 | d3 = 0.011 | 1.52110 | 1.52820 | 1.53397 | ~51 | n/a | L3 (negative) |
| 7 | +0.252 | 25.2 | d4 = 0.030 | 1.61132 | 1.61895 | 1.62514 | ~56 | n/a | L4 (positive), cemented to L3 at r6 |
| 8 | -0.367 | -36.7 | | air | | | | n/a | |

Notes:

- Paraxial trace of the table: EFL 0.9925, back focus 0.907 (f = 1). The 0.75 percent focal length slack is in the printed three-decimal radii.
- `~V_d` is estimated as (n_D - 1) / ((n_F - n_D) / 0.70), since the patent gives n_F but not n_C; 0.70 is the (n_F - n_d)/(n_F - n_C) ratio of BK7 (0.69) and F2 (0.71). Expect +/- 1.5. The two 1.61132 glasses are the same barium crown.
- Clear apertures are not printed anywhere in the patent. Entrance pupil at 1:5.5 is 0.18 f.

## 4. Double Gauss other than Tronnier: Zeiss Biotar, Merte, US 1,786,916 (1930)

- Source: `https://patents.google.com/patent/US1786916A/en`, PDF `https://patentimages.storage.googleapis.com/bc/eb/d0/e3b78766a08aab/US1786916.pdf` (saved as `double-gauss/US1786916.pdf`, 3 pages). Filed September 7, 1928 (Germany September 29, 1927), patented December 30, 1930. Assigned to Carl Zeiss. Title "Objective corrected spherically, chromatically, astigmatically, and for coma".
- Two examples, both 1:1.4, focal length 100 units. Example 1 is seven elements (the rear doublet split into three); Example 2 is the six-element Biotar form, transcribed here from the page image.

Example 2 (Fig. 2), f = 100, 1:1.4:

| # | radius | thickness | n_d | V_d | clear ap. | element |
|---|---|---|---|---|---|---|
| 1 | +83.6 | d_I = 10.75 | 1.64238 | 48.0 | n/a | I (positive meniscus) |
| 2 | +321.0 | l_1 = 1.65 | air | | n/a | |
| 3 | +44.8 | d_II = 15.55 | 1.62306 | 56.9 | n/a | II |
| 4 | -1150 | d_III = 5.05 | 1.57566 | 41.2 | n/a | III, cemented to II |
| 5 | +28.3 | l_2 = 18.9 (stop inside) | air | | n/a | |
| 6 | -38.5 | d_IV = 5.05 | 1.67270 | 32.2 | n/a | IV |
| 7 | +50.5 | d_V = 21.22 | 1.64238 | 48.0 | n/a | V, cemented to IV |
| 8 | -53.2 | l_3 = 0.97 | air | | n/a | |
| 9 | +106.0 | d_VI = 13.9 | 1.64238 | 48.0 | n/a | VI (biconvex) |
| 10 | -120.0 | | air | | n/a | |

Notes:

- Paraxial trace: EFL 99.91, back focus 65.33.
- The patent gives the total central air space l_2 = 18.9 but not where the diaphragm sits in it; the four-member Gauss convention is mid-gap, so use 9.45 / 9.45 and flag it. Clear apertures are not printed; entrance pupil at 1:1.4 is 71.4 (f = 100), so the front element needs at least that.
- Also saved, not transcribed: Rudolph's original Planar, US 583,336 (1897), `double-gauss/US583336.pdf`, PDF URL `https://patentimages.storage.googleapis.com/65/53/81/ed5b9efeb9e144/US583336.pdf`. Its examples are given in focal-length units with the old sign convention and partial glass data; the Biotar is the cleaner validation target.

## 5. Zoom: Canon New FD 70-210 mm f/4, US 4,380,377 (1983)

Candidates came from the Optical Bench Hub table (`https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBenchHub.htm`, entries US004380377 "Canon New FD70-210mm f4", US004380375 "Nikon Series E Zoom 36-72mm f/3.5", US004331389 "Ricoh smc Pentax-M 75-150mm F4", US004266860 "Nikon AI Zoom-Nikkor 35-70mm f/3.5"). The Canon patent is the one that prints the gaps at three positions; the Nikon and Pentax ones print wide and tele only.

- Source: `https://patents.justia.com/patent/4380377` (full text, used for the numbers) and the USPTO PDF `https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/4380377` (saved as `zoom/US4380377.pdf`, 14 pages, image only; Google Patents returned 503 to every fetch during this session so its patentimages link was not obtained). Title "Compact zoom lens", Canon, filed September 16, 1980 (Japan priority 1979), issued April 19, 1983. The Example 1 table was checked against the PDF page image (pages 12 and 13 of the file).
- Five groups: I positive (focusing), II negative (variator), III positive (compensator), IV and V fixed relay. All surfaces spherical; the patent mentions no aspheric. Example 1, f = 70.55 to 205.31, F/4.

| # | group | radius (mm) | thickness (mm) | n_d | V_d | clear ap. | element |
|---|---|---|---|---|---|---|---|
| 1 | I (f 108.33) | 144.296 | 2.8 | 1.80518 | 25.4 | n/a | negative, cemented |
| 2 | | 67.895 | 6.6 | 1.61272 | 58.7 | n/a | positive |
| 3 | | -4571.825 | 0.2 | air | | n/a | |
| 4 | | 94.482 | 4.7 | 1.61272 | 58.7 | n/a | positive singlet |
| 5 | | 26667.449 | d5 variable | air | | n/a | |
| 6 | II (f -34.02) | 464.312 | 1.5 | 1.713 | 53.9 | n/a | negative singlet |
| 7 | | 40.285 | 4.2 | air | | n/a | |
| 8 | | -45.202 | 1.5 | 1.713 | 53.9 | n/a | negative, cemented |
| 9 | | 44.784 | 3.4 | 1.84666 | 23.0 | n/a | positive |
| 10 | | -3270.421 | d10 variable | air | | n/a | |
| 11 | III (f 94.26) | 108.4 | 5.8 | 1.51633 | 64.1 | n/a | positive, cemented |
| 12 | | -32.441 | 1.5 | 1.7552 | 27.5 | n/a | negative |
| 13 | | -56.830 | d13 variable | air | | n/a | |
| 14 | IV (f 87.19) | 36.187 | 4.5 | 1.61272 | 58.7 | n/a | positive |
| 15 | | 797.679 | 1.95 | air | | n/a | |
| 16 | | -213.357 | 2 | 1.80518 | 25.4 | n/a | biconcave |
| 17 | | 570.712 | 47.33 | air | | n/a | |
| 18 | V (f 150.35) | -19.993 | 2 | 1.8061 | 40.9 | n/a | negative meniscus |
| 19 | | -42.359 | 0.3 | air | | n/a | |
| 20 | | 172.917 | 3.5 | 1.59551 | 39.2 | n/a | positive |
| 21 | | -61.717 | | air | | n/a | |

Variable gaps:

| position | f (mm) | d5 | d10 | d13 |
|---|---|---|---|---|
| wide | 70.55 | 0.50 | 34.12 | 18.62 |
| middle | 137.04 | 32.54 | 17.63 | 3.07 |
| tele | 205.31 | 43.86 | 0.68 | 8.70 |

Notes:

- F-number 4 at all positions (stated). Back focus is not printed; a paraxial trace of the table gives EFL 70.55 / 137.03 / 205.26 and back focus 42.69 / 42.68 / 42.66 mm, so the transcription is consistent and the image plane stays put.
- The stop position is not printed. For this type it sits in the fixed relay, in front of group IV (between surfaces 13 and 14, inside the d13 gap, on the group IV side so it does not move). Treat that as an assumption.
- Clear apertures are not printed. Entrance pupil at tele is 205.31 / 4 = 51.3 mm, so the front group needs at least about 52 mm.
- Examples 2 and 3 (f = 70.08 to 204.45 and 70.13 to 207.58) are also in the PDF and in the Justia text; not transcribed.

### Alternates saved in `zoom/`, not transcribed in full

- `US4331389.pdf`: Asahi (Pentax-M 75-150 f/4), "Compact zoom lens system", 1982, PDF `https://patentimages.storage.googleapis.com/c5/f5/17/0ee7b0f3e4dcce/US4331389.pdf`. One example, 21 surfaces, 4 groups, f = 77.064 to 144.997 (the header prints "1:41", a print garble for 1:4), three variable gaps given at wide and tele only (d5 8.109/27.579, d10 22.851/1.401, d13 3.585/5.564). Transcription from the page image traces to EFL 77.06 / 145.00, back focus 56.15.
- `US4380375.pdf`: Nikon (Series E 36-72 f/3.5), "Wide angle zoom lens of two-group construction", 1983, PDF `https://patentimages.storage.googleapis.com/c4/a9/ae/9b7fb7508823d3/US4380375.pdf`. Two embodiments, 16 and 19 surfaces, one variable gap (l = 41.4 to 0.1 and 46.6 to 0.4), wide and tele only. A trace of my reading of the first embodiment gives 39.4 to 69.7 against the stated 37 to 69, so at least one value there is misread; not used.
- `US4266860.pdf`: Nikon (AI Zoom-Nikkor 35-70 f/3.5), 1981, PDF `https://patentimages.storage.googleapis.com/2e/89/a5/8cbfe320760cf3/US4266860.pdf`. Three two-group embodiments, f = 36 to 68.8, back focus 42.7 to 65.2 printed, wide and tele only.
