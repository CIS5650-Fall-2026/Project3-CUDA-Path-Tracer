# Voigtländer lens prescriptions

Numeric prescriptions for the realistic-camera lens table, taken from the
patents. Two classic Braunschweig designs by A. W. Tronnier (Nokton 50 f/1.5,
Ultron 50 f/2) plus the Apo-Lanthar, and two modern Cosina designs (Nokton
50 f/1 Aspherical, Nokton 50 f/1.2 for X-mount). One folder per lens with the
patent PDFs and, where photonstophotos has the lens, Bill Claff's Optical
Bench data file for cross-checking.

How the data was gathered: the photonstophotos Optical Bench hub
(https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBenchHub.htm)
lists which Voigtländer and Cosina lenses have a matched patent example. Its
data files live at
`https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Data/<PatentID>_<ExampleID>.txt`.
The patent PDFs come from Google Patents
(`patentimages.storage.googleapis.com`). The US patents from 1953 are scans;
Google's OCR of their tables is garbage, so every number below was read from
the page images rendered at 220 dpi with 2x crops of the tables, and no value
comes from OCR. The Japanese publications are born-digital but their tables are
images with no text layer, same treatment. As a numeric check, a paraxial
trace of every table below reproduces the focal length and back focus the
patent states to four or five digits (see "Verification" at the end).

Conventions common to all tables:

- Surfaces are numbered from the object side. A radius is positive when its
  center of curvature lies on the image side. Thickness is the axial distance
  to the next surface. Index and Abbe number belong to the medium after the
  surface; blank means air.
- `n_d`, `V_d` are for the d line (He 587.56 nm) everywhere except the
  US 2,627,205 example, which the patent states for the F line (486.1 nm).
- The 1950 patents give the whole system for focal length F = 1 (Tronnier
  calls it "the unit"). Multiply radii and thicknesses by the focal length in
  mm. Glass data are unaffected.
- None of the patents gives clear apertures. The photonstophotos files carry
  Bill Claff's estimated diameters in their last column; treat those as his
  values, not the patent's.
- Stop rows are marked **AS**.

## Not found

- **Septon 50 mm f/2** (Bessamatic, 1959, 7 elements in 5 groups). No patent.
  lenslegend.com says there was none because the design was derived from an
  already patented lens; klassik-cameras.de describes it as the Ultron with
  the single third element split into a cemented pair and the front elements
  adapted to the 22.5 mm shutter opening. No numeric prescription exists in
  the open literature that I could find; photonstophotos has no entry.
- **Cosina Apo-Lanthar 50 mm f/2 Aspherical** (10 elements in 8 groups). Not
  on the photonstophotos hub, nothing on lensreview.xyz or asobinet that names
  a patent. Not found.
- **Cosina Nokton 50 mm f/1.2 Aspherical** (the full-frame VM/E/Z lens, 8
  elements in 6 groups, two double-sided aspheric elements). Not on the hub,
  no patent located. The JP 2025-058577 publication that the rumor sites
  called "Nokton 50mm f/1.2" is the APS-C X-mount Nokton 50 mm F1.2 (no
  aspherics); see its folder below.

## Nokton 50 mm f/1.5 (Prominent, 1950)

Folder: `nokton-50mm-f1.5-1950/`

Sources:
- https://patents.google.com/patent/US2645155A/en, PDF `US2645155.pdf`
  (Tronnier, "Photographic objective of high light-transmitting capacity of
  the Gauss type", Ser. 184,454, filed 1950-09-12, Swiss priority 1950-01-16,
  issued 1953-07-14).
- https://patents.google.com/patent/US2646721A/en, PDF `US2646721.pdf`
  (Tronnier, "Gauss type photographic objective having two lens systems on
  opposite sides of a diaphragm", Ser. 184,456, same filing and priority
  dates, issued 1953-07-28). This one cites 184,454 as the parent and adds
  the glass-index distribution claim.
- Attribution of both patents to the Nokton: rangefinderforum / lenslegend /
  Wikipedia (Voigtländer Prominent). photonstophotos has no Nokton 50/1.5
  entry, so there is no Optical Bench file for this lens.

Which embodiment: the production Nokton is 7 elements in 5 groups at f/1.5.
Both patents describe the same layout (cemented front doublet, two air-spaced
menisci, stop, cemented doublet, rear singlet; Fig. 2 in each) and each gives
one numerical example. The US 2,646,721 example is explicitly 1:1.5, uses the
same glass for L2 and L3 (the patent says this was done so the two elements
can be made from one melt for series production) and has the cheaper
thin-positive-lens layout, so it is the one to use for the production lens.
The US 2,645,155 example is 1:1.4 (entrance pupil 0.70 F) and is the parent
design. Both are transcribed. Fig. 1 of each patent shows a variant with
L1/L2 and L5/L6 air-spaced, but no numbers are given for it.

Scale: F = 1.00. For the 50 mm lens multiply by 50. The drawings (Fig. 2)
are drawn for f = 150 mm "in actual size".

Stop position: both patents only give the total air space a3 between L4 and
L5 ("air diaphragm space") and do not split it. From Fig. 2 of US 2,646,721
the diaphragm sits near the middle of that space; a split of about
b1 = 0.12 (after L4) and b2 = 0.11 (before L5), summing to 0.23362, is my
estimate from the drawing and is flagged as such. The sagittal gap in the
figure is measured vertex to vertex on the axis.

Back focus: not stated in either patent. The paraxial trace of the tables
gives 0.6002 F (US 2,646,721) and 0.6351 F (US 2,645,155).

### US 2,646,721 numerical example (f = 1.0, 1:1.5) — production match

Group focal lengths given by the patent: f12 = +1.745, f34 = -3.175,
f56 = -2.068, f7 = +0.700. Entrance pupil diameter 0.667 F.

| Surface | Element | Radius (F=1) | Thickness | n_d | V_d | Note |
|---|---|---|---|---|---|---|
| 1 | L1 | +0.66609 | 0.03364 | 1.70315 | 41.1 | d1 |
| 2 | L1/L2 cemented | +0.41154 | 0.13270 | 1.62095 | 60.3 | d2 (R2 = R2') |
| 3 | L2 | +1.96575 | 0.01028 | | | a1 |
| 4 | L3 | +0.41154 | 0.13270 | 1.62095 | 60.3 | d3 |
| 5 | L3 | +1.96575 | 0.01495 | | | a2 |
| 6 | L4 | +2.45896 | 0.04859 | 1.53250 | 46.2 | d4 (R5') |
| 7 | L4 | +0.24988 | 0.23362 | | | a3, contains the stop (R6) |
| **AS** | diaphragm | flat | | | | inside a3; split estimated 0.12 / 0.11 from Fig. 2 |
| 8 | L5 | -0.29417 | 0.01962 | 1.64783 | 33.8 | d5 (R7) |
| 9 | L5/L6 cemented | +0.98213 | 0.09812 | 1.61948 | 60.4 | d6 (R8 = R8') |
| 10 | L6 | -0.41154 | 0.00374 | | | a4 (R9) |
| 11 | L7 | +2.45896 | 0.06448 | 1.70315 | 41.1 | d7 (R10) |
| 12 | L7 | -0.60869 | BF | | | R11 |

The print shows "R4+0.41154" with the "=" dropped; the value is clear.

### US 2,645,155 numerical example (f = 1.00, entrance pupil 0.70, about 1:1.4)

Group focal lengths given by the patent: f12 = +1.844, f34 = -3.870,
f56 = -2.484, f7 = +0.761.

| Surface | Element | Radius (F=1) | Thickness | n_d | V_d | Note |
|---|---|---|---|---|---|---|
| 1 | L1 | +0.68359 | 0.03539 | 1.72713 | 28.4 | d1 |
| 2 | L1/L2 cemented | +0.42566 | 0.16176 | 1.61966 | 55.0 | d2 (R2 = R2') |
| 3 | L2 | +2.02423 | 0.00289 | | | a1 |
| 4 | L3 | +0.42566 | 0.12118 | 1.70329 | 41.1 | d3 |
| 5 | L3 | +1.64339 | 0.01596 | | | a2 |
| 6 | L4 | +2.02423 | 0.03443 | 1.54826 | 45.8 | d4 (R5') |
| 7 | L4 | +0.25716 | 0.21293 | | | a3, contains the stop (R6) |
| **AS** | diaphragm | flat | | | | inside a3, position not given |
| 8 | L5 | -0.30927 | 0.02058 | 1.64819 | 33.7 | d5 (R7) |
| 9 | L5/L6 cemented | flat (infinity) | 0.10598 | 1.61959 | 60.5 | d6 (R8 = R8' = infinity) |
| 10 | L6 | -0.42566 | 0.00385 | | | a4 (R9) |
| 11 | L7 | +2.96383 | 0.07655 | 1.69347 | 53.5 | d7 (R10) |
| 12 | L7 | -0.63533 | BF | | | R11 |

The table header in the print reads "[f=100  Diameter of aperture=0.70]"; the
text says the focal length is "assumed to be equal to 1.00" and all radii and
thicknesses refer to that unit, so the "100" is a dropped decimal point.

## Ultron 50 mm f/2 (Prominent, 1950)

Folder: `ultron-50mm-f2-1950/`

Sources:
- https://patents.google.com/patent/US2627204A/en, PDF `US2627204.pdf`
  (Tronnier, "Four-component Gauss-type photographic objective of high
  light-transmitting capacity", Ser. 203,179, filed 1950-12-28, Swiss
  priority 1950-04-29, issued 1953-02-03).
- https://patents.google.com/patent/US2627205A/en, PDF `US2627205.pdf`
  (companion, Ser. 203,180, same filing date, Swiss priority 1950-01-13).
- photonstophotos: `US002627204_Example02P`, copied here as
  `photonstophotos_US002627204_Example02P.txt`, labeled "Cosina Voigtlander
  Ultron 50mm f2", scaled to 50 mm.
- camera-wiki.org/wiki/Ultron names 2,627,204 and 2,627,205 as the Ultron
  patents.

Which embodiment: the Ultron is 6 elements in 5 groups at f/2.
US 2,627,204 has two examples. Example II is 1:2.0 with L2 and L3 air-spaced
(6 in 5 groups) and is the production lens; photonstophotos uses it too.
Example I is 1:2.3 with L2 and L3 cemented (a2 = 0, R5 = R4; 6 in 4 groups).
US 2,627,205 has one example at 1:2.3 with its glasses given for the F line;
it is the lower-aperture relative and is transcribed at the end of this
section for completeness. All three are "f = 1.0".

Scale: F = 1.0; multiply by 50 for the production lens. Drawings are for
f = 150 mm.

Stop position: given. The air space a3 between L3 and L4 is split by the
diaphragm into b1 (after L3) and b2 (before L4). Note that the photonstophotos
file puts the stop at 5.5465 / 3.8965 mm instead of the patent's
0.09393 / 0.09493 × 50 = 4.6965 / 4.7465 mm; the patent values are below.

Back focus (paraxial, object at infinity, "p0'"): Example II 0.6972 F,
Example I 0.7177 F.

### US 2,627,204 Example II (f = 1.0, 1:2.0, p0' = 0.6972) — production match

| Surface | Element | Radius (F=1) | Thickness | n_d | V_d | Note |
|---|---|---|---|---|---|---|
| 1 | L1 | +0.63214 | 0.05996 | 1.62139 | 60.3 | d1 |
| 2 | L1 | +1.76011 | 0.00400 | | | a1 |
| 3 | L2 | +0.43828 | 0.06395 | 1.65953 | 57.0 | d2 |
| 4 | L2 | +1.08680 | 0.07195 | | | a2 |
| 5 | L3 | +0.97029 | 0.04896 | 1.64691 | 33.9 | d3 |
| 6 | L3 | +0.27096 | 0.09393 | | | b1 |
| **AS** | diaphragm | flat | 0.09493 | | | b2 (b1 + b2 = a3 = 0.18886) |
| 7 | L4 | -0.26444 | 0.02198 | 1.63652 | 35.5 | d4 |
| 8 | L4/L5 cemented | +0.77003 | 0.09693 | 1.69347 | 53.5 | d5 (R8 = R9) |
| 9 | L5 | -0.37233 | 0.00300 | | | a4 (R10) |
| 10 | L6 | +3.76623 | 0.07794 | 1.72381 | 38.0 | d6 (R11) |
| 11 | L6 | -0.79382 | BF = 0.6972 | | | R12 |

### US 2,627,204 Example I (f = 1.0, 1:2.3, p0' = 0.7177)

| Surface | Element | Radius (F=1) | Thickness | n_d | V_d | Note |
|---|---|---|---|---|---|---|
| 1 | L1 | +0.60708 | 0.05695 | 1.63909 | 55.7 | d1 |
| 2 | L1 | +1.64332 | 0.00205 | | | a1 |
| 3 | L2 | +0.40395 | 0.13076 | 1.61136 | 59.0 | d2 |
| 4 | L2/L3 cemented | +0.66243 | 0.04009 | 1.64819 | 33.7 | a2 = 0, R5 = R4; d3 |
| 5 | L3 | +0.26875 | 0.12480 | | | b1 (R6) |
| **AS** | diaphragm | flat | 0.07401 | | | b2 (a3 = 0.19881) |
| 6 | L4 | -0.28160 | 0.02508 | 1.58241 | 40.6 | d4 (R7) |
| 7 | L4/L5 cemented | +0.60708 | 0.10485 | 1.61136 | 59.0 | d5 (R8 = R9) |
| 8 | L5 | -0.38692 | 0.00308 | | | a4 (R10) |
| 9 | L6 | +2.68465 | 0.04996 | 1.63909 | 55.7 | d6 (R11) |
| 10 | L6 | -0.77385 | BF = 0.7177 | | | R12 |

### US 2,627,205 numerical example (f = 1.0, 1:2.3, p0' = 0.698), indices for the F line

| Surface | Element | Radius (F=1) | Thickness | n_F | V | Note |
|---|---|---|---|---|---|---|
| 1 | L1 | +0.64174 | 0.06108 | 1.62856 | 60.3 | d1 |
| 2 | L1 | +1.78419 | 0.00394 | | | a1 |
| 3 | L2 | +0.44511 | 0.07290 | 1.66758 | 57.0 | d2 |
| 4 | L2 | +1.08963 | 0.07290 | | | a2 |
| 5 | L3 | +0.97278 | 0.04138 | 1.63245 | 36.2 | d3 |
| 6 | L3 | +0.27540 | 0.08867 | | | b1 |
| **AS** | diaphragm | flat | 0.09556 | | | b2 (a3 = 0.18423) |
| 7 | L4 | -0.27540 | 0.03941 | 1.66045 | 33.9 | d4 |
| 8 | L4/L5 cemented | +1.78419 | 0.08079 | 1.66782 | 50.9 | d5 (R9 = R8) |
| 9 | L5 | -0.40689 | 0.00197 | | | a4 (R10) |
| 10 | L6 | +3.63341 | 0.05911 | 1.70254 | 53.5 | d6 (R11) |
| 11 | L6 | -0.62343 | BF = 0.698 | | | R12 |

The Abbe column of this table was read from the full page, not the crop
(the crop cut it off); the values are the same ones the text repeats.

## Apo-Lanthar 105 mm f/4.5 (Bessa II / large format, 1950s)

Folder: `apo-lanthar-105mm-f4.5/`

Sources:
- https://patents.google.com/patent/US2645154A/en, PDF `US2645154.pdf`
  (Tronnier, "Five-lens photographic objective", Ser. 183,967, filed
  1950-09-09, Swiss priority 1949-10-17, issued 1953-07-14). German
  equivalent DE 880,802.
- Attribution: Arne Cröll, "Voigtländer Large Format Lenses from 1949-1972"
  (https://www.arnecroell.com/voigtlaender.pdf), ref. [22]: "Tronnier,
  Albrecht Wilhelm: US patent no. 2,645,154, Germany no. 880,802
  (Apo-Lanthar)". Cröll also notes the front element is a lanthanum crown
  and that the lens was made in 105, 150, 210 and 300 mm.
- photonstophotos: `US002645154_Example01P`, copied here as
  `photonstophotos_US002645154_Example01P.txt`. Bill labels it "Cosina
  Voigtlander Heliar 210mm F4.5" and scales it to 210 mm; the data are this
  same single example.

Which embodiment: the patent has one numerical example (Heliar layout,
5 elements in 3 groups, 1:4.5, stated field "almost 60 degrees"). It is the
only prescription there is, and the Apo-Lanthar line is the same design at
every focal length (Cröll: a Color-Heliar prototype "has the exact same lens
radii as a regular Apo-Lanthar"). For the 105 mm lens multiply by 105. The
patent drawing assumes f = 200 mm.

Scale: F = 1.0000. Back focus p0' = 0.8663 F (so 90.96 mm at 105 mm).
Entrance pupil 0.2222 F. Member focal lengths from the patent: f12 = 0.431 F,
f3 = -0.294 F, f45 = 0.595 F.

Stop position: given; the rear air space a2 is split into b1 (after L3) and
b2 (before L4).

| Surface | Element | Radius (F=1) | Thickness | n_d | V_d | Note |
|---|---|---|---|---|---|---|
| 1 | L1 | +0.29207 | 0.05403 | 1.65953 | 57.0 | d1 |
| 2 | L1/L2 cemented | -0.84699 | 0.01402 | 1.60266 | 38.4 | d2 |
| 3 | L2 | flat (infinite) | 0.03242 | | | a1 |
| 4 | L3 | -0.63631 | 0.01402 | 1.64282 | 47.9 | d3 |
| 5 | L3 | +0.27162 | 0.03885 | | | b1 |
| **AS** | diaphragm | flat | 0.02044 | | | b2 (a2 = 0.05929) |
| 6 | L4 | -2.42415 | 0.01402 | 1.60266 | 38.4 | d4 |
| 7 | L4/L5 cemented | +0.23511 | 0.05929 | 1.66867 | 47.5 | d5 |
| 8 | L5 | -0.40597 | BF = 0.8663 | | | |

## Cosina Voigtländer Nokton 50 mm F1 Aspherical (VM 2021, E/RF/Z later)

Folder: `nokton-50mm-f1.0-aspherical/`

Sources:
- JP 2023-063766 A, application JP 2021-173771 (Cosina), filed 2021-10-25,
  published 2023-05-10. https://patents.google.com/patent/JP2023063766A/ja,
  PDF `JP2023063766A.pdf`. J-PlatPat:
  https://www.j-platpat.inpit.go.jp/c1801/PU/JP-2021-173771/10/ja
- photonstophotos: `JP2023-063766_Example01P`, copied here as
  `photonstophotos_JP2023-063766_Example01P.txt` ("Cosina Voigtlander Nokton
  50mm F1.0 Aspherical"). Bill's note: surface 8 limits the F-number to
  f/1.04. Image height in his file 43.26 (full-frame diagonal).
- lensreview.xyz analysis 137 also picks 特開2023-063766, Example 1.

Which embodiment: Example 1 (Tables 1 and 2, PDF page 6). 9 elements in 7
groups: three menisci, stop, cemented doublet, singlet, cemented doublet,
rear meniscus; aspheric surfaces on the front element (surface 1) and both
sides of the rear element (16, 17). This matches Cosina's spec (9 elements in
7 groups, one ground-aspheric front element plus one aspheric element).
Example 2 (Tables 3 and 4, PDF pages 9-10) is a variant with the same layout
and the same three aspheric surfaces; it was not transcribed.

Given by the patent: f = 50 mm, Fno 1.0. Back focus is the last thickness,
18.74 mm. TT/f < 1.72. Units mm. `nd`, `νd` at 587.56 nm.

| Surface | Element | Radius (mm) | Thickness (mm) | n_d | V_d | Note |
|---|---|---|---|---|---|---|
| 1 | L1 | 40.765 | 4.89 | 1.90525 | 35.04 | aspheric (1A) |
| 2 | L1 | 65.488 | 1.07 | | | |
| 3 | L2 | 32.975 | 8.01 | 1.90043 | 37.37 | |
| 4 | L2 | 84.15 | 0.87 | | | |
| 5 | L3 | 58.596 | 1.65 | 1.80518 | 25.46 | |
| 6 | L3 | 19.835 | 12.05 | | | |
| **AS** (7) | stop | flat | 3.24 | | | |
| 8 | L4f | -40.995 | 1.55 | 1.76182 | 26.61 | |
| 9 | L4f/L4r cemented | 26.578 | 10.8 | 1.883 | 40.69 | |
| 10 | L4r | -90.702 | 0.31 | | | |
| 11 | L5 | 78.611 | 5.05 | 1.883 | 40.69 | |
| 12 | L5 | -125.199 | 0.31 | | | |
| 13 | L6f | 53.736 | 9.47 | 1.883 | 40.69 | |
| 14 | L6f/L6r cemented | -78.407 | 1.55 | 1.55298 | 55.07 | |
| 15 | L6r | 45.846 | 1.42 | | | |
| 16 | L7 | 3376.612 | 2.78 | 1.80835 | 40.55 | aspheric (16A) |
| 17 | L7 | 120.496 | 18.74 (BF) | | | aspheric (17A) |

Aspheric surfaces (Table 2). Sag formula as printed (数1, PDF page 7):

    Z = C H^2 / (1 + sqrt(1 - (1 + K) C^2 H^2)) + A4 H^4 + A6 H^6 + A8 H^8 + A10 H^10 + A12 H^12 + A14 H^14
    C = 1/R,  H = sqrt(X^2 + Y^2)

K is the conic constant itself (the formula has `1 + K`, so K = 0 is a
sphere); there is no H^2 term; "E" means ×10^. R is the paraxial radius from
Table 1.

| Surface | K | A4 | A6 | A8 | A10 | A12 | A14 |
|---|---|---|---|---|---|---|---|
| 1 | -0.02238 | -1.02810E-06 | 6.04388E-10 | -4.31143E-12 | 5.62572E-15 | -3.29805E-18 | -5.97602E-23 |
| 16 | -20 | 2.43710E-05 | -9.23649E-08 | 1.14851E-09 | -1.18851E-11 | 5.37423E-14 | -9.17045E-17 |
| 17 | 20 | 3.37528E-05 | -2.40831E-08 | -1.16539E-10 | 1.95240E-13 | -7.79650E-16 | 2.20734E-18 |

## Cosina Voigtländer Nokton 50 mm F1.2 (X-mount, APS-C, 2025)

Folder: `nokton-50mm-f1.2-xmount/`

This is not the full-frame "Nokton 50mm F1.2 Aspherical". It is the APS-C
X-mount lens (Cosina spec: 9 elements in 8 groups, no aspherics, angle of view
32.5°, 75 mm equivalent).

Sources:
- JP 2025-058577 A, application JP 2023-168590 (Cosina), filed 2023-09-28,
  published 2025-04-09. https://patents.google.com/patent/JP2025058577A/ja,
  PDF `JP2025058577A.pdf`.
- photonstophotos: `JP2025-058577_Example01P`, copied here as
  `photonstophotos_JP2025-058577_Example01P.txt` ("Cosina Nokton 50mm F1.2").
  Bill's note: magnification measured, glass names (Hoya FD225, E-ADF50) are
  his, not the patent's.
- fujiaddict.com (2025-04-12) via asobinet: Example 1 = 50 mm F1.2,
  Example 3 = 70 mm F1.2.

Which embodiment: Example 1 (Table 1, PDF page 9): f = 48.5 mm, F 1.23,
half angle ω = 16.28°, back focus BF = 12.57 mm, total length TTL = 65 mm,
TTL/f = 1.34, 9 elements in 8 groups, no aspheric surfaces. This is the
production X-mount lens. Example 2 (Table 2, PDF page 12) is the same layout
with an aspheric front surface and a cemented rear pair (f = 48.5, F 1.23,
ω = 16.19°, BF 15.96); it is transcribed below as well since it is the only
aspheric variant. Example 3 is the 70 mm (ω = 10.89°) and was not transcribed.

Focusing: the whole lens moves. Table 1 lists the variable gaps ZD0 (object
distance) = infinity / 369.5 mm and ZD18 (back focus) = 12.57 / 19.36 mm.
Marginal ray half-heights at infinity given in the text: Hh = 19.75 mm at
surface 1, Hs = 10.54 mm at surface 10 (the smallest beam), which bound the
clear apertures of those two surfaces.

### Example 1 (f = 48.5 mm, F 1.23, ω = 16.28°) — production match

| Surface | Element | Radius (mm) | Thickness (mm) | n_d | V_d | Note |
|---|---|---|---|---|---|---|
| 1 | L11 | 53.73 | 4.26 | 1.72916 | 54.67 | |
| 2 | L11 | 138.48 | 0.15 | | | |
| 3 | L12 | 36.67 | 3.67 | 1.72916 | 54.67 | |
| 4 | L12 | 52.00 | 0.30 | | | |
| 5 | L13 | 26.68 | 6.51 | 1.72916 | 54.67 | |
| 6 | L13 | 85.50 | 2.98 | | | |
| 7 | L14 | 204.14 | 1.40 | 1.74077 | 27.74 | |
| 8 | L14 | 38.68 | 2.24 | | | |
| 9 | L15 | 47.82 | 1.20 | 1.76182 | 26.58 | |
| 10 | L15 | 14.26 | 7.54 | | | |
| **AS** (11) | stop | flat | 1.15 | | | |
| 12 | L21 | 66.00 | 1.10 | 1.80809 | 22.76 | |
| 13 | L21/L22 cemented | 26.13 | 4.83 | 1.90043 | 37.37 | |
| 14 | L22 | -145.01 | 7.09 | | | |
| 15 | L31 | 2707.72 | 3.83 | 1.90043 | 37.37 | |
| 16 | L31 | -25.97 | 2.97 | | | |
| 17 | L32 | -26.07 | 1.20 | 1.65412 | 39.68 | |
| 18 | L32 | -300.00 | ZD18 = 12.57 (BF, infinity) / 19.36 (at 369.5 mm) | | | |

### Example 2 (f = 48.5 mm, F 1.23, ω = 16.19°), aspheric variant

| Surface | Element | Radius (mm) | Thickness (mm) | n_d | V_d | Note |
|---|---|---|---|---|---|---|
| 1 | L11 | 48.77 | 3.48 | 1.80835 | 40.55 | aspheric (1A) |
| 2 | L11 | 92.85 | 0.15 | | | |
| 3 | L12 | 33.99 | 4.28 | 1.72916 | 54.67 | |
| 4 | L12 | 52.14 | 0.30 | | | |
| 5 | L13 | 29.51 | 7.49 | 1.72916 | 54.67 | |
| 6 | L13 | 345.50 | 1.19 | | | |
| 7 | L14 | 91.40 | 1.65 | 1.74077 | 27.74 | |
| 8 | L14 | 17.45 | 3.49 | | | |
| 9 | L15 | 24.35 | 1.50 | 1.76182 | 26.58 | |
| 10 | L15 | 15.58 | 7.19 | | | |
| **AS** (11) | stop | flat | 1.00 | | | |
| 12 | L21 | 23.29 | 1.80 | 1.80518 | 25.46 | |
| 13 | L21/L22 cemented | 14.05 | 4.05 | 1.90043 | 37.37 | |
| 14 | L22 | 19.98 | 3.65 | | | |
| 15 | L31 | 54.46 | 5.94 | 1.90043 | 37.37 | |
| 16 | L31/L32 cemented | -17.55 | 1.89 | 1.62004 | 36.40 | |
| 17 | L32 | -377.56 | ZD17 = 15.96 (BF, infinity) / 22.75 (at 368.3 mm) | | | |

Aspheric surface 1 of Example 2. Sag formula as printed (数1, PDF page 11):

    Z = C H^2 / (1 + sqrt(1 - (1 + K) C^2 H^2)) + A4 H^4 + A6 H^6 + A8 H^8
    C = 1/R,  H = sqrt(X^2 + Y^2)

Same convention as the F1 patent: K is the conic constant (`1 + K` in the
root), no H^2 term, "E" = ×10^.

| Surface | K | A4 | A6 | A8 |
|---|---|---|---|---|
| 1 | 0 | -2.11597E-06 | -1.16190E-09 | -1.93368E-12 |

## Verification

A thin paraxial trace (y = 1, u = 0 at the first surface, object at
infinity) of every table above, run on the transcribed numbers:

| Table | EFL (trace) | EFL (patent) | BF (trace) | BF (patent) |
|---|---|---|---|---|
| Nokton US 2,645,155 | 1.00002 | 1.00 | 0.6351 | not stated |
| Nokton US 2,646,721 | 1.00000 | 1.0 | 0.6002 | not stated |
| Apo-Lanthar US 2,645,154 | 1.00002 | 1.0000 | 0.86636 | 0.8663 |
| Ultron US 2,627,204 Ex I | 0.99999 | 1.0 | 0.71766 | 0.7177 |
| Ultron US 2,627,204 Ex II | 1.00006 | 1.0 | 0.69730 | 0.6972 |
| US 2,627,205 example | 1.00001 | 1.0 | 0.69826 | 0.698 |
| Nokton F1 JP 2023-063766 Ex 1 | 49.998 | 50 | 18.732 | 18.74 |
| Nokton F1.2 JP 2025-058577 Ex 1 | 48.476 | 48.5 | 12.557 | 12.57 |
| Nokton F1.2 JP 2025-058577 Ex 2 | 48.546 | 48.5 | 15.983 | 15.96 |

Every table reproduces the patent's own numbers, so no transcription error of
the kind a single wrong digit would produce is present. Values that are
estimates or print oddities are flagged inline: the Nokton stop split
(estimated from the drawing), the "[f=100" header in US 2,645,155, and the
missing "=" at R4 in US 2,646,721.
