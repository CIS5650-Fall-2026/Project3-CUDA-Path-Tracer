# Zeiss lens prescriptions from patents

Collected 2026-10-04 for the realistic-camera lens table. One subfolder per lens holds the
patent PDF(s) and a copy of the photonstophotos Optical Bench data file for the same
example (Bill Claff's own transcription, used here only as a cross-check; its `[lens data]`
columns are surface, radius, thickness, n_d, diameter, V_d, all already scaled to the
production focal length).

Every number in the tables below was read off the patent page images (pdftoppm at 200 to
300 dpi), not the OCR text layer; Google Patents' OCR of the 1970s tables is badly garbled
(0.88904 came out as 0.80904, 0.68124 as 0.6824, and so on), so do not trust the text layer
of those PDFs. Where the Optical Bench file disagrees with the page image it is flagged.

## Conventions shared by all the patents here

- Light travels left to right. A radius is positive when its center of curvature lies on the
  image side (surface convex toward the object), negative otherwise. `inf` is a plane.
- Thickness on a row is the axial distance from that surface to the next one. Glass rows
  carry n_d and V_d (d-line); the 1930s patents write `nd`/`v`, the German one `nD`/`ν`.
- None of the patents gives clear apertures or element diameters. The `opticalbench_*.txt`
  files carry Claff's own diameter column (column 5); those are his estimates from the
  drawing, not patent data.
- None of the patents says where inside the stop space the iris sits. The split used by
  Optical Bench is noted per lens so the two can be reconciled; it is a choice, not data.
- Scale: the 1970s patents normalize to F = 1.0000 (every length is a fraction of the
  focal length); the 1930s and 1950s ones normalize to f = 100 mm. The "mm" columns below
  multiply by the production focal length (50, 35, 21).

## How the lenses were found

The hub page https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBenchHub.htm
embeds its table as a JS array (`PatentID`, `ExampleID`, `Description`); each row links to
`OpticalBench.htm#Data/<PatentID>_<ExampleID>[_<TaleID>].txt`, and the data file is served at
`https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Data/<same name>.txt`.
Zeiss rows relevant to the brief:

| Hub description | PatentID | Example | Fetched |
|---|---|---|---|
| Zeiss Planar T f1.4 50mm (Contax) | US 3,874,771 | Example 5 | yes |
| Zeiss Distagon T f1.4 35mm Contax | US 3,915,558 | Example 8 | yes |
| Zeiss Sonnar 50mm F1.5 Opton Contax | US 2,186,621 | Example 1 (Fig. 1 data) | yes |
| Zeiss Biogon 2.1cm f/4.5 | US 2,721,499 | Example 2 | yes |
| Zeiss Biogon 35mm F2.8 [pre-war] | US 2,084,309 | Example 1 | yes |
| Zeiss Biogon 35mm F2.8 Contax Opton | JP 1955-004837 (JP S30-4837 B) | Example 1 | no, Japanese document, not pursued |
| Zeiss Distagon T f2 28mm Contax | DE 2 359 156 | Example 12 | no |
| Zeiss Distagon T f2.8 25mm Contax | DE 1 250 153 | Example 2 | no |
| Zeiss Planar 50mm F1.4 (Praktica) | GB 2 066 504 | Example 4 | no (Jena lens, not Oberkochen) |
| Otus 1.4/55 | none listed | | see below |

Otus 55 f/1.4: no entry on the hub. The dpreview "Patents related to photo lenses" thread
(https://www.dpreview.com/forums/post/64312894) reports that no optical patent matching
the Otus lenses has been found; the only Otus-related patents found there are Cosina
mechanical ones (JP2015219385A, JP2019128471A). Google Patents assignee searches for
Carl Zeiss 2012 to 2014 "photographic lens" turned up nothing matching a 12-element f/1.4
55 mm with a double-sided asphere. Not delivered.

Patent PDFs: Google Patents stores them at `patentimages.storage.googleapis.com/<hash>/USnnnnnnn.pdf`
(link in the page HTML) and the legacy path `patentimages.storage.googleapis.com/pdfs/USnnnnnnn.pdf`
still works for old US numbers. Google Patents started returning 503/"Sorry" to both
WebFetch and curl midway through, so DE 530 843 was pulled from DEPATISnet instead
(session cookie from `depatisnet?action=bibdat&docid=DE000000530843A`, then one PDF per
page from `depatisnet/DE000000530843A_<n>.pdf?...action=showpdf...&pdfpage=<n>`, merged
with pypdf). Espacenet and the DPMA register both refuse unauthenticated fetches (403/401).

---

## 1. Planar 50 mm f/1.4 (Contax/Yashica, 1975) — `planar-50-f1.4/`

- Patent: **US 3,874,771**, "Photographic objective of the extended Gauss type",
  Karl-Heinrich Behrens and Erhard Glatzel, Carl Zeiss Stiftung (Oberkochen).
  Priority DE 22 32 101 of June 30, 1972 (the application number is from the OCR, the
  page image shows `223210` cut off at the margin); filed June 25, 1973; granted
  April 1, 1975. PDF: `planar-50-f1.4/US3874771.pdf` (9 pages), from
  https://patentimages.storage.googleapis.com/c9/bc/f1/94b1c184227c25/US3874771.pdf
- Source page: https://patents.google.com/patent/US3874771A/en
- Embodiment: **Example 5 (Table 5, PDF page 6)**. Examples 1 to 3 are f/1.5 or faster and
  list only n; examples 4 and 5 are the f/1.4 "fine-corrected forms for color photography"
  with n and V. Optical Bench maps Example 5 to the Contax Planar T* 50/1.4 (7 elements in
  6 groups, elements 4 and 5 cemented, which is the production construction).
- Given by the patent: F = 1.0000 (unit), f/1.4, back focus s'∞ = +0.70993 F,
  overall length OAL = 0.819670 F (first to last vertex). At 50 mm: back focus 35.50 mm,
  OAL 40.98 mm. No field angle, no diameters.
- Scale: F = 1; the mm columns are ×50.

| # | Surface | R (F=1) | R (mm) | t (F=1) | t (mm) | n_d | V_d | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | R1 | +0.88904 | +44.4520 | 0.101821 | 5.09105 | 1.71700 | 47.99 | L1 |
| 2 | R1' | +5.77321 | +288.6605 | 0.001177 | 0.05885 | | | air s12 |
| 3 | R2 | +0.43607 | +21.8035 | 0.097113 | 4.85565 | 1.78831 | 47.37 | L2 |
| 4 | R2' | +0.68124 | +34.0620 | 0.031782 | 1.58910 | | | air s23 |
| 5 | R3 | +0.94172 | +47.0860 | 0.023739 | 1.18695 | 1.68893 | 31.17 | L3 |
| 6 | R3' | +0.30648 | +15.3240 | 0.285453 | 14.27265 | | | **stop space CS = s34; iris inside** |
| 7 | R4 | -0.34637 | -17.3185 | 0.022954 | 1.14770 | 1.72830 | 28.68 | L4 |
| 8 | R4' = R5 | +7.48043 | +374.0215 | 0.102998 | 5.14990 | 1.78831 | 47.37 | cemented (s45 = 0), L5 |
| 9 | R5' | -0.60714 | -30.3570 | 0.005689 | 0.28445 | | | air s56 |
| 10 | R6 | -1.74852 | -87.4260 | 0.087696 | 4.38480 | 1.78831 | 47.37 | L6 |
| 11 | R6' | -0.56500 | -28.2500 | 0.002550 | 0.12750 | | | air s67 |
| 12 | R7 | +3.29359 | +164.6795 | 0.056698 | 2.83490 | 1.74400 | 44.77 | L7 |
| 13 | R7' | -2.61616 | -130.8080 | 0.70993 | 35.4965 | | | to image (s'∞) |

Stop: Optical Bench places the iris at the midpoint of CS (7.136325 mm after surface 6,
7.136325 mm before surface 7), with a stop diameter of 24.289 mm. Patent only says
"stop space". Cross-check: every Optical Bench value equals the patent value × 50 exactly.

---

## 2. Sonnar 5 cm f/1.5 (Bertele, Contax, 1932) — `sonnar-50-f1.5/`

Two patents carry a complete 7-element f/1.5 prescription, and they differ. Both are
transcribed. The founding Sonnar patent DE 530 843 is also included but contains only
4- and 5-element examples, not the production 6-element f/2 or 7-element f/1.5.

### 2a. US 1,975,678 — the original 5 cm f/1.5 (the one to use for "Sonnar 1932")

- **US 1,975,678**, "Objective", Ludwig Bertele, Zeiss Ikon AG Dresden. Priority
  Germany July 8, 1932; filed July 3, 1933; granted Oct 2, 1934. PDF:
  `sonnar-50-f1.5/US1975678.pdf` (2 pages), from
  https://patentimages.storage.googleapis.com/pdfs/US1975678.pdf
  (Google Patents page https://patents.google.com/patent/US1975678A/en was 503 at the time.)
- Single example, "1:1.5, f = 100, picture angle about 42°", 7 lenses in 3 groups
  (L1 | L2 L3 L4 cemented | L5 L6 L7 cemented), iris "y" between L4 and L5. Note the text
  calls L1 "plano-convex" and L5 "plano-concave" although r2 and r7 in the table are long
  finite radii; the table governs.
- No back focus, no diameters in the patent.
- Scale f = 100; mm columns ×0.5.

| # | Surface | r (f=100) | r (mm) | t (f=100) | t (mm) | n_d | V_d | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | r1 | +65.0 | +32.500 | 10.5 | 5.25 | 1.6375 | 56.1 | L1 |
| 2 | r2 | +416.77 | +208.385 | 0.5 | 0.25 | | | air l1 |
| 3 | r3 | +37.26 | +18.630 | 11.7 | 5.85 | 1.6727 | 47.3 | L2 |
| 4 | r4 | +104.34 | +52.170 | 7.6 | 3.80 | 1.4675 | 65.7 | L3 (cemented) |
| 5 | r5 | -247 | -123.5 | 1.9 | 0.95 | 1.6890 | 31.0 | L4 (cemented) |
| 6 | r6 | +22.14 | +11.070 | 13.9 | 6.95 | | | **air l2, iris y inside** |
| 7 | r7 | +1904.0 | +952.0 | 3.4 | 1.70 | 1.5481 | 45.9 | L5 |
| 8 | r8 | +59.85 | +29.925 | 22.4 | 11.20 | 1.6578 | 51.2 | L6 (cemented) |
| 9 | r9 | -22.06 | -11.030 | 8.4 | 4.20 | 1.5488 | 63.0 | L7 (cemented) |
| 10 | r10 | -89.06 | -44.530 | | | | | to image |

All digits read from a 300 dpi crop of page 2; the table is clean. No Optical Bench file
exists for this patent, so no independent cross-check.

### 2b. US 2,186,621, Fig. 1 data — the variant Optical Bench uses for "Sonnar 50mm F1.5 Opton Contax"

- **US 2,186,621**, "Lens system", Ludwig Bertele, Zeiss Ikon AG Dresden. Priority
  Germany July 13, 1937; filed June 15, 1938; granted Jan 9, 1940. PDF:
  `sonnar-50-f1.5/US2186621.pdf` (4 pages), from
  https://patentimages.storage.googleapis.com/c8/ee/0b/258bc74e512816/US2186621.pdf
- Source page: https://patents.google.com/patent/US2186621A/en
- Embodiment: the "Fig. 1" data on page 2 (col. 2, lines 29 to 42). The patent itself calls
  Fig. 1 prior art ("see Letters Patent No. 1,975,678") and claims the 4-unit Fig. 2 and
  Fig. 3 forms that add a rear plano-convex element to cut distortion. Optical Bench's
  `US002186621_Example01P` is this Fig. 1 table and it labels it the post-war Opton
  Sonnar 50/1.5; the Fig. 2 form was not sold as a 50 mm as far as the research found.
  Glasses differ from 2a (notably the 1.4892/70.1 fluor crown in L3 instead of 1.4675).
- Given: 1:1.5, f = 100 mm. No back focus (Optical Bench computes 22.27 mm at 50 mm).
- Scale f = 100; mm ×0.5.

| # | Surface | R (f=100) | R (mm) | t (f=100) | t (mm) | n_d | V_d | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | R1 | +69.21 | +34.605 | 9.33 | 4.665 | 1.6710 | 47.2 | L1 |
| 2 | R2 | +433.84 | +216.920 | 0.38 | 0.190 | | | air l1 |
| 3 | R3 | +35.86 | +17.930 | 11.81 | 5.905 | 1.6710 | 47.2 | L2 |
| 4 | R4 | +85.78 | +42.890 | 7.05 | 3.525 | 1.4892 | 70.1 | L3 (cemented) |
| 5 | R5 | -646.31 | -323.155 | 1.90 | 0.950 | 1.7394 | 28.2 | L4 (cemented) |
| 6 | R6 | +23.51 | +11.755 | 15.24 | 7.620 | | | **air l2, iris inside** |
| 7 | R7 | inf | inf | 2.48 | 1.240 | 1.5232 | 50.9 | L5 |
| 8 | R8 | +51.09 | +25.545 | 19.81 | 9.905 | 1.6578 | 51.2 | L6 (cemented) |
| 9 | R9 | -22.12 | -11.060 | 4.57 | 2.285 | 1.5894 | **61.2** | L7 (cemented) |
| 10 | R10 | -103.13 | -51.565 | | | | | to image |

Flag: the Optical Bench file gives V = 51.2 for L7; the page image (300 dpi crop) clearly
reads `v = 61,2`, and n = 1.5894 with V = 61 is a SK5-class dense barium crown, so 61.2 is
taken as correct. Everything else in the OB file matches the patent × 0.5. OB splits l2 as
6.49 mm + 1.13 mm around a 19.076 mm stop.

Fig. 2 data from the same patent (the claimed 4-unit form, 1:1.5, f = 100), read from the
200 dpi page-3 image, moderate confidence, no cross-check:
R1 +65.16, d1 9.30 (1.6689/48.8); R2 +322.58, l1 0.23; R3 +35.86, d2 11.10 (1.6689/48.8);
R4 +77.31, d3 7.40 (1.4645/65.7); R5 inf, d4 1.90 (1.7219/28.7); R6 +23.38, l2 15.56;
R7 +914.00, d5 3.04 (1.5325/46.2); R8 +53.13, d6 19.92 (1.6689/48.8); R9 -22.09,
d7 2.37 (1.6043/56.3); R10 -227.7, l3 0.10; R11 +272.8, d8 4.50 (1.6689/48.8); R12 inf.

### 2c. DE 530 843 — founding Sonnar patent (f/1.6 and f/2 examples, f ≈ 100)

- **DE 530 843**, "Photographisches Objektiv", Zeiss Ikon AG Dresden (Bertele is not
  named on the document). Patented in the German Reich from Aug 14, 1929; grant
  announced July 16, 1931; issued Aug 29, 1931. Klasse 42h Gruppe 4. PDF:
  `sonnar-50-f1.5/DE530843.pdf` (3 pages, merged from DEPATISnet page PDFs, document id
  DE000000530843A; https://depatisnet.dpma.de/DepatisNet/depatisnet?action=bibdat&docid=DE000000530843A).
- Both examples state "Gesamtbrennweite etwa 100 mm". The patent does not state f/2 for a
  6-element lens; its examples are:

Beispiel 1, Öffnungsverhältnis 1:1.6 (4 elements, L2 and L3 cemented), f ≈ 100:

| # | Surface | r | t | n_D | ν | Note |
|---|---|---|---|---|---|---|
| 1 | r1 | +75.90 | 10.5 | 1.6228 | 59.9 | L1 |
| 2 | r2 | +375.0 | 0.6 | | | air l1 |
| 3 | r3 | +42.3 | 24.0 | 1.5888 | 61.0 | L2 |
| 4 | r4 | -141.0 | 9.0 | 1.7174 | 29.5 | L3 (cemented) |
| 5 | r5 | +27.6 | 21.0 | | | air l2 (stop) |
| 6 | r6 | +75.0 | 7.5 | 1.6261 | 39.1 | L4 |
| 7 | r7 | -204.0 | | | | |

Beispiel 2, Öffnungsverhältnis 1:2 (5 elements, L2 L3 L4 cemented), f ≈ 100:

| # | Surface | r | t | n_D | ν | Note |
|---|---|---|---|---|---|---|
| 1 | r1 | +70.2 | 6.6 | 1.6073 | 59.5 | L1 |
| 2 | r2 | +610.2 | 0.15 | | | air l1 |
| 3 | r3 | +39.45 | 6.0 | 1.6073 | 59.5 | L2 |
| 4 | r4 | +75.00 | 21.6 | 1.5101 | 63.4 | L3 (cemented) |
| 5 | r5 | -180.0 | 1.5 | 1.7224 | 29.5 | L4 (cemented) |
| 6 | r6 | +26.25 | 27.3 | | | air l2 (stop) |
| 7 | r7 | +70.2 | 4.2 | 1.6738 | 32.1 | L5 |
| 8 | r8 | -858.33 | | | | |

Read from 300 dpi crops of the DEPATISnet scan; the old-face digits are clean. Not
cross-checked against any second source.

### 2d. US 1,998,704 (priority DE Sept 1, 1931; filed Aug 31, 1932; granted Apr 23, 1935)

Bertele / Zeiss Ikon, "Photographic objective": claims only (cemented collecting surface in
the rear member), no numeric table. PDF kept as `sonnar-50-f1.5/US1998704.pdf` from
https://patentimages.storage.googleapis.com/11/4a/f0/bb2cacc90d3aab/US1998704.pdf. The
brief's guess "DE 530,843 and US 1,998,704" is therefore half right: 530 843 exists and
is the Sonnar origin, but neither it nor 1,998,704 holds the production f/1.5 data; that
is US 1,975,678.

---

## 3. Biogon 21 mm f/4.5 (Bertele, Contax IIa/IIIa, 1954) — `biogon-21-f4.5/`

- Patent: **US 2,721,499**, "Five component wide-angle objective", Ludwig Bertele,
  Heerbrugg, Switzerland (no assignee on the face). Priority Switzerland July 12, 1951;
  filed July 5, 1952; granted Oct 25, 1955. PDF: `biogon-21-f4.5/US2721499.pdf` (4 pages),
  from https://patentimages.storage.googleapis.com/54/fc/d8/3eb1feff907bf7/US2721499.pdf
- Source page: https://patents.google.com/patent/US2721499A/en (the brief's number is
  confirmed). The hub also lists the pre-war Biogon under US 2,084,309, section 5 below.
- Embodiment: **Example 2 (PDF page 3)**, 1:4.5. Example 1 is 1:6.3, Example 3 is 1:3.4.
  All three are f = 100 mm with "image angle of about 90°". Optical Bench
  (`US002721499_Example02_Tale01`) maps Example 2 to the Biogon 2.1 cm f/4.5; the same
  numbers scale to the 38 mm (6x6), 53 mm (6x9) and 75 mm (9x12) Biogons.
- Given: f = 100 mm, 1:4.5. The text also states, for Example 2: separation of the two
  convex outer surfaces of C and D = 0.92 f (r5 to r11 = 92.0, matches), cemented radii
  0.254 f and 0.255 f, sum 0.509 f, air space C to D = 0.047 f, V sum of A+B = 137.0.
  No back focus (Optical Bench computes 10.35 mm at 21 mm, i.e. 49.3 at f = 100). The
  iris is "to be located in the third air space" l3 (between L4 and L5).
- Scale f = 100; mm ×0.21.

| # | Surface | r (f=100) | r (mm) | t (f=100) | t (mm) | n_D | V | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | r1 | +109.14 | +22.9194 | 3.7 | 0.777 | 1.50380 | 66.7 | L1 (component A) |
| 2 | r2 | +52.63 | +11.0523 | 13.2 | 2.772 | | | air l1 |
| 3 | r3 | +110.25 | +23.1525 | 3.7 | 0.777 | 1.48697 | 70.3 | L2 (B) |
| 4 | r4 | +50.72 | +10.6512 | 35.0 | 7.350 | | | air l2 |
| 5 | r5 | +56.24 | +11.8104 | 29.3 | 6.153 | 1.72050 | 50.3 | L3 (C) |
| 6 | r6 | +25.37 | +5.3277 | 13.3 | 2.793 | 1.60739 | 59.5 | L4 (cemented) |
| 7 | r7 | -194.92 | -40.9332 | 4.7 | 0.987 | | | **air l3, iris inside** |
| 8 | r8 | -252.70 | -53.0670 | 2.8 | 0.588 | 1.56993 | 57.5 | L5 (D) |
| 9 | r9 | +30.59 | +6.4239 | 23.7 | 4.977 | 1.62500 | 53.3 | L6 (cemented) |
| 10 | r10 | -25.51 | -5.3571 | 18.2 | 3.822 | 1.71966 | 29.3 | L7 (cemented) |
| 11 | r11 | -57.38 | -12.0498 | 39.0 | 8.190 | | | air l4 |
| 12 | r12 | -40.15 | -8.4315 | 9.8 | 2.058 | 1.64200 | 58.1 | L8 (E) |
| 13 | r13 | -102.64 | -21.5544 | | | | | to image |

Flags: the Optical Bench file has n = 1.72020 for L3 and 1.71965 for L7; the page image
reads 1.72050 (same glass as Example 1's L3, also printed 1.72050) and 1.71966. Patent
values are used. OB splits l3 as 0.274 + 0.713 mm around a 5.932 mm stop. All radii and
thicknesses in the OB file equal patent × 0.21.

---

## 4. Distagon 35 mm f/1.4 (Contax/Yashica, 1975) — `distagon-35-f1.4/`

- Patent: **US 3,915,558**, "High power wide-angle lens", Erhard Glatzel, Carl Zeiss
  Stiftung (Oberkochen). Priority DE 23 06 346 of Feb 9, 1973; filed Feb 1, 1974; granted
  Oct 28, 1975. PDF: `distagon-35-f1.4/US3915558.pdf` (17 pages), from
  https://patentimages.storage.googleapis.com/29/61/87/569ead4e29be3e/US3915558.pdf
- Source page: https://patents.google.com/patent/US3915558A/en
- Embodiment: **Example 8 (PDF page 12, columns 11 and 12, continued from the bottom of
  column 12)**. Eight examples, f/1.8 to f/1.2; 8 components (front member 4, rear member
  4); examples 7 and 8 are the two with V listed. Optical Bench maps Example 8 to the
  Contax Distagon T* 35/1.4. Element count: component 3 is two air-spaced elements (3a,
  3b), so 9 elements; the production lens is listed as 9 elements in 8 groups, which fits
  if the touching surfaces R6' / R7 (see note) are cemented.
- Given: F = 1.0000, f/1.4, back focus s' = +0.98567 F (34.498 mm at 35 mm). No overall
  length, no field (the general text says more than 60°), no diameters.
- Scale F = 1; mm ×35.

| # | Surface | R (F=1) | R (mm) | t (F=1) | t (mm) | n_d | V_d | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | R1 | +8.3325 | +291.6375 | 0.06327 | 2.21445 | 1.5827 | 46.5 | comp. 1 |
| 2 | R1' | +0.9281 | +32.4835 | 0.32654 | 11.42890 | | | air S1 |
| 3 | R2 | +1.3301 | +46.5535 | 0.09243 | 3.23505 | 1.5481 | 45.8 | comp. 2 |
| 4 | R2' | +0.9281 | +32.4835 | 0.33507 | 11.72745 | | | air S2 (air lens α) |
| 5 | R3a | +1.3016 | +45.5560 | 0.18569 | 6.49915 | 1.7130 | 53.9 | comp. 3a |
| 6 | R3a' | -11.273 | -394.5550 | 0.08308 | 2.90780 | | | air S3a (air lens 3i) |
| 7 | R3b | -47.565 | -1664.7750 | 0.13480 | 4.71800 | 1.7130 | 53.9 | comp. 3b |
| 8 | R3b' | -2.0928 | -73.2480 | 0.00138 | 0.04830 | | | air S3b |
| 9 | R4 | +0.7809 | +27.3315 | 0.18377 | 6.43195 | 1.7130 | 53.9 | comp. 4 |
| 10 | R4' | +0.8332 | +29.1620 | 0.23576 | 8.25160 | | | **stop space S4 = CS; iris inside** |
| 11 | R5 | -6.9100 | -241.8500 | 0.06768 | 2.36880 | 1.5481 | 45.8 | comp. 5, **aspheric** |
| 12 | R5' | +1.1271 | +39.4485 | 0.12765 | 4.46775 | | | air S5 |
| 13 | R6 | -1.1854 | -41.4890 | 0.04429 | 1.55015 | 1.8467 | 23.8 | comp. 6 |
| 14 | R6' | +1.3987 | +48.9545 | 0.00000 | 0.00000 | | | S6 = 0, see note |
| 15 | R7 | +1.3987 | +48.9545 | 0.19917 | 6.97095 | 1.7883 | 47.4 | comp. 7 |
| 16 | R7' | -0.9416 | -32.9560 | 0.01706 | 0.59710 | | | air S7 |
| 17 | R8 | +8.8250 | +308.8750 | 0.10536 | 3.68760 | 1.7883 | 47.4 | comp. 8 |
| 18 | R8' | -1.50293 | -52.6026 | 0.98567 | 34.49845 | | | to image (s') |

Note on surfaces 14/15: the patent table lists R6' and R7 as separate surfaces of equal
radius with S6 = 0.0000, and the text (col. 12, lines 59 to 68) says the two were made
equal "in length as well as in sign" so that the designer may cement them. For a
sequential ray tracer treat them as one cemented surface R = +48.9545 mm between
n = 1.8467 and n = 1.7883; a zero-thickness air gap would otherwise produce two refractions
and TIR trouble at the margin.

Stop: Optical Bench splits CS as 6.5576 mm after surface 10 and 1.6940 mm before surface
11 with a 28.454 mm stop diameter; that split is Claff's, not the patent's.

Aspheric surface (surface 11, R5). The patent (col. 15, lines 15 to 48 and col. 15 bottom,
PDF page 14) says that in all examples except 1 and 2 the front concave surface of
component 5 is aspheric, and defines the sag by the "camber expression"

    P = c1 H^2 + c2 H^4 + c3 H^6 + c4 H^8 + c5 H^10

with H the height above the axis at the surface, P the sag measured along the axis in the
same sign convention as the radius (P = c1 H^2 alone is the paraxial parabola of a surface
of radius R5), and for every example `c1 = (2 R5)^-1`, "R5 is the vertex radius of the
concave surface at the place of its surface intersection point with the optical axis".
For Example 8:

    c1 = 1 / (2 R5) = 1 / (2 × (-6.9100))   [F = 1 units]
    c2 = -7.934 0850 × 10^-1
    c3 = c4 = c5 = 0

This is a pure power series: there is no spherical base and no conic constant. In the
usual form z = c H^2 / (1 + sqrt(1 - (1 + k) c^2 H^2)) + A4 H^4 + ..., the patent's surface
is exactly reproduced by k = -1 (κ, the conic constant; in the "K = k + 1" convention
K = 0), c = 1/R5, A4 = c2, A6 = A8 = A10 = 0, because k = -1 makes the first term the
parabola H^2/(2R). Scaling to f = 35 mm: R = -241.85 mm, A4 = c2 / 35^3 = -0.79340850 /
42875 = -1.8505 × 10^-5 mm^-3. The Optical Bench file carries exactly that
(`[aspherical data] 11 -241.8500 -1 -1.8505155E-05`, i.e. surface, R, conic, A4).
Sign: with R5 negative both terms push the surface toward the object as H grows, so the
margin is more strongly curved than the vertex parabola; use the signed R directly. The
exponent of c2 was read as 10^-1 from the 200 dpi image (the OCR has "10" with the
exponent lost); it is consistent with the OB value and with the other examples' listing
(Example 4 has c3 at 10^-3, c4 and c5 at 10^-4).

Cross-check: every radius and thickness in the OB file equals the patent value × 35.

---

## 5. Biogon 3.5 cm f/2.8, pre-war (Bertele, Contax, 1936) — `biogon-35-f2.8-prewar/`

Included because the hub lists it and the PDF was at hand; the brief asked for 21 or 35.

- Patent: **US 2,084,309**, "Photographic lens system", Ludwig Bertele, Zeiss Ikon AG
  Dresden. Priority Germany June 16, 1934; filed Jan 6, 1936; granted June 22, 1937.
  PDF: `biogon-35-f2.8-prewar/US2084309.pdf` (3 pages), from
  https://patentimages.storage.googleapis.com/b6/9c/a5/9bf9ab4b474a64/US2084309.pdf
- Source page: https://patents.google.com/patent/US2084309A/en
- Embodiment: the single "Example" on PDF page 3 (col. 1, lines 10 to 27). 7 elements in
  4 units (L1 | L2 L3 L4 | L5 L6 | L7). The patent gives no f-number and no field angle in
  the example (the text says "large image angle ... at a great aperture"); Optical Bench
  lists it as 35 mm F2.8, 59°, Bf 6.48 mm (computed). Focal length is implied f = 100 by
  the Zeiss Ikon house style but is not printed on the example; flag.
- The post-war Contax Opton / Oberkochen Biogon 35/2.8 is a different computation
  (JP 1955-004837 per the hub), not fetched.
- Scale f = 100 assumed; mm ×0.35.

| # | Surface | R (f=100) | R (mm) | t (f=100) | t (mm) | n_d | V_d | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | R1 | +53.46 | +18.711 | 12.89 | 4.5115 | 1.6716 | 47.2 | L1 |
| 2 | R2 | +157.23 | +55.0305 | 1.57 | 0.5495 | | | air l1 |
| 3 | R3 | +33.33 | +11.6655 | 5.47 | 1.9145 | 1.6716 | 47.2 | L2 |
| 4 | R4 | +62.89 | +22.0115 | 5.22 | 1.8270 | 1.4645 | 65.7 | L3 (cemented) |
| 5 | R5 | inf | inf | 1.89 | 0.6615 | 1.6890 | 31.0 | L4 (cemented) |
| 6 | R6 | +25.94 | +9.0790 | 7.55 | 2.6425 | | | **air l2, iris inside** |
| 7 | R7 | +417.60 | +146.160 | 1.89 | 0.6615 | 1.4645 | 65.7 | L5 |
| 8 | R8 | +40.88 | +14.308 | 37.74 | 13.209 | 1.6716 | 47.2 | L6 (cemented) |
| 9 | R9 | -78.62 | -27.517 | 6.29 | 2.2015 | | | air l3 |
| 10 | R10 | -47.17 | -16.5095 | 25.16 | 8.806 | 1.5333 | 48.9 | L7 |
| 11 | R11 | -168.02 | -58.807 | | | | | to image |

OB file matches patent × 0.35 throughout; OB splits l2 as 1.842 + 0.800 mm around an
8.363 mm stop.

---

## Not found / not done

- Otus 1.4/55: no optical patent located (see top). The Zeiss datasheet gives only
  12 elements / 10 groups, one double-sided asphere, f/1.4 to f/16, image circle 43 mm.
- Biogon 35/2.8 Contax Opton (JP 1955-004837) and Contax Distagon 28/2 (DE 2 359 156,
  Ex. 12) and 25/2.8 (DE 1 250 153, Ex. 2): on the hub, patents not downloaded.
- Sonnar 85 mm f/2: not researched (the 50/1.5 was taken instead).
- Google Patents began rate-limiting (HTTP 503 / "Sorry") after about ten page fetches;
  the JP document and the remaining DE ones were not attempted after that.
