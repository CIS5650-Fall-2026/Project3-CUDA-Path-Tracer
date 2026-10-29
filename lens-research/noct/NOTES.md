# Nikon Noct lenses: prescription research

Two lenses. The Z 58mm f/0.95 S Noct has a published patent with a full
prescription and is fully transcribed below. The 1977 Noct-Nikkor 58mm f/1.2
has no published prescription anywhere I could find; what is known about it is
recorded in the second section so the search does not get repeated.

Files in this folder:

| File | What it is |
|---|---|
| `US20210208373A1.pdf` | US publication of the Z Noct patent (English, 25 pages, OCR text layer) |
| `WO2019229849A1.pdf` | PCT publication of the same patent (Japanese, 59 pages, image only) |
| `WO2019-229849_Example01P_photonstophotos.txt` | Bill Claff's Optical Bench data file for the same design (cross-check source) |
| `NOTES.md` | this file |

Both PDFs are encrypted with an empty user password (Google's patent images
are all like this). `pdftotext` and `pdftoppm` open them; the Read tool does
not. Rasterize a page with `pdftoppm -r 140 -f N -l N -png file.pdf out`.

---

## 1. Nikon NIKKOR Z 58mm f/0.95 S Noct (2019)

### Sources

- US 2021/0208373 A1, "Optical system, optical equipment, and manufacturing
  method for optical system", Nikon Corp. PCT/JP2018/020552, priority
  2018-05-29, published 2021-07-08.
  https://patents.google.com/patent/US20210208373A1/en
  PDF: https://patentimages.storage.googleapis.com/d2/eb/57/b798fbab0fab98/US20210208373A1.pdf
- WO 2019/229849 A1, same application, Japanese text.
  https://patents.google.com/patent/WO2019229849A1/en
  PDF: https://patentimages.storage.googleapis.com/33/b3/f1/7877a79cb02d50/WO2019229849A1.pdf
- Inventors (per cameragossip's Nikon patent database): Keisuke Tsubonoya,
  Hiroki Harada, Toshinori Take.
  https://cameragossip.github.io/nikon-lens-patents.html
- photonstophotos Optical Bench, entry "Nikon Nikkor Z 58mm f/0.95 S Noct" =
  WO2019-229849 Example 1:
  https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBench.htm#Data/WO2019-229849_Example01P.txt
  Raw data file: https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Data/WO2019-229849_Example01P.txt
- Marco Cavina, "Nikon Nikkor Z 58mm 1:0,95 S Noct e i segreti del suo schema
  ottico" (Italian), also identifies Example 1 as the production lens:
  https://www.nocsensei.com/camera/tecnica/marco-cavina/marcocavina/nikon-nikkor-z-58mm-1095-s-noct-e-i-segreti-del-suo-schema-ottico/

### Which embodiment

**Example 1 (Table 1)** is the production lens. Three independent reasons:

1. Element count. Example 1 has 17 elements in 10 groups (cemented pairs at
   surfaces 1-3, 4-6, 11-13, 15-17, 20-22, 23-25, 26-28; singlets at 7-8, 9-10,
   18-19), which is exactly Nikon's published construction for the lens.
   Example 2 has 16 elements in 9 groups, Example 3 has 32 surfaces at f/0.87,
   Example 4 is a 51.6 mm design.
2. Optical Bench maps the production lens to Example 1.
3. Cavina's article says the same.

Summary of the four examples (Examples 2 to 4 come from the WO text on Google
Patents, which is WIPO's text rather than OCR, but they were not checked
against the page images; medium confidence):

| Example | f (mm) | FNo | TL (mm) | Bf (mm) | Ymax (mm) | Aspheric surfaces |
|---|---|---|---|---|---|---|
| 1 | 59.62 | 0.98 | 160.74 | 17.10 | 21.70 | 1, 20, 28 |
| 2 | 58.93 | 0.98 | 160.71 | 17.12 | 21.70 | 1, 18, 26 |
| 3 | 56.61 | 0.87 | 185.01 | 15.72 | 21.70 | 1, 23, 32 |
| 4 | 51.60 | 0.98 | 145.01 | 17.50 | 21.70 | 2, 17, 26 |

### Example 1: stated specifications

From `[Various Data]` (WO page 20, US page 8):

| Quantity | Value |
|---|---|
| f | 59.62 mm |
| FNo | 0.98 |
| 2ω | 39.96° |
| Ymax | 21.70 mm |
| TL (surface 1 to image plane, on axis) | 160.74 mm |
| BF (last lens surface to image plane, through the filter) | 17.10 mm |
| BF, air-converted | 16.55 mm |

Lens group data (`[Lens Group Data]`, ST = starting surface):

| Group | ST | f (mm) |
|---|---|---|
| GF (front group, surfaces 1-22) | 1 | 75.60 |
| GR (rear group, surfaces 23-28) | 23 | 294.37 |
| G1 (surfaces 1-6) | 1 | -289.87 |
| G2 (surfaces 7-22) | 7 | 69.07 |

(Optical Bench lists GF as 75.37; the patent prints 75.60. The patent value
is the one transcribed above.)

### Example 1: surface data

Columns r, d, nd, vd, θgF are the patent's. The "CA" column is **not in the
patent**: it is Bill Claff's clear-aperture diameter from the Optical Bench
file, computed by his ray trace, so treat it as an estimate. nd is for
d-line 587.6 nm; air (nd = 1.0) rows are left blank. `*` marks an aspheric
surface, and the radius given for those is the paraxial radius.

| Surface | r (mm) | d (mm) | nd | vd | θgF | CA (mm, Claff) | Note |
|---|---|---|---|---|---|---|---|
| OP | ∞ | | | | | | object |
| *1 | 108.488 | 7.65 | 1.902650 | 35.77 | | 66.80 | L11, aspheric |
| 2 | -848.550 | 2.80 | 1.552981 | 55.07 | 0.54467 | 65.82 | L12 (cemented to L11) |
| 3 | 50.252 | 18.12 | | | | 57.94 | air lens La1 |
| 4 | -60.720 | 2.80 | 1.612660 | 44.46 | 0.56396 | 58.28 | L13 |
| 5 | 2497.500 | 9.15 | 1.593190 | 67.90 | | 65.32 | L14 (cemented to L13) |
| 6 | -77.239 | 0.40 | | | | 65.32 | |
| 7 | 113.763 | 10.95 | 1.848500 | 43.79 | | 70.90 | L21 |
| 8 | -178.060 | 0.40 | | | | 70.90 | |
| 9 | 70.659 | 9.74 | 1.593190 | 67.90 | | 65.00 | L22 |
| 10 | -1968.500 | 0.20 | | | | 65.00 | |
| 11 | 289.687 | 8.00 | 1.593190 | 67.90 | | 61.06 | L23 |
| 12 | -97.087 | 2.80 | 1.738000 | 32.33 | 0.58997 | 59.42 | L24 (cemented to L23) |
| 13 | 47.074 | 8.70 | | | | 50.24 | |
| **14 (S)** | ∞ | 5.29 | | | | 47.918 | **aperture stop** |
| 15 | -95.230 | 2.20 | 1.612660 | 44.46 | 0.56396 | 49.92 | L25 |
| 16 | 41.204 | 11.55 | 1.497820 | 82.57 | | 49.92 | L26 (cemented to L25) |
| 17 | -273.092 | 0.20 | | | | 49.92 | |
| 18 | 76.173 | 9.50 | 1.883000 | 40.69 | | 51.12 | L27 |
| 19 | -101.575 | 0.20 | | | | 51.12 | |
| *20 | 176.128 | 7.45 | 1.953750 | 32.33 | | 46.80 | L28, aspheric |
| 21 | -67.221 | 1.80 | 1.738000 | 32.33 | 0.58997 | 45.36 | L29 (cemented to L28) |
| 22 | 55.510 | D22 (variable) | | | | 39.84 | focusing gap |
| 23 | 71.413 | 6.35 | 1.883000 | 40.69 | | 39.46 | L31 |
| 24 | -115.025 | 1.81 | 1.698950 | 30.13 | | 39.46 | L32 (cemented to L31) |
| 25 | 46.943 | 0.80 | | | | 39.46 | |
| 26 | 55.281 | 9.11 | 1.883000 | 40.69 | | 38.94 | L33 |
| 27 | -144.041 | 3.00 | 1.765538 | 46.76 | | 38.28 | L34 (cemented to L33) |
| *28 | 52.858 | 14.50 | | | | 38.28 | aspheric, last lens surface |
| 29 | ∞ | 1.60 | 1.516800 | 64.14 | | 44.30 | filter group FL (low-pass filter block) |
| 30 | ∞ | 1.00 | | | | 44.30 | |
| I | ∞ | | | | | | image plane |

The stop diameter 47.918 mm is also Claff's number (set to reach FNo 0.98),
not a patent value. The patent gives no clear apertures at all.

Surfaces 29-30 are a 1.6 mm BK7-like filter block standing in for the
sensor cover stack. Drop it and use the air-converted BF of 16.55 mm if the
simulated sensor has no cover glass.

### Example 1: aspheric data

| Surface | κ | A4 | A6 | A8 | A10 | A12 | A14 |
|---|---|---|---|---|---|---|---|
| 1 | 0.0000 | -3.82177E-07 | -6.06486E-11 | -3.80172E-15 | -1.32266E-18 | | |
| 20 | 0.0000 | -1.15028E-06 | -4.51771E-10 | 2.72670E-13 | -7.66812E-17 | | |
| 28 | 0.0000 | 3.18645E-06 | -1.14718E-08 | 7.74567E-11 | -2.24225E-13 | 3.34790E-16 | -1.70470E-19 |

Units: r and h in mm, so A4 is mm^-3, A6 is mm^-5, and so on.

### Aspheric sag formula

Exactly as the patent writes it (WO paragraph before Table 1, US [0113]):

    x = (h^2 / r) / [ 1 + { 1 - (1 + κ) (h / r)^2 }^(1/2) ]
        + A4 h^4 + A6 h^6 + A8 h^8 + A10 h^10 + A12 h^12 + A14 h^14

- `x` is the sag: distance along the optical axis from the tangent plane at
  the vertex to the surface at height `h`.
- `r` is the paraxial radius of curvature (the value in the r column).
- `κ` is the **conic constant in the (1 + κ) form**, so κ = 0 is a sphere.
  This is the usual `k`, not `K = k + 1`.
- The polynomial starts at h^4. The patent states that "the secondary
  aspherical coefficient A2 is 0 and is omitted", so there is no h^2 term.
- `E-n` means `×10^-n`.

All three aspheres have κ = 0, so the conic part reduces to the sphere and
only the polynomial matters.

### Focusing (variable gap)

Focusing moves the front group GF (surfaces 1-22) toward the object; the rear
group GR (surfaces 23-28) stays fixed relative to the image plane. D22 is the
only variable gap.

| | Infinity | Close distance |
|---|---|---|
| f | 59.62 | |
| β (magnification) | | -0.194 |
| D22 (mm) | 2.68 | 21.29 |

Claff computes the close-distance object distance (object to surface 1) as
d0 = 320.994 mm, which is not in the patent. Total length at close focus is
160.74 + (21.29 - 2.68) = 179.35 mm.

### Conditional expression values (Example 1)

For completeness: fF/f = 1.27, (r2L1+r1L1)/(r2L1-r1L1) = 0.09,
(r2L2+r1L2)/(r2L2-r1L2) = 0.34, θgFLn + 0.0021·vdLn = 0.657 / 0.658 / 0.660,
rc/bfa = 3.19, rA/TLA = 1.79, rB/TLB = 2.52, f/fR = 0.20, Pex = 43.85,
-f1/f = 4.86, f2/f = 1.16, 2ω = 39.96°, bfa/f = 0.28, FNo = 0.98.
Pex = 43.85 mm is the exit pupil distance from the image plane, handy for a
sanity check of the ray trace.

### Confidence

Every number in the Example 1 tables above was checked against the WO page
images (WO pages 19, 20, 21 = PDF pages 21, 22, 23), the WO text on Google
Patents (which comes from the WIPO text, not OCR), and the US page images (US
pages 7-8 = PDF pages 18-19). All three agree with each other and with the
Optical Bench file, with one exception:

- **Surface 1, A4.** The US publication prints `-3.821177E-07` (seven-digit
  mantissa, every other coefficient has six). The Japanese WO original prints
  `-3.82177E-07`, and Optical Bench has `-3.82177E-07`. The US value is a
  typesetting error in the US publication; the table uses the WO value.
  High confidence.

Google's OCR of the US text layer also garbles surface 20 A6 as
`--44.51771E-10`; the page image and the WO both read `-4.51771E-10`.

The Example 2-4 summary table is from the WO text on Google Patents only.
Medium confidence; do not use those values without checking the page images
(Example 2 starts at WO page 23 = PDF page 25).

---

## 2. Noct-Nikkor 58mm f/1.2 (1977, Ai; Ai-S from 1982)

### Result: no prescription found

There is no published prescription for this lens that I could locate. The
evidence that this is a real gap and not a search failure:

- Marco Cavina's article on the lens states outright that despite decades of
  searching and direct contact with Nikon designers, no specific patent for
  the Noct-Nikkor has been identified.
  https://www.nocsensei.com/camera/tecnica/marco-cavina/marcocavina/nikon-noct-nikkor-58mm-112/
- photonstophotos' "Optical Bench and NIKKOR, The Thousand and One Nights"
  page lists Tale 16 (Ai Noct Nikkor 58mm F1.2) with a red ball, meaning no
  matching patent exists in the Optical Bench. The Tale 16 link there opens
  the AI Nikkor 50mm f/1.2S (US 4,621,909 Example 3) instead.
  https://photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Tales.htm
- The Optical Bench hub has no 58mm f/1.2 entry at all (its only 58 mm Nikon
  entries are the Z Noct and the 1966 Nikkor-S Auto 58mm f/1.4).
- cameragossip's Nikon patent database has no entry for it.
- Google Patents sweeps for Nippon Kogaku, priority 1972-1984, with
  aspheric / aspherical / 非球面 in the text (139 hits) returned only fundus
  cameras, projection lenses, aspheric fabrication tooling (US 4,151,654,
  US 4,178,720), and later zoom work. Nothing is a fast Gauss photographic
  lens with an aspheric front surface. Google Patents started serving block
  pages partway through, so a sweep by inventor name (Shimizu) did not run;
  Espacenet's REST endpoint refused unauthenticated queries.

Many 1970s Nikon lens designs exist only as Japanese applications, so if a
patent exists it is a JP S50-S54 era document that only J-PlatPat (full-text
search in Japanese) would turn up. That search was not possible from here.

### What is known about the design

From Nikon's own history article (Tale 16, by Kouichi Ohshita):
https://imaging.nikon.com/history/story/0016/index.htm

- Designer: Yoshiyuki Shimizu (清水義之). Released 1977, Ai-S revision 1982.
- 7 elements in 6 groups, modified Gauss type.
- The aspheric is on the first (object side) element. Cavina specifies the
  outer (object facing) radius of the first element. The surface was polished
  by hand using a method borrowed from telescope mirror making.
- Shimizu's stated method: weaken each surface's curvature with high-index
  glass, then put the aspheric on the largest-diameter, most object-side lens
  to kill sagittal coma flare.
- No floating elements. Focus is by unit extension (Cavina confirms, in
  contrast to Canon's 55/1.2 AL which has a floating group).
- Front doublet is air-spaced; two single rear menisci (Cavina's description
  of the cross-section).

Minimum focus 0.5 m, 52 mm filter, about 2,000 Ai units made. None of this
gives a prescription.

### The predecessor patent (not the Noct)

Cavina mentions a Shimizu patent filed December 25, 1970 for a fast double
Gauss with an air-spaced front doublet and identifies it as a predecessor
design, explicitly not the Noct-Nikkor. Google Patents shows two Nippon
Kogaku documents with that priority date and Yoshiyuki Shimizu as inventor:
DE 2163431 ("Gauss lens") and DE 2163430 ("Apochromatic telephoto lens").
DE 2163431 is the one Cavina means. The Optical Bench maps the 1978 Nikkor
50mm f/1.2 to US 3,738,736 Example 3, which is probably the US member of the
same family (same era, same type), but Google Patents blocked the page before
I could confirm the priority date, so treat that link as unconfirmed. Either
way it is a spherical f/1.2 Gauss, not the aspheric 58 mm.

### Stand-in options if a 58/1.2-class lens is still wanted

- US 4,621,909 Example 3 (Hamanishi, 1982): the AI Nikkor 50mm f/1.2S, 7
  elements, 51.6 mm f/1.2, no aspheric, with a close-focus correction scheme.
  Optical Bench already hosts it:
  https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Data/US004621909_Example03_Tale16.txt
  This is what photonstophotos shows for the Noct's Tale, so it is the
  closest published Nikon design of that era.
- The 2012-2013 Nikon JP applications for a new 58mm f/1.2 (JP 2012-230133,
  JP 2012-230340, JP 2013-019992, JP 2013-011831) are 9-element designs with
  three aspheres and are unrelated to the 1977 lens.

Neither was transcribed here since neither is the Noct-Nikkor.
