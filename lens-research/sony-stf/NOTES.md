# Sony / Minolta STF lens research

Collected 2026-10-04 for the realistic-camera lens table. Two lenses:

- Sony FE 100mm F2.8 STF GM OSS (SEL100F28GM, 2017). Target. **No optical prescription found.** Only the block diagram and the APD element position are documented below.
- Minolta/Sony STF 135mm F2.8 [T4.5] (1999, A-mount, later SAL135F28). Fallback. **Full prescription from the Minolta patent JP H11-231209 A**, transcribed and cross-checked below.

## 1. Sony FE 100mm F2.8 STF GM OSS: what was found and what was not

### Searches that came up empty

| Where | Query | Result |
|---|---|---|
| Google Patents (xhr/query JSON) | `apodization` + assignee Sony, priority 2013..2018 | 9 hits, none a lens design (birefringent-device camera, exposure control, projector, ultrasound) |
| Google Patents | `アポダイゼーション` + assignee ソニー, 2013..2018 | 6 hits: JP2015-079128A (optical filter, see below), exposure-control family JP2015-049296/297/298, projector, ultrasound |
| Google Patents | CPC G02B13 + assignee Sony, priority 2015-01..2018-07 (every Sony lens-design filing of the era, 78 hits) | Candidates with the right date (EP3255473B1 2015-02, WO2017138250A1 2016-02, CN109073862B 2016-05 = F1.6-2.0 lens, JP6883226B2 2016-11, JP7207327B2 2017-12) fetched and grepped: zero occurrences of apodization / transmittance / filter-with-gradient in any of them |
| Google Patents | `アポダイゼーション` + CPC G02B13, 2014..2018 (any assignee) | 17 hits: Canon (US10281735B2, JP2019045631A), Fujifilm 56mm APD (US9904040B2 / US20160274335A1), Apple, Zeiss. No Sony |
| J-PlatPat full text | 全文 アポダイゼーション AND 書誌 ソニー | 35 domestic hits; the only Sony optics ones are JP2015-079128 (optical filter, 2013), the 2013 exposure-control trio, and old optical-pickup patents. No lens design 2015-2018 |
| J-PlatPat full text | 全文 アポダイゼイション / アポダイズ / アポダイザ / アポダイゼーションフィルタ AND 書誌 ソニー | 21 hits, same set, nothing new |
| J-PlatPat full text | 全文 (ＮＤフィルタ / 減光フィルタ / 透過率分布 / アポダイゼーション / ボケ味) AND 書誌 ソニー AND 名称 (撮像レンズ / 光学系 / レンズ) | 64 hits, mostly pre-2014 zoom and ND-iris patents. The only 2016-2018 ones are lens/body communication patents (特許6269874 filed 2017-03-03, 特開2018-146952, 再表2018/173902); 特許6269874 was read on J-PlatPat: its "transmittance" is a liquid-crystal variable ND filter in the barrel, nothing about apodization or the STF |
| photonstophotos Optical Bench hub | grep of the hub's lens list for `STF`, `apodiz`, `Sony 100mm` | Only one STF entry: "Konica Minolta 135mm F2.8 T4.5 STF" = JP1999-231209 Example 1. No Sony 100 STF |
| asobinet.com, sonyalpharumors, sonyaddict, Justia, FreePatentsOnline | Sony STF / apodization patent articles | No article on a Sony 100 STF design patent; asobinet only has the Canon APD patents (US2018-0067333) |
| Espacenet, Lens.org | | Both put up a human-verification challenge; not attempted |

Conclusion: as of 2026-10-04 there is no published Sony patent or application that carries the FE 100mm F2.8 STF GM OSS prescription under any apodization wording, and nobody (Bill Claff included) has one. The design appears to be unpublished. If it ever surfaces it will most likely be an "撮像レンズおよび撮像装置" filing with a 2015-2016 priority; the G02B13 sweep above is the list to re-check.

### Sony-published facts about the lens

Sources: product pages https://www.sony.jp/ichigan/products/SEL100F28GM/feature_1.html and https://www.sony.net/Products/di_photo-gallery/lens/SEL100F28GM/ ; Imaging-Resource CP+ 2017 interview with Yasuyuki Nagata (Sony Imaging Products & Solutions), https://www.imaging-resource.com/news/2017/02/25/sony-interview-cpplus-2017-insights-design-team-100mm-stf-apodization-lens (read via the Wayback Machine, the live page returns 403); alphauniverse.com/stories/stf-demystified ; dpreview.com/news/2820818690.

- Focal length 100 mm, F2.8, T5.6 at full aperture. The aperture ring is marked in T-stops; from T8 (f/8) onward T = F because the beam only passes the clear center of the APD element. Minimum T22.
- 11-blade circular aperture. OSS. Direct Drive SSM internal focus. Macro ring (0.57 m, 0.25x).
- Sony's construction count: 10 groups, 13 elements (sony.jp: 10群13枚, the APD element counted). Reviews that count the glass separately say "13 elements in 10 groups plus the APD element".
- Sony's description of the APD element (feature page): a special filter whose transmitted light decreases toward the lens periphery, placed near the aperture mechanism (絞り機構の近く), softening the outline of point images and suppressing double-line bokeh. Alpha Universe says it sits between the aperture and the sensor, i.e. just behind the stop. Sony does not publish the stop position.
- Nagata, on construction: "the apodization filter is like an ND filter, but it has a density graduation from center to edge"; how it is manufactured is "classified". On the profile: the gradation curve was chosen by simulating bokeh on a computer; T5.6 "is just a result", they "prioritized the graduation". The lens's physical front aperture is about an f/2-size pupil so that the APD-shaped pupil is not clipped by vignetting at the frame edge (no cat's-eye).
- Comparison he drew: the A-mount 135 STF is F2.8 T4.5 (lighter apodization), Fujifilm's 56mm APD is F1.2 T1.7.

### Block diagram (saved as `sony_y_SEL100F28GM_lenscomposition.jpg`, 1200x529)

Legend: blue = アポダイゼーション光学エレメント (APD), purple = aspherical, green = ED glass. Object side on the left. Pieces of glass, counted from the front:

| # | Element as drawn | Note |
|---|---|---|
| 1 | thin positive meniscus, convex front | |
| 2 | thick biconvex, ED (green) | cemented to 3 |
| 3 | negative meniscus | |
| 4 | biconcave | |
| 5 | positive meniscus | cemented to 6 |
| 6 | thin negative meniscus | |
| 7 | **APD element: plane-parallel plate (blue)**, diameter about the same as 5/6, thickness roughly 0.25 of its diameter in the drawing | the stop is not drawn; Sony says it is adjacent |
| 8 | small thin biconvex | |
| 9 | plano-concave | cemented to 10 |
| 10 | positive meniscus | |
| (gap) | long air space | focus / OSS groups live here |
| 11 | large biconvex | |
| 12 | thick plano-convex | |
| 13 | negative meniscus | cemented to 14 |
| 14 | aspheric meniscus (purple), rear element | |

That is 13 glass elements + 1 APD plate in 10 groups (9 glass groups + the APD), which reconciles Sony's "13 elements, 10 groups" with the drawing. Two things matter for the camera model: the Sony APD is drawn as a flat plate with no optical power, unlike the Minolta plano-concave ND doublet, so Sony's gradient is in the material or a coating, not in a thickness profile; and it sits in the middle of the front half of the lens, about 45 % of the way from the front vertex to the rear vertex in the drawing, with the stop next to it.

`sony_original_SEL100F28GM_APD.jpg` is Sony's rendering of the element's density: clear center, smooth falloff to near-black at the rim. No numbers are published for the curve. Treat the profile as an unknown; a Gaussian pupil apodization with a pupil-averaged transmittance of (2.8/5.6)^2 = 0.25 reproduces the stated T5.6 and is the right first guess (the Minolta profile below is Gaussian in form).

### Related Sony filings (not the lens, but about the element)

- JP2015-079128A, 光学フィルタおよび撮像装置, Sony, filed 2013-10-17, published 2015-04-23, deemed withdrawn. PDF saved as `JP2015079128A_sony_optical_filter.pdf`. A gradient ND filter made from two transparent substrates whose facing surfaces are shaped so the gap, filled with an absorbing "transmittance characteristic changing member", grows from center to edge; the text says it functions as an apodization filter. No lens data, no transmittance numbers. It shows Sony was working on a flat-plate gradient element rather than Minolta's ND lens, which is consistent with the block diagram.
- WO2016063849A1 (JP priority 2014-10-21), 光学素子および撮像装置, Sony: a solid-state electrochromic element with externally controllable apodization. Not what shipped.
- JP2015-049296/049297/049298 (US9632393B2), Sony 2013-08-30: exposure control and flash control for a camera carrying an apodization filter (how to handle T vs F). Not fetched.

## 2. Minolta STF 135mm F2.8 [T4.5]: patent JP H11-231209 A

### Bibliographic

- Publication: 特開平11-231209 (JPH11231209A), published 1999-08-27. Application 特願平10-035859, filed 1998-02-18 (priority). Title 撮影レンズ系 / Photographic lens system. Applicant ミノルタ株式会社 (Minolta Co., Ltd.). Inventor 工藤芳信 (Yoshinobu Kudo).
- Status on J-PlatPat: 出願のみなし取下げ (deemed withdrawn, examination never requested). So it is a published application only; there is no granted JP patent and no US/EP family member (Google Patents family: JP3585998A only).
- Sources: Google Patents https://patents.google.com/patent/JPH11231209A/en (text layer, machine translation; the page has no PDF for this document) and J-PlatPat https://www.j-platpat.inpit.go.jp/ (特許・実用新案番号照会 for H11-231209, 11 page PDFs merged into `JPH11231209A_minolta_stf135_jplatpat.pdf`). photonstophotos: https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBench.htm#Data/JP1999-231209_Example01P.txt (data file saved as `p2p_JP1999-231209_Example01P.txt`, figures `p2p_JP1999-231209_Figure01.png` and `..._Figure01P.png`).
- Which embodiment is the product: **Example 1**. Bill Claff lists it as "Konica Minolta 135mm F2.8 T4.5 STF". Sony's SAL135F28 construction diagram (`sony_SAL135F28_lensconstruction.jpg`, from sony.jp/ichigan/products/SAL135F28/) shows the same topology: plano-convex front, cemented meniscus pair, negative meniscus, the thick APD block, biconvex, rear negative meniscus = 8 elements in 6 groups counting the APD doublet, which is Sony's published count for that lens. Example 2 has a negative rear group and a different front doublet; it is not the product.
- Claims: (1) front group + rear group with a sufficient air space between them holding the apodization filter, and 0.0001 < |φF/φR| x |φAF/φT| < 0.5; (2) the filter is a first member of ND glass plus a second member of glass with nearly the same index, 0.0001 < |nd_ND - nd_B| < 0.05; (3) 0.00001 < |φAF/φT| < 0.5. Para. 0023 adds a placement condition 0.1 < (a + t_AF + b)/Σd < 0.8.

### Stated figures (both examples)

- f = 135.0 mm, FNO = 2.83, close-focus magnification β = -1/4, ND absorption coefficient α = 0.55 (the patent gives no unit; it is per mm, since thicknesses are in mm and the resulting curve matches Fig. 11).
- Example 1 power data: φF = 0.00370, φR = 0.00671, φAF = -0.000267, φT = 0.00740 (1/mm). Condition values: (1) 0.02, (2) |nd_ND - nd_B| = 0.00553, (3) 0.036, (4) (a + t_AF + b)/Σd = 42.78/93.86 = 0.46. Focusing: the whole lens moves toward the object with floating; front-group travel M1 : (stop+AF+rear) travel M2 = 1.11 : 1. The stop and the APD block move with the rear group.
- Example 2: φF = 0.00760, φR = -0.00302, φAF = -0.000267, φT = 0.00740; (1) 0.091, (2) 0.00553, (3) 0.036, (4) 39.099/82.70 = 0.47; M1/M2 = 0.97.
- T-number is not in the patent (the product is marked T4.5, manual T-ring T4.5-T6.7, plus a 9-blade auto iris).
- Back focus is not stated in the patent. My paraxial trace of Example 1 gives f = 135.002 and BFL = 46.800 mm at infinity, which is what photonstophotos lists (Bf 46.80 infinity, 80.01 at β = -0.25, the latter with d0 = 694.05 "measured" by Bill, i.e. not patent data). Overall length first vertex to last vertex at infinity Σd = 93.86 (patent), so vertex-to-image = 140.66 mm.

Surface conventions: surfaces numbered from the object side; r > 0 means the center of curvature is on the image side; d_i is the axial distance from surface i to i+1; N and ν are d-line values. All surfaces are spherical, there are no aspheres in this patent, and no clear apertures are given. The CA column below is Bill Claff's estimate from the patent drawing (his data file), not patent data.

### Example 1 prescription (the production 135 STF)

| Surface | Element | r (mm) | d (mm) | n_d | V_d | CA dia (mm, p2p estimate) | Note |
|---|---|---|---|---|---|---|---|
| 1 | L1 | 75.063 | 9.000 | 1.60311 | 60.74 | 64.36 | plano-convex, front group Gr1 |
| 2 | | inf | 0.250 | | | 64.36 | |
| 3 | L2 | 37.102 | 3.000 | 1.72342 | 37.99 | 47.42 | cemented to L3 |
| 4 | L3 | 26.475 | 16.000 | 1.51680 | 64.20 | 47.42 | |
| 5 | | 104.186 | 2.050 | | | 47.42 | |
| 6 | L4 | 678.979 | 2.000 | 1.61293 | 36.96 | 36.96 | negative meniscus |
| 7 | | 30.584 | 14.397 (inf) to 18.198 (β=-1/4) | | | 35.42 | variable gap, floating |
| **8** | **STOP (AS)** | inf | 1.500 | | | 28.756 | aperture stop; my paraxial marginal ray height here is 14.16 mm (dia 28.3) at F2.83 |
| **9** | **L5 = APD, ND glass** | inf | 0.300 | 1.50690 | 58.94 | 31.32 | plano-concave absorbing element, center thickness 0.300 |
| **10** | **L6 = APD, clear glass** | 20.731 | 9.330 | 1.50137 | 56.46 | 31.32 | cemented, concave/convex radius 20.731; plano-convex |
| 11 | | inf | 17.250 | | | 31.32 | exit face of the APD block |
| 12 | L7 | 97.643 | 7.500 | 1.58913 | 61.11 | 38.28 | rear group Gr2 |
| 13 | | -47.530 | 9.500 | | | 38.28 | |
| 14 | L8 | -41.707 | 1.800 | 1.62041 | 60.29 | 35.00 | |
| 15 | | -923.182 | BFL 46.80 | | | 35.00 | image at 46.80 (infinity) |

### Example 2 prescription (not the product)

| Surface | Element | r (mm) | d (mm) | n_d | V_d | Note |
|---|---|---|---|---|---|---|
| 1 | L1 | 136.245 | 7.799 | 1.62041 | 60.29 | biconvex |
| 2 | | -2351.337 | 0.250 | | | |
| 3 | L2 | 37.142 | 2.802 | 1.75000 | 25.14 | cemented to L3 |
| 4 | L3 | 30.487 | 13.000 | 1.51680 | 64.20 | |
| 5 | | 138.452 | 1.450 | | | |
| 6 | L4 | 698.983 | 1.999 | 1.67339 | 29.25 | |
| 7 | | 66.753 | 12.000 (inf) to 10.983 (β=-1/4) | | | variable |
| **8** | **STOP** | inf | 1.500 | | | |
| **9** | **L5 = APD, ND** | inf | 0.300 | 1.50690 | 58.94 | |
| **10** | **L6 = APD, clear** | 20.731 | 9.300 | 1.50137 | 56.46 | |
| 11 | | inf | 15.999 | | | |
| 12 | L7 | 155.942 | 5.000 | 1.62588 | 35.70 | |
| 13 | | -83.733 | 9.500 | | | |
| 14 | L8 | -33.259 | 1.801 | 1.61272 | 58.52 | negative rear group |
| 15 | | -256.843 | | | | |

### The apodization filter (paras. 0036-0045, Figs. 9-11)

Construction: a plano-concave lens L5 of ND (absorbing) glass, concave toward the image, cemented to a plano-convex lens L6 of clear glass with the same convex radius and nearly the same index, so the block has flat entry and exit faces and almost no power. The absorption is a thickness effect, Beer-Lambert through the ND element:

- I = I0 exp(-α t)  (eq. C), α the ND glass absorption coefficient, t the local thickness.
- Thickness of the plano-concave element at height h: t(h) = t0 + r - sqrt(r^2 - h^2) (eq. D, plus the center thickness t0 which the patent leaves out because it only discusses the shape); binomial approximation t ≈ h^2 / 2r (eq. E).
- Transmittance, normalized to the axis: τ(h) = exp(-α h^2 / 2r) (eq. G). The patent calls this "approximately Gaussian"; the exact sag makes the real curve fall a little faster than Gaussian at the rim.
- With the absolute center loss included: τ(h) = exp(-α (t0 + r - sqrt(r^2 - h^2))).

Design values for the filter alone, para. 0044 (this table is an image in the publication; Google's text layer drops it, read from `JPH11231209A_minolta_stf135_jplatpat.pdf` page 7):

| Surface | R (mm) | T (mm) | n_d | V_d | α |
|---|---|---|---|---|---|
| R1 | inf | T1 = 0.35 (ND lens 1) | 1.507 | 59.0 | 0.55 |
| R2 | 20.5 | T2 = 9.28 (clear lens 2) | 1.507 | 59.0 | |
| R3 | inf | | | | |

Effective radius 15.5 mm. In the lens examples the same filter is used with t0 = 0.300, r = 20.731, nd 1.50690/1.50137.

Computed profile for the Example 1 filter (α = 0.55 /mm, t0 = 0.300, r = 20.731), exact sag / Gaussian approximation:

| h (mm) | 0 | 5 | 10 | 14.0 (marginal ray) | 15.5 (rim) |
|---|---|---|---|---|---|
| τ exact | 0.848 | 0.606 | 0.206 | 0.040 | 0.018 |
| τ Gaussian (eq. G, times 0.848) | 0.848 | 0.609 | 0.225 | 0.059 | 0.035 |

Fig. 11 (page 11 of the PDF) plots the para. 0044 filter: peak about 83 % on axis (= exp(-0.55 x 0.35) = 0.825) falling to a few percent at the ends. Flag: its x axis is labelled 入射高 (h) [mm] with ticks at ±1.0 while the filter's effective radius is 15.5 mm; the curve matches the formula only if the axis is read in units of 10 mm (about 20-25 % at "1.0"). The plot's unit label is wrong or the ticks are cm.

T-number check: the pupil-averaged transmittance of the Example 1 filter over the marginal ray radius 14.0 mm is 0.286, i.e. T = 2.83 / sqrt(0.286) = T5.3 (T5.4 for the para. 0044 values). The shipping lens is T4.5 (average transmittance 0.40), so the production filter is about 0.4 stop lighter than the patent example: a smaller α, a thinner ND element, or a larger r. For the simulator, keep the patent geometry and scale α down to about 0.38 /mm if the product's T4.5 should be matched; the shape of the falloff stays Gaussian-like either way. (Fig. 2 of the patent labels the close-focus state as 有効FNO = 3.94, which is geometric effective F-number at β = -1/4, not a T-stop.)

Para. 0045 notes alternatives the patent does not use: an absorbing layer vacuum-deposited on a flat plate with a radial profile, or a photographic emulsion exposed to a radial density; both are thinner but introduce phase and scattering errors from the varying film thickness. This is the family the Sony flat-plate element presumably belongs to.

### Verification

- Google Patents text (Example 1, Example 2, condition values, eqs. C-G) was compared line by line against the J-PlatPat page images (pages 6-9 of the saved PDF). All numbers agree; no garbled values were found. The para. 0044 filter table exists only in the page image.
- photonstophotos' data file agrees with the patent for every r, d, n, V. Its diameters, d0 = 694.05 and the Bf values are Bill's additions.
- Paraxial trace of Example 1 (own code): f = 135.002, BFL = 46.800, marginal ray at the stop 14.156 mm.

## 3. Files in this folder

| File | What |
|---|---|
| `JPH11231209A_minolta_stf135_jplatpat.pdf` | Minolta patent, 11 pages, merged from J-PlatPat's per-page PDFs (owner-password encryption removed so readers can open it) |
| `p2p_JP1999-231209_Example01P.txt` | photonstophotos Optical Bench data file for Example 1 |
| `p2p_JP1999-231209_Figure01.png`, `p2p_JP1999-231209_Figure01P.png` | photonstophotos copies of Fig. 1 and the product cutaway |
| `sony_SAL135F28_lensconstruction.jpg` | Sony A-mount 135 STF block diagram (sony.jp) |
| `sony_y_SEL100F28GM_lenscomposition.jpg` | Sony FE 100 STF block diagram with the APD element in blue (sony.jp) |
| `sony_original_SEL100F28GM_APD.jpg` | Sony's picture of the APD element density |
| `JP2015079128A_sony_optical_filter.pdf` | Sony gradient optical filter application (2013), Google Patents PDF |
