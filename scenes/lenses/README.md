# Lens files

Prescriptions for the real-lens camera. A scene names one with `"LENS": "<file>"` in its Camera block, or the command line does with `--lens <file>`; a bare name resolves to `scenes/lenses/<name>.json`. All lengths are millimeters.

## Format

Lens files are `.json`, in this project's own format. The top level has `name`, `source` (where the numbers came from), `apertures` (where the clear apertures came from, since patents rarely print them), the stated `focalLength` and `fNumber`, the `sensor` size as `[width, height]`, the aperture `blades` (0 for a round stop), and `closeFocusMagnification` when the patent states one. `surfaces` lists the table from the scene side to the film, one object per surface:

| Key | Meaning | Default |
|---|---|---|
| `radius` | curvature radius, positive bulging toward the scene, 0 for flat | required |
| `thickness` | gap to the next surface; on the last surface, to the film at infinity focus (0 when not known: the load fills in the paraxial back focus) | required |
| `aperture` | clear diameter | required |
| `ior` | n_d of the glass behind the surface | 1 (air) |
| `abbe` | V_d of that glass | 0 (unknown, no dispersion) |
| `stop` | `true` on the aperture stop | false |
| `conic` | k in the sag formula below | 0 |
| `aspheric` | `[A4, A6, ...]`, as many terms as the source lists | none |
| `focus` | `[gap at infinity, gap at the closest focus]` on the one surface whose gap moves to focus | none |

Aspheric sag, with c = 1 / radius and h the distance from the axis: `z = c h² / (1 + sqrt(1 - (1 + k) c² h²)) + A4 h⁴ + A6 h⁶ + ...`. Patents that write the conic as K = k + 1, or that include an h² term, are converted when the file is made, and the file's `source` says so.

Prescriptions from other sources (PBRT-v3's `.dat` tables, Optical Bench exports, patents) are converted into this format by hand; the renderer reads nothing else.

## How the renderer places a lens

The camera's `EYE` is the lens's entrance pupil (the stop as the scene sees it), so switching between the pinhole, the thin lens and a real lens keeps the framing. The lens extends from there toward and away from the scene by its table, in millimeters scaled by the scene's `UNITS_PER_METER` (a top-level scene key, 1 when absent: one scene unit is one meter). The film's height is the file's sensor height and its width follows the render's aspect ratio; `FOVY` is not used. A last gap of 0 is filled at load with the paraxial back focus, so a distant object is in focus; any other last gap is used as written. At load the renderer prints what a trace of the table gives: focal length, f-number, film distance and the entrance pupil's position and radius.

## Lenses

| File | Lens | Source | Notes |
|---|---|---|---|
| `dgauss.50mm.json` | Double Gauss 50 mm f/2 | PBRT-v3's `dgauss.50mm.dat`, from Tronnier US 2,673,491, plus the patent's Abbe numbers and the traced film distance | 11 surfaces, stop at surface 6. The reference lens: a paraxial trace of the table gives EFL 50.36 mm, f/2.02, the film 68.146 mm behind the front vertex (back focus 36.11), and an entrance pupil radius of 12.48 mm against a stop radius of 8.55. The host test checks every surface against PBRT's rows |
| `noct-z58.json` | Nikon NIKKOR Z 58mm f/0.95 S Noct | WO 2019/229849 A1, Example 1 (`lens-research/noct/NOTES.md`) | 28 surfaces, 17 elements, stop at surface 14, aspheres on surfaces 1, 20 and 28, one focusing gap (surface 22: 2.68 mm at infinity, 21.29 at the closest focus, magnification -0.194). The patent's 1.6 mm filter block is dropped and the last gap is the air-converted back focus, 16.55 mm. Clear apertures are Bill Claff's ray-traced estimates, not patent values |
