# bioshrek-export

Final export pipeline for Project Bioshrek.

Renders the Cell Slicer pixel art animation using fluorescent protein colors,
matching the agar pixel art palette from `project-bioshrek/source-art/flourescent-color-map.md`.

## Color mapping

| Label | Cell type | FP | Hex |
|-------|-----------|----|-----|
| 0 | Background | — | `#111111` |
| 1 | RBC interior | mKO2 | `#FF8000` |
| 2 | RBC edge | mRFP | `#FF0000` |
| 3 | Neutrophil | mTurquoise2 | `#00C8FF` |
| 4 | Bacterium | Venus | `#AAFF00` |

## Scripts

- `render_fp.py` — render pixel art JSON → MP4 using FP colors (upscaled, no interpolation)

## Outputs

- `output/` — rendered MP4 files
