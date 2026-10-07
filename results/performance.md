| Scene | Compact | Sort | Median ms | Repetition range | Compaction speedup | Sorting speedup |
| --- | --- | --- | --- | --- | --- | --- |
| open | 0 | 0 | 6.344 | 6.337–6.379 | — | — |
| open | 0 | 1 | 18.479 | 17.861–18.499 | — | 0.343× |
| open | 1 | 0 | 10.691 | 10.686–10.914 | 0.593× | — |
| open | 1 | 1 | 21.421 | 21.406–21.494 | 0.863× | 0.499× |
| closed | 0 | 0 | 6.407 | 6.396–6.416 | — | — |
| closed | 0 | 1 | 18.748 | 18.702–18.860 | — | 0.342× |
| closed | 1 | 0 | 12.229 | 12.213–12.357 | 0.524× | — |
| closed | 1 | 1 | 24.524 | 24.363–24.598 | 0.764× | 0.499× |

| Feature | Reference ms | Enabled ms | Whole-scene change |
| --- | --- | --- | --- |
| Standard GGX | 21.559 | 21.421 | -0.6% |
| Oren–Nayar | 21.503 | 21.421 | -0.4% |
| Smooth glass | 21.218 | 21.421 | +1.0% |
| Metal Fresnel | 7.195 | 7.199 | +0.1% |
