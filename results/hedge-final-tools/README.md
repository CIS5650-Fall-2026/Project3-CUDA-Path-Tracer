These records contain local tooling self-tests and read-only deterministic glTF asset verification. They are not GPU correctness tests, rendered images, convergence measurements, or performance evidence. No renderer or compiler is executed by this run.

The measurement-driver fixtures in `measure-hedge/` and `measure-gltf/` are explicitly synthetic. The plotting self-test creates six synthetic SVGs in a temporary directory; those temporary SVGs are not retained as project figures. Its structured result is retained under `plot-hedge/`.

`summary.json` records every executed command, timestamps, exit code, elapsed wall time, log hashes, and frozen source identity. `logs/` retains stdout and stderr. The independently authored `run_local_checks.py` refuses to overwrite an earlier run and stops on a nonzero command exit. Scene and asset bytes and modification times are compared before and after all checks; the generator is invoked only with `--self-test` and `--verify`, never its writing mode.

This retention run is not an independent final reviewer audit.
