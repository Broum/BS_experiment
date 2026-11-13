# BS_experiment

Minimal setup to run the truncated beam-splitter circuits from
`GBS4EO___Simulating_bosonic_interferences_on_a_superconducting_quantum_computer-1.pdf` on the VLQ quantum computer
(IT4I, IQM), on IQM Sirius (IQM Resonance) and on the noisy simulator FakeVLQ.

## Layout

| Path | What |
|---|---|
| `bs_11_experiment.ipynb` | Main experiment: general BS circuit (symbolic θ, full or simplified, input \|N1>\|N2>) evaluated at any list `THETAS`; `BACKENDS` = any of FakeVLQ, VLQ, Sirius; qubit selection; plots and saving |
| `vlq_connection_test.ipynb` | Bell circuit on FakeVLQ, VLQ and Sirius: checks login, resource and backend |
| `utils/bs_circuits.py` | Circuits from the PDF; `make_bs(circuit=..., n1=, n2=)` gives the general circuit with symbolic `THETA`, `bind(qc, thetas)` fills in values; decoding, theory, exact unitary |
| `utils/vlq.py` | `get_backend("VLQ" / "Sirius" / "FakeVLQ")`, `transpile()` (same layout for all circuits), `submit()` + `collect()` or `run()`, `run_ideal()` |
| `utils/qubit_select.py` | Qubit selection from calibration files: per-qubit errors, layouts ranked by ESP |
| `utils/fake_vlq_backend.py` | **Generated** FakeVLQ backend (do not edit) |
| `utils/metrics.py`, `utils/results_io.py` | SBC/TVD metrics; `Results/<name>_<date>/` run folders |
| `scripts/generate_fake_vlq_backend.py` | FakeVLQ generator |
| `scripts/plot_tilted_bs.py` | Plotly figures of a \|1>\|1> run (probabilities, post-selected, fidelity) with error bars, in the style of `Tomev/gbs-compiler` `experiments/tilted_bs` |
| `scripts/recover_run_2026-11-09_163711.py` | Rebuilds the unsaved `results.json` of VLQ job 275 (VLQ from the printed table, FakeVLQ re-run) |
| `scripts/check_move_validity.py` | Noise-free check of the transpiled circuits on the star: illegal MOVEs (qubit and resonator both excited), gates on the resonator, HF |
| `scripts/fit_move_phases.py` | Fits one residual Z phase per qubit per MOVE sandwich to the stored VLQ counts (coherent-error model) |
| `scripts/move_sandwich_test.py` | Ramsey tests of the MOVE sandwich on VLQ: `--test single` (phase vs resonator CZs held), `--test multi` (idle crosstalk and resonator-frame carry-over from another qubit's sandwiches); per frame-tracking mode; `--dry-run`, `--fetch`, `--analyse` |
| `scripts/endpoint_position_test.py` | One VLQ job separating the θ = 0, π/2 failure: simplified (level 3) vs unsimplified (level 1) endpoint circuits, at the batch edges and in the middle |
| `scripts/cz_motif_test.py` | Population test of MOVE H,R · CZ P,R · MOVE H,R for all qubit pairs (does the CZ swap?), plus the BS endpoints with barriers |
| `scripts/refetch_vlq_jobs.py` | Re-fetches finished VLQ jobs: raw counts with per-circuit `measurement_keys`, matched to the stored run; `--scan` lists job ids |
| `calibration/VLQ/`, `calibration/SIRIUS/` | Downloaded calibration (quality-metric-set) files; the newest one is used |
| `tests/` | Circuits against the exact truncated BS unitary; qubit selection on the stored calibration files |
| `tmp/` | Reference notebook from another project (GHZ5 on best qubits, VLQ vs Sirius); not used here |
| `old_scripts/` | Older notebooks and their helper modules, kept for reference only (not runnable as is) |

Encoding: mode 1 = (q0, q1), mode 2 = (q2, q3), n = 2·q_msb + q_lsb. In Qiskit's count keys (`q3q2q1q0`),
|1>|1> is `1010`. `bs_11_circuit` is the even-sector-only version for input |1>|1>: it has ~22 two-qubit gates after
transpilation, against ~74 for the full `U_BS`. Both give the same ideal distribution (tested).

## Environment

The environment is a uv project pinned to the current VLQ software: `qaas` 1.0 (module `qheappe_client`), `py4lexis`
7.1 (IT4I index), `iqm-client` 36 (from the `qaas[iqm-qiskit]` extra), `qiskit` 2.1, `qiskit-aer` and `gbs-compiler`.

```bash
cd /mnt/proj1/open-36-43/GBS4EO/BS_experiment
export UV_CACHE_DIR=/mnt/proj1/open-36-43/software/uv/cache UV_PYTHON_INSTALL_DIR=/mnt/proj1/open-36-43/software/uv/python
deactivate 2>/dev/null; uv sync          # creates .venv (Python 3.12)
.venv/bin/python -m unittest discover -v # circuit checks
```

In VS Code, select `.venv/bin/python` as the notebook kernel. `~/.bashrc` activates the `Quantula/` venv: do not use
that one here.

`qaas` 1.0 changed the import from `qaas.client` to `qheappe_client`. The old `qaas.client` is kept for legacy
backends only. `QProvider.get_backend()` refuses a backend whose IQM version does not match the installed
iqm-client and names the extra to install. If that happens, change the `qaas[...]` extra in `pyproject.toml` and run
`uv sync` again.

## VLQ access

Copy `.env.example` to `.env` (already done) and set:

- `VLQ_PROJECT`: LEXIS project, e.g. `OPEN-37-13`.
- `VLQ_RESOURCE`: LEXIS resource of VLQ. It was `VLQ-CZ` with qaas 0.5. If it is empty, qaas picks the resource itself.
  `vlq_connection_test.ipynb` lists what is available.
- `VLQ_TOKEN` (optional): a LEXIS token. If it is empty, `vlq.get_token()` logs in once per kernel through
  `LexisSession`. It prints *"Open provided url in your browser, please: …"*: open that link and log in within about
  2 minutes, otherwise the login times out. The session renews the token by itself, so leave `VLQ_TOKEN` empty for
  long runs: a pasted token expires. To get a token in a shell:

  ```bash
  export VLQ_TOKEN=$(.venv/bin/python -c 'import sys, contextlib
  from py4lexis.session import LexisSession
  with contextlib.redirect_stdout(sys.stderr):
      print(LexisSession().get_access_token(), file=sys.__stdout__)')
  ```

  The redirect keeps the login banner out of the token. A valid token starts with `eyJ`.

In code:

```python
from utils import vlq
backend = vlq.vlq_backend()                       # or vlq.fake_vlq()
tr, layout = vlq.transpile(circuits, backend)     # layout = qubit names, reused for every circuit
counts = vlq.run(backend, tr, shots=4000)         # one VLQ job for the whole list
```

Every `run()` on VLQ is one queued job, so pass all θ values in one list. `run()` prints the job id. The notebook
submits all jobs first (`submit()`) and then waits for them (`collect()`), so VLQ and Sirius queue in parallel.

## IQM Sirius

`vlq.sirius_backend()` connects to `sirius` on IQM Resonance (`https://resonance.iqm.tech/`) with `IQMProvider` from
`iqm-client`. The token is read from the file `.IQM_token` (gitignored) or from `IQM_TOKEN`. `iqm-client` refuses a
token given both as an argument and in the environment, so the function removes `IQM_TOKEN`/`IQM_TOKENS_FILE` from the
kernel environment after reading it. Sirius is also a star (16 qubits + resonator), so transpiling, qubit selection and
decoding are the same as for VLQ.

Sirius uses IQM Resonance credits, so the notebook runs only `THETAS_PER_BACKEND["Sirius"] = [0, π/8, π/4]` there:
no splitting, P(1,1) = ½, and the HOM dip. Other backends run the full `THETAS`. The bar charts show the θ values that
every backend ran.

## Qubit selection

`utils/qubit_select.py`, used in the notebook section *Backends and qubit selection*:

1. `load_calibration(folder_or_file)` reads a quality-metric-set JSON (newest file of a folder). It returns the metrics,
   the calibration set id (`describes_id`) and the file name. `check_calibration(backend, id)` warns if the machine
   now runs another calibration set: then download a new file.
2. `qubit_errors()` gives per qubit the error 1 − F of `prx` (RB), CZ with the resonator (IRB; `crf_crf` or `tgss`,
   whichever is calibrated), one MOVE (IRB of 2 MOVEs reduced to one), and readout (SSRO); plus T1, T2.
3. `best_layouts()` takes the 6 best qubits by `readout + prx + cz + 2·move` and tries every placement of the 4
   circuit qubits on them (360 transpilations, ~8 s). Each one is scored by its ESP, the product of 1 − error over all
   gates and measurements of the transpiled circuit. Idling and crosstalk are ignored. The best layout is used for all θ.

FakeVLQ uses the snapshot it was generated from, or VLQ's qubits when both run. `LAYOUT = {"VLQ": [...]}` fixes the
qubits by hand; `SELECT_QUBITS = False` leaves the choice to the transpiler. The FakeVLQ target has no error data, so
the transpiler alone picks QB1–QB4 there, which includes QB3 with the worst readout. On FakeVLQ, the selected qubits
raise the ESP of the full circuit from 0.61 to 0.80, and the kept fraction from ~0.70 to ~0.83.

For VLQ, `qs.fetch_vlq_calibration(backend)` downloads the current metrics through qaas and saves them to
`calibration/VLQ/` (not tested yet). Pass its result to `load_calibration`.

Sanity check for real runs: at θ = 0 the output must be mostly (1,1). If it comes out as (2,2), the bit order is
reversed.

**VLQ bit order.** qaas 1.0's `QJob.result()` defaults to `key_order_left_to_right=False`, which reverses every IQM
bitstring. `vlq.collect()` sets it to `True` and checks the measurement keys since 2026-11-12. All VLQ counts saved
before that (every VLQ run in `Results/` up to 2026-11-12) have reversed keys: reverse them before any analysis.
Findings and open points: [Results/FINDINGS_VLQ_BS_systematic_error_2026-11-12.md](Results/FINDINGS_VLQ_BS_systematic_error_2026-11-12.md).

## FakeVLQ

`fake_vlq()` is an `IQMFakeBackend` with VLQ's star topology (24 qubits + resonator CR1) and noise from one calibration
snapshot (currently `2026-10-23T05_13_19.912718Z.json`). The noise is T1/T2 relaxation, depolarizing `prx`/`cz`/`move`
errors and readout errors. `vlq.run()` runs it on Aer with a seed and with MOVE noise applied: the plain
`IQMFakeBackend.run` cannot be seeded and drops MOVE noise. This setup is taken from the
`worktrees/Estimation_of_VLQ_parameters` branch.

To regenerate it for a new calibration:

1. Get the VLQ quality-metric observation-set JSON export (`"observation_set_type": "quality-metric-set"`). See the IT4I
   [calibration health page](https://docs.it4i.cz/en/docs/clusters/vlq/calibration_health). Copy it into
   `calibration/VLQ/` without renaming it.
2. Run:

   ```bash
   .venv/bin/python scripts/generate_fake_vlq_backend.py calibration/VLQ/<snapshot>.json   # -> utils/fake_vlq_backend.py
   ```

3. Read the `note:` lines on stderr (also written into the module header). For example, a gate whose depolarizing
   parameter was clipped to 0 because T1/T2 relaxation alone already explains its benchmarked error.

Gate durations (`prx` 32 ns, `cz` 88 ns, `move` 64 ns) and the qubit count are constants at the top of the
generator. Details: `worktrees/Estimation_of_VLQ_parameters/README_FakeVLQ_regeneration.md` and
`README_FakeVLQ_noise_fix.md`. FakeVLQ results from different snapshots are not comparable: the notebook saves the
snapshot name in `results.json`.

## Next: GBS compiler

`gbs-compiler` is installed (`from gbs_compiler import CompilerGBS`). `CompilerGBS(fock_cutoff=4).compile_to_qasm(...)`
turns a Blackbird program into OpenQASM: load it with `QuantumCircuit.from_qasm_str`, then use `vlq.transpile` and
`vlq.run` as above. `compiler.binary_to_photon_meas(samples)` decodes the measured bits back to photon numbers.
