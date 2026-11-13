# VLQ beam-splitter: systematic error investigation (state 2026-11-13, §7 added)

**Starting point.** On VLQ the tilted beam splitter (BS) looked almost like noise:
- At θ = 0 the ideal output is `1010` = |1,1>, but one run gave `1111` in 43 % of shots.
- The kept fraction was lowest at θ = 0.
- The same circuits on IQM Emerald came out close to theory.
- 4-qubit GBS on VLQ reached HF 0.77.

So TN3 §7.1 ("VLQ at uniform-noise level") was suspected to be a bug rather than hardware noise. Three candidate
causes were checked: bit order, the measurement map, and the level-3 MOVE transpilation.

**Summary**

| # | Finding | Status |
|---|---|---|
| 1 | qaas 1.0 reverses every VLQ bitstring by default. **All VLQ counts saved before 2026-11-12 are affected.** | Confirmed, fixed in code, confirmed on hardware |
| 2 | Level-3 transpilation onto the star (MOVE/CZ) is correct: no illegal MOVEs, HF 1.000 without noise | Confirmed (simulation) |
| 3 | A single MOVE sandwich adds no significant phase (≤ 0.15 rad) and keeps its coherence | Confirmed on hardware (job 930) |
| 3b | Another qubit's sandwiches leave no phase on the next qubit through the resonator. Idle qubits drift in phase by up to −0.33 rad/µs (QB13), which reaches about 1–1.7 rad over a 5 µs BS circuit. | Confirmed on hardware (job 959); crosstalk vs plain detuning not yet separated |
| 4 | Even with reversed keys, VLQ data stays far below Sirius/Garnet/Emerald. The θ = 0 and θ = π/2 circuits fail on VLQ under any bit order, though FakeVLQ runs them fine. The server decoding is not the cause: re-fetched job 275 lists its keys in clbit order in all 17 circuits. | **Open**, see §5 |
| 5 | The coherent-phase fit (`fit_move_phases.py`) explains part of the data, but its phases are not real MOVE phases | Ruled out by §3; the idle drift of §3b is a candidate for what they stand in for |

The metric throughout is the **squared Bhattacharyya coefficient (SBC)** between the measured Fock distribution and
the exact truncated BS: raw (no post-selection), averaged over the 17 θ values in [0, π/2].

## 1. Bit-order bug in qaas 1.0 (`qheappe_client`)

- **Cause.** `QJob.result()` defaults to `key_order_left_to_right=False`, but
  `QBackendIQM.result_to_qiskit[_legacy]()` default to `True`. IQM lists `measurement_keys` in clbit order, so
  `True` is the setting that gives correct Qiskit keys. `utils/vlq.py` called `job.result(qiskit_legacy=True)`, so
  every key came back reversed. No error is raised. Symmetric states (GHZ, Bell) still look correct, so the bug hides.
- **When it started.** qaas 1.0 was installed in the BS venv on 2026-11-08 at 10:36, before the first VLQ BS run
  (2026-11-08 19:31). qaas 0.5 decoded correctly on the server side. **Every VLQ BS run is affected.**
- **Fix.** `vlq.collect()` now calls `job.result(qiskit_legacy=True, key_order_left_to_right=True)` and raises if
  the `measurement_keys` are not in clbit order. Checked on hardware: in job 930 the key-order circuit gives `01`
  in 91 % of shots, as expected.
- **Other branches** (only those with qaas 1.0 / iqm-client 36):
  - `quantula/resources.py` `VLQDevice` uses the legacy `qaas.client`, so it is not affected.
  - HyperView `scripts/run_hyperview_msi_vlq.py` and `scripts/tiny_vlq_replay.py`: already fixed.
  - **Fixed 2026-11-12 (uncommitted):**
    - HyperView `scripts/fake_vs_real_vlq.py`. Its 12 Nov run
      ([vlq_stagerun_20261112_094228](../../worktrees/Hyperview_msi_experiment/results/fake_vs_real_vlq/vlq_stagerun_20261112_094228),
      HEAppE jobs 649–657) is stored with reversed keys. GHZ5 is unaffected; the GBS HF must be recomputed.
    - Pavia `scripts/run_hyperview_msi_vlq.py`. Its 9 Nov HyperView VLQ runs
      ([hyperview_msi_vlq](../../worktrees/Hyperview_msi_experiment/results/hyperview_msi_vlq), HEAppE 326–336)
      and the selected bands in the shared MLflow store were computed from reversed counts.
  - The GBS effect is documented in
    [README_vlq_replay_20261112.md](../../worktrees/Hyperview_msi_experiment/results/tiny_vlq_replay/README_vlq_replay_20261112.md).
- **Upstream.** An issue text for the qaas repository is drafted: change the default to `True`, or derive the order
  from `measurement_keys`.

### Effect on the archived runs

"Stored" = keys as saved; "reversed" = every key reversed (the fix applied after the fact); "best" = best of all 24
bit permutations. Runs on other backends used `IQMProvider` directly and are not affected.

| Run | Backend | Input | Physical qubits (q0..q3) | SBC stored | SBC reversed | Best permutation |
|---|---|---|---|---|---|---|
| [bs_full_in11_2026-11-08_193135](bs_full_in11_2026-11-08_193135) | VLQ | \|1,1> | QB10 QB13 QB14 QB9 | 0.114 | **0.352** | reversal |
| [bs_full_in02_2026-11-08_200127](bs_full_in02_2026-11-08_200127) | VLQ | \|0,2> | QB10 QB13 QB14 QB9 | 0.063 | 0.259 | 0.367, not reversal (§5) |
| [bs_full_in20_2026-11-08_200400](bs_full_in20_2026-11-08_200400) | VLQ | \|2,0> | QB14 QB17 QB13 QB9 | 0.060 | **0.234** | reversal |
| [bs_full_in11_2026-11-09_125141](bs_full_in11_2026-11-09_125141) | VLQ | \|1,1> | QB2 QB17 QB18 QB20 | 0.148 | 0.171 | reversal |
| [bs_full_in11_2026-11-09_125551](bs_full_in11_2026-11-09_125551) | VLQ | \|1,1> | QB15 QB16 QB21 QB24 | 0.160 | 0.136 | none better than 0.160 |
| [bs_full_in11_2026-11-09_163711](bs_full_in11_2026-11-09_163711) | VLQ (job 275) | \|1,1> | QB11 QB13 QB17 QB14 | 0.126 | **0.417** ([re-fetched](bs_full_in11_2026-11-09_163711/vlq_refetch_job275.json)) | not computed |
| [bs_full_in11_2026-11-12_093835](bs_full_in11_2026-11-12_093835) | VLQ (job 427) | \|1,1> | — | no data: the job `FAILED` on the server (re-fetch 2026-11-13) | | |
| [bs_full_in11_2026-11-09_141455-Sirius](bs_full_in11_2026-11-09_141455-Sirius) | Sirius | \|1,1> | | 0.415 | | |
| [bs_full_in02_2026-11-10_114839-Sirius](bs_full_in02_2026-11-10_114839-Sirius) | Sirius | \|0,2> | | 0.361 | | |
| [bs_full_in20_2026-11-10_115300-Sirius](bs_full_in20_2026-11-10_115300-Sirius) | Sirius | \|2,0> | | 0.376 | | |
| [bs_full_in11_2026-11-11_123419-Garnet](bs_full_in11_2026-11-11_123419-Garnet) | Garnet | \|1,1> | | 0.461 | | |
| [bs_full_in02_2026-11-11_124446-Garnet](bs_full_in02_2026-11-11_124446-Garnet) | Garnet | \|0,2> | | 0.490 | | |
| [bs_full_in20_2026-11-11_125224-Garnet](bs_full_in20_2026-11-11_125224-Garnet) | Garnet | \|2,0> | | 0.539 | | |
| [bs_full_in11_2026-11-11_123807-Emerald](bs_full_in11_2026-11-11_123807-Emerald) | Emerald | \|1,1> | | 0.754 | | |
| [bs_full_in02_2026-11-11_125911-Emerald](bs_full_in02_2026-11-11_125911-Emerald) | Emerald | \|0,2> | | 0.691 | | |
| [bs_full_in20_2026-11-11_130825-Emerald](bs_full_in20_2026-11-11_130825-Emerald) | Emerald | \|2,0> | | 0.735 | | |
| FakeVLQ (stored next to each VLQ run) | FakeVLQ | all | as the VLQ run | 0.54–0.81 | | |

Reversal fixes most of the gap on 8 Nov: the best VLQ run, 0.35, comes close to Sirius at 0.42. It barely helps on
9 Nov, except for job 275 (0.13 → 0.42). That 0.42 overstates job 275, though: a θ-independent distribution (its
θ-average) already scores 0.36, because a (2,0) peak sits at every θ. **None of the VLQ runs at TN3 §7.1 is valid as stored.** Re-analyse them with reversed keys, or better,
re-fetch them (§6).

## 2. Transpilation is correct

[`scripts/check_move_validity.py`](../scripts/check_move_validity.py) re-transpiles the circuits (level 3, seed 42,
FakeVLQ topology) for 6 layouts/inputs at θ = nπ/32. It then steps a state vector through each circuit. The physical
MOVE is undefined when the qubit and the resonator are both excited, and FakeVLQ's MOVE unitary hides that case.

Result: **no illegal MOVEs, no 1-qubit gates on the resonator, and noise-free HF 1.000 in every case.** The
diag_star concern from TN3 does not apply to the BS circuits.

## 3. A MOVE sandwich alone adds no phase (hardware test)

The test is [`scripts/move_sandwich_test.py --test single`](../scripts/move_sandwich_test.py). Each qubit q runs a
Ramsey sequence: r(π/2) q, then MOVE q→R, k CZs between a spectator s and R, MOVE R→q, then r(π/2, φ) q.

Run: [move_sandwich_2026-11-12_212311](move_sandwich_2026-11-12_212311), VLQ job 930, calibration set
`615046c5-ddb2-48a0-8bdf-69af92721e15`, frame tracking `full`, 1000 shots. The FakeVLQ baseline is in
[move_sandwich_2026-11-12_205003](move_sandwich_2026-11-12_205003) and gives 0 ± 0.05 rad, the shot-noise floor.

| Qubit | Phase per sandwich (rad), k = 0 / 1 / 2 / 4 | CZ with s in \|1>, minus π | Phase change per CZ | Ramsey contrast, ref → held |
|---|---|---|---|---|
| QB9 | −0.06 / −0.07 / −0.03 / −0.02 | −0.02 | +0.01 | 0.47 → 0.45 |
| QB10 | −0.02 / +0.01 / −0.03 / −0.03 | 0.00 | −0.01 | 0.47 → 0.46 |
| QB13 | +0.04 / +0.06 / +0.13 / +0.05 | +0.07 | +0.01 | 0.47 → 0.47 |
| QB14 | +0.07 / −0.03 / +0.01 / −0.02 | +0.02 | −0.02 | 0.49 → 0.48 |
| QB17 | −0.02 / −0.14 / −0.14 / −0.15 | −0.12 | −0.03 | 0.47 → 0.47 |

- **No residual MOVE phase:** most values are within ±0.03 rad. QB17 has an offset of about −0.14 rad once a CZ is
  held.
- **Frame tracking works:** the phase does not grow with the hold time.
- **The CZ is calibrated:** the conditional phase is π to within 0.1 rad.
- **Coherence is kept:** contrast barely drops across the sandwich.

Caveat: the BS runs were on 8–9 Nov and stored no calibration-set id, so a miscalibration on those days that has
since been fixed cannot be excluded.

## 3b. Resonator traffic from other qubits (hardware test)

The test is [`move_sandwich_test.py --test multi`](../scripts/move_sandwich_test.py). Qubit q runs a Ramsey sequence
while a second qubit a runs K sandwiches (MOVE · 1 CZ with s · MOVE, about 216 ns each):
- `idleK`: during q's idle time (crosstalk or detuning);
- `carryK`: before q's own sandwich (resonator frame carry-over);
- `*2x`: the same with a in |1>.

Run: [move_sandwich_multi_2026-11-12_220742](move_sandwich_multi_2026-11-12_220742), VLQ job 959, same calibration
set as §3, frame tracking `full`, 1000 shots. The FakeVLQ baseline is in
[move_sandwich_multi_2026-11-12_214236](move_sandwich_multi_2026-11-12_214236) and gives 0 ± 0.05 rad. The key-order
check gives `01` at 0.91.

| q (a, s) | own | idle 1 / 2 / 4 | carry 1 / 2 / 4 | idle2x / carry2x | idle per sandwich | ≈ rad/µs | carry per sandwich |
|---|---|---|---|---|---|---|---|
| QB9 (QB10, QB13) | −0.01 | −0.06 / −0.08 / −0.16 | −0.08 / −0.08 / −0.05 | −0.06 / −0.06 | −0.038 | −0.18 | −0.006 |
| QB10 (QB13, QB14) | +0.05 | −0.04 / −0.11 / −0.21 | −0.01 / 0.00 / 0.00 | −0.11 / −0.02 | −0.054 | −0.25 | −0.009 |
| QB13 (QB14, QB17) | +0.01 | −0.08 / −0.13 / −0.29 | −0.06 / −0.03 / −0.06 | −0.19 / +0.02 | −0.072 | −0.33 | −0.014 |
| QB14 (QB17, QB9) | +0.06 | +0.01 / +0.03 / +0.07 | +0.07 / +0.05 / +0.03 | +0.04 / +0.05 | +0.018 | +0.08 | −0.010 |
| QB17 (QB9, QB10) | +0.08 | 0.00 / +0.10 / +0.09 | +0.03 / +0.01 / +0.02 | +0.06 / +0.03 | +0.025 | +0.12 | −0.013 |

- **No carry-over:** the phase does not grow with the number of earlier sandwiches (≤ 0.014 rad per sandwich). The
  resonator frame is reset correctly between qubits.
- **Idle drift:** the phase grows linearly with q's idle time. The rate is up to −0.33 rad/µs on QB13, i.e. a
  detuning of about −50 kHz. The values depend on the qubit and are reproducible across K. Exciting a changes little
  (`idle2x` vs `idle2`), so this is mostly not ZZ coupling to a.
- **Size in the BS:** the BS circuit lasts about 5 µs (2 µs at θ = 0; ASAP estimate with the generator's gate
  durations). At these rates an idle qubit in superposition picks up about 0.4–1.7 rad. That is the same size as the
  phases `fit_move_phases.py` needed (§4). The fitted values do not match these rates qubit by qubit, but the fit
  could not determine them uniquely anyway.
- **Not yet separated:** a constant frequency offset (calibration drift) and a Stark shift from the other qubits'
  flux pulses look the same here. A plain `delay` of the same length would tell them apart.

## 4. Coherent-phase fit: it fits, but the phases are not MOVE phases

[`scripts/fit_move_phases.py`](../scripts/fit_move_phases.py) fits a model to the reversed counts:
p = (1 − ε)·p_circuit(phases) + ε/16, with one residual Z phase per qubit per MOVE sandwich.

| Group | Runs | ε without → with phases | Fidelity to data | Fitted phases (rad) |
|---|---|---|---|---|
| 08 Nov | in11/in02/in20 | 0.84 → 0.39 | 0.795 → 0.916 | QB9 +0.66, QB10 −0.90, QB13 −0.42, QB14 +1.91, QB17 +0.17 |
| 09 Nov a | 125141 | 0.96 → 0.45 | 0.811 → 0.902 | QB2 −1.98, QB17 +0.33, QB18 −0.43, QB20 −1.11 |
| 09 Nov b | 125551 | 0.99 → 0.39 | 0.731 → 0.863 | QB15 +2.35, QB16 +0.31, QB21 +1.43, QB24 +1.15 |

- **Out of sample:** phases fitted on in11 alone carry over to in02 (ε 0.87 → 0.60) and in20 (ε 0.90 → 0.48).
- **Not identifiable:** the individual phase values change between fits.
- **Contradicted by §3:** the fit needs phases of 0.4–1.9 rad on QB9/10/13/14, but the hardware shows ≤ 0.15 rad.
  The fitted phases only stand in for some other coherent error.

## 5. Open: VLQ data that fits no single bit order

The run [bs_full_in11_2026-11-08_193135](bs_full_in11_2026-11-08_193135), with reversed keys, matches theory for
0 < θ ≲ 1.1 (SBC 0.36–0.54 per θ). Beyond that it degrades:
- **`1111` near π/2:** it grows to about 0.30, so SBC falls to 0.14 at θ = 1.47.
- **θ = 0:** it matches only *without* reversal (`1010`: 0.32; reversed gives SBC 0.06).
- **θ = π/2:** neither order fits (SBC ≤ 0.11).

The transpiler builds exactly these two circuits differently: 10 and 23 CZs instead of 28. Other anomalies:
- **in02** decodes mode-swapped under reversal at every θ.
- **in20** has about 40 % `0000` at every θ.
- **The 9 Nov runs** stay near 0.15 under any bit order, e.g. `1111` = 0.43 at θ = 0 on QB2/17/18/20.

**Re-fetch of job 275** (9 Nov, QB11/13/17/14, [vlq_refetch_job275.json](bs_full_in11_2026-11-09_163711/vlq_refetch_job275.json)):
- The server lists `measurement_keys` as `meas_4_0_0 … meas_4_0_3` (clbit order) in all 17 circuits. The old
  decoding is exactly the reverse of the fixed one.
- Bitstring SBC against FakeVLQ (same layout, seeded re-run): fixed decoding 0.79–0.87 for 0 < θ < π/2, mean 0.78.
  Old decoding: mean 0.43. So the fixed decoding is right, and the interior circuits behave as the noise model
  predicts.
- The endpoints do not:

  | θ | ideal | FakeVLQ top outcome | VLQ (fixed) top outcomes |
  |---|---|---|---|
  | 0 | `1010` | `1010` 0.87 | `0001` 0.39, `1011` 0.26, `1010` 0.04 |
  | 0.098 | `1010` 0.96 | `1010` 0.74 | `1010` 0.42 |
  | π/2 | `1010` | `1010` 0.75 | `0000` 0.34, `1010` 0.18 |

The endpoint failure therefore comes from what the hardware does with those two differently transpiled circuits, not
from decoding. The 8 Nov jobs are not re-fetched yet (ids unknown), so for them hypothesis 1 below is unlikely
rather than excluded.

Hypotheses:
1. ~~The server's `measurement_keys` order was **not the same in every circuit**.~~ Ruled out for job 275.
2. A coherent error **involving several qubits** that the single-qubit sandwich test cannot see. §3b rules out
   resonator-frame carry-over. It finds an idle phase drift of up to 0.33 rad/µs, enough for 1–1.7 rad per BS
   circuit, which makes it the main candidate.

The real-VLQ transpiled circuits of these runs were not saved, and §2 used the FakeVLQ target. A difference between
the two targets is not excluded.

## 6. Next steps (scripts ready)

1. **Re-fetch the old jobs** with [`scripts/refetch_vlq_jobs.py`](../scripts/refetch_vlq_jobs.py). It saves the
   server's raw counts and per-circuit `measurement_keys` as `vlq_refetch_job<ID>.json` in the matching run folder.
   It also prints SBC as stored, fixed and reversed for each θ.
   - Done 2026-11-13: 275 (keys in clbit order, §5) and 427 (`FAILED` on the server, no data).
   - Remaining: the 8 Nov and 9 Nov 12:51/12:55 jobs. Their ids are not stored; find them with `--scan <first> <last>`,
     which lists state and times per id (job 275 = 9 Nov 15:36 UTC, so the earlier ones are below 275).
   ```bash
   .venv/bin/python scripts/refetch_vlq_jobs.py 275 427
   .venv/bin/python scripts/refetch_vlq_jobs.py --scan 150 300
   ```
2. **Multi-qubit sandwich test:** done (§3b, job 959). Follow-ups:
   - a `delay` variant of the same length, to separate frequency offset from flux crosstalk;
   - a simulation of the BS circuit with the measured idle rates, to check whether they reproduce the VLQ
     distributions.
3. **Repeat the BS** with the fixed `collect()`, saving the transpiled circuits and the calibration-set id. Run θ = 0
   and θ = π/2 on the 8 Nov qubits first. Also compare their transpiled circuits with the θ = 0.098 one: the
   endpoints are what fails (§5).
4. **Rewrite TN3 §7.1** once steps 1–3 are in.

## 7. 13 Nov runs (fixed key order): only the endpoints fail

Runs [in11](bs_full_in11_2026-11-13_091109), [in02](bs_full_in02_2026-11-13_092111),
[in20](bs_full_in20_2026-11-13_092357): QB17 QB9 QB7 QB14, calibration set `615046c5…` (same as jobs 930/959).

| Input | SBC_post θ = 0 | SBC_post 0 < θ < π/2 | SBC_post θ = π/2 | kept (interior), VLQ / FakeVLQ |
|---|---|---|---|---|
| \|1,1> | 0.48 | 0.89–0.99 | 0.16 | 0.41–0.48 / 0.79–0.83 |
| \|0,2> | 0.48 | 0.81–1.00 | 0.23 | 0.44–0.49 / 0.82–0.85 |
| \|2,0> | 0.55 | 0.82–0.98 | 0.12 | 0.39–0.43 / 0.82–0.85 |

- The BS itself works on VLQ: after post-selection the interior matches theory. The interior gap to FakeVLQ is
  leakage out of the photon-number sector (kept ≈ 0.45 vs 0.82), not a wrong distribution.
- Endpoints fail in **all 8 VLQ runs on 5 layouts** (8, 9 and 13 Nov, keys corrected), never on FakeVLQ.
- **Not generic coherent noise.** A gate-list simulation of the level-3 circuits (ASAP schedule; per-qubit detuning,
  random Z after each 2q gate, prx over-rotation, CZ phase, MOVE-sandwich phase) makes θ = 0 the *most* robust
  circuit (2.1 µs vs 5.3 µs), never the worst.
- **Not a mislabelled result.** No other input, θ, or qubit permutation fits the endpoint counts much better than the
  intended circuit plus noise.
- The endpoint circuits are the only ones level 3 simplifies (θ = 0: 10 CZ / 18 MOVE, π/2: 23 / 38; others 28 / 42),
  but they contain no MOVE-sandwich motif that the interior circuits lack. They are also always the **first and last
  circuit of the job**.
- Sirius/Garnet/Emerald pass at the endpoints, but those runs came from another pipeline (different `results.json`
  format, layout and transpilation not recorded), so they do not show that our level-3 endpoint circuits run fine on a
  star device.

Open: (A) the simplified circuits fail on VLQ hardware, or (B) the batch edges do. Test:
[`scripts/endpoint_position_test.py`](../scripts/endpoint_position_test.py), one job of 12 circuits: the level-3
endpoints at the edges and in the middle, the level-1 (28 CZ) endpoints in the middle, θ = π/4 controls. FakeVLQ
baseline [endpoint_position_in11_2026-11-13_094033](endpoint_position_in11_2026-11-13_094033): SBC_post ≥ 0.95
everywhere. Workaround meanwhile: transpile the endpoints at level 1, or pad the batch with a dummy circuit at each end.

### 7.1 Job 1059 (endpoint test): content, not position

[endpoint_position_in11_2026-11-13_094714](endpoint_position_in11_2026-11-13_094714), calibration set `2df63f7d…`.
The level-3 θ = 0 circuit gives the same counts at positions 0, 2, 8 (SBC_post 0.51–0.53, `0000` 0.34); level-3 π/2
gives the same at 3, 9, 11 (0.18–0.20). **Batch position is ruled out.** Level 1 helps but does not fully fix:
θ = 0 0.65, π/2 0.80 (interior 0.86–0.90).

- The circuits sent to VLQ (saved QPY) are gate-for-gate identical to the FakeVLQ transpilation, and noise-free
  correct. Nothing is lost client-side.
- **Ruled out by simulation:** decay (T1 30–47 µs, also the resonator), a residual excitation in the resonator or a
  qubit at shot start, CZ/MOVE local-phase miscalibration, and |11> load at CZs.
- **Skeleton, not angles.** Level-1 π/2 has the interior CZ/MOVE skeleton (only angles differ) and works (0.80).
  The three bad circuits are exactly those with a different skeleton.
- **The bad skeletons share one motif:** `MOVE QB7,R · CZ QB17,R · MOVE QB7,R`. No working circuit has it; in the
  interior circuits QB17 is only ever moved, never a CZ partner. QB7 was never the held qubit in jobs 930/959.
- **Swap fit.** Replacing only that CZ by a full |01>↔|10> exchange (plus a conditional phase) fits the data:
  θ = 0 rises from 0.77–0.87 to 0.92–0.96 for all inputs (fit on the 13 Nov runs). Out of sample, on job 1059, the
  level-3 θ = 0 rises from 0.73 to 0.93 and the level-1 θ = 0 from 0.83 to 0.92. π/2 is only partly explained
  (0.80–0.85 → 0.84–0.90).
- **Job 930 shows CZ(QB17,R) correct** when QB14's state is in R. So the fault would be specific to the pair
  (held QB7, CZ QB17), or to the circuit context.

Test: [`scripts/cz_motif_test.py`](../scripts/cz_motif_test.py) runs the motif `MOVE H · CZ P · MOVE H` for all ordered
pairs of QB17/QB9/QB7/QB14 as a population test: H in |1> stays in H under a CZ, but moves to P under a swap. It also
runs the BS endpoints with a barrier after every gate. FakeVLQ baseline
[cz_motif_2026-11-13_100846](cz_motif_2026-11-13_100846): ideal ≥ 0.93, swapped 0.00.
