"""Generates ``utils/fake_vlq_backend.py`` from a VLQ calibration (observation-set) JSON export.

Usage::

    python scripts/generate_fake_vlq_backend.py calibration/VLQ/<snapshot>.json [-o utils/fake_vlq_backend.py]

Units are converted to those of :class:`IQMErrorProfile`. Benchmarked average gate fidelities are converted to
depolarizing parameters ``lam`` such that T1/T2 relaxation followed by ``depolarizing_error(lam)`` (the channel
:class:`IQMFakeBackend` builds per gate) reproduces the benchmarked fidelity, see :func:`depolarizing_parameter`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

NUM_QUBITS: int = 24
QUBITS: list[str] = [f"QB{i}" for i in range(1, NUM_QUBITS + 1)]
RESONATOR_FAKE: str = "CR1"  # label used in the generated backend
RESONATOR_JSON: str = "COMPR1"  # label used in the calibration JSON

# Gate durations (ns); not present in the JSON. They set the relaxation subtracted from the benchmarked error.
GATE_DURATIONS_SINGLE: dict[str, float] = {"prx": 32.0}
GATE_DURATIONS_TWO: dict[str, float] = {"cz": 88.0, "move": 64.0}

# Decimal places of the generated values.
T_NS_DECIMALS: int = 1
PRX_DECIMALS: int = 5
TWO_Q_DECIMALS: int = 6
READOUT_DECIMALS: int = 4

# T2 below this (s) is a failed fit (e.g. COMPR1 reports 17 ns with zero uncertainty on 2026-10-23).
MIN_PLAUSIBLE_T2_S: float = 1e-6

# MOVE(b -> CR), n x CZ(a, CR), MOVE(CR -> b): b is stored in the resonator during each CZ. Source:
# https://docs.it4i.cz/en/docs/clusters/vlq/calibration_health ("MOVE-CZ-MOVE pair fidelity").
PAIR_FIELD: re.Pattern[str] = re.compile(
    rf"metrics\.irb\.circuit\.move_ncz_move\.(QB\d+)__{RESONATOR_JSON}__(QB\d+)"
    r"\.fidelity:n_interleaved_gates=([01])$"
)

Observations = dict[str, float]
Pair = tuple[str, str]


@dataclass
class Calibration:
    """Error-profile values extracted from one calibration snapshot."""

    t1s: dict[str, float]
    t2s: dict[str, float]
    prx: dict[str, float]
    cz: dict[Pair, float]
    move: dict[Pair, float]
    readout: dict[str, dict[str, float]]
    timestamp: str
    source_file: str
    notes: list[str] = field(default_factory=list)


def load_observations(path: Path) -> tuple[Observations, str]:
    """Returns a flat ``{dut_field: value}`` map of the valid observations and the snapshot timestamp."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    observations: Observations = {
        str(obs["dut_field"]): float(obs["value"])
        for obs in data["observations"]
        if not obs.get("invalid") and obs.get("value") is not None
    }
    timestamp: str = data.get("created_timestamp") or data.get("end_timestamp") or ""
    return observations, timestamp


def require(obs: Observations, name: str) -> float:
    """Returns ``obs[name]``, raising :class:`KeyError` with the field name if it is missing."""
    if name not in obs:
        raise KeyError(f"missing calibration field: {name}")
    return obs[name]


def _note(notes: list[str], msg: str) -> None:
    """Records a note for the generated file header and echoes it to stderr."""
    notes.append(msg)
    print(f"note: {msg}", file=sys.stderr)


def _plausible_t2(val: float | None, label: str, notes: list[str], floor_s: float = MIN_PLAUSIBLE_T2_S) -> float | None:
    """Returns ``val``, or ``None`` (with a note) if it is below ``floor_s``, i.e. a failed fit."""
    if val is not None and val < floor_s:
        _note(
            notes,
            f"{label} t2_time = {val * 1e9:.1f} ns is below the plausible floor {floor_s * 1e6:.1f} us, "
            "treated as a failed fit",
        )
        return None
    return val


def extract_t1(obs: Observations) -> dict[str, float]:
    """T1 times, s -> ns."""
    result = {
        RESONATOR_FAKE: round(require(obs, f"characterization.model.{RESONATOR_JSON}.t1_time") * 1e9, T_NS_DECIMALS)
    }
    for qb in QUBITS:
        result[qb] = round(require(obs, f"characterization.model.{qb}.t1_time") * 1e9, T_NS_DECIMALS)
    return result


def extract_move_cz_move_pairs(obs: Observations) -> dict[Pair, tuple[float, float]]:
    """MOVE-CZ-MOVE IRB fidelities as ``{(a, b): (F_n0, F_n1)}``; pairs missing either ``n`` are skipped."""
    by_n: dict[str, dict[Pair, float]] = {"0": {}, "1": {}}
    for name, val in obs.items():
        match = PAIR_FIELD.match(name)
        if match:
            by_n[match[3]][(match[1], match[2])] = val
    return {pair: (by_n["0"][pair], f1) for pair, f1 in by_n["1"].items() if pair in by_n["0"]}


def resonator_t2_lower_bound(pairs: dict[Pair, tuple[float, float]], t1_s: float) -> float | None:
    """Median over pairs of the lowest resonator T2 (s) consistent with the MOVE-CZ-MOVE benchmarks.

    The extra error of n=1 over n=0 (ratio of 2-qubit depolarizing parameters ``p = (4F - 1) / 3``) is attributed
    to storing b in the resonator for one CZ duration. This ignores the CZ's own error and storage during the
    MOVEs, so T2 is underestimated. The storage channel has process fidelity
    ``F_pro = (1 + e^{-t/T1} + 2 e^{-t/T2}) / 4`` and 2-qubit average infidelity ``eps = 4/5 (1 - F_pro)``.
    Returns ``None`` without pair data.
    """
    duration_s: float = GATE_DURATIONS_TWO["cz"] * 1e-9
    bounds: list[float] = []
    for f0, f1 in pairs.values():
        eps: float = 0.75 * (1.0 - (4 * f1 - 1) / (4 * f0 - 1))
        exp_t_t2: float = 1.0 - (5 * eps - (1.0 - math.exp(-duration_s / t1_s))) / 2
        if exp_t_t2 >= 1.0:
            bounds.append(math.inf)  # no extra error: any T2 fits
        elif exp_t_t2 <= 0.0:
            bounds.append(0.0)  # error too large to constrain T2
        else:
            bounds.append(-duration_s / math.log(exp_t_t2))
    return statistics.median(bounds) if bounds else None


def extract_t2(obs: Observations, notes: list[str]) -> dict[str, float]:
    """T2 times, s -> ns.

    An implausible qubit ``t2_time`` falls back to ``t2_echo_time``; the resonator falls back to its T1. The
    resonator's plausibility floor is raised to the MOVE-CZ-MOVE bound (capped at T1).
    """
    cr_t1: float = require(obs, f"characterization.model.{RESONATOR_JSON}.t1_time")
    pairs = extract_move_cz_move_pairs(obs)
    bound = resonator_t2_lower_bound(pairs, cr_t1)
    cr_floor: float = MIN_PLAUSIBLE_T2_S
    if bound is not None:
        _note(
            notes,
            f"{RESONATOR_JSON} T2 >= {bound * 1e6:.1f} us "
            f"(median bound from {len(pairs)} MOVE-CZ-MOVE pair benchmarks)",
        )
        cr_floor = max(cr_floor, min(bound, cr_t1))  # never demand more than T1, the fallback value
    cr_val = _plausible_t2(obs.get(f"characterization.model.{RESONATOR_JSON}.t2_time"), RESONATOR_JSON, notes, cr_floor)
    if cr_val is None:
        cr_val = cr_t1
        _note(notes, f"{RESONATOR_JSON} T2 set to its T1 = {cr_t1 * 1e6:.1f} us")
    result: dict[str, float] = {RESONATOR_FAKE: round(cr_val * 1e9, T_NS_DECIMALS)}
    for qb in QUBITS:
        val = _plausible_t2(obs.get(f"characterization.model.{qb}.t2_time"), qb, notes)
        if val is None:
            val = require(obs, f"characterization.model.{qb}.t2_echo_time")
            _note(notes, f"{qb} T2 taken from t2_echo_time = {val * 1e6:.1f} us")
        result[qb] = round(val * 1e9, T_NS_DECIMALS)
    return result


def relaxation_process_fidelity(t1_ns: float, t2_ns: float, duration_ns: float) -> float:
    """Process fidelity of Aer's ``thermal_relaxation_error``: ``(1 + e^{-t/T1} + 2 e^{-t/T2}) / 4``."""
    return (1.0 + math.exp(-duration_ns / t1_ns) + 2.0 * math.exp(-duration_ns / t2_ns)) / 4.0


def depolarizing_parameter(fidelity: float, relaxation_fidelities: Sequence[float]) -> float:
    """Aer depolarizing parameter ``lam`` such that relaxation + ``depolarizing_error(lam, n)`` has average fidelity
    ``fidelity``; ``n = len(relaxation_fidelities)`` (per-component relaxation process fidelities).

    Uses ``F_avg = (d F_pro + 1) / (d + 1)``, ``F_pro`` multiplicative under tensor products and
    ``F_pro(depol after E) = (1 - lam) F_pro(E) + lam / d**2``. Without relaxation, ``lam = d / (d - 1) * (1 - F)``.
    Negative if relaxation alone exceeds the benchmarked error.
    """
    dim: int = 2 ** len(relaxation_fidelities)
    target: float = ((dim + 1) * fidelity - 1) / dim
    relaxation: float = math.prod(relaxation_fidelities)
    return (relaxation - target) / (relaxation - 1 / dim**2)


def per_gate_fidelity(fidelity: float, n_gates: int, dim: int) -> float:
    """Average fidelity of one gate from that of ``n_gates`` repetitions (depolarizing polarizations multiply)."""
    polarization: float = (dim * fidelity - 1) / (dim - 1)
    root: float = polarization ** (1 / n_gates)
    return (root * (dim - 1) + 1) / dim


def gate_depolarizing(
    gate: str,
    fidelity: float,
    locus: Sequence[str],
    coherence: tuple[dict[str, float], dict[str, float]],
    notes: list[str],
) -> float:
    """:func:`depolarizing_parameter` for ``gate`` on ``locus`` given ``(t1s, t2s)`` in ns; clipped at 0 with a note."""
    duration: float = (GATE_DURATIONS_SINGLE | GATE_DURATIONS_TWO)[gate]
    t1s, t2s = coherence
    lam: float = depolarizing_parameter(
        fidelity, [relaxation_process_fidelity(t1s[c], t2s[c], duration) for c in locus]
    )
    if lam < 0.0:
        _note(
            notes,
            f"{gate} {'__'.join(locus)}: T1/T2 relaxation alone exceeds the benchmarked error "
            f"1 - F = {1.0 - fidelity:.2e}, depolarizing parameter set to 0",
        )
        return 0.0
    return lam


def extract_prx_errors(
    obs: Observations, coherence: tuple[dict[str, float], dict[str, float]], notes: list[str]
) -> dict[str, float]:
    """Single-qubit ``prx`` depolarizing parameter from the RB fidelity."""
    result: dict[str, float] = {}
    for qb in QUBITS:
        fidelity: float = require(obs, f"metrics.rb.prx.drag_crf.{qb}.fidelity:par=d1")
        result[qb] = round(gate_depolarizing("prx", fidelity, [qb], coherence, notes), PRX_DECIMALS)
    return result


def extract_cz_errors(
    obs: Observations, coherence: tuple[dict[str, float], dict[str, float]], notes: list[str]
) -> dict[Pair, float]:
    """CZ depolarizing parameter from the 2-qubit IRB fidelity, keyed by ``(QB, CR1)``."""
    result: dict[Pair, float] = {}
    for qb in QUBITS:
        fidelity: float = require(obs, f"metrics.irb.cz.crf_crf.{qb}__{RESONATOR_JSON}.fidelity")
        lam: float = gate_depolarizing("cz", fidelity, [qb, RESONATOR_FAKE], coherence, notes)
        result[(qb, RESONATOR_FAKE)] = round(lam, TWO_Q_DECIMALS)
    return result


def extract_move_errors(
    obs: Observations, coherence: tuple[dict[str, float], dict[str, float]], notes: list[str]
) -> dict[Pair, float]:
    """MOVE depolarizing parameter; the IRB metric has ``n_interleaved_gates=2``, so it is reduced to one gate first."""
    result: dict[Pair, float] = {}
    for qb in QUBITS:
        pair_fidelity: float = require(
            obs, f"metrics.irb.move.crf_crf.{qb}__{RESONATOR_JSON}.fidelity:n_interleaved_gates=2"
        )
        fidelity: float = per_gate_fidelity(pair_fidelity, n_gates=2, dim=4)
        lam: float = gate_depolarizing("move", fidelity, [qb, RESONATOR_FAKE], coherence, notes)
        result[(qb, RESONATOR_FAKE)] = round(lam, TWO_Q_DECIMALS)
    return result


def extract_readout_errors(obs: Observations) -> dict[str, dict[str, float]]:
    """Readout misclassification per qubit from ``ssro.measure.constant``."""
    return {
        qb: {
            "0": round(require(obs, f"metrics.ssro.measure.constant.{qb}.error_0_to_1"), READOUT_DECIMALS),
            "1": round(require(obs, f"metrics.ssro.measure.constant.{qb}.error_1_to_0"), READOUT_DECIMALS),
        }
        for qb in QUBITS
    }


def extract_calibration(obs: Observations, timestamp: str, source_file: str) -> Calibration:
    """Extracts all error-profile values from the observations."""
    notes: list[str] = []
    t1s = extract_t1(obs)
    t2s = extract_t2(obs, notes)
    return Calibration(
        t1s=t1s,
        t2s=t2s,
        prx=extract_prx_errors(obs, (t1s, t2s), notes),
        cz=extract_cz_errors(obs, (t1s, t2s), notes),
        move=extract_move_errors(obs, (t1s, t2s), notes),
        readout=extract_readout_errors(obs),
        timestamp=timestamp,
        source_file=source_file,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Code generation (output is stable under ``ruff format``)
# ---------------------------------------------------------------------------


def _fmt_scalar_dict(values: dict[str, float], indent: int) -> str:
    pad = " " * indent
    return "\n".join(f'{pad}"{key}": {val!r},' for key, val in values.items())


def _fmt_pair_dict(values: dict[Pair, float], indent: int) -> str:
    pad = " " * indent
    return "\n".join(f'{pad}("{a}", "{b}"): {val!r},' for (a, b), val in values.items())


def _fmt_readout_dict(values: dict[str, dict[str, float]], indent: int) -> str:
    pad = " " * indent
    return "\n".join(f'{pad}"{qb}": {{"0": {val["0"]!r}, "1": {val["1"]!r}}},' for qb, val in values.items())


def _fmt_inline(values: dict[str, float]) -> str:
    return "{" + ", ".join(f'"{key}": {val!r}' for key, val in values.items()) + "}"


TEMPLATE: str = '''# {header}
#
# Copyright 2022-2025 Qiskit on IQM developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fake backend for IQM's {n_qubits}-qubit VLQ architecture.

Gate error parameters are Aer depolarizing parameters with the per-gate T1/T2 relaxation already subtracted, so
relaxation + depolarizing reproduces the benchmarked fidelities. Generated by ``scripts/generate_fake_vlq_backend.py``
-- do not edit by hand.
"""

from iqm.iqm_client import StaticQuantumArchitecture
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMErrorProfile, IQMFakeBackend

CALIBRATION_FILE: str = "{source_file}"
CALIBRATION_TIMESTAMP: str = "{timestamp}"


def fake_vlq_backend() -> IQMFakeBackend:
    """Returns an :class:`IQMFakeBackend` with VLQ's star topology and the calibration above."""
    qubits: list[str] = [f"QB{{i}}" for i in range(1, {n_qubits_plus_one})]
    architecture = StaticQuantumArchitecture(
        dut_label="FakeVLQ",
        qubits=qubits,
        computational_resonators=["{resonator}"],
        connectivity=[("{resonator}", qubit) for qubit in qubits],
    )
    error_profile = IQMErrorProfile(
        t1s={{
{t1s}
        }},
        t2s={{
{t2s}
        }},
        single_qubit_gate_depolarizing_error_parameters={{
            "prx": {{
{prx}
            }},
        }},
        two_qubit_gate_depolarizing_error_parameters={{
            "cz": {{
{cz}
            }},
            "move": {{
{move}
            }},
        }},
        single_qubit_gate_durations={single_durations},
        two_qubit_gate_durations={two_durations},
        readout_errors={{
{readout}
        }},
        name="VLQ",
    )
    return IQMFakeBackend(architecture, error_profile, name="FakeVLQBackend")
'''


def render(calibration: Calibration) -> str:
    """Renders the backend module source for ``calibration``."""
    try:
        snapshot = datetime.fromisoformat(calibration.timestamp.replace("Z", "+00:00"))
        header = snapshot.astimezone(UTC).strftime("%B %d, %Y calibration data from VLQ")
    except ValueError:
        header = "Calibration data from VLQ"
    header += "".join(f"\n# Note: {note}" for note in calibration.notes)

    return TEMPLATE.format(
        header=header,
        n_qubits=NUM_QUBITS,
        n_qubits_plus_one=NUM_QUBITS + 1,
        source_file=calibration.source_file,
        timestamp=calibration.timestamp,
        resonator=RESONATOR_FAKE,
        t1s=_fmt_scalar_dict(calibration.t1s, 12),
        t2s=_fmt_scalar_dict(calibration.t2s, 12),
        prx=_fmt_scalar_dict(calibration.prx, 16),
        cz=_fmt_pair_dict(calibration.cz, 16),
        move=_fmt_pair_dict(calibration.move, 16),
        single_durations=_fmt_inline(GATE_DURATIONS_SINGLE),
        two_durations=_fmt_inline(GATE_DURATIONS_TWO),
        readout=_fmt_readout_dict(calibration.readout, 12),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate the FakeVLQ backend module from a VLQ calibration JSON.")
    parser.add_argument("calibration_json", type=Path, help="Path to the VLQ calibration JSON export.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("utils/fake_vlq_backend.py"),
        help="Destination module (default: utils/fake_vlq_backend.py).",
    )
    args = parser.parse_args(argv)

    obs, timestamp = load_observations(args.calibration_json)
    calibration = extract_calibration(obs, timestamp, args.calibration_json.name)
    args.output.write_text(render(calibration), encoding="utf-8")
    print(f"Wrote {args.output} from {args.calibration_json} (snapshot {timestamp})")


if __name__ == "__main__":
    main()
