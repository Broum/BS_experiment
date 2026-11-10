"""Extract IQM VLQ device parameters for a qiskit-dynamics ``DynamicsBackend``.

The Hamiltonian parameters a ``Solver`` / ``DynamicsBackend`` needs (drive
frequency, anharmonicity, drive strength, sample time ``dt``) live in the live
Pulla settings tree and calibration stash, reached through the APIs verified
against iqm-pulla 13.0.3 below.

**Coherence times come from a different object.** The station keeps two sibling
observation sets per calibration:

* the *calibration set* -- gate and pulse parameters; this is what
  ``Pulla.get_calibration_stash()`` returns, and it carries **no** t1/t2;
* the *quality-metric set* -- ``characterization.model.<c>.t{1,2}_time``,
  ``t2_echo_time`` and the rb/irb/ssro benchmark metrics, fetched from
  ``calibration-sets/{qc}/{id}/metrics``.

Looking t1_time up in the stash therefore always misses. Use
:func:`fetch_quality_metrics` (live) or :func:`load_quality_metrics` (the
hand-exported ``FakeVLQ_parser/*.json``, which are dumps of exactly this set)
and pass the result to the extractors as ``metrics=``.

Architecture note: VLQ is a *resonator-mediated star* -- every qubit couples
only to the central computational resonator ``COMPR1`` through a tunable
coupler that idles at ~zero coupling. There is therefore no static qubit-qubit
J; a two-qubit CZ is a qubit<->resonator gate activated by a flux pulse on the
coupler (see :func:`find_coupler` and the notebook's Milestone C).

Typical use (needs a live connection)::

    from iqm.pulla.pulla import Pulla
    p = Pulla(iqm_server_url)
    compiler = p.get_standard_compiler()
    params = extract_device_params(p, compiler, ["QB1", "QB2"], dim=3)
    save_snapshot(params, "iqm_params_snapshot.json")

``load_snapshot`` reads that file back with no live connection, so downstream
Hamiltonian construction and tests can run offline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import math


class AnharmonicityUnavailable(RuntimeError):
    """Raised when a qubit's anharmonicity is not a real calibrated value.

    Per project policy the simulation must not fall back to a generic guessed
    anharmonicity -- obtain a measured value (request the chip's ``anharmonicity``
    / ``charging_energy`` from IQM, or run a 1<->2 (EF) spectroscopy through
    Pulla) before building the Hamiltonian.
    """


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class QubitParams:
    """Per-qubit Hamiltonian inputs, all in SI units (Hz, s)."""

    name: str
    nu: float                       # drive / transition frequency f_01, Hz
    alpha: float                    # anharmonicity f_12 - f_01, Hz (negative)
    r: float                        # max Rabi frequency at unit drive amplitude, Hz
    T1: float = float("nan")        # relaxation time, s
    T2: float = float("nan")        # dephasing time (echo or ramsey), s
    alpha_source: str = ""          # provenance of alpha (audit trail)
    coherence_source: str = ""      # provenance of T1/T2 (audit trail)
    coupler: str = ""               # coupler joining this qubit to the resonator


@dataclass
class ResonatorParams:
    """Computational-resonator subsystem inputs (used for CZ / MOVE in Milestone C)."""

    name: str
    nu: float                       # resonator frequency, Hz
    T1: float = float("nan")
    T2: float = float("nan")
    nu_source: str = ""             # provenance of nu (audit trail)
    coherence_source: str = ""      # provenance of T1/T2 (audit trail)


@dataclass
class DeviceParams:
    """Everything needed to build the DynamicsBackend for a chosen qubit subset."""

    qubits: list[QubitParams]
    dt: float                                   # sample time, s
    dim: int = 3                                # truncation dim per subsystem
    resonator: ResonatorParams | None = None
    calibration_set_id: str = ""               # provenance
    chip_label: str = ""

    # convenience -------------------------------------------------------------
    def qubit(self, name: str) -> QubitParams:
        for q in self.qubits:
            if q.name == name:
                return q
        raise KeyError(name)


# --------------------------------------------------------------------------- #
# Low-level readers (verified against iqm-pulla 13.0.3)
# --------------------------------------------------------------------------- #

def _unwrap(x: Any) -> float | None:
    """Coerce a Pulla ``Setting`` / ``ObservationLite`` / scalar to a float."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    for attr in ("value", "magnitude"):
        if hasattr(x, attr):
            return _unwrap(getattr(x, attr))
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _node_get(node: Any, key: str) -> Any:
    """Access ``node[key]`` or ``node.key`` on a SettingNode, else None."""
    try:
        return node[key]
    except (TypeError, KeyError, AttributeError):
        pass
    return getattr(node, key, None)


def cal_value(stash: Any, dut_field: str) -> float | None:
    """Value of a calibration observation, or None.

    ``stash`` is the ``PullaStash`` returned by ``Pulla.get_calibration_stash()``:
    a dataclass with ``.observations: dict[dut_field -> ObservationLite]`` and a
    ``get_latest_observation(name)`` accessor (``iqm/pulla/pulla.py``).
    """
    getter = getattr(stash, "get_latest_observation", None)
    obs = getter(dut_field) if callable(getter) else None
    if obs is None:
        observations = getattr(stash, "observations", None)
        if isinstance(observations, Mapping):
            obs = observations.get(dut_field)
    return _unwrap(obs)


def _observations_to_map(obj: Any) -> dict[str, float]:
    """``{dut_field: value}`` from an observation set, skipping invalid entries.

    Accepts either the pydantic ``ObservationSetWithObservations`` returned by
    the server (``.observations`` -> list of ``ObservationLite``) or the plain
    dict parsed from a JSON export -- both carry the same fields.
    """
    observations = obj.get("observations") if isinstance(obj, Mapping) else getattr(obj, "observations", None)
    if observations is None:
        raise TypeError(f"not an observation set: {type(obj).__name__}")
    out: dict[str, float] = {}
    for o in observations:
        get = o.get if isinstance(o, Mapping) else lambda k, d=None, _o=o: getattr(_o, k, d)
        if get("invalid", False):
            continue
        field, value = get("dut_field"), get("value")
        if field is not None and isinstance(value, (int, float)):
            out[str(field)] = float(value)
    return out


def fetch_quality_metrics(p: Any, calibration_set_id: str = "default") -> dict[str, float]:
    """``{dut_field: value}`` of the live quality-metric set (t1/t2 + benchmarks).

    This is the set the calibration stash does *not* contain. Endpoint:
    ``_IQMServerClient.get_calibration_set_quality_metric_set`` -> GET
    ``calibration-sets/{qc}/{id}/metrics`` (``iqm/iqm_server_client/iqm_server_client.py``);
    ``IQMClient.get_quality_metric_set()`` is the public wrapper over the same call.
    """
    client = getattr(p, "_iqm_server_client", None)
    fn = getattr(client, "get_calibration_set_quality_metric_set", None)
    if callable(fn):
        return _observations_to_map(fn(calibration_set_id))
    fn = getattr(p, "get_quality_metric_set", None)          # IQMClient-style handle
    if callable(fn):
        return _observations_to_map(fn())
    raise RuntimeError(
        f"{type(p).__name__} exposes no quality-metric-set accessor "
        "(expected _iqm_server_client.get_calibration_set_quality_metric_set)"
    )


def load_quality_metrics(path: str | Path) -> dict[str, float]:
    """``{dut_field: value}`` from a hand-exported observation-set JSON.

    The ``FakeVLQ_parser/*.json`` files are dumps of a quality-metric set
    (``observation_set_type: 'quality-metric-set'``), so they are a drop-in
    offline stand-in for :func:`fetch_quality_metrics` -- with the caveat that
    they are frozen at their ``created_timestamp``, not the current operating point.
    """
    with open(path) as fh:
        return _observations_to_map(json.load(fh))


def metric_value(metrics: Mapping[str, float] | None, dut_field: str) -> float | None:
    """Value from a quality-metric map, or None when absent/empty."""
    if not metrics:
        return None
    v = metrics.get(dut_field)
    return float(v) if isinstance(v, (int, float)) else None


def _resolve_coherence(stash: Any, metrics: Mapping[str, float] | None,
                       settings: Any, component: str) -> tuple[float | None, float | None, str]:
    """(T1, T2, provenance) for one component, quality-metric set first.

    T2 prefers ``t2_time`` and falls back to ``t2_echo_time`` (a handful of VLQ
    qubits only ever get the echo variant -- see FakeVLQ_parser/CALIBRATION_NOTES.md).
    """
    base = f"characterization.model.{component}"
    T1 = metric_value(metrics, f"{base}.t1_time")
    T2 = metric_value(metrics, f"{base}.t2_time")
    T2_echo = metric_value(metrics, f"{base}.t2_echo_time")
    if T1 is not None or T2 is not None or T2_echo is not None:
        src = "quality-metric-set" + ("" if T2 is not None else ":t2_echo_time")
        return T1, (T2 if T2 is not None else T2_echo), src

    T1 = cal_value(stash, f"{base}.t1_time")
    T2 = cal_value(stash, f"{base}.t2_time") or cal_value(stash, f"{base}.t2_echo_time")
    if T1 is not None or T2 is not None:
        return T1, T2, "calibration-stash"

    T1, _ = read_model_param(settings, component, "t1_time")
    T2, _ = read_model_param(settings, component, "t2_time", "t2_echo_time")
    if T1 is not None or T2 is not None:
        return T1, T2, "settings.characterization.model"
    return None, None, "unavailable"


def probe_calibration_fields(stash: Any, prefix: str = "") -> list[str]:
    """Sorted dut_fields in the stash starting with ``prefix`` (discovery helper).

    Use this to inventory what a device actually calibrates -- e.g.
    ``probe_calibration_fields(stash, "characterization.model.QB1")`` or, for the
    CZ/coupler model (Milestone C0), ``"flux"`` / ``"cz"`` / the coupler name.
    """
    observations = getattr(stash, "observations", {})
    keys = observations.keys() if isinstance(observations, Mapping) else []
    return sorted(k for k in keys if k.startswith(prefix))


def resolve_dt(compiler: Any, settings: Any = None) -> float:
    """Sample time ``dt`` (s) from the drive AWG sample rate.

    ``compiler.channel_properties`` does not exist; the channel properties come
    from ``get_default_channel_properties(settings, chip_topology)``
    (``iqm/cpc/compiler/_utils/stages.py:409``).

    Pass an already-built ``settings`` tree to avoid a second
    ``compiler.get_settings()`` call -- that call reloads every observation from
    the server and is the dominant cost of extraction (calling it twice is what
    makes the one-shot extraction appear to hang).
    """
    from iqm.cpc.compiler._utils.stages import get_default_channel_properties

    if settings is None:
        settings = compiler.get_settings(circuits=[])
    props, comp_to_chan = get_default_channel_properties(settings, compiler.chip_topology)
    # Prefer a real qubit drive channel; fall back to any channel with a rate.
    for comp, funcs in comp_to_chan.items():
        chan = funcs.get("drive")
        if chan is not None and chan in props:
            return 1.0 / props[chan].sample_rate
    raise RuntimeError(
        "No drive channel with a sample_rate found in channel properties; "
        f"available components: {list(comp_to_chan)}"
    )


def _compiled_to_playlist(compiled: Any) -> Any:
    """Unwrap the ``Playlist`` from whatever ``compiler.compile()`` returns.

    On iqm-pulla 13.x ``compiler.compile()`` returns a ``RunDefinition`` whose
    playlist is at ``.sweep_definition.playlist``; a Playlist (identified by
    ``.channel_descriptions``) may also be passed through directly. Returns None
    if no playlist is reachable.
    """
    if compiled is None or hasattr(compiled, "channel_descriptions"):
        return compiled
    sweep = getattr(compiled, "sweep_definition", None)
    if sweep is not None and getattr(sweep, "playlist", None) is not None:
        return sweep.playlist
    return getattr(compiled, "playlist", None)


def _drive_pulse_area(playlist: Any, channel: str) -> float:
    """Sum over pulses on ``channel`` of ``|Σ_k w[k]|``, in sample-amplitude units.

    ``w`` is the scaled complex drive envelope (``scale·shape``) of each
    ``RealPulse`` / ``IQPulse`` on the channel (``wave.sample()`` gives the
    unit-peak shape, the ``scale``/``scale_i,scale_q`` fields the amplitude in
    [-1, 1]). Per-pulse magnitudes are summed so a constant carrier phase (or the
    small DRAG Q-quadrature, which integrates to ~0) does not cancel the area.
    Multiply the result by ``dt`` to get ``|∫ w dt|`` in [amplitude·s].
    """
    import numpy as np

    chan_desc = getattr(playlist, "channel_descriptions", {}).get(channel)
    if chan_desc is None:
        return 0.0
    table = chan_desc.instruction_table
    area = 0.0
    for segment in playlist.segments:
        for idx in segment.instructions.get(channel, []):
            op = table[idx].operation
            cls = op.__class__.__name__
            if cls == "IQPulse":
                w = op.wave_i.sample() * op.scale_i + 1j * op.wave_q.sample() * op.scale_q
                area += float(abs(np.sum(w)))
            elif cls == "RealPulse":
                area += float(abs(np.sum(op.wave.sample() * op.scale)))
    return area


def drive_strength_from_prx(compiler: Any, qubit: str, dt: float, *,
                            angle: float = math.pi, drive_channel: str | None = None,
                            settings: Any = None) -> float:
    """Drive strength ``r`` (Hz, max Rabi frequency at unit amplitude) from the PRX pulse.

    Fallback for when ``max_rabi_frequency`` is not calibrated (the case on this
    VLQ deployment). Compiles a single ``prx(angle, phase=0)`` on ``qubit`` with
    the live compiler, integrates the sampled drive envelope ``w(t)`` of the
    resulting *calibrated* pulse, and inverts the qiskit-dynamics drive relation
    ``theta = 2*pi*r*∫w(t)dt``::

        r = |angle| / (2*pi * |∫ w dt|),   |∫ w dt| = _drive_pulse_area(...) * dt

    ``angle`` defaults to pi (a π pulse). Pass ``drive_channel`` (from
    ``get_default_channel_properties``) and/or ``settings`` to avoid rebuilding
    the settings tree. Needs a live connection (compiles a circuit).
    """
    import numpy as np
    from iqm.pulse import Circuit, CircuitOperation

    if drive_channel is None:
        from iqm.cpc.compiler._utils.stages import get_default_channel_properties
        if settings is None:
            settings = compiler.get_settings(circuits=[])
        _, comp_to_chan = get_default_channel_properties(settings, compiler.chip_topology)
        drive_channel = comp_to_chan.get(qubit, {}).get("drive")
    if not drive_channel:
        raise RuntimeError(f"{qubit}: no drive channel found in the channel mapping")

    circ = Circuit(
        name="prx_r_cal",
        instructions=(CircuitOperation(name="prx", locus=(qubit,),
                                       args={"angle": float(angle), "phase": 0.0}),),
    )
    compiled, _ctx = compiler.compile([circ])
    playlist = _compiled_to_playlist(compiled)
    if playlist is None:
        raise RuntimeError(
            f"{qubit}: compiler.compile() returned {type(compiled).__name__} with no "
            f"reachable Playlist (expected .sweep_definition.playlist)."
        )

    area = _drive_pulse_area(playlist, drive_channel) * dt          # [amplitude·s]
    if area <= 0.0:
        raise RuntimeError(
            f"{qubit}: no drive pulse found on channel '{drive_channel}' in the "
            f"compiled prx({angle:.4f}) playlist — cannot derive r."
        )
    return abs(angle) / (2.0 * np.pi * area)


def find_coupler(chip_topology: Any, comp_a: str, comp_b: str) -> str | None:
    """Name of the coupler joining two components, or None.

    Tries the topology's own accessor first, then the
    ``coupler_to_components`` map (``{coupler -> [comp_a, comp_b]}``).
    On VLQ each coupler joins a qubit to ``COMPR1``; two qubits are never
    directly coupled, so this returns None for a qubit-qubit pair.
    """
    getter = getattr(chip_topology, "get_coupler_for", None)
    if callable(getter):
        try:
            result = getter(comp_a, comp_b)
            name = result if isinstance(result, str) else getattr(result, "name", None)
            if name:
                return name
        except Exception:
            pass
    mapping = getattr(chip_topology, "coupler_to_components", {})
    target = {comp_a, comp_b}
    for coupler, comps in mapping.items():
        if target.issubset(set(comps)):
            return coupler
    return None


def read_model_param(settings: Any, component: str, *fields: str) -> tuple[float | None, str]:
    """Read an estimated Hamiltonian parameter from the settings tree.

    Per the IQM docs, ``settings.characterization.model.{component}`` holds
    "certain estimated qubit and coupler Hamiltonian parameters" (anharmonicity,
    coupler coupling, …). This is a *different* structure from the calibration
    observation stash and is populated even when a parameter was not promoted
    into the default calibration set. Returns ``(value, source_path)`` or
    ``(None, "")``.
    """
    if settings is None:
        return None, ""
    char = _node_get(settings, "characterization")
    model = _node_get(char, "model") if char is not None else None
    comp_node = _node_get(model, component) if model is not None else None
    if comp_node is None:
        return None, ""
    for f in fields:
        v = _unwrap(_node_get(comp_node, f))
        if v is not None:
            return v, f"settings.characterization.model.{component}.{f}"
    return None, ""


def model_component_fields(settings: Any, component: str) -> list[str]:
    """Public child field names under ``settings.characterization.model.{component}``.

    Discovery helper — shows which estimated Hamiltonian parameters the device
    actually exposes for a component (empty if the node is missing).
    """
    char = _node_get(settings, "characterization")
    model = _node_get(char, "model") if char is not None else None
    comp_node = _node_get(model, component) if model is not None else None
    if comp_node is None:
        return []
    names = getattr(comp_node, "children", None) or getattr(comp_node, "child_nodes", None)
    if isinstance(names, dict):
        names = list(names.keys())
    elif names is not None and not isinstance(names, (list, tuple)):
        names = [a for a in dir(comp_node) if not a.startswith("_")]
    return sorted(str(n) for n in (names or []))


def read_chad_design(p: Any) -> dict[str, dict]:
    """Per-component design-physics from the raw chip design record (CHAD).

    Returns ``{component_name -> design_dict}`` gathered from
    ``p._iqm_server_client.get_chip_design_records()[0]["content"]["components"]``.
    On VLQ each qubit's design block carries ``alpha_coupled`` (the anharmonicity),
    ``e_c`` / ``e_j``, ``f_ge_bare`` / ``f_ge_coupled``, ``g_rr``, ``chi_coupled``,
    etc. These are DESIGN-target values, *not* live measurements -- used only as a
    fallback source for the anharmonicity gate after the live settings tree and
    the observation stash both come up empty. Returns ``{}`` if the record cannot
    be fetched or parsed (the gate then blocks, as before).
    """
    try:
        client = getattr(p, "_iqm_server_client", None)
        records = client.get_chip_design_records() if client is not None else None
        components = records[0]["content"]["components"]
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for items in components.values():
        if not isinstance(items, list):
            continue
        for comp in items:
            if isinstance(comp, dict) and comp.get("name"):
                out[comp["name"]] = (comp.get("properties") or {}).get("design", {}) or {}
    return out


def resolve_anharmonicity(stash: Any, qubit: str, settings: Any = None,
                          chad_design: Mapping[str, dict] | None = None) -> tuple[float, str]:
    """Anharmonicity (Hz, negative) and its provenance, or raise.

    Priority (measured always beats design):
      1. measured ``anharmonicity`` -- ``settings.characterization.model``
         (documented Hamiltonian-parameter structure), then the calibration stash;
      2. measured ``charging_energy`` -> alpha ~= -E_C (settings, then stash);
      3. design ``alpha_coupled`` from the chip design record (CHAD);
      4. design ``e_c`` from the CHAD -> alpha ~= -E_C;
      else :class:`AnharmonicityUnavailable`. No generic default is substituted;
      design values are per-qubit and chip-specific (project policy).
    """
    # 1. measured anharmonicity.
    alpha, src = read_model_param(settings, qubit, "anharmonicity")
    if alpha is not None:
        return alpha, src
    base = f"characterization.model.{qubit}"
    alpha = cal_value(stash, f"{base}.anharmonicity")
    if alpha is not None:
        return alpha, "stash:anharmonicity"

    # 2. measured charging energy -> alpha ~= -E_C.
    ec, ec_src = read_model_param(settings, qubit, "charging_energy")
    if ec is None:
        ec, ec_src = cal_value(stash, f"{base}.charging_energy"), "stash:charging_energy"
    if ec is not None:
        return -abs(ec), f"{ec_src}(-E_C)"

    # 3-4. design fallback from the CHAD (per-qubit, chip-specific): prefer the
    # purpose-computed `alpha_coupled`, then -e_c.
    design = (chad_design or {}).get(qubit, {})
    a_coupled = _unwrap(design.get("alpha_coupled"))
    if a_coupled is not None:
        return a_coupled, "chad:alpha_coupled(design)"
    ec_design = _unwrap(design.get("e_c"))
    if ec_design is not None:
        return -abs(ec_design), "chad:-e_c(design)"

    raise AnharmonicityUnavailable(
        f"{qubit}: no measured 'anharmonicity'/'charging_energy' in "
        f"settings.characterization.model nor the calibration stash, and no design "
        f"'alpha_coupled'/'e_c' in the chip design record. Obtain a value (IQM "
        f"request, or 1<->2 EF spectroscopy via Pulla) before building the "
        f"Hamiltonian. model fields present: {model_component_fields(settings, qubit)}; "
        f"stash fields present: {probe_calibration_fields(stash, base)}; "
        f"design fields present: {sorted(design)}"
    )


# --------------------------------------------------------------------------- #
# Top-level extraction
# --------------------------------------------------------------------------- #

def extract_qubit_params(settings: Any, stash: Any, qubit: str,
                         chip_topology: Any = None,
                         resonator: str = "COMPR1",
                         chad_design: Mapping[str, dict] | None = None,
                         metrics: Mapping[str, float] | None = None) -> QubitParams:
    """Assemble one :class:`QubitParams` from the live settings tree + stash.

    ``chad_design`` (from :func:`read_chad_design`) supplies the design-value
    anharmonicity fallback when the live sources are empty. ``metrics`` (from
    :func:`fetch_quality_metrics` / :func:`load_quality_metrics`) is the only
    source that actually carries T1/T2 -- without it they come back NaN.
    """
    # controllers.<q>.drive.frequency is a leaf Setting carrying .value (Hz)
    drive_node = _node_get(_node_get(settings.controllers, qubit), "drive")
    nu = _unwrap(_node_get(drive_node, "frequency"))
    if nu is None:
        raise RuntimeError(f"{qubit}: drive frequency not found in settings tree")

    base = f"characterization.model.{qubit}"
    # Preferred drive-strength source. If absent, leave NaN -- the notebook's
    # Phase 5 derives r from the PRX pulse integral as a documented fallback.
    r = cal_value(stash, f"{base}.max_rabi_frequency")
    if r is None:
        r = metric_value(metrics, f"{base}.max_rabi_frequency")
    if r is None:
        r, _ = read_model_param(settings, qubit, "max_rabi_frequency")

    T1, T2, coherence_source = _resolve_coherence(stash, metrics, settings, qubit)

    alpha, alpha_source = resolve_anharmonicity(stash, qubit, settings=settings,
                                                chad_design=chad_design)

    coupler = ""
    if chip_topology is not None:
        coupler = find_coupler(chip_topology, qubit, resonator) or ""

    return QubitParams(
        name=qubit, nu=nu, alpha=alpha,
        r=r if r is not None else float("nan"),
        T1=T1 if T1 is not None else float("nan"),
        T2=T2 if T2 is not None else float("nan"),
        alpha_source=alpha_source, coherence_source=coherence_source, coupler=coupler,
    )


def resolve_resonator_frequency(settings: Any, qubits: Any,
                                resonator: str = "COMPR1",
                                *, tol: float = 1e6) -> tuple[float | None, str]:
    """Recover the computational-resonator frequency from the MOVE gate calibration.

    ``characterization.model.<resonator>.frequency`` is unpopulated on the VLQ
    default calibration set, but every ``move`` gate stores the qubit-resonator
    detuning it needs for phase bookkeeping: ``iqm/pulse/gates/move.py`` declares
    it as ``Parameter("", "Qubit - resonator detuning", "Hz")`` and applies
    ``detuning_phase = 2*pi * detuning * duration`` to the moved state. Hence::

        nu_r = nu_q - move_detuning(q, resonator)

    Every qubit couples to the *same* resonator, so the estimate must agree across
    qubits -- the spread is the built-in validity check (verified live 2026-07-23:
    QB1-QB4 all gave 4.0685 GHz from detunings spanning 96-146 MHz).

    Returns ``(nu_r, source)``, or ``(None, reason)`` when it cannot be resolved.
    """
    est: dict[str, float] = {}
    for q in qubits:
        try:
            node = settings.get_gate_node_for_locus("move", (q, resonator))
        except Exception:
            continue
        detuning = _unwrap(_node_get(node, "detuning"))
        drive_node = _node_get(_node_get(settings.controllers, q), "drive")
        nu_q = _unwrap(_node_get(drive_node, "frequency"))
        if detuning is None or nu_q is None:
            continue
        est[q] = nu_q - detuning
    if not est:
        return None, f"move.detuning unavailable for {resonator}"
    vals = list(est.values())
    spread = max(vals) - min(vals)
    if len(vals) > 1 and spread > tol:
        return None, (f"move.detuning inconsistent across qubits: nu_r spread "
                      f"{spread / 1e6:.3f} MHz exceeds tol {tol / 1e6:.3f} MHz")
    return (sum(vals) / len(vals),
            f"move.detuning(gate-cal, n={len(vals)}, spread={spread:.1f} Hz)")


def read_gate_calibration(settings: Any, gate: str, locus: Any,
                          implementation: str | None = None) -> dict[str, Any]:
    """Flat ``{leaf-name: value}`` of the calibrated gate node for ``locus``.

    Reads ``settings.gates.<gate>.<impl>.<locus>`` (see
    :meth:`SettingNode.get_gate_node_for_locus`). Keys are relative to the locus
    node, e.g. ``duration``, ``coupler.full_width``, ``qubit.amplitude``, ``rz.QB1``.
    """
    locus = tuple(locus)
    node = settings.get_gate_node_for_locus(gate, locus, implementation)
    prefix = "__".join(locus) + "."
    out: dict[str, Any] = {}
    for setting in node.all_settings:
        path = getattr(setting, "path", "") or ""
        name = getattr(getattr(setting, "parameter", None), "name", "") or ""
        key = path.split(prefix, 1)[-1] if prefix in path else (path or name)
        out[key] = setting.value
    return out


def cosine_rise_fall_area(full_width: float, rise_time: float) -> float:
    """Time integral (s) of a unit-amplitude ``CosineRiseFall`` envelope.

    The IQM waveform (``iqm/models/playlist/waveforms.py``) is a flat top of width
    ``p = full_width - 2*rise_time`` with sinusoidal edges
    ``0.5*(1 + cos(pi*u/rise_time))``. Each edge integrates to ``rise_time/2``, so::

        area = p + 2*(rise_time/2) = full_width - rise_time
    """
    fw, rt = abs(full_width), abs(rise_time)
    if fw < 2 * rt:
        raise ValueError(
            f"full_width {fw} < 2*rise_time {2 * rt}: the envelope never reaches its top"
        )
    return fw - rt


def swap_coupling_from_move(full_width: float, rise_time: float) -> float:
    """Exchange coupling ``g`` (Hz) implied by a calibrated MOVE (qubit<->resonator swap).

    With the qubit flux-tuned into resonance with the resonator, the exchange term
    ``H = 2*pi*g*w(t)*(a_q^dag a_r + h.c.)`` acts inside the single-excitation
    subspace ``{|1,0>, |0,1>}`` as ``2*pi*g*w(t)*sigma_x``. The ladder matrix element
    is exactly 1 there, so this is independent of the anharmonicity and of the
    truncation dim. A complete population transfer is a pi/2 rotation::

        2*pi*g*Area = pi/2   =>   g = 1 / (4*Area)

    with ``Area = cosine_rise_fall_area(full_width, rise_time)``. For VLQ
    QB1<->COMPR1 MOVE (full_width 60 ns, rise_time 20 ns) => Area 40 ns => g = 6.25 MHz.

    NOTE this is a *fitted* effective coupling, anchored to the device's own
    calibrated gate -- not a measured chip parameter. The flux->frequency map needed
    to derive ``g`` from first principles is absent from this calibration set.
    """
    return 1.0 / (4.0 * cosine_rise_fall_area(full_width, rise_time))


def extract_resonator_params(settings: Any, stash: Any,
                             resonator: str = "COMPR1",
                             qubits: Any = None,
                             metrics: Mapping[str, float] | None = None) -> ResonatorParams | None:
    """Assemble the computational-resonator subsystem params, or None if absent.

    Frequency resolution order: measured ``characterization.model`` value ->
    ``controllers`` tree -> inferred from the MOVE gate calibration
    (:func:`resolve_resonator_frequency`, needs ``qubits``).
    """
    base = f"characterization.model.{resonator}"
    nu_source = f"characterization.model.{resonator}.frequency"
    nu = cal_value(stash, f"{base}.frequency") or metric_value(metrics, f"{base}.frequency")
    if nu is None:
        # some tiers expose the resonator frequency in the controllers tree
        node = _node_get(settings.controllers, resonator)
        nu = _unwrap(_node_get(node, "frequency")) if node is not None else None
        nu_source = f"controllers.{resonator}.frequency"
    if nu is None and qubits:
        nu, nu_source = resolve_resonator_frequency(settings, qubits, resonator)
    if nu is None:
        return None
    T1, T2, coherence_source = _resolve_coherence(stash, metrics, settings, resonator)
    return ResonatorParams(
        name=resonator, nu=nu, nu_source=nu_source,
        coherence_source=coherence_source,
        T1=T1 if T1 is not None else float("nan"),
        T2=T2 if T2 is not None else float("nan"),
    )


def extract_device_params(p: Any, compiler: Any, qubits: list[str], *,
                          dim: int = 3, resonator: str = "COMPR1",
                          calibration_set_id: str = "default",
                          metrics: Mapping[str, float] | None = None) -> DeviceParams:
    """Extract all DynamicsBackend inputs for ``qubits`` from a live connection.

    Raises :class:`AnharmonicityUnavailable` if any chosen qubit lacks a real
    anharmonicity -- everything downstream is gated on that, by design.

    ``metrics`` overrides the automatic quality-metric-set fetch (pass
    :func:`load_quality_metrics` output to pin T1/T2 to an exported set). A
    failed fetch is non-fatal: T1/T2 simply come back NaN, which only affects
    the open-system (Lindblad) model.
    """
    settings = compiler.get_settings(circuits=[])
    stash = p.get_calibration_stash(calibration_set_id)
    chad_design = read_chad_design(p)
    if metrics is None:
        try:
            metrics = fetch_quality_metrics(p, calibration_set_id)
        except Exception:                                    # noqa: BLE001 - optional source
            metrics = None

    qparams = [
        extract_qubit_params(settings, stash, q,
                             chip_topology=compiler.chip_topology, resonator=resonator,
                             chad_design=chad_design, metrics=metrics)
        for q in qubits
    ]
    chip_label = getattr(compiler, "dut_label", "") or ""
    return DeviceParams(
        qubits=qparams,
        dt=resolve_dt(compiler, settings=settings),
        dim=dim,
        resonator=extract_resonator_params(settings, stash, resonator, qubits=qubits,
                                           metrics=metrics),
        calibration_set_id=str(calibration_set_id),
        chip_label=chip_label,
    )


# --------------------------------------------------------------------------- #
# Snapshot I/O (offline)
# --------------------------------------------------------------------------- #

def to_dict(params: DeviceParams) -> dict:
    """JSON-serialisable dict for ``DeviceParams`` (NaN -> null)."""
    def clean(x):
        if isinstance(x, float) and math.isnan(x):
            return None
        return x
    d = asdict(params)
    for q in d["qubits"]:
        for k, v in list(q.items()):
            q[k] = clean(v)
    if d.get("resonator"):
        for k, v in list(d["resonator"].items()):
            d["resonator"][k] = clean(v)
    return d


def save_snapshot(params: DeviceParams, path: str | Path) -> Path:
    """Freeze the extracted operating point to JSON for reproducibility."""
    path = Path(path)
    path.write_text(json.dumps(to_dict(params), indent=2))
    return path


def load_snapshot(path: str | Path) -> DeviceParams:
    """Reconstruct :class:`DeviceParams` from a saved snapshot (no live connection)."""
    d = json.loads(Path(path).read_text())

    def nan_if_none(v):
        return float("nan") if v is None else v

    qubits = []
    for q in d["qubits"]:
        qubits.append(QubitParams(
            name=q["name"], nu=q["nu"], alpha=q["alpha"], r=q["r"],
            T1=nan_if_none(q.get("T1")), T2=nan_if_none(q.get("T2")),
            alpha_source=q.get("alpha_source", ""), coupler=q.get("coupler", ""),
            coherence_source=q.get("coherence_source", ""),
        ))
    res = None
    if d.get("resonator"):
        rr = d["resonator"]
        res = ResonatorParams(
            name=rr["name"], nu=rr["nu"],
            T1=nan_if_none(rr.get("T1")), T2=nan_if_none(rr.get("T2")),
            nu_source=rr.get("nu_source", ""),
            coherence_source=rr.get("coherence_source", ""),
        )
    return DeviceParams(
        qubits=qubits, dt=d["dt"], dim=d.get("dim", 3), resonator=res,
        calibration_set_id=d.get("calibration_set_id", ""),
        chip_label=d.get("chip_label", ""),
    )
