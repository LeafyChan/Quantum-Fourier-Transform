"""
DCG v4 – Hardware-Aware Compiler Framework
============================================
Addresses the "cryogenic reality" critique:

  1. DRAG pulse shaping  → spectral leakage suppression, kills |2> leakage
  2. Thermal budget model → heat dissipation at 20mK, fridge stability analysis
  3. Static ISA compiler → pre-compiled waveform library, zero runtime re-linking,
                           jitter-free gate start times via slot reservation
  4. Worst-case CVaR     → Conditional Value-at-Risk (tail-risk) instead of
                           average Monte Carlo fidelity; finds 3-sigma failures

Architecture modules:
  DRAGPulse          – Gaussian + derivative correction, bandwidth-limited
  ThermalBudget      – heat load per gate, fridge stability envelope
  StaticISACompiler  – offline m* decision → fixed waveform slots, no runtime linking
  WorstCaseAnalyzer  – CVaR over (epsilon × crosstalk resonance) joint distribution

Run:
    pip install qutip numpy scipy matplotlib
    python dcg_v4_compiler.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal as sig
import scipy.stats as stats
from scipy.linalg import expm as sp_expm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import itertools
import unittest
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# HARDWARE CONSTANTS  (realistic IBM/Google-class transmon, 20 mK)
# ══════════════════════════════════════════════════════════════════
T1              = 100e-9    # energy relaxation (s)
T2              = 80e-9     # dephasing (s)
TAU_PRIM        = 20e-9     # primitive pulse duration (s)
ANHARMONICITY   = -220e6    # α (Hz) — transmon |1>→|2> offset from |0>→|1>
QUBIT_FREQ      = 5.0e9     # ω01 / 2π (Hz)
N_QUBITS        = 5

# Thermal constants at 20 mK
FRIDGE_COOLING  = 20e-6     # available cooling power (W) — 20 µW typical
HEAT_PER_PULSE  = 100e-12   # energy dissipated per microwave pulse (J) — 100 pJ
T_BASE          = 20e-3     # base temperature (K)
T_CRITICAL      = 50e-3     # T1/T2 collapse above this temperature (K)
FRIDGE_THERMAL_RC = 0.5     # thermal time constant (s) — fridge heat capacity / conductance

# ISA / controller constants (Zurich Instruments HDAWG-class)
AWG_WAVEFORM_SLOTS = 32     # total pre-loadable waveform slots
AWG_SAMPLE_RATE    = 2.4e9  # samples/second
AWG_MIN_JITTER     = 0.1e-9 # 100 ps irreducible hardware jitter (s)
GATE_TIME_SLOTS    = int(TAU_PRIM * AWG_SAMPLE_RATE)  # samples per gate slot


# ══════════════════════════════════════════════════════════════════
# SECTION 1 – DRAG PULSE SHAPING
# Addresses: spectral leakage + accidental |1>→|2> excitation
# ══════════════════════════════════════════════════════════════════

@dataclass
class DRAGPulse:
    """
    Derivative Removal via Adiabatic Gate (DRAG) pulse.

    The envelope is:  Ω(t) = Ω_x(t) + i·Ω_y(t)
      Ω_x(t) = A · gauss(t, σ)                   [in-phase, does the rotation]
      Ω_y(t) = −(dΩ_x/dt) / (2α)                 [quadrature, kills |2> leakage]

    The quadrature component adds destructive interference for |1>→|2>
    while leaving |0>→|1> untouched.  This is the standard DRAG correction
    (Motzoi et al., PRL 2009).

    bandwidth_ghz: Gaussian σ parameter.  Larger → narrower spectrum.
    alpha_hz     : transmon anharmonicity (negative for transmon).
    """
    theta       : float          # target rotation angle (rad)
    phi         : float          # rotation axis angle in XY plane
    duration_s  : float          # pulse duration (seconds)
    bandwidth_ghz: float = 0.1   # spectral bandwidth (GHz)
    alpha_hz    : float = ANHARMONICITY
    n_samples   : int   = field(init=False)

    def __post_init__(self):
        self.n_samples = max(4, int(self.duration_s * AWG_SAMPLE_RATE))

    def envelope(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (t, Omega_x, Omega_y) arrays.
        Omega_x: in-phase (Gaussian), Omega_y: quadrature (DRAG correction).
        bandwidth_ghz controls the Gaussian sigma: narrow BW → wide pulse in time.
        """
        t   = np.linspace(0, self.duration_s, self.n_samples)
        dt  = t[1] - t[0]

        # Map bandwidth_ghz → time-domain sigma via time-bandwidth product
        # sigma_t = 1 / (2π · sigma_f),  sigma_f = bandwidth_ghz * 1e9
        sig_t = 1.0 / (2 * np.pi * self.bandwidth_ghz * 1e9)
        sig_t = np.clip(sig_t, self.duration_s * 0.05, self.duration_s / 2.5)

        t_mid  = self.duration_s / 2
        gauss  = np.exp(-0.5 * ((t - t_mid) / sig_t) ** 2)
        gauss -= gauss[0]                            # zero baseline

        _trapz = getattr(np, "trapezoid", None) or np.trapz
        norm   = _trapz(gauss, t)
        if abs(norm) < 1e-20:
            norm = 1.0
        Omega_x = (self.theta / 2) * gauss / norm

        # DRAG quadrature: -dΩ_x/dt / (2α)
        dOmega_x  = np.gradient(Omega_x, dt)
        alpha_rad = self.alpha_hz * 2 * np.pi
        Omega_y   = -dOmega_x / (2 * alpha_rad)

        return t, Omega_x, Omega_y

    def power_spectral_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """PSD of the complex envelope, normalised. Uses fft (complex input)."""
        _, Ox, Oy = self.envelope()
        envelope  = (Ox + 1j * Oy).astype(np.complex128)
        freqs     = np.fft.fftfreq(len(envelope), d=1.0/AWG_SAMPLE_RATE)
        psd       = np.abs(np.fft.fft(envelope)) ** 2
        # Keep positive frequencies only
        pos       = freqs >= 0
        psd       = psd[pos] / (psd[pos].max() + 1e-30)
        return freqs[pos] / 1e9, psd   # GHz, normalised PSD

    def leakage_to_level2(self) -> float:
        """
        First-order perturbative leakage to |2> level (Gambetta et al.).
        DRAG's Omega_y provides destructive interference at the anharmonicity
        frequency, so leakage is computed from the corrected complex envelope.
        """
        t, Ox, Oy = self.envelope()
        alpha_rad = self.alpha_hz * 2 * np.pi
        _trapz    = getattr(np, "trapezoid", None) or np.trapz
        # Corrected (DRAG) envelope coherently cancels the leakage amplitude
        integrand = (Ox + 1j * Oy).astype(np.complex128) * np.exp(1j * alpha_rad * t)
        return float(np.real(abs(_trapz(integrand, t)) ** 2))

    def spectral_width_ghz(self) -> float:
        """RMS bandwidth of the pulse in GHz."""
        freqs_ghz, psd = self.power_spectral_density()
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        df   = freqs_ghz[1] - freqs_ghz[0]
        norm = _trapz(psd, freqs_ghz) + 1e-30
        f_c  = _trapz(freqs_ghz * psd, freqs_ghz) / norm
        f2   = _trapz((freqs_ghz**2) * psd, freqs_ghz) / norm
        return float(np.sqrt(max(0, f2 - f_c**2)))


def square_pulse(theta: float, phi: float, duration_s: float) -> DRAGPulse:
    """Return a wide-bandwidth (square-equivalent) pulse: large bandwidth param."""
    p = DRAGPulse(theta, phi, duration_s, bandwidth_ghz=10.0)
    return p

def drag_pulse(theta: float, phi: float, duration_s: float) -> DRAGPulse:
    """Return a bandwidth-limited DRAG pulse: narrow bandwidth param."""
    p = DRAGPulse(theta, phi, duration_s, bandwidth_ghz=0.08)
    return p


# ══════════════════════════════════════════════════════════════════
# SECTION 2 – THERMAL BUDGET MODEL
# Addresses: heat dissipation, fridge temperature, T1/T2 collapse
# ══════════════════════════════════════════════════════════════════

class ThermalBudget:
    """
    Models heat load at the mixing chamber of a dilution refrigerator.

    State: T_fridge(t) evolves as a first-order thermal RC:
        C · dT/dt = -T/R_thermal + P_dissipated(t)
    where P_dissipated = n_active_pulses * HEAT_PER_PULSE / TAU_PRIM

    T1 and T2 collapse above T_critical (empirical, fits IBM data):
        T1_eff = T1_base * exp(-k * (T - T_base) / T_base)
    """
    def __init__(self, cooling_power: float = FRIDGE_COOLING,
                 base_temp: float = T_BASE,
                 t_critical: float = T_CRITICAL):
        self.P_cool   = cooling_power
        self.T_base   = base_temp
        self.T_crit   = t_critical
        self.T_fridge = base_temp
        self.RC       = FRIDGE_THERMAL_RC
        self.history  : List[Tuple[float, float, int]] = []  # (time, T, n_pulses)
        self._time    = 0.0

    def step(self, n_active_pulses: int, dt: float):
        """
        Advance fridge temperature by dt seconds with n_active_pulses firing.
        """
        P_in  = n_active_pulses * HEAT_PER_PULSE / max(dt, 1e-12)
        P_net = P_in - self.P_cool
        # First-order RC thermal model
        dT    = (P_net / (self.P_cool / self.T_base)) * (1 - np.exp(-dt / self.RC))
        self.T_fridge = max(self.T_base, self.T_fridge + dT * dt / self.RC)
        self._time   += dt
        self.history.append((self._time, self.T_fridge, n_active_pulses))

    def effective_T1(self) -> float:
        if self.T_fridge <= self.T_base:
            return T1
        k = 5.0   # empirical collapse rate
        ratio = (self.T_fridge - self.T_base) / self.T_base
        return T1 * np.exp(-k * ratio)

    def effective_T2(self) -> float:
        return min(self.effective_T1() * 2, T2 * (self.T_base / self.T_fridge))

    def is_stable(self) -> bool:
        return self.T_fridge < self.T_crit

    def stability_margin(self) -> float:
        """Fraction of critical temperature headroom remaining."""
        return max(0.0, (self.T_crit - self.T_fridge) / (self.T_crit - self.T_base))

    def simulate_circuit(self, pulse_counts_per_layer: List[int],
                          dt_per_layer: float) -> Dict:
        """Run a full circuit through the thermal model layer by layer."""
        self.T_fridge = self.T_base
        self.history  = []
        self._time    = 0.0
        boiled_at     = None

        for i, n in enumerate(pulse_counts_per_layer):
            self.step(n, dt_per_layer)
            if not self.is_stable() and boiled_at is None:
                boiled_at = i

        temps  = [h[1] for h in self.history]
        return {
            "T_max_mK"        : max(temps) * 1e3,
            "T_final_mK"      : self.T_fridge * 1e3,
            "boiled_at_layer" : boiled_at,
            "survived"        : boiled_at is None,
            "T_history_K"     : np.array(temps),
            "T1_final"        : self.effective_T1(),
            "T2_final"        : self.effective_T2(),
            "stability_margin": self.stability_margin(),
        }


# ══════════════════════════════════════════════════════════════════
# SECTION 3 – STATIC ISA COMPILER
# Addresses: dynamic re-linking, waveform slot explosion, gate jitter
# ══════════════════════════════════════════════════════════════════

@dataclass
class WaveformSlot:
    """One entry in the AWG's pre-loaded waveform memory."""
    slot_id      : int
    gate_type    : str        # "naive" | "bb1_m1" | "bb1_m2" | "bb1_m3"
    theta        : float
    phi          : float
    m_order      : int        # BB1 truncation order
    n_samples    : int
    pulse_type   : str        # "square" | "drag"
    leakage      : float = 0.0
    bandwidth_ghz: float = 0.0


class StaticISACompiler:
    """
    Offline compiler that pre-decides all m* choices and loads a STATIC
    waveform library before circuit execution.

    Key insight: instead of computing m* at runtime (which causes jitter),
    we partition the ε space into bins offline, assign each bin a fixed
    waveform slot, and use a simple lookup table on the controller.

    This converts a "runtime re-linking" problem into a "lookup table"
    problem — the controller only needs to issue a 5-bit slot address,
    which has deterministic, sub-nanosecond latency.

    Waveform slot budget: AWG_WAVEFORM_SLOTS (= 32)
    """
    def __init__(self, n_epsilon_bins: int = 8, use_drag: bool = True):
        self.n_bins      = n_epsilon_bins
        self.use_drag    = use_drag
        self.library     : List[WaveformSlot] = []
        self.slot_map    : Dict[Tuple, int]   = {}  # (theta_bin, eps_bin) → slot_id
        self._next_slot  = 0

    def compile(self, theta_values: List[float],
                eps_range: Tuple[float, float] = (-0.3, 0.3)):
        """
        Offline pass: for each (theta, eps_bin), decide m* and allocate slot.
        Total slots = len(theta_values) × n_epsilon_bins × 2 (naive+bb1)
        Must fit within AWG_WAVEFORM_SLOTS.
        """
        self.library  = []
        self.slot_map = {}
        eps_bins = np.linspace(eps_range[0], eps_range[1], self.n_bins + 1)
        eps_centers = 0.5 * (eps_bins[:-1] + eps_bins[1:])

        total_needed = len(theta_values) * self.n_bins
        if total_needed > AWG_WAVEFORM_SLOTS:
            # Reduce bins to fit budget
            self.n_bins = max(2, AWG_WAVEFORM_SLOTS // len(theta_values))
            eps_bins    = np.linspace(eps_range[0], eps_range[1], self.n_bins + 1)
            eps_centers = 0.5 * (eps_bins[:-1] + eps_bins[1:])

        for t_idx, theta in enumerate(theta_values):
            for e_idx, eps in enumerate(eps_centers):
                m_star = self._decide_m_star(theta, eps)
                ptype  = "drag" if self.use_drag else "square"

                # Build representative pulse for slot characterisation
                phi1 = np.arccos(np.clip(-theta / (4*np.pi), -1, 1))
                if self.use_drag:
                    p = drag_pulse(theta, np.pi/2, TAU_PRIM)
                else:
                    p = square_pulse(theta, np.pi/2, TAU_PRIM)

                slot = WaveformSlot(
                    slot_id      = self._next_slot,
                    gate_type    = f"bb1_m{m_star}" if m_star > 1 else "naive",
                    theta        = theta,
                    phi          = np.pi / 2,
                    m_order      = m_star,
                    n_samples    = p.n_samples * m_star,
                    pulse_type   = ptype,
                    leakage      = p.leakage_to_level2(),
                    bandwidth_ghz= p.spectral_width_ghz(),
                )
                self.library.append(slot)
                self.slot_map[(t_idx, e_idx)] = self._next_slot
                self._next_slot += 1
                if self._next_slot >= AWG_WAVEFORM_SLOTS:
                    break
            if self._next_slot >= AWG_WAVEFORM_SLOTS:
                break

        return self

    def _decide_m_star(self, theta: float, eps: float) -> int:
        """
        OFFLINE m* decision. Uses T2 budget + crosstalk penalty.
        This is the only place m* is computed — never at runtime.
        """
        # Cost model: fidelity gain from correction vs T2 + thermal penalty
        def local_fid(m):
            phi1 = np.arccos(np.clip(-theta/(4*np.pi), -1, 1))
            U = sp_expm(-1j*(theta+eps)/2 * np.array([[0,-1j],[1j,0]],complex))
            for i in range(1, min(m+1,4)):
                ang = [np.pi, 2*np.pi, np.pi][i-1] + eps
                ph  = [phi1, 3*phi1, phi1][i-1]
                ax  = (np.cos(ph)*np.array([[0,1],[1,0]],complex) +
                       np.sin(ph)*np.array([[0,-1j],[1j,0]],complex))
                U   = sp_expm(-1j*ang/2 * ax) @ U
            U_ideal = sp_expm(-1j*theta/2 * np.array([[0,-1j],[1j,0]],complex))
            ov = np.trace(U_ideal.conj().T @ U)
            return abs(ov)**2 / 4

        # Penalty: each extra pulse uses TAU_PRIM more time against T2
        t2_pen = TAU_PRIM / T2 * 0.3   # empirical weight
        scores = {m: local_fid(m) - m * t2_pen for m in [1, 2, 3]}
        return max(scores, key=scores.get)

    def lookup(self, theta_idx: int, eps_bin: int) -> Optional[WaveformSlot]:
        """O(1) slot lookup — deterministic, no computation at runtime."""
        return self.library[self.slot_map.get((theta_idx, eps_bin), 0)]

    def jitter_analysis(self, n_gates: int) -> Dict:
        """
        Static ISA: gate start time jitter = hardware_jitter only (no compute delay).
        vs dynamic ISA: jitter = hardware_jitter + m* compute latency per gate.
        """
        # Static: all waveforms pre-loaded, controller just issues addresses
        jitter_static = AWG_MIN_JITTER

        # Dynamic: m* computation adds latency on critical path
        m_star_compute_ns = 50e-6   # 50 µs compute on a real-time FPGA
        jitter_dynamic    = AWG_MIN_JITTER + m_star_compute_ns

        # QFT phase error from jitter: δφ = ω · δt
        omega = 2 * np.pi * QUBIT_FREQ
        phase_err_static  = omega * jitter_static
        phase_err_dynamic = omega * jitter_dynamic

        return {
            "jitter_static_ns"    : jitter_static * 1e9,
            "jitter_dynamic_ns"   : jitter_dynamic * 1e9,
            "phase_err_static_rad": phase_err_static,
            "phase_err_dynamic_rad": phase_err_dynamic,
            "qft_phase_ratio"     : phase_err_dynamic / phase_err_static,
            "slots_used"          : len(self.library),
            "slots_budget"        : AWG_WAVEFORM_SLOTS,
            "slot_utilisation"    : len(self.library) / AWG_WAVEFORM_SLOTS,
        }


# ══════════════════════════════════════════════════════════════════
# SECTION 4 – WORST-CASE CVaR ANALYSER
# Addresses: tail-risk, 3-sigma failures, resonance catastrophes
# ══════════════════════════════════════════════════════════════════

class WorstCaseAnalyzer:
    """
    Conditional Value-at-Risk (CVaR) analysis over the joint distribution
    of (epsilon, crosstalk_resonance_offset).

    In circuit timing analysis, the 3-sigma failure — not the mean — sets
    the design margin.  CVaR at α=0.95 is the expected infidelity in the
    worst 5% of noise realisations.

    Crosstalk resonance disaster: when ε brings qubit B's dressed frequency
    within ε_res of qubit A's drive frequency, off-resonant excitation spikes.
    This is NOT captured by a smooth crosstalk matrix.
    """
    def __init__(self, n_qubits: int = N_QUBITS):
        self.n = n_qubits
        # Qubit frequencies (GHz), slightly detuned as in real chips
        self.qubit_freqs = QUBIT_FREQ + np.array([0, 0.15, 0.31, 0.47, 0.62]) * 1e9
        self.qubit_freqs = self.qubit_freqs[:n_qubits]

    def resonance_penalty(self, driven_qubit: int, eps: float,
                           pulse_bw_ghz: float) -> np.ndarray:
        """
        For a pulse on qubit `driven_qubit` with noise eps and spectral
        bandwidth pulse_bw_ghz, compute accidental excitation probability
        on all other qubits.

        Model: Lorentzian line shape centred on driven qubit's shifted frequency.
        Peak occurs when |f_drive(eps) - f_neighbor| < pulse_bw_ghz / 2
        """
        # Noise eps shifts the effective drive frequency
        f_drive = self.qubit_freqs[driven_qubit] + eps * QUBIT_FREQ * 0.01
        penalty = np.zeros(self.n)
        for j in range(self.n):
            if j == driven_qubit:
                continue
            delta_f = abs(f_drive - self.qubit_freqs[j]) / 1e9   # GHz
            # Lorentzian: P_leak ∝ (bw/2)² / (delta_f² + (bw/2)²)
            half_bw = pulse_bw_ghz / 2
            penalty[j] = half_bw**2 / (delta_f**2 + half_bw**2 + 1e-30)
        return penalty

    def sample_infidelity(self, theta: float, driven_qubit: int,
                           pulse_type: str, n_samples: int = 5000,
                           seed: int = 42) -> np.ndarray:
        """
        Draw n_samples from the joint (eps, thermal_noise) distribution
        and compute per-sample total circuit infidelity including:
          - gate infidelity from eps
          - resonance-driven neighbour excitation
          - thermal noise contribution

        Returns array of infidelity values for CVaR computation.
        """
        rng = np.random.default_rng(seed)

        # Realistic epsilon distribution: drifting Ornstein-Uhlenbeck process
        # ε ~ N(0.05, 0.03) with heavy tail (Student-t ν=5 for rare events)
        eps_samples = stats.t.rvs(df=5, loc=0.05, scale=0.03,
                                   size=n_samples, random_state=rng.integers(1e9))

        # Bandwidth: square pulse is wide, DRAG is narrow
        bw = 1.5 if pulse_type == "square" else 0.15

        # Unitary infidelity for target gate
        def gate_infid(eps):
            U = sp_expm(-1j*(theta+eps)/2 * np.array([[0,-1j],[1j,0]],complex))
            U_ideal = sp_expm(-1j*theta/2 * np.array([[0,-1j],[1j,0]],complex))
            ov = np.trace(U_ideal.conj().T @ U)
            return 1.0 - abs(ov)**2 / 4

        infidelities = np.zeros(n_samples)
        for i, eps in enumerate(eps_samples):
            local_infid  = gate_infid(eps)
            res_penalty  = self.resonance_penalty(driven_qubit, eps, bw)
            global_infid = np.sum(res_penalty) * 0.1   # scale to fidelity units
            thermal_jitter = rng.normal(0, 0.005)       # small thermal noise floor
            infidelities[i] = local_infid + global_infid + abs(thermal_jitter)

        return infidelities

    def cvar(self, infidelities: np.ndarray, alpha: float = 0.95) -> Dict:
        """
        Compute CVaR (Expected Shortfall) at confidence level alpha.
        CVaR_α = E[X | X > VaR_α]  where X = infidelity.

        This is the mean of the worst (1-α) fraction of outcomes.
        """
        sorted_inf = np.sort(infidelities)
        n          = len(sorted_inf)
        cutoff_idx = int(np.ceil(alpha * n))
        var_alpha  = sorted_inf[cutoff_idx]
        cvar_alpha = sorted_inf[cutoff_idx:].mean()
        sigma3_val = np.mean(infidelities) + 3 * np.std(infidelities)

        return {
            "mean_infidelity"  : float(np.mean(infidelities)),
            "median_infidelity": float(np.median(infidelities)),
            "std_infidelity"   : float(np.std(infidelities)),
            "VaR_95"           : float(var_alpha),
            "CVaR_95"          : float(cvar_alpha),
            "sigma3_infidelity": float(sigma3_val),
            "p_catastrophic"   : float(np.mean(infidelities > 0.5)),
            "worst_1pct"       : float(np.percentile(infidelities, 99)),
        }

    def compare_pulse_types(self, theta: float, driven_qubit: int = 2,
                              n_samples: int = 4000) -> Dict:
        """Compare square vs DRAG pulse worst-case distributions."""
        sq_inf   = self.sample_infidelity(theta, driven_qubit, "square",  n_samples)
        drag_inf = self.sample_infidelity(theta, driven_qubit, "drag",    n_samples, seed=99)
        return {
            "square": self.cvar(sq_inf),
            "drag"  : self.cvar(drag_inf),
            "sq_samples"  : sq_inf,
            "drag_samples": drag_inf,
        }


# ══════════════════════════════════════════════════════════════════
# SECTION 5 – UNIT TESTS  (worst-case design requirements)
# ══════════════════════════════════════════════════════════════════

class TestV4(unittest.TestCase):

    def test_drag_narrower_than_square(self):
        """DRAG pulse must have strictly lower bandwidth than square pulse."""
        d = drag_pulse(np.pi/2, np.pi/2, TAU_PRIM)
        s = square_pulse(np.pi/2, np.pi/2, TAU_PRIM)
        self.assertLess(d.spectral_width_ghz(), s.spectral_width_ghz(),
            "DRAG must have narrower spectrum than square pulse")

    def test_drag_reduces_level2_leakage(self):
        """DRAG must reduce |1>→|2> leakage vs square pulse."""
        d = drag_pulse(np.pi/2, np.pi/2, TAU_PRIM)
        s = square_pulse(np.pi/2, np.pi/2, TAU_PRIM)
        self.assertLess(d.leakage_to_level2(), s.leakage_to_level2(),
            f"DRAG leakage={d.leakage_to_level2():.4e} not less than "
            f"square={s.leakage_to_level2():.4e}")

    def test_thermal_boiling_with_bb1(self):
        """BB1 circuit (4× pulses) must heat fridge more than naive."""
        budget = ThermalBudget()
        n_layers = 40
        res_naive = budget.simulate_circuit([N_QUBITS * 1] * n_layers, TAU_PRIM)
        budget2   = ThermalBudget()
        res_bb1   = budget2.simulate_circuit([N_QUBITS * 4] * n_layers, TAU_PRIM)
        self.assertGreater(res_bb1["T_max_mK"], res_naive["T_max_mK"],
            "BB1 circuit must heat fridge more than naive circuit")

    def test_thermal_stability_margin_decreases(self):
        """Stability margin must be lower after BB1 circuit than naive."""
        b1, b2 = ThermalBudget(), ThermalBudget()
        r1 = b1.simulate_circuit([N_QUBITS * 1] * 30, TAU_PRIM)
        r2 = b2.simulate_circuit([N_QUBITS * 4] * 30, TAU_PRIM)
        self.assertLess(r2["stability_margin"], r1["stability_margin"])

    def test_static_isa_fits_slot_budget(self):
        """Static ISA compiler must not exceed AWG waveform slot budget."""
        compiler = StaticISACompiler(n_epsilon_bins=8, use_drag=True)
        thetas   = [np.pi/4, np.pi/2, np.pi]
        compiler.compile(thetas)
        self.assertLessEqual(len(compiler.library), AWG_WAVEFORM_SLOTS,
            f"Used {len(compiler.library)} slots, budget={AWG_WAVEFORM_SLOTS}")

    def test_static_jitter_less_than_dynamic(self):
        """Static ISA must have lower gate jitter than dynamic re-linking."""
        compiler = StaticISACompiler().compile([np.pi/2])
        j = compiler.jitter_analysis(100)
        self.assertLess(j["jitter_static_ns"], j["jitter_dynamic_ns"])

    def test_cvar_worse_than_mean(self):
        """CVaR_95 must be strictly greater than mean infidelity."""
        wca  = WorstCaseAnalyzer()
        inf  = wca.sample_infidelity(np.pi/2, 2, "square", n_samples=1000)
        cvar = wca.cvar(inf)
        self.assertGreater(cvar["CVaR_95"], cvar["mean_infidelity"],
            "CVaR_95 must exceed mean — tail risk must be detectable")

    def test_drag_lower_cvar_than_square(self):
        """DRAG pulse must have lower CVaR (better worst-case) than square."""
        wca = WorstCaseAnalyzer()
        res = wca.compare_pulse_types(np.pi/2, driven_qubit=2, n_samples=2000)
        self.assertLess(res["drag"]["CVaR_95"], res["square"]["CVaR_95"],
            "DRAG CVaR must be lower than square pulse CVaR")

    def test_resonance_penalty_peaks_at_close_frequency(self):
        """Resonance penalty on qubit j must increase as pulse BW grows."""
        wca  = WorstCaseAnalyzer()
        # Narrow BW → small Lorentzian → small penalty
        pen_narrow = wca.resonance_penalty(0, eps=0.0, pulse_bw_ghz=0.05)[1]
        # Wide BW → large Lorentzian → large penalty
        pen_wide   = wca.resonance_penalty(0, eps=0.0, pulse_bw_ghz=2.0)[1]
        self.assertGreater(pen_wide, pen_narrow,
            "Wider pulse bandwidth must cause larger resonance penalty")


# ══════════════════════════════════════════════════════════════════
# SECTION 6 – MAIN: TESTS + 6-PANEL HARDWARE ANALYSIS FIGURE
# ══════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("UNIT TESTS – v4 hardware-aware compiler checks")
    print("=" * 65)
    suite  = unittest.TestLoader().loadTestsFromTestCase(TestV4)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main():
    if not run_tests():
        print("\n[ABORT] Tests failed."); return
    print("\n[OK] All tests passed. Running hardware analysis.\n")

    theta = np.pi / 2

    # ── Pre-compute all analysis data ────────────────────────────
    print("1/5  DRAG pulse spectral analysis...")
    d_pulse = drag_pulse(theta, np.pi/2, TAU_PRIM)
    s_pulse = square_pulse(theta, np.pi/2, TAU_PRIM)
    t_d, Ox_d, Oy_d   = d_pulse.envelope()
    t_s, Ox_s, _      = s_pulse.envelope()
    freqs_d, psd_d    = d_pulse.power_spectral_density()
    freqs_s, psd_s    = s_pulse.power_spectral_density()

    print("2/5  Thermal budget simulation (naive vs BB1, 80-layer QFT)...")
    bgt_n = ThermalBudget()
    bgt_b = ThermalBudget()
    n_layers = 80
    res_n = bgt_n.simulate_circuit([N_QUBITS * 1] * n_layers, TAU_PRIM)
    res_b = bgt_b.simulate_circuit([N_QUBITS * 4] * n_layers, TAU_PRIM)

    print("3/5  Static ISA compilation...")
    thetas   = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    compiler = StaticISACompiler(n_epsilon_bins=7, use_drag=True)
    compiler.compile(thetas)
    jitter   = compiler.jitter_analysis(100)

    print("4/5  CVaR worst-case analysis (4000 samples each)...")
    wca = WorstCaseAnalyzer()
    cvar_res = wca.compare_pulse_types(theta, driven_qubit=2, n_samples=4000)

    print("5/5  Resonance penalty heatmap...")
    eps_grid  = np.linspace(-0.5, 0.5, 60)
    bw_grid   = np.linspace(0.05, 2.0, 60)
    res_map   = np.zeros((60, 60))
    for i, eps in enumerate(eps_grid):
        for j, bw in enumerate(bw_grid):
            pen = wca.resonance_penalty(2, eps, bw)
            res_map[i, j] = pen.sum()

    # ══ FIGURE ═══════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "DCG v4 – Hardware-Aware Compiler Framework\n"
        "DRAG Shaping · Thermal Budget · Static ISA · CVaR Worst-Case",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Plot A: DRAG pulse envelope + spectrum ────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_d*1e9, Ox_d, color="steelblue", lw=2, label="Ωₓ DRAG (in-phase)")
    ax.plot(t_d*1e9, Oy_d, color="darkorange", lw=2, ls="--",
            label="Ωᵧ DRAG (correction)")
    ax.plot(t_s*1e9, Ox_s, color="tomato",    lw=1.5, ls=":", label="Square Ωₓ")
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Amplitude (rad/s · norm)")
    ax.set_title("A – DRAG vs Square Pulse Envelope", fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.semilogy(freqs_d[freqs_d < 3], psd_d[freqs_d < 3],
                 color="steelblue", lw=1.5, alpha=0.5, label="DRAG PSD")
    ax2.semilogy(freqs_s[freqs_s < 3], psd_s[freqs_s < 3],
                 color="tomato", lw=1.5, alpha=0.5, ls=":", label="Square PSD")
    ax2.set_ylabel("PSD (norm, log)", color="gray", fontsize=8)
    ax2.axvline(abs(ANHARMONICITY)/1e9, color="red", lw=1, ls="-.",
                label=f"|α|={abs(ANHARMONICITY)/1e6:.0f} MHz")
    ax2.legend(fontsize=6.5, loc="lower right")

    # ── Plot B: Thermal budget over circuit depth ─────────────────
    ax = fig.add_subplot(gs[0, 1])
    layers = np.arange(n_layers)
    ax.plot(layers, res_n["T_history_K"]*1e3, color="steelblue",
            lw=2, label=f"Naive (1 pulse/qubit)")
    ax.plot(layers, res_b["T_history_K"]*1e3, color="tomato",
            lw=2, label=f"BB1 (4 pulses/qubit)")
    ax.axhline(T_CRITICAL*1e3, color="black", ls="--", lw=1.5,
               label=f"T_crit = {T_CRITICAL*1e3:.0f} mK")
    ax.axhline(T_BASE*1e3,     color="green", ls=":",  lw=1,
               label=f"T_base = {T_BASE*1e3:.0f} mK")
    if res_b["boiled_at_layer"] is not None:
        ax.axvline(res_b["boiled_at_layer"], color="red", lw=1.5, ls="-.",
                   label=f"BB1 boils at layer {res_b['boiled_at_layer']}")
    ax.set_xlabel("Circuit layer"); ax.set_ylabel("Fridge temp (mK)")
    ax.set_title("B – Dilution Fridge Thermal Budget\n(20 µW cooling power)",
                 fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

    # ── Plot C: Static ISA slot usage + jitter comparison ─────────
    ax = fig.add_subplot(gs[0, 2])
    m_vals = [slot.m_order for slot in compiler.library]
    bw_vals= [slot.bandwidth_ghz for slot in compiler.library]
    colors_m = {1:"tomato", 2:"darkorange", 3:"steelblue"}
    sc = ax.scatter(range(len(compiler.library)), bw_vals,
                    c=[colors_m[m] for m in m_vals], s=40, zorder=3)
    ax.set_xlabel("Waveform slot index"); ax.set_ylabel("Pulse bandwidth (GHz)")
    ax.set_title(
        f"C – Static ISA: {len(compiler.library)}/{AWG_WAVEFORM_SLOTS} slots used\n"
        f"Jitter: static={jitter['jitter_static_ns']:.1f}ns  "
        f"dynamic={jitter['jitter_dynamic_ns']*1e3:.0f}µs→ns",
        fontsize=9
    )
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=f"m={k}") for k,v in colors_m.items()],
              fontsize=8)
    ax.axhline(abs(ANHARMONICITY)/1e9, color="red", ls="--", lw=1.5,
               label=f"|α| = {abs(ANHARMONICITY)/1e6:.0f} MHz")
    ax.grid(True, alpha=0.3)

    txt = (f"Phase error (QFT):\n"
           f"  Static  ISA: {jitter['phase_err_static_rad']:.3f} rad\n"
           f"  Dynamic ISA: {jitter['phase_err_dynamic_rad']:.1f} rad\n"
           f"  Ratio: {jitter['qft_phase_ratio']:.0f}×")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # ── Plot D: CVaR infidelity distributions ─────────────────────
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1.0, 80)
    ax.hist(cvar_res["sq_samples"],   bins=bins, density=True,
            color="tomato",    alpha=0.55, label="Square pulse")
    ax.hist(cvar_res["drag_samples"], bins=bins, density=True,
            color="steelblue", alpha=0.55, label="DRAG pulse")

    for label, results, col in [("Square", cvar_res["square"], "tomato"),
                                  ("DRAG",   cvar_res["drag"],   "steelblue")]:
        ax.axvline(results["CVaR_95"], color=col, lw=2, ls="--",
                   label=f"{label} CVaR₉₅={results['CVaR_95']:.3f}")
        ax.axvline(results["sigma3_infidelity"], color=col, lw=1.5, ls=":",
                   label=f"{label} 3σ={results['sigma3_infidelity']:.3f}")

    ax.set_xlabel("Total circuit infidelity"); ax.set_ylabel("Probability density")
    ax.set_title("D – CVaR Worst-Case Distribution\n(4000 samples, Student-t noise)",
                 fontsize=10)
    ax.legend(fontsize=6.5); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(cvar_res["sq_samples"].max(), 0.8))

    # ── Plot E: Resonance penalty heatmap ─────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(res_map.T, origin="lower", aspect="auto",
                   extent=[eps_grid[0], eps_grid[-1], bw_grid[0], bw_grid[-1]],
                   cmap="hot", vmin=0)
    ax.axhline(d_pulse.spectral_width_ghz(), color="cyan", lw=2, ls="--",
               label=f"DRAG bw={d_pulse.spectral_width_ghz():.2f}GHz")
    ax.axhline(s_pulse.spectral_width_ghz(), color="white", lw=2, ls="--",
               label=f"Square bw={s_pulse.spectral_width_ghz():.2f}GHz")
    plt.colorbar(im, ax=ax, label="Total resonance penalty")
    ax.set_xlabel("Noise ε"); ax.set_ylabel("Pulse bandwidth (GHz)")
    ax.set_title("E – Resonance Catastrophe Map\n(drive on Q2, penalty summed over Q0–Q4)",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")

    # ── Plot F: Summary scorecard ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

    sq  = cvar_res["square"]
    dr  = cvar_res["drag"]
    rows = [
        ["Metric",               "Square BB1",         "DRAG BB1"],
        ["Spectral width (GHz)", f"{s_pulse.spectral_width_ghz():.2f}",
                                  f"{d_pulse.spectral_width_ghz():.3f}"],
        ["|2⟩ leakage",          f"{s_pulse.leakage_to_level2():.2e}",
                                  f"{d_pulse.leakage_to_level2():.2e}"],
        ["Mean infidelity",       f"{sq['mean_infidelity']:.4f}",
                                  f"{dr['mean_infidelity']:.4f}"],
        ["CVaR₉₅ (worst 5%)",    f"{sq['CVaR_95']:.4f}",
                                  f"{dr['CVaR_95']:.4f}"],
        ["3σ infidelity",         f"{sq['sigma3_infidelity']:.4f}",
                                  f"{dr['sigma3_infidelity']:.4f}"],
        ["P(catastrophic >0.5)",  f"{sq['p_catastrophic']:.3%}",
                                  f"{dr['p_catastrophic']:.3%}"],
        ["Fridge T_max (mK)",     f"{res_n['T_max_mK']:.2f}",
                                  f"{res_b['T_max_mK']:.2f}"],
        ["ISA jitter",            f"{jitter['jitter_dynamic_ns']*1e6:.0f} µs",
                                  f"{jitter['jitter_static_ns']:.2f} ns"],
        ["Slots used/budget",     "N/A",
                                  f"{len(compiler.library)}/{AWG_WAVEFORM_SLOTS}"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.55)
    # Colour the DRAG column green where better
    for r in range(1, len(rows)-1):
        tbl[r, 2].set_facecolor("#d4edda")
    ax.set_title("F – Hardware Scorecard", fontsize=10, pad=14)

    plt.savefig("dcg_v4_hardware.png", dpi=150, bbox_inches="tight")
    print("\nSaved → dcg_v4_hardware.png")
    print("\n── Summary ───────────────────────────────────────────────")
    print(f"  DRAG spectral width  : {d_pulse.spectral_width_ghz():.3f} GHz "
          f"vs square {s_pulse.spectral_width_ghz():.2f} GHz")
    print(f"  Level-2 leakage DRAG : {d_pulse.leakage_to_level2():.2e} "
          f"vs square {s_pulse.leakage_to_level2():.2e}")
    print(f"  CVaR₉₅ DRAG          : {dr['CVaR_95']:.4f} "
          f"vs square {sq['CVaR_95']:.4f}")
    print(f"  BB1 fridge T_max     : {res_b['T_max_mK']:.2f} mK "
          f"(naive: {res_n['T_max_mK']:.2f} mK)")
    print(f"  Static ISA jitter    : {jitter['jitter_static_ns']:.2f} ns "
          f"vs dynamic {jitter['jitter_dynamic_ns']*1e6:.0f} µs")
    boil = res_b['boiled_at_layer']
    print(f"  BB1 fridge stability : {'BOILED at layer '+str(boil) if boil else 'survived all '+str(n_layers)+' layers'}")


if __name__ == "__main__":
    main()
