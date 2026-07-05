"""
Basic energy-consistency tests for LaserPulse.

These tests intentionally check only the most fundamental Grid/Pulse behavior.
They are meant to be a safety net for future refactoring, not a full physical
validation of the regenerative-amplifier model.
"""

from pathlib import Path
import sys

import numpy as np

# Make the repository root importable when pytest is run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.grid import Grid
from core.pulse import Pulse


def make_gaussian_pulse(target_energy=1e-9):
    """Create a simple Gaussian pulse normalized to a target energy [J]."""
    grid = Grid(points=1024, central_wl=1030e-9, max_wl=1100e-9)
    pulse = Pulse(grid)

    tau = 200e-15  # [s], only used to build a smooth test pulse
    envelope = np.exp(-0.5 * (grid.time_window / tau) ** 2)
    pulse.A_t = envelope.astype(np.complex128)

    current_energy = pulse.get_energy()
    pulse.A_t *= np.sqrt(target_energy / current_energy)
    pulse.to_freq_domain()
    return pulse


def test_time_frequency_round_trip_preserves_energy():
    """A_t -> A_f -> A_t should preserve pulse energy."""
    pulse = make_gaussian_pulse(target_energy=1e-9)
    energy_before = pulse.get_energy()

    pulse.to_freq_domain()
    pulse.to_time_domain()
    energy_after = pulse.get_energy()

    assert np.isclose(energy_after, energy_before, rtol=1e-10, atol=0.0)


def test_frequency_round_trip_preserves_field_shape():
    """FFT followed by IFFT should recover the original time-domain field.

    We compare the relative L2 error instead of using a very small absolute
    point-by-point tolerance. FFT/IFFT round trips naturally leave tiny floating
    point tails near zero-valued samples, especially on different Python/NumPy
    versions.
    """
    pulse = make_gaussian_pulse(target_energy=1e-9)
    original_field = pulse.A_t.copy()

    pulse.to_freq_domain()
    pulse.to_time_domain()

    relative_error = np.linalg.norm(pulse.A_t - original_field) / np.linalg.norm(original_field)
    assert relative_error < 1e-12


def test_ftl_pulse_does_not_modify_original_pulse():
    """get_ftl_pulse should return a new pulse and keep the original A_f unchanged."""
    pulse = make_gaussian_pulse(target_energy=1e-9)
    original_spectrum = pulse.A_f.copy()

    ftl_pulse = pulse.get_ftl_pulse()

    assert ftl_pulse is not pulse
    assert np.allclose(pulse.A_f, original_spectrum)
    assert ftl_pulse.get_energy() > 0
