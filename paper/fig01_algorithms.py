import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from paper.paths import DATA_ROOT, PLOTS_ROOT
except ModuleNotFoundError:
    from paths import DATA_ROOT, PLOTS_ROOT

OUTPUT_DIR = PLOTS_ROOT / "Fig01"

FIGURE_STYLE = {
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def exp_kernel(tau_s: float, dt_s: float, duration_s: float = 1.0) -> np.ndarray:
    """Exponential kernel normalized to sum=1 for discrete convolution."""
    t = np.arange(0, duration_s + dt_s, dt_s, dtype=np.float32)
    k = np.exp(-t / tau_s).astype(np.float32)
    s = float(k.sum())
    if s > 0:
        k /= s
    return k


def gaussian_depth_profile(z_um: np.ndarray, center_um: float, sigma_um: float) -> np.ndarray:
    """Gaussian profile normalized to peak=1."""
    prof = np.exp(-0.5 * ((z_um - center_um) / sigma_um) ** 2).astype(np.float32)
    m = float(prof.max())
    if m > 0:
        prof /= m
    return prof


def make_depth_trace_flat_with_smooth_transitions(
    t_s: np.ndarray,
    sigma: float = 1.0,
    smooth: bool = True,
) -> np.ndarray:
    """
    Flat segments with a few jumps; then smooth transitions via Gaussian-window convolution.
    """
    n = t_s.size
    dt = float(t_s[1] - t_s[0])

    # These change points reproduce the piecewise depth trajectory in Figure 1.
    change_times_s = np.array([0, 2, 5, 7, 11, 13, 14], dtype=float)
    change_idx = np.clip((change_times_s / dt).astype(int), 0, n - 1)

    target_depths = np.array([0, -1, -4, -2, 3, 5, 0], dtype=float)

    depth = np.empty(n, dtype=np.float32)
    for k in range(len(change_idx) - 1):
        depth[change_idx[k] : change_idx[k + 1]] = target_depths[k]
    depth[change_idx[-1] :] = target_depths[-1]

    # Smooth the jumps with a Gaussian kernel implemented in NumPy.
    if smooth:
        win_len = int(np.ceil(6.0 * sigma / dt))
        win_len += win_len % 2 == 0  # make odd
        x = np.linspace(-3 * sigma, 3 * sigma, win_len, dtype=np.float32)
        g = np.exp(-(x * x) / (2 * sigma * sigma)).astype(np.float32)
        g /= float(g.sum())
        depth = np.convolve(depth, g, mode="same").astype(np.float32)

    return depth


def save_panel(fig, out_path: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_figures(input_root, output_dir, *, style=None, seed=0) -> None:
    """Generate the deterministic Figure 1 algorithm panels.

    Parameters
    ----------
    input_root
        Existing manuscript-data root. Figure 1 uses synthetic data but still
        requires an explicit root for a consistent paper-script interface.
    output_dir
        Directory receiving the SVG panel files.
    style
        Optional Matplotlib style name or file applied after manuscript
        defaults.
    seed
        Seed for the Poisson event simulation. Identical seeds produce
        identical numerical traces.
    """
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Figure 1 input root not found: {input_root}")
    mpl.rcParams.update(FIGURE_STYLE)
    if style:
        plt.style.use(style)
    # Use a short, deterministic simulation suitable for regenerating the figure.
    fs_hz = 50.0
    dt_s = 1.0 / fs_hz
    duration_s = 15.0
    tau_s = 0.1

    rate_hz = 5.0

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Time base
    t = np.arange(0, duration_s, dt_s, dtype=np.float32)

    # Simulate events, then apply the indicator's exponential response kernel.
    rng = np.random.default_rng(seed)
    p = rate_hz * dt_s
    events = (rng.random(t.size) < p).astype(np.float32)

    k = exp_kernel(tau_s=tau_s, dt_s=dt_s, duration_s=1.0)
    calcium = np.convolve(events, k, mode="full")[: t.size].astype(np.float32)

    # 2) Depth trace (-5..5 µm), flat segments with smooth transitions
    depth_um = make_depth_trace_flat_with_smooth_transitions(t_s=t, sigma=0.4, smooth=True)

    # 3) Depth profile (Gaussian)
    z = np.linspace(-15.0, 15.0, 301, dtype=np.float32)
    z0_um = 3.0
    sigma_um = 5.0
    profile = gaussian_depth_profile(z_um=z, center_um=z0_um, sigma_um=sigma_um)
    ref_val = float(np.interp(0, z, profile))
    profile /= ref_val

    # 4) Amplitude over time from profile(depth(t))
    amp = np.interp(depth_um, z, profile).astype(np.float32)

    # 5) Depth-modulated calcium
    calcium_mod = (calcium * amp).astype(np.float32)

    # 6a) Plot + save files for neuron
    # Depth trace
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(t, depth_um, color="tab:blue", linewidth=1)
    ax.axhline(0, color="darkred")
    ax.set_title("Depth trace")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Depth ($\mu$m)")
    save_panel(fig, out_dir / "sim_neurons_depth_trace.pdf")
    # Amplitude trace
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(t, amp, color="tab:green", linewidth=1)
    ax.axhline(1, color="darkred")
    ax.set_title("Amplitude from depth profile at depth(t)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gain")
    save_panel(fig, out_dir / "sim_neurons_amplitude_trace.pdf")
    # Depth profile
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.plot(profile, z, color="tab:purple", linewidth=2)
    ax.axhline(0, color="darkred", linestyle="--", linewidth=1)
    ax.set_title("Depth profile (Gaussian)")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel(r"Depth ($\mu$m)")
    save_panel(fig, out_dir / "sim_neurons_depth_profile.pdf")
    # Calcium multiplied by amplitude trace
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(t, calcium_mod, color="tab:red", linewidth=1)
    ax.plot(t, calcium, color="k", linewidth=1)
    ax.set_title("Depth-modulated calcium")
    ax.set_xlabel("Time (s)")
    save_panel(fig, out_dir / "sim_neurons_calcium_traces.pdf")

    # Now for boutons:
    # 2) Depth trace (-5..5 µm), flat segments with smooth transitions
    depth_um = make_depth_trace_flat_with_smooth_transitions(t_s=t, sigma=0.1, smooth=True)
    depth_um_hard = make_depth_trace_flat_with_smooth_transitions(t_s=t, smooth=False)

    # 3) Depth profile (Gaussian)
    z = np.linspace(-15.0, 15.0, 301, dtype=np.float32)
    z0_um = 0.0
    sigma_um = 2.0
    profile = gaussian_depth_profile(z_um=z, center_um=z0_um, sigma_um=sigma_um)

    # 4) Amplitude over time from profile(depth(t))
    amp = np.interp(depth_um, z, profile).astype(np.float32)

    # 5) Depth-modulated calcium
    calcium_mod = (calcium * amp).astype(np.float32)

    # 6b) Plot + save files for boutons
    # Depth trace
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(t, depth_um_hard, color="tab:blue", linewidth=1)
    ax.axhline(0, color="darkred")
    ax.set_title("Depth trace")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Depth ($\mu$m)")
    save_panel(fig, out_dir / "sim_boutons_depth_trace.pdf")
    # Amplitude trace
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(t, amp, color="tab:green", linewidth=1)
    ax.axhline(1, color="darkred")
    ax.set_title("Amplitude from depth profile at depth(t)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gain")
    save_panel(fig, out_dir / "sim_boutons_amplitude_trace.pdf")
    # Depth profile
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.plot(profile, z, color="tab:purple", linewidth=2)
    ax.axhline(0, color="darkred", linestyle="--", linewidth=1)
    ax.set_title("Depth profile (Gaussian)")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel(r"Depth ($\mu$m)")
    save_panel(fig, out_dir / "sim_boutons_depth_profile.pdf")
    # Calcium multiplied by amplitude trace
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(t, calcium_mod, color="tab:red", linewidth=1)
    ax.plot(t, calcium, color="k", linewidth=1)
    ax.set_title("Depth-modulated calcium")
    ax.set_xlabel("Time (s)")
    save_panel(fig, out_dir / "sim_boutons_calcium_traces.pdf")


def main() -> None:
    generate_figures(DATA_ROOT, OUTPUT_DIR, style=None, seed=0)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 1 algorithm panels.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_figures(args.input_root, args.output_dir, style=args.style, seed=args.seed)


if __name__ == "__main__":
    cli()
