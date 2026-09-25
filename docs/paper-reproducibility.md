# Paper figure reproducibility

Paper scripts remain repository scripts and are not installed as runtime modules. Run them from a
source checkout with `MPLBACKEND=Agg`. Set `SUITE2P_IN_DEPTH_DATA_ROOT` to the private manuscript
`Data` directory; do not place it in this repository. The default is
`~/suite2p-in-depth-paper-data`. Plotting scripts use that root for inputs and write to its
`plots/FigXX` subdirectories. Figures 1, 2, and 4 require explicit input and output paths on the
command line; the remaining scripts use `paper/paths.py`. `fig05_preprocess.py` uses the same root
for its output and has separate private source locations configured through
`SUITE2P_IN_DEPTH_PRIVATE_ROOT`, `SUITE2P_IN_DEPTH_PROCESSED_ROOT`,
`SUITE2P_IN_DEPTH_BONSAI_ROOT`, and `SUITE2P_IN_DEPTH_REGISTERED_ROOT`. The private root defaults
to a sibling `suite2p-in-depth_private` directory, and the other three roots default to its
`processed`, `bonsai`, and `registered` subdirectories. Generated high-resolution panels are
untracked.

The correction cohorts are fixed in code: Figure 3 uses Quille (2023-07-24), Uma
(2023-12-06), Dublin (2024-05-01), Ely (2024-07-01), and SS126 (2024-07-30);
Figure 5 uses Oephelia (2023-07-21), Memphis (2023-08-21), Styx (2024-02-26),
Stereopes (2024-02-13), Vesta (2024-05-08), and SS123 (2024-09-12). The CSV files
must contain these recordings, but their `Process` flags do not change the cohort.
Figure 3 ground truth uses Quille (2023-10-12, experiment 5) and Tara
(2023-11-01, experiment 6); Figure 5 ground truth uses Stereopes (2024-01-29,
experiment 3) and Styx (2024-03-05, experiment 4). Extra data folders are ignored.
`fig05_preprocess.py` runs once without arguments: it exports matched reference traces
for all six bouton recordings and all-plane z profiles for Stereopes (2024-02-13).
Both products are written under the manuscript `Data/boutons` directory, which
`fig05_zregistration_correction.py` reads. Registered movies remain on the separate
source drives and are not copied into manuscript Data.

```shell
python -m pip install ".[dev]"
export MPLBACKEND=Agg
export SUITE2P_IN_DEPTH_DATA_ROOT=/path/to/private/Data
python paper/fig01_algorithms.py --input-root "$SUITE2P_IN_DEPTH_DATA_ROOT" --output-dir "$SUITE2P_IN_DEPTH_DATA_ROOT/plots/Fig01"
python paper/fig02_zstack.py --input-root "$SUITE2P_IN_DEPTH_DATA_ROOT" --output-dir "$SUITE2P_IN_DEPTH_DATA_ROOT/plots/Fig02"
python paper/fig04_zregistration.py --input-root "$SUITE2P_IN_DEPTH_DATA_ROOT" --output-dir "$SUITE2P_IN_DEPTH_DATA_ROOT/plots/Fig04"
```

PowerShell uses `$env:MPLBACKEND = "Agg"` and
`$env:SUITE2P_IN_DEPTH_DATA_ROOT = "C:\\path\\to\\private\\Data"`; pass the same paths with
`--input-root` and `--output-dir`. Figure 1 uses the fixed simulation seed `0`.

## Figure 1: `fig01_algorithms.py`

Figure 1 uses deterministic simulations and only requires that `DATA_ROOT` exist. It generates
eight PDF panels: depth trace, amplitude trace, depth profile, and calcium traces for both neurons
and boutons (`sim_neurons_*.pdf` and `sim_boutons_*.pdf`). The fixed seed controls the Poisson event
simulation.

## Figure 2: `fig02_zstack.py`

The script expects `DATA_ROOT/Fig02/Ely_2024-07-01/plane 1`. Required dataset files are
`piezo_per_plane.npy`, `piezo_time.npy`, `2pRois.zProfiles.npy`, `2pRois.ids.npy`,
`2pCalcium.F_zcorrected.npy`, `2pCalcium.N_zcorrected.npy`, and `2pCalcium.dff.npy`. The plane
folder requires `ops.npy`, `stack_plane1_chan1.tif`, `bestSlice_plane1.npy`, `zcorr_plane1.npy`,
`db_zcorrect.npy`, `settings_zcorrect.npy`, `detect_outputs_zcorrect.npy`, `stat.npy`, `F.npy`,
and `Fneu.npy`.

It writes `piezo_per_plane.pdf`, `mean_image.pdf`, `zstack_reference_slice.pdf`,
`overlay_ref_zstack.pdf`, `depth_trace.pdf`, `axial_profile.pdf`, `zstack_ROI.pdf`,
`correction_trace.pdf`, `calcium_traces.pdf`, `signal-neuropil_scatter.pdf`,
`neuropil-corrected_traces.pdf`, and `dFF_traces.pdf`.

## Figure 4: `fig04_zregistration.py`

The script expects `DATA_ROOT/Fig04/Io_2023-02-13/plane_z`. The dataset folder requires
`frame_times.npy`, `piezo_time.npy`, `piezo_trace.npy`,
`piezo_per_plane_shifted_good_+011.npy`, and `piezo_per_plane_shifted_bad_+015.npy`; `plane_z`
requires `db.npy` and `reg_outputs.npy`.

It writes one `reference_planeNN.png` per valid plane plus `corrs_refs-vs-planes.pdf`,
`best_plane_per_reference.pdf`, `shifts.pdf`, `piezo_trace.pdf`, `piezo_per_plane.pdf`, and
`piezo_per_plane_shifted.pdf`.

## Reproducibility checks

The test suite generates small synthetic inputs for every script, invokes the functions with Agg,
and checks exact output names and non-empty files. Those fixtures test execution, not manuscript
values. For an accepted paper build, record the source commit, environment, data snapshot,
command, style, seed, and checksums of generated panels in the private manuscript-data repository.
