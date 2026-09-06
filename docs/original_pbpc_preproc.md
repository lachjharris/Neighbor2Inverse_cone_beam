# Neighbor2Inverse PBPC Preprocessing Notes

Baseline repository commit: `f298081`

This note summarises the original authors' preprocessing / phase-retrieval / reconstruction pipeline as currently implemented in the pristine repository. It is intended as a practical map of the code rather than a derivation of the underlying mathematics.

---

## Relevant repository structure

```text
code/
├── README.md
├── Reconstruction/
│   ├── 0_preprocessing.py
│   ├── 1_RingArtifactRemoval.py
│   ├── 2_PhaseRetrieval.py
│   ├── 3_reconstruction.py
│   ├── phase_retrieval.py
│   ├── reco_utils.py
│   ├── stitching.py
│   └── utilRingArtifactRemoval.py
└── Neighbor2Inverse/
    └── PBPC_data/
        ├── 0_calculateStats.py
        ├── 1_trainNeighbor2Inverse.py
        ├── trainparamsNeighbor2InverseProjSub.yml
        ├── trainparamsNeighbor2InverseSinoSub.yml
        ├── trainparamsDataFidelityOrigSino.yml
        ├── trainparamsDataFidelityVirtSino.yml
        ├── trainparamsSparse.yml
        ├── trainSplit.csv
        ├── valSplit.csv
        └── testSplit.csv
```

The raw calf data are **not included in the Git repository**. The README points to a separate data download.

---

# Overall preprocessing / reconstruction pipeline

The authors provide four main scripts under `Reconstruction/`:

```text
raw measurement data
   ↓
0_preprocessing.py
   ↓
projStitched_*.npy
   ↓
1_RingArtifactRemoval.py
   ↓
projFiltered_*.npy
   ↓
2_PhaseRetrieval.py
   ↓
projPR_*.npy
   ↓
3_reconstruction.py
   ↓
reco_*.npy
```

The README also says that it is possible to run only:

```text
0_preprocessing.py
   ↓
3_reconstruction.py
```

and enable phase retrieval and/or ring removal inside `3_reconstruction.py`.

---

# 1. `0_preprocessing.py`

## Purpose

Handles the raw synchrotron projection data and produces flat-field corrected, stitched projection images.

## Inputs

The script expects raw measurement folders such as:

```text
../Measurements/Calf_31_pos<position>_<exposure>ms/
```

with files including:

```text
SAMPLE.hdf
DF_BEFORE.hdf
DF_AFTER.hdf
BG_BEFORE.hdf
BG_AFTER.hdf
```

`DF_*` are dark-field measurements and `BG_*` are flat/background measurements.

## Pipeline

```text
SAMPLE.hdf
   ↓
discard first 10 faulty detector pixels
   ↓
load before/after darks and flats
   ↓
dark / flat-field correction
   ↓
keep projections [20:3620]
= 3600 projections
   ↓
split into:
first 1800 projections
second 1800 projections
   ↓
flip second half horizontally
   ↓
stitch opposing views to enlarge the field of view
   ↓
despeckle
   ↓
save:
projStitched_<exposure>ms_pos<position>.npy
```

## Important details

- The original acquisition contains approximately 360° of data.
- After trimming, the code keeps 3600 projections.
- These are split into two sets of 1800.
- The second half is horizontally flipped and stitched to the first half.
- This produces 1800 stitched projections over approximately 180°.
- Stitching overlap is estimated from the 200 ms scan for each position and then reused for the other exposure times at that position.
- Output is stored as `float16`.

---

# 2. `1_RingArtifactRemoval.py`

## Purpose

Applies the sorting-based sinogram correction described by Vo et al.

## Important note from the README

The README states that this filtering was used **only for the testing data, not for the training split**.

We have not yet inspected the implementation of this script in detail, so no further assumptions are recorded here.

---

# 3. `2_PhaseRetrieval.py`

## Purpose

This is the **driver script** for phase retrieval.

It handles file paths, loops over positions/exposure times, loads projection stacks, calls the Paganin implementation, and saves the result.

## Pipeline

```text
projFiltered_<exposure>ms_pos<position>.npy
   ↓
load as float32 torch tensor
   ↓
call:
compute_paganin_batch(...)
from phase_retrieval.py
   ↓
receive projected-thickness stack
   ↓
save:
projPR_<exposure>ms_pos<position>.npy
```

## Parameters used

```text
delta      = 0.8e-8
beta       = 1.0e-11
z          = 5000 mm
pixel size = 0.009 mm
energy     = 70 keV
```

## Current loop in the script

The current `exp_list` is:

```python
[25, 33, 50, 67, 100, 200]
```

so `15 ms` is not included in this standalone script as currently written.

---

# 4. `phase_retrieval.py`

## Purpose

This file contains the **actual Paganin phase-retrieval implementation** used by both preprocessing and Neighbor2Inverse training.

`2_PhaseRetrieval.py` answers:

> Which files should be processed?

`phase_retrieval.py` answers:

> How is Paganin phase retrieval performed?

## High-level pipeline inside `compute_paganin_batch()`

```text
input projection stack
   ↓
calculate wavelength from X-ray energy
   ↓
calculate linear attenuation coefficient μ
   ↓
calculate Paganin filter parameter
   ↓
process projections in batches
   ↓
add small constant (2e-6)
to avoid log(0)
   ↓
pad each projection
   ↓
construct 2-D Paganin Fourier kernel
   ↓
FFT
   ↓
apply Paganin low-pass filter
   ↓
inverse FFT
   ↓
remove padding
   ↓
optional clipping
   ↓
convert filtered intensity to projected thickness:
t = -(1/μ) log(I_filtered)
   ↓
stack processed projections
   ↓
return projected-thickness stack
```

## Important details

- The Paganin filtering is performed in **2-D on each detector projection**.
- `compute_paganin_batch()` is decorated with `@torch.no_grad()`.
- The returned quantity is a **projected thickness map**, not merely a filtered intensity image.

---

# 5. `3_reconstruction.py`

## Purpose

Driver script for tomographic reconstruction.

With the file exactly as currently written:

```python
load_path = '../ProcessedData/projPR/'
phase_retrieval = False
ring_removal = False
```

the script assumes phase retrieval has already been performed.

## Default pipeline

```text
projPR_<exposure>ms_pos<position>.npy
   ↓
load projected-thickness projections
   ↓
swap axes to form a stack of sinograms
   ↓
call:
recon_batch(...)
from reco_utils.py
   ↓
save:
reco_<exposure>ms_pos<position>.npy
```

The key axis operation is:

```python
sin_stack = proj.swapaxes(0, 2)
```

This converts the projection stack into a stack of independent sinograms for slice-by-slice reconstruction.

## Optional modes

The script can also perform earlier steps itself.

### Phase retrieval only

Conceptually:

```text
projFiltered
   ↓
Paganin phase retrieval
   ↓
reconstruction
```

### Ring removal + phase retrieval

Conceptually:

```text
projStitched
   ↓
ring-artifact removal
   ↓
Paganin phase retrieval
   ↓
reconstruction
```

---

# 6. `reco_utils.py` → `recon_batch()`

## Purpose

Contains the actual parallel-beam FBP implementation used by `3_reconstruction.py`.

## Expected input

```text
[num_slices, n_angles, detector_pixels]
```

Each item in the first dimension is treated as an independent 2-D sinogram.

## Reconstruction pipeline

```text
phase-retrieved sinogram stack
   ↓
choose projection angles
(default: 1800 equally spaced angles from 0 to π)
   ↓
pad detector-width direction
by half the original width on each side
   ↓
create TorchRadon `Radon` operator
   ↓
process slices in batches
   ↓
apply `rivers(...)` ring correction
   ↓
Ram-Lak filter
   ↓
parallel-beam backprojection
   ↓
crop / mask back to original size
   ↓
move slices to CPU
   ↓
concatenate batches
   ↓
return reconstructed slice stack
```

## Geometry

The code uses:

```python
Radon(...)
```

rather than:

```python
RadonFanbeam(...)
```

so this reconstruction is explicitly **parallel-beam**.

The reconstruction model is therefore:

```text
one detector row across all projection angles
   ↓
one 2-D sinogram
   ↓
one reconstructed axial slice
```

This is one of the main acquisition-specific assumptions that will eventually need to change for cone-beam reconstruction.

---

# Important distinction for Neighbor2Inverse training

The standalone preprocessing pipeline above can generate precomputed phase-retrieved projections and reconstructions.

However, the default Neighbor2Inverse projection-subsampling training YAML currently contains:

```yaml
path_proj: '../../ProcessedData/projStitched/'
path_reco: '../../ProcessedData/recos/'
doPhaseRetrieval: True
```

This means the training code loads the **pre-phase-retrieval `projStitched` projections** and performs phase retrieval during training.

That is important because Neighbor2Inverse requires the ordering:

```text
measured / corrected projection
   ↓
Neighbor subsampling
   ↙            ↘
  g1            g2
   ↓             ↓
phase retrieval  phase retrieval
   ↓             ↓
reconstruction   reconstruction
```

rather than:

```text
phase retrieval
   ↓
Neighbor subsampling
```

The neighbour split therefore happens **before** the Paganin filtering.

---

# Useful repository quirks noted so far

These are observations from the pristine `f298081` checkout and should be documented rather than silently changed.

## README vs YAML regularizer setting

The README says:

```text
trainparamsNeighbor2InverseProjSub.yml
→ projection subsampling with regularization
```

but the current YAML contains:

```yaml
regularizer: False
```

So the README and current configuration are not perfectly synchronized.

## Standalone phase-retrieval exposure list

`2_PhaseRetrieval.py` currently processes:

```python
[25, 33, 50, 67, 100, 200]
```

and does not include `15 ms`.

## Reconstruction is not cone-beam

The supplied reconstruction uses parallel-beam TorchRadon FBP and independent detector-row sinograms.

This is an implementation detail of the authors' calf workflow, not a fundamental requirement of the Neighbor2Inverse idea.

---

# Current mental model

The authors' original calf processing can be thought of as:

```text
RAW SYNCHROTRON DATA
        ↓
flat/dark correction
        ↓
360° → stitched 180° projections
        ↓
optional ring correction
        ↓
Paganin thickness retrieval
        ↓
parallel-beam sinogram extraction
        ↓
parallel-beam FBP
        ↓
reconstructed volume
```

For Neighbor2Inverse training specifically, the important modification is:

```text
corrected/stiched projections
        ↓
Neighbor subsampling FIRST
      ↙                 ↘
     g1                 g2
      ↓                  ↓
Paganin retrieval   Paganin retrieval
      ↓                  ↓
reconstruction      reconstruction
      ↓                  ↓
network input       noisy target
```

---

# What has not yet been inspected

To keep the investigation incremental, the following have not yet been worked through in detail:

- `1_RingArtifactRemoval.py`
- `0_calculateStats.py`
- the internals of Neighbor subsampling in `modelLightning.py`
- the exact `L_Nei` implementation
- `L_reg`
- the data-fidelity variants
- the clinical CT implementation

Those can be handled separately rather than mixing them into the preprocessing notes.
