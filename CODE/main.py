#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper-code.py - robust IoT tensor completion: seven methods, two datasets,
four results.

--------------------------------------------------------------------------------
WHAT THIS FILE IS
--------------------------------------------------------------------------------
One self-contained script.  It takes one or more incomplete 3-way tensors in the
layout (L, A, T) - mode 1 = locations, mode 2 = attributes, mode 3 = time,
float32, NaN = never observed - and runs the same sweep on each of them:

    Dataset I    tao_mode3_LxAxT.npy     (47, 5, 13248)
                 TAO/TRITON moorings x (SST, AIRT, RH, T125, T200) x
                 ten-minute instants, 74.4 % observed
    Dataset II   airquality_LxAxT.npy    (376, 6, 21000)
                 AirKorea stations x (SO2, CO, O3, NO2, PM10, PM25) x hourly
                 instants, 74.0 % observed (prepare_airquality.py)

Each tensor is corrupted under a controlled protocol (missing ratio x outlier
bound x noise level, see THE CORRUPTION PROTOCOL), and every method is asked to
return the clean, complete tensor.  Every method sees the bit-identical
corrupted input for a given configuration.

    Main comparison (default --methods; the order given is M1, M2, ... in the
    results)
    M1  dltd         Deep-TD (this project): robust Tucker with low-rank
                     spatial / attribute modes and a full-resolution temporal
                     factor, regularised by a bidirectional causal TCN prior,
                     with an AR(1) residual stage and outlier test.  Section 6.
    M2  datc         DATC, Deep Adversarial Tensor Completion
                     (Xie et al., IEEE/ACM Trans. Networking 2023).  Section 8.
    M3  deeptensor   DeepTensor, low-rank decomposition with deep network
                     priors (Saragadam et al., IEEE TPAMI 2024).  Section 9.
    M4  neuralcp     NeuralCP, Bayesian neural CP decomposition
                     (Liu et al., Cognitive Computation 2018).  Section 10.
    M5  trpca        TRPCA with the t-SVD tensor nuclear norm
                     (Lu et al., IEEE TPAMI 2020).  Section 11.
    M6  costco       CoSTCo, convolutional sparse tensor completion
                     (Liu et al., KDD 2019).  Section 12.
    M7  ntc          NTC, Neural Tensor Completion: outer-product interaction
                     map + 3-D CNN (Xie et al., IEEE INFOCOM 2021).  Section 13.

    Ablations (add them with --methods; they share M1's code path exactly)
    dltd_core        TD + TCN, no residual stage
    td               pure Tucker: same ranks, no TCN prior, no residual stage
    td_res           Tucker + residual stage, no TCN prior
    cp               pure CP: Khatri-Rao dictionary of CP-ALS factors, no prior
    cp_tcn           CP + the same TCN prior, no residual stage

--------------------------------------------------------------------------------
FIDELITY TO THE SOURCE PAPERS  (what each baseline implements, and where it
had to adapt)
--------------------------------------------------------------------------------
Each method's section opens with the equations it implements.  In brief:

  M2 DATC        3-D conv autoencoder G with the missing-data-aware first layer
                 (eq. 4); n = 2 down-sampling units of a (2,2,1)- then a
                 (1,1,2)-strided conv, a valid conv to the code, one valid and
                 n strided deconvs; kernels [4,8,16,32,64,32,8,1]; layer norm
                 + ReLU after every layer, sigmoid last; encoder-decoder
                 discriminator emitting one probability per entry; eqs. (9)
                 and (10) alternated per sub-tensor (Algorithm 1); w = 12,
                 lambda = 0.1, Adam 1e-3.  Adapted: mini-batches of windows, and
                 an epoch visits the windows at a stride of w from a random
                 offset (all T - w + 1 windows over the epochs) so an epoch is
                 one pass over the tensor.
  M3 DeepTensor  eq. (8): PARAFAC whose three factors are outputs of 1-D
                 Deep-Image-Prior networks (skip connections, BN, LeakyReLU,
                 reflection padding, output channels = rank) fed fixed noise;
                 Adam 1e-3 on the masked l2 loss (--dt-loss l1 for the paper's
                 sparse-noise variant).  The stopping point is the
                 regulariser; it is chosen on held-out cells.
  M4 NeuralCP    Gaussian variational posterior per factor row (eq. 5), shared
                 learnt Gaussian prior, tanh hidden layer (eq. 4), mean and
                 log-variance heads (eq. 3); the negative ELBO of eq. (9) with
                 the KL of *every* factor row; R = 50, K = 100 (the paper's
                 real-data settings).  Adapted: batch 4096 with Adam 1e-3
                 (the paper: batch 30, 1e-4).
  M5 TRPCA-TNN   Algorithm 4 with its published constants (mu0 1e-3, rho 1.1,
                 mu_max 1e10, eps 1e-8), t-SVT of Algorithm 3, lambda of
                 Theorem 4.1.  Adapted: TRPCA has no missing-data model, so the
                 constraint is imposed on the observed set only.
  M6 CoSTCo      embedding module (eq. 4), a (1, N) then an (r, 1)
                 convolution with ReLU (eq. 5), MLP aggregation with nc hidden
                 units (eq. 6, Appendix A.1, nc = 2r as for the paper's sensor
                 tensors), squared loss (eq. 7), Adam.  Adapted: batch 1024 with
                 Adam 1e-3 (the paper: 32, 1e-4), a linear output unit (the
                 data are standardised, so signed).
  M7 NTC         embeddings (eq. 5), outer-product 3-D interaction map (eq. 6),
                 half-size-tower 3-D CNN of 2x2x2 stride-2 kernels with ReLU
                 (eqs. 7-9), sigmoid inferring layer (eq. 10), squared loss with
                 l2 on the embeddings (eq. 4), mini-batch Adam.  Adapted: R = 8
                 (3 conv layers, 32 kernels) instead of the paper's 16 or 32,
                 because the CNN runs once per entry and R = 16 costs hours per
                 fit on Dataset II; --ntc-rank 16 restores the paper's network.

  Corrections made in this revision to the four baselines that were already
  here: DATC's decoder had four deconvolutions and no normalisation, used
  LeakyReLU and a linear output, a 32-wide window and lambda 0.05 (now the
  paper's structure, LN + ReLU, sigmoid, w = 12, lambda = 0.1); DeepTensor's
  factor networks were plain up-sampling decoders (now the DIP skip
  architecture the paper uses for imputation); NeuralCP's KL term covered only
  the rows a batch touched and was down-weighted by an arbitrary 1e-3, and its
  rank / hidden size were 20 / 50 (now the full KL of eq. 9 and R = 50,
  K = 100); TRPCA's tolerance was 1e-7 (now the paper's 1e-8).

  Shared by every method, so that the comparison is of models and not of
  conventions: the same robust per-attribute standardisation of the input,
  the same 3 % of held-out readable cells for choosing the returned iterate,
  and the same output rule (:func:`robust_estimate`): measured value where a
  cell was readable and its residual is within --flag-kappa robust sigmas,
  the model everywhere else.

--------------------------------------------------------------------------------
THE CORRUPTION PROTOCOL  (one configuration = ratio x bound x noise)
--------------------------------------------------------------------------------
Omega0 is the set of cells the raw array observed.

  missing ratio p      A uniformly random share p of Omega0 is hidden (MCAR).
  (--ratios)           The rest, Omega, is what a method may read.

  noise level eta      Dense noise on every readable cell:
  (--noise-levels)     x~ = x + eta * sigma_a * N(0, 1), sigma_a the standard
                       deviation of attribute a over Omega0.

  outlier bound k      Tukey-Huber epsilon-contamination: a share
  (--outlier-bounds)   eps = --outlier-ratio (default 5 %) of Omega gets
                       x~ = x + s * m * sigma_a, s = +-1, m ~ U[k, (1 + spread) k],
                       so every impulse is at least k standard deviations
                       (by Chebyshev at most 1/k^2 of genuine readings reach
                       that).  The same cells, signs and uniforms at every k.

  evaluation           Scored against the ORIGINAL clean values (--eval-set:
                       observed, hidden or outliers).

--------------------------------------------------------------------------------
THE BUDGETS  (uniform iterations)
--------------------------------------------------------------------------------
Every method runs exactly --iters iterations (default 50), and Results 2 and 3
plot all of them on the same linear x axis 1 ... --iters.  An iteration is one
step of the method's own outer loop, sized so that --iters of them take every
method to convergence:

  M1 Deep-TD     1 alternating iteration (S-step, factor or network step,
                 per-instant solve).  The knee of its error curve is ~40-50.
  M2 DATC        1 epoch over the sliding windows (the paper's unit; it uses
                 at most 20, here 50).
  M3 DeepTensor  --dt-steps-per-iter (20) full-batch Adam steps, i.e. 1,000
                 steps in all (the paper's fits run 100s to 1000s of steps).
  M4 NeuralCP    1 epoch of mini-batches over the observed entries, capped at
                 --ncp-entries-per-iter (2^19) entries.
  M5 TRPCA-TNN   --trpca-steps-per-iter (5) ADMM steps, i.e. at most 250; it
                 stops earlier only if the paper's tolerance is met.
  M6 CoSTCo      1 epoch over the observed entries, capped at
                 --costco-entries-per-iter (2^19) entries (the paper: 50 epochs).
  M7 NTC         1 epoch over the observed entries, capped at
                 --ntc-entries-per-iter (2^17) entries.

The cap makes "epoch" mean the same thing on both datasets: an uncapped epoch
of Dataset II at 10 % missing is 1.3 x 10^7 entries.

What is returned: every method except M5 keeps the iterate with the lowest
clipped squared error on the same held-out 3 % of the readable cells
(--val-frac) - early stopping evaluated after the fact, without cutting any
curve short (CoSTCo's own protocol is early stopping on held-out entries; for
DeepTensor the stopping point is the regulariser).  No ground truth is read.
M5 is convex and returns its last iterate.

--------------------------------------------------------------------------------
OUTPUTS  (under --out)
--------------------------------------------------------------------------------
  result1_nrmse_tables.md          Result 1.  One table per dataset (and per
                                   noise level): rows = outlier bound x missing
                                   ratio, columns = M1 ... M7, NRMSE averaged
                                   over the attributes on --eval-set, best of
                                   every row in bold.
  result2_convergence_all.png      Result 2.  Grid: one row per dataset, one
                                   column per missing ratio; every panel draws
                                   every method's NRMSE against iterations at
                                   --conv-bound / --conv-noise (default: the
                                   first of each).
  result3_convergence_deeptd.png   Result 3.  Grid: one row per dataset, one
                                   column per outlier bound; every panel draws
                                   Deep-TD's NRMSE against iterations, one line
                                   per missing ratio, at --conv-noise.
  result4_runtime_memory.png       Result 4.  Bar charts, methods on the x
                                   axis, one bar per dataset: (a) runtime of
                                   one fit, (b) peak CPU memory of one fit (RSS
                                   above the job's start), (c) peak GPU memory
                                   when the fits ran on a GPU.
  Every panel of every figure labels both its axes.
  Plumbing: run.log, environment.json, and the job cache
  <out>/<dataset tag>/raw/<job>.json that makes an interrupted run resumable.

The NRMSE of Results 2 and 3 is the Result-1 number - output rule, NRMSE per
attribute, mean over attributes - computed at the end of every iteration on a
fixed random sample of --trace-cells (10^6) evaluation cells, the same sample
for every method; the large marker at the end of each line is the exact
Result-1 value of the estimate the fit returned.  Only the jobs a figure draws
record a curve (all methods at the convergence scenario, Deep-TD at every
outlier bound of the convergence noise level), and the time the curve takes is
subtracted from the runtime Result 4 reports; Result 4 averages over untraced
jobs whenever there are any.

--------------------------------------------------------------------------------
RUN
--------------------------------------------------------------------------------
    pip install -r requirements.txt

    # 0. smoke test: tiny budgets, truncated time axes, every output appears
    python3 paper-code.py --quick --out results_quick

    # 1. paper run: 2 datasets x 7 methods x 5 ratios x 3 bounds x 3 noise levels
    python3 paper-code.py \
        --data tao_mode3_LxAxT.npy,airquality_LxAxT.npy \
        --device cpu --out results_paper \
        --methods dltd,datc,deeptensor,neuralcp,trpca,costco,ntc \
        --ratios 0.1,0.2,0.4,0.6,0.8 --outlier-bounds 5,10,20 \
        --noise-levels 0.05,0.1,0.5 \
        --attrs "SST,AIRT,RH,T125,T200;SO2,CO,O3,NO2,PM10,PM25" \
        --t-limit 0,8760 --eval-set observed \
        --jobs 6 --threads-per-job 4

    # redraw the four results from the cache, no recomputation (same flags)
    python3 paper-code.py --stage results ...

Per-dataset flags: --attrs and --attr-names take one group per dataset
separated by ';'; --t-limit and --slots-per-day one number per dataset
separated by ','.  Cached rows carry a schema number, the attribute list, the
tensor shape and --iters; rows that do not match are recomputed, not mixed in.
Rows written by earlier revisions (schema < 9) are therefore all recomputed.

--------------------------------------------------------------------------------
WHEN THE DEVICE IS FULL
--------------------------------------------------------------------------------
On a MIG-partitioned GPU that another tenant has mostly filled, the CUDA
allocator fails with an NVML internal assertion instead of "out of memory".
Such a job is re-run on the CPU, its row records `device_fallback`, and Result
4 leaves it out (--no-cpu-fallback lets it fail instead); `main` names every
missing configuration and exits 2.  On a GPU the default is one worker
(--gpu-jobs 1): processes sharing a device time-slice it.

--------------------------------------------------------------------------------
PRIOR WEIGHT - the one number that matters most in M1
--------------------------------------------------------------------------------
The temporal solve is a ridge system per instant in which the TCN's prediction
enters as a Gaussian prior of weight W = sigma^2 / v (sigma^2 the residual
variance on the trusted cells, v the network's prediction error per
coefficient), capped at --prior-cap times the mean observation density.  On
TAO the measured W is far above the cap, so the cap is the prior weight.
Measured on the real tensor (T = 3,000, the TCN replaced by a two-point
average, which is a floor on what a trained network does), pooled NRMSE:

    cap        0.5     1.0     2.0     4.0     8.0    16.0    32.0
     0%     0.0256  0.0227  0.0211  0.0201  0.0199  0.0199  0.0200
    20%     0.1850  0.1732  0.1657  0.1605  0.1574  0.1557  0.1550
    40%     0.2930  0.2750  0.2640  0.2572  0.2531  0.2511  0.2503
    60%     0.4516  0.4164  0.3968  0.3878  0.3846  0.3839  0.3838
    80%     0.6664  0.6212  0.5887  0.5732  0.5701  0.5714  0.5732

Eight is the default (an interior optimum at 80 %).

--------------------------------------------------------------------------------
RANKS AND ITERATIONS - where the other M1 defaults come from
--------------------------------------------------------------------------------
Same probe, pooled NRMSE.  --ranks first entry against --rank-obs-frac:

              R1=24                  R1=32                  R1=40
    frac   0 %    40 %   80 %     0 %    40 %   80 %     0 %    40 %   80 %
    0.40 0.0194 0.2539 0.5639   0.0161 0.2438 0.5389   0.0157 0.2412 0.5355
    0.50 0.0198 0.2438 0.5565   0.0153 0.2246 0.5461   0.0139 0.2148 0.5429
    0.60 0.0194 0.3441 0.5407   0.0131 0.3525 0.5466   0.0151 0.2609 0.5433
    0.70 0.0176 0.3432 0.5291   0.0237 0.3658 0.5438   0.0180 0.3725 0.5441

`--ranks 0.85x1.0x0` (shares of each mode; 40 x 5 on TAO) and
`--rank-obs-frac 0.5` priced against the first quartile of the per-instant
observation counts.  --spike-kappa 6, --spatial-iters 2.  Iterations:

    iteration      12      18      24      32      40      48      64      80
        0 %    0.0144  0.0144  0.0135  0.0134  0.0134  0.0134  0.0134  0.0134
       40 %    0.2281  0.2143  0.2086  0.2051  0.2035  0.2026  0.2018  0.2015
       60 %    0.4000  0.3622  0.3450  0.3334  0.3274  0.3240  0.3209  0.3196
       80 %    0.6103  0.5675  0.5457  0.5296  0.5203  0.5145  0.5078  0.5045

so the common budget of 50 sits at M1's knee at every ratio.  Every number on
this page was read off the ground truth on one seed and one 3,000-instant
window; the defaults are taken from the shape of these surfaces, not their
exact optima.

--------------------------------------------------------------------------------
M1 RUNTIME AND MEMORY  (what this revision changed, all exact rewrites)
--------------------------------------------------------------------------------
The previous run measured M1 at 6.4 min / 627 MB per fit on Dataset I and
8.6 min / 1,568 MB on Dataset II, the most memory of any method.  The changes,
none of which alters what is computed:

  * the two temporal networks (forward and backward) run as one *grouped*
    network - one convolution call per layer instead of two, the same
    parameters and the same Adam updates (:class:`TCN`).  The TCN is the
    largest cost on Dataset I;
  * the per-instant Gram matrices are symmetric, so only their lower
    triangles are formed: the cached outer-product table and the GEMM that
    dominates the solve on Dataset II (T P R3^2) both shrink by ~half;
  * the three full-size masks are bool instead of float32 (a quarter of the
    bytes each); the sparse term S is no longer stored (it is non-zero only
    where the working mask is zero, so it never enters the solve, the working
    tensor or the noise estimate); the solve's right-hand side is formed per
    chunk instead of as a full-size temporary; the model is written into one
    pre-allocated buffer, and the selected iterate is copied into one buffer
    allocated once, instead of a new tensor every iteration;
  * the noise variance of the prior weight is a by-product of the reject step
    (it used to be three more passes over the tensor), and the held-out loss
    is a gather of the 3 % held-out cells instead of a pass over all cells;
  * the relative-change diagnostic, which kept a second copy of the model
    alive every iteration and was not drawn, is gone;
  * standardise / destandardise / the validation draw work in float32 (they
    promoted whole tensors to float64 temporaries before);
  * the convergence curve reads the model at the sampled cells only, instead
    of destandardising and re-scoring the whole tensor every iteration.

--------------------------------------------------------------------------------
COMPLEXITY OF M1  (Deep-TD)
--------------------------------------------------------------------------------
Symbols: L, A, T the mode sizes, P = L A, N = P T, R1 x R2 x R3 the
multilinear rank, Q = R1 R2, K = --iters, Kw = --spatial-iters, S =
--tcn-steps, B blocks, C channels and k kernel width of the TCN, D = 2
directions.

  stage                            runs    time                            space
  ----------------------------------------------------------------------------------
  standardise, layout, ranks       once    O(N + T log T)                  O(N)
  reject step + noise variance     K       O(N)                            O(N)
  factor step (HOSVD)              Kw      O(L^2 A T + Q^2 T + P R3^2)     O(N + P R3^2 / 2)
  network step, S Adam steps       K - Kw  O(D S (B T C^2 k + T C R3))     O(D (B T C + G))
  prior and its weight             K - Kw  O(D B T C^2 k + T R3)           O(D T R3)
  u3 solve, batched Cholesky       K       O(T P R3^2 / 2 + T R3^3)        O(c R3^2 + T R3)
  completion L = U3 Phi^T          K       O(T P R3)                       O(N)
  held-out loss                    K       O(|held|)                       O(|held|)
  residual stage (AR(1) bridge)    once    O(N)                            O(N)

    TC = O( K T (D S B C^2 k + L A R3^2) )      SC = O( N + P R3^2 + D B C T )

Both are linear in T.  R3 is chosen by the data (capped by --max-rank3), so as
the missing ratio rises the per-instant systems shrink and the dominant solve
term falls quadratically.

All seven methods, whole fit, in the same symbols (E = entries per iteration,
R_x = that method's rank, H = hidden width, C_x = channels, w = DATC window):

  method          time, whole fit                              space
  ------------------------------------------------------------------------------
  M1 Deep-TD      O(K T (D S B C^2 k + L A R3^2))              O(N)
  M2 DATC         O(K (T / w) L A w C_d^2 k^3)                  O(N)
  M3 DeepTensor   O(K s (N R_dt + (L + A + T) W^2))             O(N + T W)
  M4 NeuralCP     O(K E R_cp H + N R_cp H)                      O(N + (L+A+T) R_cp)
  M5 TRPCA-TNN    O(K s (N log T + T L A^2))                    O(N)
  M6 CoSTCo       O(K E R_c C_c^2 + N R_c C_c^2)                O(N + (L+A+T) R_c)
  M7 NTC          O(K E R^3 C_n + N R^3 C_n)                    O(N + (L+A+T) R)

(s = steps per iteration.)  M4, M6 and M7 pay a term in N once, for the
prediction of the whole tensor; M7's per-entry R^3 C_n is what sets its rank.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# One BLAS thread per process by default: the worker pool re-sets this to the
# planned value as it starts, and a default of "all cores" here would have every
# forked worker fighting for the whole machine.  These have to be set before
# torch and numpy pick up their backends, hence the position.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("dltd")


# ==============================================================================
# 0.  Constants of the record
# ==============================================================================

#: mode-2 labels of the tensor being run.  Rebound by :func:`apply_attributes`
#: for every dataset, in the main process and in every worker, so every
#: function reads it at call time.
DEFAULT_ATTRS: Tuple[str, ...] = ("SST", "AIRT", "RH", "T125", "T200")   # T125/T200 = sea temperature at 125 m / 200 m
ATTRS: Tuple[str, ...] = DEFAULT_ATTRS

#: What the script knows about a tensor from its file name alone, so that the
#: two datasets of the paper run with no per-dataset flags.  ``key`` is matched
#: against the lower-cased file stem.  ``slots_per_day`` is how many samples of
#: mode 3 make one day - the period of the two diurnal channels M1's temporal
#: network is given (ten-minute sampling on TAO, hourly on Air Quality).
DATASET_PROFILES: Tuple[Dict[str, object], ...] = (
    {"key": "tao", "title": "TAO/TRITON moorings",
     "attr_names": ("SST", "AIRT", "RH", "T125", "T200"), "slots_per_day": 144},
    {"key": "air", "title": "Korean Air Quality",
     "attr_names": ("SO2", "CO", "O3", "NO2", "PM10", "PM25"), "slots_per_day": 24},
)
#: the tensors looked for next to this file when --data is not given, in order
DATA_NAMES: Tuple[str, ...] = ("tao_mode3_LxAxT.npy", "airquality_LxAxT.npy")
ROMAN: Tuple[str, ...] = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                          "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII")

#: the evaluation sets every job is scored on; see THE CORRUPTION PROTOCOL
EVAL_SETS: Tuple[str, ...] = ("observed", "hidden", "outliers")

#: the per-attribute measures every job records
METRICS: Tuple[str, ...] = ("nrmse", "nrmse_sd", "rmse", "mae", "r2")

#: the missing ratios swept by default: the share of observed cells
#: additionally hidden.  It starts at 10 % because at 0 % nothing is hidden and
#: every method scores zero by construction.
RATIOS: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8)

#: quantile of the per-instant observation counts that the temporal rank is
#: priced against (see :meth:`TuckerTCN._ranks`).  The first quartile rather
#: than the median, so that the "observations per coefficient" ratio the budget
#: assumes holds for three quarters of the instants instead of half of them.
RANK_QUANTILE: float = 0.25

#: the main comparison, run by default
MAIN_METHODS: Tuple[str, ...] = ("dltd", "datc", "deeptensor", "neuralcp", "trpca",
                                 "costco", "ntc")
#: the ablation ladder; every one runs through M1's own code path
ABLATIONS: Tuple[str, ...] = ("dltd_core", "td", "td_res", "cp", "cp_tcn")
METHODS: Tuple[str, ...] = MAIN_METHODS + ABLATIONS

METHOD_LABEL: Dict[str, str] = {
    "dltd": "Deep-TD",
    "datc": "DATC",
    "deeptensor": "DeepTensor",
    "neuralcp": "NeuralCP",
    "trpca": "TRPCA-TNN",
    "costco": "CoSTCo",
    "ntc": "NTC",
    "dltd_core": "TD+TCN (no residual stage)",
    "td": "Pure TD",
    "td_res": "TD + residual stage",
    "cp": "Pure CP",
    "cp_tcn": "CP+TCN",
}

#: Results 2 and 3, x axis: what one of the --iters iterations is in each
#: method.  See THE BUDGETS in the header.
ITERATION_UNIT: Dict[str, str] = {
    "dltd": "alternating iteration",
    "datc": "epoch over the windows",
    "deeptensor": "block of Adam steps",
    "neuralcp": "capped epoch",
    "trpca": "block of ADMM steps",
    "costco": "capped epoch",
    "ntc": "capped epoch",
    "dltd_core": "alternating iteration",
    "td": "alternating iteration",
    "td_res": "alternating iteration",
    "cp": "alternating iteration",
    "cp_tcn": "alternating iteration",
}

# Categorical slots of the reference palette, in that fixed order.  A colour
# belongs to a method (its position in METHODS) or to a dataset (its position in
# --data), never to a rank, so it does not move when the sweep changes.  The
# order is the colour-vision mechanism, not decoration: every adjacent pair
# clears the CVD and normal-vision gates on the light surface.  Three of the
# first five sit below 3:1 contrast, so identity never rests on colour alone:
# the bars carry hatches and printed values, the lines carry dash patterns and
# markers, and Result 1 ships the numbers as a table.
C_SERIES: Tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                              "#7a5bd1", "#6b6a64", "#3fa7c4", "#b8862b", "#c2477a")
C_EDGE: Tuple[str, ...] = ("#184f95", "#a63c17", "#0e7250", "#996800", "#b44a73",
                           "#4f3699", "#3d3c38", "#1f6f85", "#7d5a17", "#862a52")
HATCHES: Tuple[str, ...] = ("", "///", "...", "\\\\\\", "xxx", "--", "++", "oo", "||", "**")
DASHES: Tuple[object, ...] = ("-", (0, (5, 1.6)), (0, (1.2, 1.2)), (0, (4, 1.4, 1, 1.4)),
                              (0, (7, 2)), (0, (3, 1, 1, 1, 1, 1)), (0, (2, 2)),
                              (0, (6, 1.5, 2, 1.5)), (0, (1, 2.5)), (0, (8, 1.5, 1, 1.5)))
MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK_2 = "#52514e"
C_GRID = "#dcdbd6"


# ==============================================================================
# 1.  Configuration
# ==============================================================================

@dataclass
class Config:
    """Every knob.  Populated from the command line; see ``--help``."""

    # ---- plumbing ------------------------------------------------------------
    data: str = ""                     # comma-separated .npy files; "" = look for
                                       # DATA_NAMES next to this file
    out: str = "results"
    stage: str = "all"                 # all | run | results
    seed: int = 0
    jobs: int = 0                      # 0 = size the pool from the machine
    threads_per_job: int = 0           # 0 = size it from the machine
    device: str = "auto"               # auto | cuda | cpu
    gpu_index: int = 0                 # index into the *visible* CUDA devices
    gpu_jobs: int = 1                  # 1 = the device to itself; 0 = from memory
    gpu_job_gb: float = 2.5            # device memory reserved per concurrent job
    cudnn_benchmark: bool = False      # let cuDNN autotune its convolutions
    resume: bool = True                # skip jobs already in the cache
    quick: bool = False                # smoke-test budgets
    methods: Tuple[str, ...] = MAIN_METHODS
    ratios: Tuple[float, ...] = RATIOS

    # ---- per-dataset arguments, as typed (one group per --data entry) ---------
    attr_names_arg: str = ""           # ';'-separated groups of mode-2 labels
    attrs_arg: str = ""                # ';'-separated groups of labels to keep
    t_limit_arg: str = "0"             # ','-separated instants to keep (0 = all)
    slots_arg: str = ""                # ','-separated samples per day

    # ---- the dataset in hand: filled by :func:`dataset_config`, one copy of the
    #      configuration per dataset, and that copy is what a worker receives ----
    dataset: str = ""                  # tag of the dataset, also its cache folder
    attr_names: Tuple[str, ...] = DEFAULT_ATTRS   # labels of mode 2, file order
    attrs: Tuple[str, ...] = ()        # subset of attr_names to keep (() = all)
    t_limit: int = 0                   # 0 = whole time axis
    slots_per_day: int = 144           # samples of mode 3 in one day (0 = unknown)
    shape: Tuple[int, ...] = ()        # (L, A, T) after --attrs and --t-limit

    # ---- corruption protocol (see THE CORRUPTION PROTOCOL in the header) ------
    outlier_bounds: Tuple[float, ...] = (0.0, 5.0, 10.0)   # Chebyshev k; 0 = none
    outlier_ratio: float = 0.05        # epsilon: share of the readable cells hit
    outlier_spread: float = 1.0        # magnitudes ~ U[k, (1 + spread) k] sigma_a
    noise_levels: Tuple[float, ...] = (0.0,)               # eta, in sigma_a units

    # ---- scoring -------------------------------------------------------------
    flag_kappa: float = 4.0            # robust sigmas of the attribute before a
                                       # measurement is treated as a spike
                                       # (0 = score the bare model, no copy-through)
    eval_set: str = "observed"         # which evaluation set the results show

    # ---- Results 2 and 3 -----------------------------------------------------
    conv_bound: float = -1.0           # outlier bound of Result 2's grid
    conv_noise: float = -1.0           # noise level of Results 2 and 3
                                       # (-1 = the first listed)
    trace_all: bool = False            # record the NRMSE curve in every job
    trace_cells: int = 1 << 20         # evaluation cells the curve is computed on
                                       # (0 = all of them)

    # ---- budgets (see THE BUDGETS in the header) -------------------------------
    iters: int = 50                    # iterations of EVERY method (x axis of
                                       # Results 2 and 3)
    val_frac: float = 0.03             # share of the readable cells held out of
                                       # every fit to pick its returned iterate
                                       # (0 = off: the last iterate is returned)
    val_clip: float = 3.0              # clip on a standardised validation residual,
                                       # so a corrupted held-out cell cannot vote

    # ---- M1  Deep-TD ----------------------------------------------------
    # R1 and R2 are shares of their own mode when <= 1, counts when > 1, and
    # the whole mode when 0.  Written as shares so that no rank in this file is
    # a number that only means something on a 47-mooring array.
    ranks: Tuple[float, float, float] = (0.85, 1.0, 0.0)
    rank_obs_frac: float = 0.5         # R3 budget, as observations-per-coefficient
                                       # at the RANK_QUANTILE instant: 0.5 is two
                                       # observations per coefficient there
    min_rank3: int = 8
    max_rank3: int = 96                # ceiling on R3 (0 = none).  The solve costs
                                       # T L A R3^2, so on an array with thousands
                                       # of channels the data-driven budget alone
                                       # is not affordable; 96 is above anything
                                       # that budget produces on TAO.
    spatial_iters: int = 2             # of the --iters iterations, how many before the factors freeze
    clip_sigma: float = 6.0            # hard cap that seeds the sparse term
    spike_kappa: float = 6.0           # robust residual sigmas before a cell is
                                       # trimmed out of the fit and moved into S.
                                       # 6 costs 4 % less error than 3 did; see
                                       # RANKS AND ITERATIONS in the header.
    lambda_prior: float = 0.0          # 0 = the empirical-Bayes weight, measured
    prior_cap: float = 8.0             # ceiling on that weight, as a multiple of
                                       # what a typical instant's data contributes.
                                       # On TAO the measured weight sits far above
                                       # this, so the cap *is* the prior weight in
                                       # practice; see PRIOR WEIGHT in the header.
    lambda_ridge: float = 1e-3
    tcn_channels: int = 32
    tcn_blocks: int = 8
    tcn_kernel: int = 3
    tcn_dropout: float = 0.05
    tcn_steps: int = 20                # network steps per alternating iteration
    tcn_warmup: int = 3                # multiplier on the first of them
    bidirectional: bool = True         # a second TCN reading the sequence backwards
    min_count_frac: float = 0.5        # share of the median count a TCN target needs
    lr: float = 1e-3
    solve_chunk: int = 0               # instants per batched solve (0 = from R3)
    innov_kappa: float = 6.0           # robust sigmas of the leave-one-out
                                       # innovation before a cell is an outlier
    ar_max_phi: float = 0.999          # cap on the residual AR(1) coefficient
    cp_iters: int = 10                 # CP-ALS sweeps per basis refresh (cp, cp_tcn)

    # ---- M2  DATC  (paper values: w = 12, n = 2, width 4, lambda 0.1, 1e-3) ----
    datc_window: int = 12
    datc_stride: int = 0               # window stride within an epoch (0 = w)
    datc_units: int = 2                # down-sampling units n on the encoder side
    datc_width: int = 4                # kernels in the first convolution
    datc_batch: int = 16               # sub-tensors per update
    datc_lr: float = 1e-3
    datc_lambda: float = 0.1           # weight of the adversarial term, eq. (9)

    # ---- M3  DeepTensor ------------------------------------------------------
    dt_rank: int = 20
    dt_width: int = 64                 # channels of the DIP networks
    dt_levels: int = 5                 # scales of the DIP encoder-decoder
    dt_skip: int = 4                   # channels of each skip branch
    dt_input: int = 32                 # channels of the fixed noise code
    dt_steps_per_iter: int = 20        # Adam steps in one iteration
    dt_lr: float = 1e-3
    dt_loss: str = "l2"                # l2 (eq. 8) or l1 (sparse-noise variant)

    # ---- M4  NeuralCP --------------------------------------------------------
    # A job killed by a device allocation failure is re-run on the CPU rather
    # than left as a hole in the figure; see :func:`_run_job`.
    cpu_fallback: bool = True

    ncp_rank: int = 50                 # R, the paper's real-data value
    ncp_hidden: int = 100              # K in eq. (4), the paper's real-data value
    ncp_entries_per_iter: int = 1 << 19
    ncp_batch: int = 4096
    ncp_lr: float = 1e-3
    ncp_kl: float = 1.0                # weight on the KL term of eq. (9)

    # ---- M5  TRPCA-TNN  (Algorithm 4's constants) -----------------------------
    trpca_steps_per_iter: int = 5      # ADMM steps in one iteration
    trpca_rho: float = 1.1
    trpca_mu0: float = 1e-3
    trpca_mu_max: float = 1e10
    trpca_tol: float = 1e-8

    # ---- M6  CoSTCo ------------------------------------------------------------
    costco_rank: int = 20
    costco_channels: int = 0           # nc (0 = 2 x rank, the paper's setting
                                       # for sensor tensors)
    costco_entries_per_iter: int = 1 << 19
    costco_batch: int = 1024
    costco_lr: float = 1e-3

    # ---- M7  NTC ----------------------------------------------------------------
    ntc_rank: int = 8                  # R, a power of two (paper: 16 or 32)
    ntc_kernels: int = 32              # T, kernels per convolutional layer
    ntc_entries_per_iter: int = 1 << 17
    ntc_batch: int = 512
    ntc_lr: float = 1e-3
    ntc_reg: float = 1e-4              # alpha = beta = gamma of eq. (4)

    def quicken(self) -> None:
        """Shrink every budget so the whole run finishes in a few minutes."""
        self.iters = 6
        self.spatial_iters = 2
        self.tcn_blocks = 4
        self.tcn_steps = 4
        self.tcn_warmup = 2
        self.max_rank3 = min(self.max_rank3 or 32, 32)
        self.dt_steps_per_iter = 5
        self.dt_levels = 2
        self.ncp_entries_per_iter = 1 << 14
        self.costco_entries_per_iter = 1 << 14
        self.ntc_entries_per_iter = 1 << 12
        self.trpca_steps_per_iter = 2
        self.cp_iters = 4
        self.trace_cells = 1 << 16


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(
        prog="paper-code.py",
        description="Robust tensor completion, seven methods, one or more "
                    "datasets; four results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("plumbing")
    g.add_argument("--data", default=d.data,
                   help="comma-separated .npy tensors in (L, A, T) layout, run back "
                        "to back as Dataset I, II, ... (default: "
                        + " and ".join(DATA_NAMES) + " next to this script)")
    g.add_argument("--out", default=d.out, help="output directory")
    g.add_argument("--stage", default=d.stage, choices=("all", "run", "results"))
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--jobs", type=int, default=d.jobs,
                   help="worker processes (0 = size from the machine)")
    g.add_argument("--threads-per-job", type=int, default=d.threads_per_job,
                   help="torch threads inside each worker (0 = size from the machine)")
    g.add_argument("--device", choices=("auto", "cuda", "cpu"), default=d.device,
                   help="compute device; auto takes CUDA when it is available")
    g.add_argument("--gpu-index", type=int, default=d.gpu_index,
                   help="which visible CUDA device to use; under MIG this is an "
                        "index into CUDA_VISIBLE_DEVICES, not the GPU number")
    g.add_argument("--gpu-jobs", type=int, default=d.gpu_jobs,
                   help="worker processes sharing the GPU; 1, the default, gives "
                        "each job the device to itself (0 = size from free memory)")
    g.add_argument("--gpu-job-gb", type=float, default=d.gpu_job_gb,
                   help="device memory to reserve for each concurrent GPU job")
    g.add_argument("--cudnn-benchmark", action="store_true",
                   default=d.cudnn_benchmark,
                   help="let cuDNN autotune its convolutions; worth it on a "
                        "whole GPU, not on a part-full MIG slice")
    g.add_argument("--no-resume", dest="resume", action="store_false", default=d.resume,
                   help="recompute jobs already present in the cache")
    g.add_argument("--quick", action="store_true", help="smoke-test budgets")
    g.add_argument("--methods", default=",".join(d.methods),
                   help="comma-separated subset of " + ",".join(METHODS)
                        + "; the order given is M1, M2, ... in the results")
    g.add_argument("--ratios", default=",".join(f"{r:g}" for r in d.ratios),
                   help="comma-separated missing ratios, the share of the "
                        "observed cells to hide")

    g = p.add_argument_group("per-dataset arguments (one group per --data entry)")
    g.add_argument("--attr-names", dest="attr_names_arg", default=d.attr_names_arg,
                   help="labels of mode 2 of each .npy, in file order; groups "
                        "separated by ';', e.g. \"SST,AIRT,RH,T125,T200;SO2,CO,"
                        "O3,NO2,PM10,PM25\" (default: the file's own profile)")
    g.add_argument("--attrs", dest="attrs_arg", default=d.attrs_arg,
                   help="attributes to keep, a subset of --attr-names; groups "
                        "separated by ';'; a missing or empty group keeps all")
    g.add_argument("--t-limit", dest="t_limit_arg", default=d.t_limit_arg,
                   help="truncate the time axis to this many instants (0 = all); "
                        "one value for every dataset, or one per dataset "
                        "separated by ','")
    g.add_argument("--slots-per-day", dest="slots_arg", default=d.slots_arg,
                   help="samples of the time axis in one day, the period of the "
                        "diurnal channels M1's temporal network is given; one "
                        "per dataset separated by ',' (default: the file's own "
                        "profile; 0 = no diurnal information)")

    g = p.add_argument_group("corruption protocol")
    g.add_argument("--outlier-bounds", default=",".join(f"{v:g}" for v in d.outlier_bounds),
                   help="Chebyshev outlier bounds k to sweep (impulses of at least "
                        "k standard deviations); 0 = no outliers")
    g.add_argument("--outlier-ratio", type=float, default=d.outlier_ratio,
                   help="epsilon of the contamination model: share of the readable "
                        "cells that receive an impulse")
    g.add_argument("--outlier-spread", type=float, default=d.outlier_spread,
                   help="impulse magnitudes are Uniform[k, (1 + spread) k] sigma_a")
    g.add_argument("--noise-levels", default=",".join(f"{v:g}" for v in d.noise_levels),
                   help="additive Gaussian noise levels eta to sweep, in units of "
                        "each attribute's standard deviation")

    g = p.add_argument_group("scoring and results")
    g.add_argument("--flag-kappa", type=float, default=d.flag_kappa,
                   help="robust sigmas of the attribute before a measurement is "
                        "treated as a spike; 0 scores the bare model with no "
                        "copy-through")
    g.add_argument("--raw-output", dest="flag_kappa", action="store_const", const=0.0,
                   default=argparse.SUPPRESS,
                   help="shorthand for --flag-kappa 0: score the bare models")
    g.add_argument("--eval-set", choices=EVAL_SETS, default=d.eval_set,
                   help="cells the NRMSE of Results 1 and 3 is computed on: "
                        "observed = every cell the raw array observed, hidden = "
                        "only the cells removed by the missing ratio, outliers = "
                        "only the corrupted cells")
    g.add_argument("--conv-bound", type=float, default=d.conv_bound,
                   help="outlier bound of the configurations Result 2 draws "
                        "(-1 = the first of --outlier-bounds)")
    g.add_argument("--conv-noise", type=float, default=d.conv_noise,
                   help="noise level of the configurations Results 2 and 3 draw "
                        "(-1 = the first of --noise-levels)")
    g.add_argument("--trace-all", action="store_true", default=d.trace_all,
                   help="record the NRMSE curve in every job, not only in the "
                        "configurations Results 2 and 3 draw (slower; the cost "
                        "is taken off the reported runtimes either way)")
    g.add_argument("--trace-cells", type=int, default=d.trace_cells,
                   help="evaluation cells sampled for the convergence curves "
                        "(0 = all)")

    g = p.add_argument_group("budgets")
    g.add_argument("--iters", type=int, default=d.iters,
                   help="iterations of every method; see THE BUDGETS")
    g.add_argument("--val-frac", type=float, default=d.val_frac,
                   help="share of the readable cells held out of every fit to "
                        "pick the iterate it returns; 0 = the last iterate")

    g = p.add_argument_group("M1  Deep-TD")
    g.add_argument("--ranks", default="x".join(map(str, d.ranks)),
                   help="R1xR2xR3.  Each entry is a share of its mode when it is "
                        "<= 1, a number of directions when > 1, and the whole "
                        "mode when 0; 0 for R3 means R1*R2, a full-resolution "
                        "time mode")
    g.add_argument("--rank-obs-frac", type=float, default=d.rank_obs_frac,
                   help="R3 budget as a share of the first-quartile per-instant "
                        "observation count; 0.5 is two observations per "
                        "coefficient at that instant")
    g.add_argument("--max-rank3", type=int, default=d.max_rank3,
                   help="ceiling on R3 (0 = none); bounds the T*L*A*R3^2 cost of "
                        "the per-instant solve on arrays with many channels")
    g.add_argument("--spatial-iters", type=int, default=d.spatial_iters,
                   help="warm-up iterations before U1, U2 and the core freeze")
    g.add_argument("--tcn-steps", type=int, default=d.tcn_steps)
    g.add_argument("--spike-kappa", type=float, default=d.spike_kappa)
    g.add_argument("--lambda-prior", type=float, default=d.lambda_prior,
                   help="weight of the TCN prior (0 = measured, empirical Bayes)")
    g.add_argument("--prior-cap", type=float, default=d.prior_cap,
                   help="ceiling on that weight, as a multiple of what a typical "
                        "instant's measurements are worth; on this data it binds "
                        "at every iteration, so it is the prior's weight")
    g.add_argument("--one-directional", dest="bidirectional", action="store_false",
                   default=d.bidirectional,
                   help="drop the backward TCN and use the causal one alone")
    g.add_argument("--innov-kappa", type=float, default=d.innov_kappa,
                   help="robust sigmas of the leave-one-out innovation before M1 "
                        "declares an available cell an outlier")
    g.add_argument("--clip-sigma", type=float, default=d.clip_sigma,
                   help="standardised level above which a cell starts in S")
    g.add_argument("--tcn-channels", type=int, default=d.tcn_channels)
    g.add_argument("--tcn-blocks", type=int, default=d.tcn_blocks)
    g.add_argument("--cp-iters", type=int, default=d.cp_iters,
                   help="CP-ALS sweeps per basis refresh in the cp / cp_tcn ablations")

    g = p.add_argument_group("M2  DATC")
    g.add_argument("--datc-window", type=int, default=d.datc_window)
    g.add_argument("--datc-stride", type=int, default=d.datc_stride,
                   help="stride of the training windows within an epoch (0 = the "
                        "window; 1 = every window of the paper, every epoch)")
    g.add_argument("--datc-batch", type=int, default=d.datc_batch)
    g.add_argument("--datc-lambda", type=float, default=d.datc_lambda)

    g = p.add_argument_group("M3  DeepTensor")
    g.add_argument("--dt-rank", type=int, default=d.dt_rank)
    g.add_argument("--dt-width", type=int, default=d.dt_width)
    g.add_argument("--dt-steps-per-iter", type=int, default=d.dt_steps_per_iter)
    g.add_argument("--dt-loss", choices=("l2", "l1"), default=d.dt_loss)

    g = p.add_argument_group("M4  NeuralCP")
    g.add_argument("--no-cpu-fallback", dest="cpu_fallback",
                   action="store_false", default=d.cpu_fallback,
                   help="let a job that exhausts the device fail, instead of "
                        "re-running it on the CPU and marking the row")
    g.add_argument("--ncp-rank", type=int, default=d.ncp_rank)
    g.add_argument("--ncp-hidden", type=int, default=d.ncp_hidden)
    g.add_argument("--ncp-entries-per-iter", type=int, default=d.ncp_entries_per_iter)
    g.add_argument("--ncp-batch", type=int, default=d.ncp_batch)

    g = p.add_argument_group("M5  TRPCA-TNN")
    g.add_argument("--trpca-steps-per-iter", type=int, default=d.trpca_steps_per_iter,
                   help="ADMM steps in one iteration")

    g = p.add_argument_group("M6  CoSTCo")
    g.add_argument("--costco-rank", type=int, default=d.costco_rank)
    g.add_argument("--costco-channels", type=int, default=d.costco_channels)
    g.add_argument("--costco-entries-per-iter", type=int,
                   default=d.costco_entries_per_iter)
    g.add_argument("--costco-batch", type=int, default=d.costco_batch)

    g = p.add_argument_group("M7  NTC")
    g.add_argument("--ntc-rank", type=int, default=d.ntc_rank,
                   help="embedding size R, a power of two (paper: 16 or 32)")
    g.add_argument("--ntc-kernels", type=int, default=d.ntc_kernels)
    g.add_argument("--ntc-entries-per-iter", type=int, default=d.ntc_entries_per_iter)
    g.add_argument("--ntc-batch", type=int, default=d.ntc_batch)
    return p


def config_from_args(argv: Optional[Sequence[str]] = None) -> Config:
    args = build_parser().parse_args(argv)
    cfg = Config()
    listed = ("ranks", "methods", "ratios", "outlier_bounds", "noise_levels")
    for key, value in vars(args).items():
        if key in listed:
            continue
        setattr(cfg, key, value)
    cfg.outlier_bounds = tuple(float(v) for v in args.outlier_bounds.split(",")
                               if v.strip()) or (0.0,)
    cfg.noise_levels = tuple(float(v) for v in args.noise_levels.split(",")
                             if v.strip()) or (0.0,)
    if any(v < 0 for v in cfg.outlier_bounds + cfg.noise_levels):
        raise SystemExit("--outlier-bounds and --noise-levels must be >= 0")
    if not 0.0 <= cfg.outlier_ratio < 1.0:
        raise SystemExit("--outlier-ratio wants a share in [0, 1)")
    if not 0.0 <= cfg.val_frac < 0.5:
        raise SystemExit("--val-frac wants a share in [0, 0.5)")
    # "40x5x0" and "0.85x1.0x0" both parse; see TuckerTCN._mode_rank for what
    # each form means.
    cfg.ranks = tuple(float(v) for v in args.ranks.lower().split("x"))    # type: ignore[assignment]
    if len(cfg.ranks) != 3:
        raise SystemExit("--ranks wants three numbers, e.g. 10x3x20")
    chosen = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = [m for m in chosen if m not in METHODS]
    if unknown:
        raise SystemExit("unknown method(s): " + ", ".join(unknown))
    if len(set(chosen)) != len(chosen) or not chosen:
        raise SystemExit("--methods wants each method once")
    cfg.methods = chosen
    cfg.ratios = tuple(float(v) for v in args.ratios.split(",") if v.strip())
    if not cfg.ratios or any(not 0.0 <= r < 1.0 for r in cfg.ratios):
        raise SystemExit("--ratios wants shares in [0, 1), e.g. 0.1,0.2,0.4,0.6,0.8")
    for name, wanted, swept in (("--conv-bound", cfg.conv_bound, cfg.outlier_bounds),
                                ("--conv-noise", cfg.conv_noise, cfg.noise_levels)):
        if wanted >= 0 and not any(abs(wanted - v) < 1e-9 for v in swept):
            raise SystemExit(f"{name} {wanted:g} is not one of the values swept "
                             f"({', '.join(f'{v:g}' for v in swept)})")
    if cfg.quick:
        cfg.quicken()
    return cfg


def pct(ratio: float) -> int:
    """A missing ratio as a whole percentage, the key every table is indexed by."""
    return int(round(float(ratio) * 100))


# ---- scenarios: (outlier bound, noise level) pairs ----------------------------

def scenarios(cfg: Config) -> List[Tuple[float, float]]:
    """Every (outlier bound k, noise level eta) pair the sweep runs, in order."""
    return [(float(k), float(eta)) for k in cfg.outlier_bounds
            for eta in cfg.noise_levels]


def convergence_scenario(cfg: Config) -> Tuple[float, float]:
    """The one (outlier bound, noise level) pair Result 3 is drawn for.

    Only the jobs of this pair record an NRMSE curve, so the other
    configurations do not pay for it.
    """
    bound = cfg.conv_bound if cfg.conv_bound >= 0 else cfg.outlier_bounds[0]
    noise = cfg.conv_noise if cfg.conv_noise >= 0 else cfg.noise_levels[0]
    return float(bound), float(noise)


def is_traced(cfg: Config, method: str, ratio: float, bound: float,
              noise: float) -> bool:
    """Does this job record a convergence curve?

    Result 2 draws every method at the convergence scenario; Result 3 draws
    the lead method (Deep-TD) at every outlier bound of the convergence noise
    level.  ``ratio`` is unused (every ratio is drawn) but keeps the signature
    that of a :data:`Job`.
    """
    if cfg.trace_all:
        return True
    k, eta = convergence_scenario(cfg)
    if abs(noise - eta) >= 1e-9:
        return False
    lead = "dltd" if "dltd" in cfg.methods else cfg.methods[0]
    return method == lead or abs(bound - k) < 1e-9


def scenario_tag(bound: float, noise: float) -> str:
    """File-name tag of one scenario, e.g. ``k5_n0``."""
    return f"k{bound:g}_n{noise:g}".replace(".", "p")


def scenario_of(row: Dict[str, object]) -> Tuple[float, float]:
    return (float(row.get("outlier_bound", 0.0)), float(row.get("noise_level", 0.0)))


def rows_for(rows: Sequence[Dict[str, object]], bound: float, noise: float
             ) -> List[Dict[str, object]]:
    """The cached rows of one scenario."""
    return [r for r in rows if abs(scenario_of(r)[0] - bound) < 1e-9
            and abs(scenario_of(r)[1] - noise) < 1e-9]


# ---- attributes ----------------------------------------------------------------

def kept_attribute_index(cfg: Config) -> List[int]:
    """Mode-2 indices of the attributes the run keeps, in file order."""
    if not cfg.attrs:
        return list(range(len(cfg.attr_names)))
    return [cfg.attr_names.index(a) for a in cfg.attr_names if a in cfg.attrs]


def apply_attributes(cfg: Config) -> None:
    """Rebind ATTRS to the attributes of the dataset this configuration is for.

    Called in the main process before each dataset and in every worker,
    because a spawned worker imports this module afresh and would otherwise
    see the defaults.
    """
    global ATTRS
    ATTRS = tuple(cfg.attr_names[i] for i in kept_attribute_index(cfg))


# ==============================================================================
# 2.  Plumbing: logging, seeding, timing, memory, machine probe
# ==============================================================================

def setup_logging(path: Optional[Path] = None) -> None:
    """Progress goes to stderr and to a log file; the three results go to stdout."""
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a"))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname).1s %(message)s",
                        datefmt="%H:%M:%S")
    # matplotlib subsets a font for every PDF it writes and narrates all of it
    # at INFO; left alone it buries the run in several thousand glyph lines.
    for noisy in ("matplotlib", "matplotlib.font_manager", "fontTools", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def configure_device(cfg: Config) -> "torch.device":
    """Pick the compute device and make new tensors land on it by default.

    Every tensor in Sections 5 and 6 is built by a torch factory call
    (``torch.as_tensor``, ``zeros``, ``eye``, ...), and ``set_default_device``
    intercepts all of them, so this one call moves all five methods onto the
    GPU.  The only boundary that has to be crossed by hand is the far end:
    each runner hands back a NumPy array, and those five sites go through
    ``.detach().cpu().numpy()``.

    Under MIG a visible device is one *instance*, not the whole H100, so
    ``--gpu-index`` indexes ``CUDA_VISIBLE_DEVICES``.  With MIG enabled and
    that variable unset, torch enumerates nothing and this falls back to the
    CPU; ``--device cuda`` turns that silent fallback into an error.
    """
    if cfg.device in ("auto", "cuda") and torch.cuda.is_available():
        count = torch.cuda.device_count()
        if not 0 <= cfg.gpu_index < count:
            raise SystemExit(f"--gpu-index {cfg.gpu_index} is out of range; "
                             f"{count} CUDA device(s) visible")
        device = torch.device(f"cuda:{cfg.gpu_index}")
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        if torch.backends.cudnn.is_available():
            # Autotuning pays for itself only when a shape recurs often enough
            # to amortise the search, and the search itself probes algorithms
            # with large workspaces - costly on a MIG slice that is part-full
            # of somebody else's job.  Off by default; --cudnn-benchmark asks
            # for it on a whole GPU, where the TCN shape never changes.
            torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
        LOG.info("compute device: %s (%s)", device,
                 torch.cuda.get_device_name(cfg.gpu_index))
        return device
    if cfg.device == "cuda":
        raise SystemExit(
            "--device cuda was asked for but torch.cuda.is_available() is False. "
            "Under MIG, export CUDA_VISIBLE_DEVICES=MIG-<uuid> from `nvidia-smi -L`.")
    device = torch.device("cpu")
    torch.set_default_device(device)
    LOG.info("compute device: CPU")
    return device


#: Marks of a device *allocation* failure, as opposed to a bug in a method.
#: The first is what CUDA says when it can say it.  The others are what a MIG
#: instance says instead: the caching allocator builds its out-of-memory report
#: by asking NVML about the device, NVML's per-device queries are restricted on
#: a MIG compute instance, and the internal assertion therefore fires before the
#: out-of-memory message is ever constructed.  Run 4 failed this way on all five
#: M4 jobs, with 0.86 of 19.62 GB of the slice free:
#:
#:     RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
#:     "../c10/cuda/CUDACachingAllocator.cpp":995, please report a bug to PyTorch
#:
#: which is not a bug to report to PyTorch; it is this machine's slice being
#: 96 % occupied by somebody else's job.
_ALLOCATOR_MARKS: Tuple[str, ...] = (
    "out of memory",
    "CUDACachingAllocator",
    "NVML_SUCCESS",
    "all CUDA-capable devices are busy or unavailable",
    "CUDA_ERROR_OUT_OF_MEMORY",
)


def is_allocator_failure(error: BaseException) -> bool:
    """Did this exception come from the device running out of room?"""
    text = f"{type(error).__name__}: {error}"
    return any(mark in text for mark in _ALLOCATOR_MARKS)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def job_seed(base: int, ratio: float) -> int:
    """A seed that depends on the missing ratio but not on the method, so the
    thinned input is bit-identical for all five methods.  The same value rev5
    used, so the hidden cells are the ones rev5 hid."""
    return (base * 1_000_003 + int(round(ratio * 1000)) * 7919) % (2 ** 31 - 1)


def warm_up_device() -> None:
    """Load every library the five methods call, once, before any job is timed.

    CUDA loads cuBLAS, cuSOLVER, cuFFT and cuDNN lazily, on the first call that
    needs each one, and loading one costs seconds of wall clock and hundreds of
    MB of host memory.  Without this, that one-time cost lands on whichever job
    happens to call the library first and shows up in Result 2 as a property of
    that method: in rev5 the first DATC job took 25.7 s against about 16 s for
    the rest, and the first Deep-TD job - the first job of the sweep to
    call cuSOLVER, and most likely for that reason - put 543 MB on the host
    against 24 to 48 MB for the other four.

    One tiny call of each kind runs on whatever device is the default.  It is
    harmless on the CPU.  A failure here is logged and otherwise ignored: the
    run is still correct without the warm-up, only less evenly timed.
    """
    try:
        a = torch.randn(32, 32)
        gram = a @ a.transpose(0, 1) + 32.0 * torch.eye(32)            # cuBLAS
        torch.linalg.eigh(gram)                                        # cuSOLVER
        torch.linalg.eigh(gram.to(torch.complex64).unsqueeze(0).repeat(4, 1, 1))
        factor, _ = torch.linalg.cholesky_ex(gram.unsqueeze(0))
        torch.cholesky_solve(torch.ones(1, 32, 1), factor)
        torch.linalg.solve(gram, torch.ones(32, 1))
        torch.fft.irfft(torch.fft.rfft(torch.randn(4, 4, 64), dim=2), n=64, dim=2)
        x = torch.randn(1, 4, 64, requires_grad=True)                  # cuDNN
        F.conv1d(F.pad(x, (2, 0)), torch.randn(4, 4, 3)).sum().backward()
        v = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
        kernel = torch.randn(2, 1, 3, 3, 3)
        F.conv_transpose3d(F.conv3d(v, kernel, padding=1), kernel,
                           padding=1).sum().backward()
        sample = torch.rand(256)
        torch.quantile(sample, 0.25)
        sample.median()
        if a.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as error:                       # never fatal
        LOG.warning("device warm-up failed (%s: %s); the first job of each "
                    "method may carry library start-up time",
                    type(error).__name__, error)


def _rss_bytes() -> int:
    """Resident set size of this process, in bytes."""
    try:
        with open("/proc/self/statm", "r") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        try:
            import resource
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:
            return 0


class PeakMemory:
    """Sample this process's RSS while a block runs and keep the maximum.

    A sampling thread rather than tracemalloc: most of the memory these methods
    touch is allocated by BLAS and by torch's caching allocator, which Python's
    own allocator never sees.
    """

    def __init__(self, period: float = 0.02):
        self.period = period
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "PeakMemory":
        self.baseline = _rss_bytes()
        self.peak = self.baseline
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.period):
            self.peak = max(self.peak, _rss_bytes())

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak = max(self.peak, _rss_bytes())

    @property
    def delta_mb(self) -> float:
        return max(0.0, (self.peak - self.baseline)) / 2 ** 20

    @property
    def peak_mb(self) -> float:
        return self.peak / 2 ** 20


def machine(gpu_index: int = 0) -> Dict[str, object]:
    """What the run has to work with: cores, RAM, free disk, CUDA device."""
    info: Dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count() or 1,
    }
    try:
        info["affinity_cpus"] = len(os.sched_getaffinity(0))
    except Exception:
        info["affinity_cpus"] = info["logical_cpus"]
    try:
        model = [line.split(":", 1)[1].strip()
                 for line in open("/proc/cpuinfo") if line.startswith("model name")]
        info["cpu_model"] = model[0] if model else "unknown"
        cores = {line.split(":", 1)[1].strip()
                 for line in open("/proc/cpuinfo") if line.startswith("core id")}
        info["physical_cores"] = len(cores) or info["affinity_cpus"]
    except Exception:
        info["cpu_model"] = platform.processor() or "unknown"
        info["physical_cores"] = info["affinity_cpus"]
    try:
        mem = {k.strip(): v for k, v in
               (line.split(":", 1) for line in open("/proc/meminfo"))}
        info["ram_gb"] = round(int(mem["MemTotal"].split()[0]) / 2 ** 20, 1)
        info["ram_available_gb"] = round(int(mem["MemAvailable"].split()[0]) / 2 ** 20, 1)
    except Exception:
        info["ram_gb"] = info["ram_available_gb"] = -1.0
    try:
        usage = shutil.disk_usage(str(Path.cwd()))
        info["disk_free_gb"] = round(usage.free / 2 ** 30, 1)
    except Exception:
        info["disk_free_gb"] = -1.0
    # mem_get_info reports the free memory of one MIG instance when that is
    # what is visible, which is the number the GPU pool has to be sized
    # against: the instance can be most of the way full of somebody else's job.
    # Which torch built the run.  Not decoration: the failure mode in
    # :data:`_ALLOCATOR_MARKS` is a torch-version-and-MIG interaction, and run
    # 4's environment.json did not record either half of that.
    info["torch"] = getattr(torch, "__version__", "?")
    info["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    info["cuda_available"] = False
    info["cuda_free_gb"] = info["cuda_total_gb"] = -1.0
    try:
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_devices"] = [torch.cuda.get_device_name(i)
                                    for i in range(torch.cuda.device_count())]
            if 0 <= gpu_index < torch.cuda.device_count():
                free, total = torch.cuda.mem_get_info(gpu_index)
                info["cuda_free_gb"] = round(free / 2 ** 30, 2)
                info["cuda_total_gb"] = round(total / 2 ** 30, 2)
    except Exception:
        pass
    return info


def plan_pool(cfg: Config, n_jobs: int, info: Dict[str, object]) -> Tuple[int, int]:
    """Split the cores between processes and BLAS threads.

    Tensor completion on a CPU is a pile of medium-sized GEMMs and FFTs: torch
    scales to about four threads on a problem this size and then flattens, so
    the machine is used far better by running several configurations at once
    than by giving one of them every core.  RAM is the other limit - each worker
    holds its own copy of the working tensors - so the pool is also capped by
    available memory.
    """
    cores = int(info.get("affinity_cpus") or 1)
    if cfg.device != "cpu" and info.get("cuda_available"):
        # The default here is one worker, which the caller above runs in
        # process with no pool.  Processes sharing a CUDA device do not
        # partition it, they time-slice it, and one job already saturates a
        # 2g.20gb MIG slice, so a pool buys no throughput and costs M5
        # everything: a batched cuSOLVER SVD fenced by host synchronisations
        # loses its time slice on every switch.  --gpu-jobs 0 asks for the
        # memory-sized pool instead, for a whole GPU with SMs to spare: each
        # worker carries its own CUDA context plus its own copy of the working
        # tensors, which is what --gpu-job-gb reserves, and a tenth of the free
        # memory is held back so that a job whose peak runs over the
        # reservation fails to grow rather than taking a neighbour down.
        if cfg.gpu_jobs > 0:
            workers = cfg.gpu_jobs
        else:
            free = float(info.get("cuda_free_gb") or 0.0)
            workers = (max(1, int(0.9 * free / cfg.gpu_job_gb))
                       if free > 0 and cfg.gpu_job_gb > 0 else 1)
        workers = max(1, min(workers, n_jobs, cores))
        threads = cfg.threads_per_job or max(1, min(4, cores // max(1, workers)))
        return workers, threads
    if cfg.jobs > 0 and cfg.threads_per_job > 0:
        return cfg.jobs, cfg.threads_per_job
    threads = cfg.threads_per_job or max(1, min(4, cores // 4))
    jobs = cfg.jobs or max(1, cores // threads)
    ram = float(info.get("ram_available_gb") or 0.0)
    if ram > 0:
        jobs = max(1, min(jobs, int(ram // 1.5)))      # ~1.5 GB of headroom each
    jobs = max(1, min(jobs, n_jobs))
    return jobs, threads




# ==============================================================================
# 3.  The data and the experiment it is turned into
# ==============================================================================

@dataclass
class Dataset:
    """One tensor of the sweep, with everything that is specific to it."""

    index: int                       # 0-based position in --data
    path: Path
    title: str                       # e.g. "TAO/TRITON moorings"
    attr_names: Tuple[str, ...]      # labels of mode 2, file order
    attrs: Tuple[str, ...]           # the ones kept (() = all)
    t_limit: int                     # 0 = whole time axis
    slots_per_day: int
    full_shape: Tuple[int, int, int]             # as stored
    shape: Tuple[int, int, int]                  # after --attrs and --t-limit
    rows: List[Dict[str, object]] = field(default_factory=list)

    @property
    def label(self) -> str:
        """``Dataset I``, ``Dataset II``, ... - the name the results use."""
        return f"Dataset {ROMAN[self.index] if self.index < len(ROMAN) else self.index + 1}"

    @property
    def tag(self) -> str:
        """Folder name of this dataset's cache, e.g. ``dataset1_tao_mode3_LxAxT``."""
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.path.stem)
        return f"dataset{self.index + 1}_{stem}"

    @property
    def kept(self) -> Tuple[str, ...]:
        return tuple(a for a in self.attr_names if not self.attrs or a in self.attrs)

    def describe(self) -> str:
        L, A, T = self.shape
        return (f"{self.label} = {self.title} ({self.path.name}; {L} x {A} x {T:,}; "
                f"attributes {', '.join(self.kept)})")


def find_data(explicit: str) -> List[Path]:
    """Locate the tensors without hard-coding anything.

    An explicit --data is a comma-separated list of files.  Without one, the
    tensors named in :data:`DATA_NAMES` are looked for next to this script,
    then in the working directory, then one level below either of those, and
    every one that is found is run, in that order.  Nothing outside the
    checkout is ever consulted, so the script moves between machines untouched.
    """
    if explicit:
        paths = []
        for item in (v.strip() for v in explicit.split(",")):
            if not item:
                continue
            path = Path(item).expanduser()
            if not path.is_file():
                raise SystemExit(f"--data {path} does not exist")
            paths.append(path.resolve())
        if not paths:
            raise SystemExit("--data names no file")
        if len(set(paths)) != len(paths):
            raise SystemExit("--data names the same file twice")
        return paths

    here = Path(__file__).resolve().parent
    found: List[Path] = []
    for name in DATA_NAMES:
        candidates = [here / name, Path.cwd() / name]
        for root in (here, Path.cwd()):
            candidates.extend(sorted(root.glob("*/" + name)))
        for candidate in candidates:
            if candidate.exists():
                found.append(candidate.resolve())
                break
    if not found:
        raise SystemExit(
            f"cannot find {' or '.join(DATA_NAMES)}.  Put them next to "
            f"{Path(__file__).name} or pass --data file1.npy,file2.npy")
    return found


def _groups(text: str, n: int) -> List[Tuple[str, ...]]:
    """``"a,b;c,d"`` -> one tuple of labels per dataset, padded with ()."""
    groups = [tuple(v.strip() for v in part.split(",") if v.strip())
              for part in text.split(";")] if text.strip() else []
    if len(groups) > n:
        raise SystemExit(f"{len(groups)} ';'-separated groups were given for "
                         f"{n} dataset(s): {text!r}")
    return groups + [()] * (n - len(groups))


def _numbers(text: str, n: int, flag: str, spread: bool) -> List[Optional[int]]:
    """``"0,8760"`` -> one number per dataset; None where none was given.

    With ``spread`` a single value applies to every dataset.
    """
    try:
        values: List[Optional[int]] = [int(v) if v.strip() else None
                                       for v in text.split(",")] if text.strip() else []
    except ValueError:
        raise SystemExit(f"{flag} wants whole numbers separated by ',', got {text!r}")
    if len(values) > n:
        raise SystemExit(f"{flag} lists {len(values)} values for {n} dataset(s)")
    if spread and len(values) == 1:
        values = values * n
    return values + [None] * (n - len(values))


def resolve_datasets(cfg: Config) -> List[Dataset]:
    """Turn --data and the per-dataset arguments into one :class:`Dataset` each."""
    paths = find_data(cfg.data)
    n = len(paths)
    names = _groups(cfg.attr_names_arg, n)
    keeps = _groups(cfg.attrs_arg, n)
    limits = _numbers(cfg.t_limit_arg, n, "--t-limit", spread=True)
    slots = _numbers(cfg.slots_arg, n, "--slots-per-day", spread=False)
    out: List[Dataset] = []
    for i, path in enumerate(paths):
        stored = np.load(str(path), mmap_mode="r").shape
        if len(stored) != 3:
            raise SystemExit(f"{path} holds a {len(stored)}-way array; a 3-way "
                             f"(L, A, T) one is needed")
        profile = next((p for p in DATASET_PROFILES
                        if str(p["key"]) in path.stem.lower()), None)
        if names[i]:
            attr_names = names[i]
        elif profile is not None and len(profile["attr_names"]) == stored[1]:   # type: ignore[arg-type]
            attr_names = tuple(profile["attr_names"])                           # type: ignore[arg-type]
        else:
            attr_names = tuple(f"A{a + 1}" for a in range(stored[1]))
        if len(attr_names) != stored[1] or len(set(attr_names)) != len(attr_names):
            raise SystemExit(
                f"{path.name} has {stored[1]} attributes on mode 2 but its "
                f"--attr-names group lists {len(attr_names)} "
                f"({','.join(attr_names)}); give one distinct label per attribute")
        unknown = [a for a in keeps[i] if a not in attr_names]
        if unknown:
            raise SystemExit(
                f"--attrs names {','.join(unknown)} for {path.name}, whose "
                f"attributes are {','.join(attr_names)}.  Groups are positional: "
                f"one per --data entry, separated by ';'")
        limit = max(0, limits[i] or 0)
        if cfg.quick and not limit:
            limit = 2048
        per_day = slots[i] if slots[i] is not None else (
            int(profile["slots_per_day"]) if profile is not None else 0)     # type: ignore[arg-type]
        kept = len(keeps[i]) if keeps[i] else stored[1]
        length = min(int(stored[2]), limit) if limit else int(stored[2])
        out.append(Dataset(
            index=i, path=path,
            title=str(profile["title"]) if profile is not None else path.stem,
            attr_names=attr_names, attrs=keeps[i], t_limit=limit,
            slots_per_day=max(0, int(per_day)),
            full_shape=(int(stored[0]), int(stored[1]), int(stored[2])),
            shape=(int(stored[0]), int(kept), int(length))))
    return out


def dataset_config(cfg: Config, dataset: Dataset) -> Config:
    """The configuration of one dataset's sweep: what its workers receive."""
    return replace(cfg, dataset=dataset.tag, attr_names=dataset.attr_names,
                   attrs=dataset.attrs, t_limit=dataset.t_limit,
                   slots_per_day=dataset.slots_per_day, shape=dataset.shape)


def load_tensor(path: Path, t_limit: int = 0) -> np.ndarray:
    X = np.load(str(path)).astype(np.float32, copy=False)
    if X.ndim != 3:
        raise SystemExit(f"{path} holds a {X.ndim}-way array; a 3-way one is needed")
    if t_limit and t_limit < X.shape[2]:
        X = X[:, :, :t_limit].copy()
    return np.ascontiguousarray(X)


def prepare_tensor(path: Path, cfg: Config) -> np.ndarray:
    """Load the tensor and keep the attributes --attrs asks for."""
    X = load_tensor(path, cfg.t_limit)
    if X.shape[1] != len(cfg.attr_names):
        raise SystemExit(
            f"{path} has {X.shape[1]} attributes on mode 2 but {len(cfg.attr_names)} "
            f"labels ({','.join(cfg.attr_names)}); pass the labels with --attr-names")
    keep = kept_attribute_index(cfg)
    if len(keep) != X.shape[1]:
        X = np.ascontiguousarray(X[:, keep, :])
    return X


@dataclass
class Experiment:
    """One configuration (ratio, outlier bound, noise), identical for all methods."""

    X_true: np.ndarray          # the raw tensor, NaN where never observed
    X_in: np.ndarray            # what a method sees: NaN where unavailable,
                                # corrupted values where outliers / noise were added
    obs0: np.ndarray            # cells the raw array observed  (eval set "observed")
    mask_in: np.ndarray         # cells a method may read
    ratio: float
    outliers: Optional[np.ndarray] = None   # cells that received an impulse
    bound: float = 0.0          # Chebyshev outlier bound k
    noise: float = 0.0          # noise level eta
    trace: bool = False         # record the Result-3 NRMSE curve during the fit

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.X_true.shape)              # type: ignore[return-value]

    @property
    def hidden(self) -> np.ndarray:
        """Cells removed by the missing ratio (eval set "hidden")."""
        return self.obs0 & ~self.mask_in


def attribute_sigma(X: np.ndarray) -> np.ndarray:
    """Sample standard deviation of each attribute over the cells observed.

    The scale of the corruption protocol.  It is a property of the clean
    record, computed by the experimenter, and no method ever sees it.
    """
    out = np.ones(X.shape[1], dtype=np.float64)
    for a in range(X.shape[1]):
        values = X[:, a, :]
        values = values[np.isfinite(values)].astype(np.float64)
        if values.size > 1:
            out[a] = float(values.std(ddof=1)) or 1.0
    return out


def build_experiment(X: np.ndarray, ratio: float, seed: int, bound: float = 0.0,
                     outlier_ratio: float = 0.0, spread: float = 1.0,
                     noise: float = 0.0) -> Experiment:
    """Corrupt the tensor exactly once per configuration.

    1. Missing ratio (MCAR).  A share ``ratio`` of the observed cells is drawn
       uniformly and removed.  The random stream is the one rev6 used, so the
       hidden cells are bit-identical to rev6's at every ratio.
    2. Noise level.  Every readable cell gets ``noise * sigma_a * N(0, 1)``.
    3. Outlier Addition, the Tukey-Huber epsilon-contamination model.  A share
       ``outlier_ratio`` of the readable cells is drawn uniformly and each gets
       ``s * m * sigma_a`` with s = +-1 and m ~ U[bound, (1 + spread) bound].
       Every impulse is at least ``bound`` standard deviations, which is the
       Chebyshev outlier bound: P(|X - mu| >= k sigma) <= 1 / k^2 for any
       distribution, so at most 1/k^2 of genuine readings deviate that much.
       The cells, signs and uniforms come from a stream that does not depend
       on ``bound``, so every bound corrupts the same cells and only the
       magnitude changes.

    Noise and outliers each have their own stream, independent of the missing
    mask's, so adding them never changes which cells are hidden.
    """
    rng = np.random.default_rng(seed)
    obs0 = np.isfinite(X)

    keep = obs0.copy()
    if ratio > 0:
        flat = np.flatnonzero(obs0.reshape(-1))
        n_drop = int(round(ratio * flat.size))
        if n_drop:
            drop = rng.choice(flat, size=n_drop, replace=False)
            keep.reshape(-1)[drop] = False

    X_in = np.where(keep, X, np.nan).astype(np.float32)
    outliers = np.zeros_like(keep)
    T = X.shape[2]
    A = X.shape[1]
    if noise > 0 or (bound > 0 and outlier_ratio > 0):
        sigma = attribute_sigma(X)
        readable = np.flatnonzero(keep.reshape(-1))
        attr_of = (readable // T) % A                 # C order: ((l*A)+a)*T + t
        flat_in = X_in.reshape(-1)
        if noise > 0:
            rng_noise = np.random.default_rng([seed, 1])
            flat_in[readable] += (noise * sigma[attr_of]
                                  * rng_noise.standard_normal(readable.size)
                                  ).astype(np.float32)
        if bound > 0 and outlier_ratio > 0:
            rng_out = np.random.default_rng([seed, 2])
            n_out = int(round(outlier_ratio * readable.size))
            pick = rng_out.choice(readable.size, size=n_out, replace=False)
            sign = rng_out.choice(np.array([-1.0, 1.0]), size=n_out)
            u = rng_out.random(n_out)
            magnitude = bound * (1.0 + spread * u)
            cells = readable[pick]
            flat_in[cells] += (sign * magnitude * sigma[attr_of[pick]]).astype(np.float32)
            outliers.reshape(-1)[cells] = True
    return Experiment(X_true=X, X_in=X_in, obs0=obs0, mask_in=keep,
                      ratio=float(ratio), outliers=outliers,
                      bound=float(bound), noise=float(noise))


# ==============================================================================
# 4.  Scaling and the error measure
# ==============================================================================

def robust_sigma(values: np.ndarray) -> float:
    """1.4826 x median absolute deviation.

    A scale estimate that a handful of large spikes cannot inflate, which an
    ordinary standard deviation emphatically can: spikes on 0.2 % of the cells,
    at ten times the spread, would alone widen a plain sigma by about a tenth.
    """
    if values.size == 0:
        return 1.0
    centre = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - centre)))
    if scale > 1e-8:
        return scale
    return max(1e-8, float(values.std()))


def standardise(X_in: np.ndarray, mask: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and scale each attribute slice using the cells a method may read.

    RH lives in tens of percent and the winds in signed m/s, so an unnormalised
    Frobenius objective would be almost entirely an RH objective.  The statistics
    come from the thinned input, because that is all any method is given, and
    they are the median and the MAD rather than the mean and the standard
    deviation, so that the spikes the record carries do not set the scale every
    method then works in.
    """
    A = X_in.shape[1]
    mu = np.zeros(A, dtype=np.float64)
    sd = np.ones(A, dtype=np.float64)
    for a in range(A):
        values = X_in[:, a][mask[:, a]]
        if values.size:
            mu[a] = float(np.median(values))
            sd[a] = robust_sigma(values)
    # In float32 and in place: one tensor-sized buffer, no float64 temporaries.
    X_std = np.nan_to_num(np.asarray(X_in, dtype=np.float32), copy=True)
    X_std -= mu.astype(np.float32)[None, :, None]
    X_std /= sd.astype(np.float32)[None, :, None]
    return X_std, mu, sd


def destandardise(X_std: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    out = np.asarray(X_std, dtype=np.float32) * sd.astype(np.float32)[None, :, None]
    out += mu.astype(np.float32)[None, :, None]
    return out


def robust_estimate(X_in: np.ndarray, mask_in: np.ndarray, model: np.ndarray,
                    kappa: float, own_flags: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a method's model output into the tensor it is scored on.

    Every method here is asked for the same thing - the clean tensor - and every
    one of them is finished off by this same rule, so the comparison stays a
    comparison of models rather than of output conventions:

        * a cell that was never available is the model's value;
        * an available cell whose residual against the model exceeds ``kappa``
          robust sigmas is declared corrupted and is also the model's value;
        * every other available cell is its own measurement.

    This is what the source papers do themselves.  paper2 completes with
    ``M o X + (1 - M) o G(X)``; paper5 recovers ``L = X - S`` on the observed
    set, which is the measurement wherever its sparse term is zero.  Scoring the
    bare model everywhere instead - the first version of this script did - turns
    the figure into a ranking of how closely a model can reproduce measurements
    it was handed, and hands the methods with an explicit sparse term a
    structural advantage that has nothing to do with completion.

    The threshold is ``kappa`` times the robust spread of that attribute's own
    measurements - a property of the data and the same number for every
    method.  It is deliberately not a multiple of the method's residual spread:
    that version was tried and is a trap, because the better a method fits the
    bulk of the cells the smaller its residual spread becomes, so the same rule
    rejects more and more honest measurements the more the fit improves.  On
    this data it turned a converging fit into a diverging error curve.

    A false flag costs one model value in place of one measurement, which the
    error measure counts like any other error.

    ``own_flags`` is a method's own outlier decision, for a method that has an
    explicit detector (M1's leave-one-out innovation test).  Those cells are
    replaced as well: the shared rule is the floor every method gets, and a
    method with a better detector is allowed to use it.

    Returns the estimate and the mask of cells it replaced, which is also what
    the outlier-detection scores are computed from.
    """
    out = np.array(model, dtype=np.float32, copy=True)
    flagged = np.zeros_like(mask_in)
    if kappa <= 0 and own_flags is None:  # --raw-output: score the model as it is
        return out, flagged
    if kappa > 0:
        residual = np.where(mask_in, np.nan_to_num(X_in) - model, 0.0)
        for a in range(X_in.shape[1]):
            here = mask_in[:, a, :]
            if not here.any():
                continue
            scale = robust_sigma(X_in[:, a, :][here])      # the attribute's spread
            flagged[:, a, :] = here & (np.abs(residual[:, a, :]) > kappa * scale)
    if own_flags is not None:
        flagged |= (own_flags & mask_in)
    if kappa <= 0:
        # --raw-output with a detector: the bare model everywhere is what was
        # asked for, so the detector only feeds the detection scores.
        return out, flagged
    trusted = mask_in & ~flagged
    out[trusted] = np.nan_to_num(X_in)[trusted]
    return out, flagged


def metrics_per_attribute(X_true: np.ndarray, X_hat: np.ndarray,
                          mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Every measure, per attribute, over the cells in ``mask``.

        nrmse     ||e||_2 / ||x||_2                   paper4 eq. (16)
        nrmse_sd  RMSE / std(x)                       shift-invariant
        rmse      sqrt(mean e^2)                      physical units
        mae       mean |e|                            physical units
        r2        1 - sum e^2 / sum (x - mean x)^2

    ``nrmse`` divides by the root mean square of the raw values, so it depends
    on where the attribute's zero is: SST sits near 29 degC with a spread of
    0.9, so its NRMSE is roughly RMSE / 29, while the wind components are
    centred near zero and their NRMSE is roughly RMSE / 4.5.  ``nrmse_sd``
    divides by the spread instead, which makes attributes comparable with one
    another; ``r2`` is 1 - nrmse_sd^2 over the same cells.
    """
    out: Dict[str, Dict[str, float]] = {m: {} for m in METRICS}
    out["cells"] = {}
    truth = np.nan_to_num(X_true)
    for a, name in enumerate(ATTRS):
        m = mask[:, a, :]
        n = int(m.sum())
        out["cells"][name] = float(n)
        if n == 0:
            for key in METRICS:
                out[key][name] = float("nan")
            continue
        base = truth[:, a, :][m].astype(np.float64)
        diff = base - X_hat[:, a, :][m].astype(np.float64)
        sse = float((diff ** 2).sum())
        rms_truth = math.sqrt(float((base ** 2).sum()))
        spread = float(base.std())
        rmse = math.sqrt(sse / n)
        out["nrmse"][name] = math.sqrt(sse) / rms_truth if rms_truth > 0 else float("nan")
        out["nrmse_sd"][name] = rmse / spread if spread > 0 else float("nan")
        out["rmse"][name] = rmse
        out["mae"][name] = float(np.abs(diff).mean())
        out["r2"][name] = (1.0 - sse / (spread ** 2 * n)) if spread > 0 else float("nan")
    return out


def detection_scores(flagged: np.ndarray, injected: Optional[np.ndarray]
                     ) -> Dict[str, float]:
    """Precision, recall and F1 of the replaced cells against the injected ones."""
    n_flag = int(flagged.sum())
    if injected is None or not injected.any():
        return {"injected": 0.0, "flagged": float(n_flag), "caught": 0.0,
                "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan")}
    caught = int((flagged & injected).sum())
    n_inj = int(injected.sum())
    precision = caught / n_flag if n_flag else 0.0
    recall = caught / n_inj if n_inj else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return {"injected": float(n_inj), "flagged": float(n_flag),
            "caught": float(caught), "precision": precision, "recall": recall,
            "f1": f1}


def nrmse_per_attribute(X_true: np.ndarray, X_hat: np.ndarray,
                        mask: np.ndarray) -> Dict[str, float]:
    """paper4 eq. (16) restricted to one attribute, over the cells in ``mask``."""
    out: Dict[str, float] = {}
    truth = np.nan_to_num(X_true)
    for a, name in enumerate(ATTRS):
        m = mask[:, a, :]
        if not m.any():
            out[name] = float("nan")
            continue
        diff = (truth[:, a, :] - X_hat[:, a, :])[m]
        base = truth[:, a, :][m]
        denominator = float(np.sqrt((base.astype(np.float64) ** 2).sum()))
        numerator = float(np.sqrt((diff.astype(np.float64) ** 2).sum()))
        out[name] = numerator / denominator if denominator > 0 else float("nan")
    return out


def evaluation_mask(exp: Experiment, name: str) -> np.ndarray:
    """The cells of one evaluation set; see THE CORRUPTION PROTOCOL."""
    if name == "hidden":
        return exp.hidden
    if name == "outliers":
        return exp.outliers if exp.outliers is not None else np.zeros_like(exp.obs0)
    return exp.obs0


def mean_nrmse(X_true: np.ndarray, X_hat: np.ndarray, mask: np.ndarray) -> float:
    """NRMSE averaged over the attributes: the one number of Results 1 and 3.

    Each attribute's NRMSE is a ratio of norms in that attribute's own unit, so
    the plain mean over attributes weights them equally whatever their scale
    and however many cells each one has.  Attributes with no cell in ``mask``
    are left out of the mean.
    """
    values = [v for v in nrmse_per_attribute(X_true, X_hat, mask).values()
              if math.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def validation_split(exp: Experiment, cfg: Config) -> np.ndarray:
    """The readable cells a fit holds out to judge itself by.

    A random share ``--val-frac`` of the cells a method may read.  They are
    inputs, not ground truth: the method simply does not fit them, watches the
    clipped squared error of its model on them, and keeps the iterate where
    that error is lowest.  At the output they are copied through like any other
    readable cell.

    Every validated method calls this as the first consumer of the NumPy
    stream after :func:`_run_job` has seeded it, so M1 and the baselines hold
    out the *same* cells in a given configuration.
    """
    held = np.zeros_like(exp.mask_in)
    if cfg.val_frac > 0:
        rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
        held = exp.mask_in & (rng.random(exp.mask_in.shape, dtype=np.float32)
                              < cfg.val_frac)
    return held


def nrmse_pooled(X_true: np.ndarray, X_hat: np.ndarray, mask: np.ndarray) -> float:
    """The same error over every attribute at once, in standardised units.

    Pooling raw units would make the number an RH number, so the attributes are
    put on a common scale first.  Logged per job, as a second opinion on the
    attribute-averaged NRMSE the results report.
    """
    truth = np.nan_to_num(X_true)
    numerator = 0.0
    cells = 0
    for a in range(X_true.shape[1]):
        m = mask[:, a, :]
        if not m.any():
            continue
        base = truth[:, a, :][m].astype(np.float64)
        scale = float(base.std()) or 1.0
        diff = (truth[:, a, :] - X_hat[:, a, :])[m].astype(np.float64) / scale
        numerator += float((diff ** 2).sum())
        cells += int(base.size)                 # ||standardised truth||^2 == cells
    return math.sqrt(numerator / cells) if cells else float("nan")


# ==============================================================================
# 5.  Threads
# ==============================================================================

def set_threads(threads: int) -> None:
    """Pin this process's share of the machine."""
    if threads > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)
        torch.set_num_threads(threads)


# ==============================================================================
# 6.  M1  Deep-TD - this project's method
# ==============================================================================
#
# THE MODEL
#
#     X  ~=  L + S                      on the cells the method is allowed to read
#     L   =  G x1 U1 x2 U2 x3 U3        Tucker, multilinear rank (R1, R2, R3)
#     S      sparse, supported on the observed cells        (the gross spikes)
#
# Where the ranks go is the whole design.  The array is 47 moorings x 5
# attributes x 13,248 instants: the first two modes really are low-rank - the
# moorings sit in one ocean and the attributes are physically coupled - but the
# time mode is not.  Compressing 13,248 instants into a handful of global
# temporal profiles forces every mooring's entire time course to be a linear
# combination of those few profiles, which is hopeless for the wind components,
# and it was the single biggest reason the first version of this method lost to
# every baseline it was compared with.
#
# So the temporal factor is kept at (or near) full resolution: one coefficient
# row u3(t) per instant, with R3 defaulting to R1 R2 and then capped by how much
# data a typical instant actually carries (never fit more coefficients per
# instant than roughly half the cells observed in it).  What stops the model
# from simply memorising the observed cells is not a rank constraint any more -
# it is the network.  A causal dilated TCN is trained to predict u3(t) from its
# own past, a second one to predict it from its future, and their average enters
# the next solve as a Gaussian prior on u3(t):
#
#     u3(t)  =  argmin_u  || M_t o (x_t - Phi u) ||^2
#                         + lambda || u - p(t) ||^2 + eps || u ||^2
#
# with Phi = (U2 (x) U1) V the (L A) x R3 dictionary of the Tucker model.  The
# prior weight is not a magic number: lambda = sigma_noise^2 / sigma_pred^2 is
# the empirical-Bayes value, both variances being measured on the fly.  Where an
# instant is densely observed the first term dominates and u3 follows the
# measurements; where it is nearly empty the prior dominates and u3 follows the
# learnt dynamics.  That is the Kalman-like behaviour the method was always
# meant to have.
#
# THE LOOP  (one "iteration" of Result 3)
#
#     1. S-step       soft-threshold the observed residual at kappa robust
#                     sigmas; the spikes land in S instead of bending L
#     2. factor step  refresh U1, U2 and the core from the current completion
#                     (warm-up iterations only, then they are frozen so that the
#                     network keeps talking about a fixed coordinate frame)
#     3. network step Adam on the one-step-ahead prediction loss of both TCNs
#     4. u3-step      the per-instant ridge solve above, batched over t
#     5. record       NRMSE of the current estimate - a diagnostic, never read
#                     by the fit, and what Result 3 plots
# ------------------------------------------------------------------------------

# ---- 6.1  tensor plumbing ----------------------------------------------------

def unfold(tensor, mode: int):
    """Mode-``mode`` unfolding: that axis becomes the rows, the rest the columns."""
    return tensor.movedim(mode, 0).reshape(tensor.shape[mode], -1)


def mode_product(tensor, matrix, mode: int):
    """``tensor x_mode matrix`` with ``matrix`` of shape (out, tensor.shape[mode])."""
    out = torch.tensordot(matrix, tensor, dims=([1], [mode]))
    return out.movedim(0, mode)


def top_left_singular(matrix, rank: int):
    """Leading ``rank`` left singular vectors.

    Short-and-wide matrices (every unfolding this file takes) go through the
    Gram matrix, which is 47 x 47 instead of 47 x 66,240.
    """
    rows, cols = matrix.shape
    rank = max(1, min(rank, rows))
    if rows <= cols and rows <= 4096:
        gram = matrix @ matrix.transpose(0, 1)
        gram = 0.5 * (gram + gram.transpose(0, 1))
        _, vectors = torch.linalg.eigh(gram)
        return vectors.flip(1)[:, :rank].contiguous()
    left, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return left[:, :rank].contiguous()


def initial_fill(X_std, mask):
    """Start every unavailable cell at its own channel's observed mean."""
    count = mask.sum(dim=2, keepdim=True).clamp_min(1)
    total = (X_std * mask).sum(dim=2, keepdim=True)
    channel_mean = total / count
    return torch.where(mask, X_std, channel_mean.expand_as(X_std)).contiguous()


def diurnal_channels(T: int, dtype, slots_per_day: int):
    """Two channels carrying the position of each instant within its day.

    ``slots_per_day`` is the number of samples in a day: 144 at ten-minute
    sampling, 24 at hourly.  With 0 (unknown) the two channels are constant
    and carry nothing.
    """
    if slots_per_day > 0:
        angle = 2.0 * np.pi * (np.arange(T) % slots_per_day) / slots_per_day
    else:
        angle = np.zeros(T)
    return torch.as_tensor(np.stack([np.sin(angle), np.cos(angle)], axis=0), dtype=dtype)


def dense_instants(mask: np.ndarray, frac: float) -> np.ndarray:
    """Instants observed at least ``frac`` times the median instant.

    Only these teach the network.  A coefficient row solved from a nearly empty
    instant is mostly the prior talking, and training the prior on its own
    output is a closed loop.  The threshold is relative to the median rather
    than absolute so that it still selects something sensible at 80 % missing.
    """
    counts = mask.reshape(-1, mask.shape[2]).sum(axis=0)
    live = counts[counts > 0]
    typical = float(np.median(live)) if live.size else 0.0
    chosen = np.flatnonzero(counts >= max(1.0, frac * typical))
    if chosen.size < 32:
        chosen = np.flatnonzero(counts > 0)
    if chosen.size == 0:
        chosen = np.arange(mask.shape[2])
    return chosen.astype(np.int64)


def torch_robust_sigma(values) -> float:
    """1.4826 x MAD of a torch vector; see :func:`robust_sigma`."""
    if values.numel() == 0:
        return 1.0
    centre = values.median()
    scale = float(1.4826 * (values - centre).abs().median())
    if scale > 1e-8:
        return scale
    return max(1e-8, float(values.std(unbiased=False)))


def ar_bridge(resid, trusted, max_phi: float = 0.999, min_pairs: int = 50,
              chunk: int = 64):
    """Per-channel AR(1) prediction of a residual from its trusted neighbours.

    ``resid`` and ``trusted`` are (T, P): one column per channel.  Each column
    is modelled as  r_t = m + s_t + n_t,  s a stationary AR(1) (a sampled
    Ornstein-Uhlenbeck process) with coefficient phi, n white noise, and
    gamma = var(s) / var(s + n) the signal share.  Then the lag-h
    autocorrelation of r is gamma phi^h for h >= 1, so

        phi = rho2 / rho1,      gamma = rho1 / phi,

    which is the moment estimator that is not biased by the white noise (rho1
    alone would be gamma phi).  Every cell t then gets the best linear unbiased
    predictor of r_t from the nearest trusted observation strictly before it
    (distance d1) and strictly after it (distance d2):

        g = 1 / gamma,  a = phi^d1,  b = phi^d2,  D = d1 + d2
        w_left  = (g a - phi^D b) / (g^2 - phi^(2D))
        w_right = (g b - phi^D a) / (g^2 - phi^(2D))
        r_hat   = m + w_left (r_left - m) + w_right (r_right - m)

    With gamma = 1 this is the AR(1) bridge, the exact conditional mean
    E[s_t | s_{t-d1}, s_{t+d2}], which by the Markov property equals the
    conditional mean given the whole channel.  With one side missing it falls
    back to gamma phi^d times the one neighbour, and with neither to m.

    "Strictly" matters: for a trusted cell the prediction never reads the cell
    itself, so ``resid - r_hat`` is a leave-one-out innovation, which is what
    the outlier test in :meth:`TuckerTCN._residual_stage` thresholds.

    Columns are processed ``chunk`` at a time so the working set stays small.
    Returns the prediction (T, P) and the per-channel phi and gamma.
    """
    T, P = resid.shape
    out = torch.empty_like(resid)
    phis = torch.zeros(P)
    gammas = torch.zeros(P)
    index = torch.arange(T).unsqueeze(1)
    for begin in range(0, P, chunk):
        stop = min(P, begin + chunk)
        w = trusted[:, begin:stop]
        wf = w.to(resid.dtype)
        r = resid[:, begin:stop]
        n = wf.sum(0).clamp_min(1.0)
        mean = (r * wf).sum(0) / n                     # the channel's bias
        z = (r - mean) * wf                            # centred, 0 off the trusted set

        def rho(lag: int):
            if T <= lag:
                return torch.zeros(stop - begin), torch.zeros(stop - begin)
            pair = wf[lag:] * wf[:-lag]
            num = (z[lag:] * z[:-lag] * pair).sum(0)
            den = torch.sqrt((z[lag:] ** 2 * pair).sum(0)
                             * (z[:-lag] ** 2 * pair).sum(0)).clamp_min(1e-12)
            return num / den, pair.sum(0)

        rho1, n1 = rho(1)
        rho2, _ = rho(2)
        usable = (n1 >= min_pairs) & (rho1 > 0.05)
        phi = torch.where(rho2 > 0, rho2 / rho1.clamp_min(1e-6), rho1)
        phi = torch.maximum(phi, rho1).clamp(0.0, max_phi)
        gamma = (rho1 / phi.clamp_min(1e-6)).clamp(0.0, 1.0)
        phi = torch.where(usable, phi, torch.zeros_like(phi))
        gamma = torch.where(usable, gamma, torch.zeros_like(gamma))
        phis[begin:stop] = phi
        gammas[begin:stop] = gamma

        # nearest trusted instant strictly before / strictly after each instant
        idx = index.expand(T, stop - begin)
        before = torch.where(w, idx, torch.full_like(idx, -1))
        before = torch.cummax(before, dim=0).values
        prev = torch.cat([torch.full_like(before[:1], -1), before[:-1]], dim=0)
        after = torch.where(w, idx, torch.full_like(idx, T))
        after = torch.flip(torch.cummin(torch.flip(after, [0]), dim=0).values, [0])
        nxt = torch.cat([after[1:], torch.full_like(after[:1], T)], dim=0)
        has_l = prev >= 0
        has_r = nxt < T
        z_l = torch.gather(z, 0, prev.clamp_min(0))
        z_r = torch.gather(z, 0, nxt.clamp_max(T - 1))
        log_phi = torch.log(phi.clamp_min(1e-12))
        a = torch.exp((idx - prev).to(resid.dtype) * log_phi)
        b = torch.exp((nxt - idx).to(resid.dtype) * log_phi)
        ab = a * b
        g = 1.0 / gamma.clamp_min(1e-6)
        den = (g * g - ab * ab).clamp_min(1e-12)
        both = has_l & has_r
        w_l = torch.where(both, (g * a - ab * b) / den,
                          torch.where(has_l, gamma * a, torch.zeros_like(a)))
        w_r = torch.where(both, (g * b - ab * a) / den,
                          torch.where(has_r, gamma * b, torch.zeros_like(b)))
        out[:, begin:stop] = mean + w_l * z_l + w_r * z_r
    return out, phis, gammas


def cp_als(Y, rank: int, iters: int, init=None, ridge: float = 1e-4):
    """CP decomposition of a complete (L, A, T) tensor by alternating least squares.

    Used only by the cp / cp_tcn ablations, to build a CP dictionary
    Phi = U1 (Khatri-Rao) U2 with the same number of columns as M1's Tucker
    dictionary.  Every step is a GEMM against the (L A, T) unfolding.  The
    ridge (relative to the Gram's mean diagonal) keeps the R x R systems
    solvable when CP factors become nearly collinear - the degeneracy that
    CP, unlike Tucker, is prone to.
    """
    L, A, T = Y.shape
    Ymat = Y.reshape(L * A, T)
    if init is None:
        F1, F2, F3 = torch.randn(L, rank), torch.randn(A, rank), torch.randn(T, rank)
    else:
        F1, F2, F3 = init
    eye = torch.eye(rank)

    def solve(gram, rhs):                              # rhs (n, R) -> (n, R)
        lift = ridge * float(gram.diagonal().mean().clamp_min(1e-12))
        return torch.linalg.solve(gram + lift * eye, rhs.transpose(0, 1)).transpose(0, 1)

    for _ in range(max(1, iters)):
        kr12 = (F1.unsqueeze(1) * F2.unsqueeze(0)).reshape(L * A, rank)
        F3 = solve((F1.T @ F1) * (F2.T @ F2), Ymat.T @ kr12)
        YF3 = (Ymat @ F3).reshape(L, A, rank)
        F1 = solve((F2.T @ F2) * (F3.T @ F3), (YF3 * F2.unsqueeze(0)).sum(1))
        F2 = solve((F1.T @ F1) * (F3.T @ F3), (YF3 * F1.unsqueeze(1)).sum(0))
        n1 = F1.norm(dim=0).clamp_min(1e-12)
        n2 = F2.norm(dim=0).clamp_min(1e-12)
        F1, F2, F3 = F1 / n1, F2 / n2, F3 * (n1 * n2)
    return F1, F2, F3


# ---- 6.1b  checkpoints: held-out selection and the convergence curves --------

def drain_device() -> None:
    """Wait for the device's queue, so a clock started next times only what follows."""
    if torch.cuda.is_available() and torch.empty(0).is_cuda:
        torch.cuda.synchronize()


def clone_state(module: nn.Module) -> Dict[str, object]:
    """A detached copy of a module's parameters and buffers."""
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


class TraceScorer:
    """The convergence measure of Results 2 and 3, on a fixed sample of cells.

    It is the number Result 1 reports - the model finished by the shared output
    rule (:func:`robust_estimate`), then the NRMSE against the clean values per
    attribute, averaged over the attributes - evaluated on a fixed random sample
    of ``--trace-cells`` cells of the evaluation set instead of on all of them.
    The sample is drawn from its own stream, so every method of a configuration
    is scored on the same cells, and a per-entry network (M4, M6, M7) only has
    to predict the sample at a checkpoint instead of the whole tensor.  With
    10^6 cells the sampling error of an NRMSE is far below the gaps between the
    methods; the last point of every curve is the exact Result-1 value.

    Everything that does not change during a fit - the sample, its clean
    values, its measurements and the per-attribute thresholds of the output
    rule - is computed once here, so a checkpoint is one pass over the sample.
    """

    def __init__(self, exp: Experiment, cfg: Config, mu: np.ndarray, sd: np.ndarray):
        L, A, T = exp.shape
        flat = np.flatnonzero(evaluation_mask(exp, cfg.eval_set).reshape(-1))
        if 0 < cfg.trace_cells < flat.size:
            rng = np.random.default_rng([int(cfg.seed), 7])
            flat = np.sort(rng.choice(flat, size=int(cfg.trace_cells), replace=False))
        self.flat = flat.astype(np.int64)
        attr = (self.flat // T) % A
        self.truth = np.nan_to_num(exp.X_true.reshape(-1)[self.flat]).astype(np.float64)
        self.available = exp.mask_in.reshape(-1)[self.flat]
        self.measured = np.nan_to_num(exp.X_in.reshape(-1)[self.flat]).astype(np.float32)
        self.mu = mu.astype(np.float32)[attr]
        self.sd = sd.astype(np.float32)[attr]
        self.kappa = float(cfg.flag_kappa)
        threshold = np.full(A, np.inf, dtype=np.float32)
        if self.kappa > 0:
            for a in range(A):
                here = exp.mask_in[:, a, :]
                if here.any():
                    threshold[a] = self.kappa * robust_sigma(exp.X_in[:, a, :][here])
        self.threshold = threshold[attr]
        self.groups = [np.flatnonzero(attr == a) for a in range(A)]
        self.norms = [math.sqrt(float((self.truth[g] ** 2).sum())) for g in self.groups]

    def score(self, standardised: np.ndarray) -> float:
        """Attribute-averaged NRMSE of a model given at the sampled cells."""
        model = np.asarray(standardised, dtype=np.float32).reshape(-1) * self.sd + self.mu
        if self.kappa > 0:
            keep = self.available & (np.abs(self.measured - model) <= self.threshold)
            estimate = np.where(keep, self.measured, model)
        else:
            estimate = model
        error = (self.truth - estimate.astype(np.float64)) ** 2
        values = [math.sqrt(float(error[g].sum())) / norm
                  for g, norm in zip(self.groups, self.norms) if g.size and norm > 0]
        return float(np.mean(values)) if values else float("nan")


class Tracker:
    """Which iterate a fit returns, and the curve Results 2 and 3 draw.

    Every method runs for exactly ``--iters`` iterations (THE BUDGETS in the
    header); at the end of each one it calls the tracker.

    *Validation* is part of the method, and its time stays on the clock.  The
    fit holds out the cells :func:`validation_split` draws - the same cells for
    every method - and the clipped squared error of the model on them is

        loss = mean over held-out cells of min(r^2, c^2),   c = --val-clip

    a bounded loss, so a held-out cell that carries an outlier adds a constant
    instead of deciding the outcome.  The method keeps the parameters of the
    iteration with the lowest loss and returns those: early stopping evaluated
    after the fact, which is what CoSTCo prescribes and what DeepTensor's
    stopping-epoch analysis asks for, without cutting any curve short.  No
    ground truth is read.

    *Tracing* is a diagnostic, and its time is taken off the clock
    (:attr:`seconds`).  When ``exp.trace`` is set the model is scored by a
    :class:`TraceScorer` at every iteration.
    """

    def __init__(self, exp: Experiment, cfg: Config, X_std: np.ndarray,
                 mu: np.ndarray, sd: np.ndarray, held: Optional[np.ndarray]):
        started = time.perf_counter()
        self.clip2 = float(cfg.val_clip) ** 2
        self.held_idx: Optional[np.ndarray] = None
        self.held_target: Optional[np.ndarray] = None
        if held is not None and held.any():
            self.held_idx = np.flatnonzero(held.reshape(-1))
            self.held_target = np.asarray(X_std.reshape(-1)[self.held_idx], dtype=np.float32)
        self.best = float("inf")
        self.best_x = 0.0
        self.checks = 0
        self.val_curve: List[float] = []
        self.points: List[Dict[str, float]] = []
        self.scorer = TraceScorer(exp, cfg, mu, sd) if exp.trace else None
        self.seconds = (time.perf_counter() - started) if self.scorer is not None else 0.0

    @property
    def validating(self) -> bool:
        return self.held_idx is not None

    @property
    def tracing(self) -> bool:
        return self.scorer is not None

    def validate(self, x: float, predicted: np.ndarray) -> bool:
        """Score the held-out cells; True when this is the best iterate so far."""
        residual = self.held_target - np.asarray(predicted, dtype=np.float32).reshape(-1)
        loss = float(np.minimum(residual * residual, self.clip2).mean())
        self.checks += 1
        self.val_curve.append(loss)
        if math.isfinite(loss) and loss < self.best:
            self.best, self.best_x = loss, float(x)
            return True
        return False

    def record(self, x: float, best: float, best_x: float) -> None:
        """Book a selection made on the device (M1 validates there itself)."""
        self.checks += 1
        self.best, self.best_x = best, float(best_x)

    def trace(self, x: float, values: Callable[[], np.ndarray]) -> None:
        """One point of the curve; ``values()`` is the model at the sampled cells."""
        if self.scorer is None:
            return
        drain_device()                    # the fit's own time, not the diagnostic's
        started = time.perf_counter()
        value = self.scorer.score(values())
        if math.isfinite(value):
            self.points.append({"x": float(x), "nrmse": value})
        self.seconds += time.perf_counter() - started

    def info(self, used: float) -> Dict[str, object]:
        return {"budget_used": float(used),
                "best_x": float(self.best_x if self.validating else used),
                "val_loss": float(self.best) if self.validating else float("nan"),
                "checks": int(self.checks), "stopped": "budget",
                "validated": bool(self.validating)}


def full_checkpoint(tracker: Tracker, x: float,
                    full_std: Callable[[], np.ndarray]) -> bool:
    """Checkpoint of a method whose model is a whole tensor (M2, M3, M5).

    ``full_std()`` returns the model on every cell, standardised, (L, A, T).
    It is built at most once per checkpoint, and only when it is needed.
    """
    improved = False
    full: Optional[np.ndarray] = None
    if tracker.validating:
        full = full_std()
        improved = tracker.validate(x, full.reshape(-1)[tracker.held_idx])
    if tracker.tracing:
        tracker.trace(x, lambda: (full if full is not None else full_std()
                                  ).reshape(-1)[tracker.scorer.flat])
    return improved


def entry_checkpoint(tracker: Tracker, x: float,
                     predict_at: Callable[[np.ndarray], np.ndarray]) -> bool:
    """Checkpoint of a per-entry network (M4, M6, M7): predict only what is scored."""
    improved = (tracker.validate(x, predict_at(tracker.held_idx))
                if tracker.validating else False)
    if tracker.tracing:
        tracker.trace(x, lambda: predict_at(tracker.scorer.flat))
    return improved


# ---- 6.2  the temporal network -----------------------------------------------

class CausalConv1d(nn.Module):
    """One dilated convolution that never reads a later instant.

    ``groups`` independent convolutions are evaluated as one grouped
    convolution; see :class:`TCN`.
    """

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int,
                 groups: int = 1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in * groups, c_out * groups, kernel,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class ResidualBlock(nn.Module):
    """Two causal convolutions at the same dilation, with a skip connection."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int,
                 dropout: float, groups: int = 1):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, kernel, dilation, groups)
        self.conv2 = CausalConv1d(c_out, c_out, kernel, dilation, groups)
        self.dropout = nn.Dropout(dropout)
        self.skip = (nn.Conv1d(c_in * groups, c_out * groups, 1, groups=groups)
                     if c_in != c_out else None)

    def forward(self, x):
        hidden = self.dropout(F.relu(self.conv1(x)))
        hidden = self.conv2(hidden)
        bypass = x if self.skip is None else self.skip(x)
        return F.relu(hidden + bypass)


class TCN(nn.Module):
    """Causal dilated residual network over the temporal factor.

    The dilation doubles every block, so ``blocks`` layers reach
    ``1 + 2 (kernel - 1) (2 ** blocks - 1)`` instants back.  The read-out is a
    1x1 convolution, so one pass over the whole sequence yields every
    one-step-ahead prediction at once.

    ``groups`` networks with separate weights are run as one *grouped*
    network: channel block g of every layer belongs to network g and never
    sees another block.  Deep-TD's forward and backward networks are the two
    groups.  This is the same pair of networks - same shapes, same
    initialisation distribution (a grouped convolution's fan-in is that of one
    group), and Adam is element-wise, so the optimisation is unchanged - but
    every layer is one convolution call instead of two, which halves the
    dispatch overhead and lets one GEMM use all of the worker's threads.
    """

    def __init__(self, c_in: int, c_out: int, hidden: int = 32, blocks: int = 8,
                 kernel: int = 3, dropout: float = 0.1, groups: int = 1):
        super().__init__()
        self.kernel, self.blocks, self.groups = kernel, blocks, groups
        layers = [ResidualBlock(c_in if i == 0 else hidden, hidden, kernel,
                                2 ** i, dropout, groups) for i in range(blocks)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden * groups, c_out * groups, 1, groups=groups)

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel - 1) * (2 ** self.blocks - 1)

    def forward(self, z):
        """(1, G C_in, T) -> (1, G R3, T); column tau predicts the row at tau + 1."""
        return self.head(self.body(z))


# ---- 6.3  the alternating fit ------------------------------------------------

class TuckerTCN:
    """Robust Deep-TD, fitted by alternating minimisation.

    The ground truth is never touched by the fit; it is read only by the
    :class:`TraceScorer` that draws the convergence curves.

    Memory.  Everything lives in the (T, P) layout, P = L A.  Of the full-size
    buffers only three are float - the data Xt, the model Lt and the selected
    iterate - and the three masks (fit cells, held-out cells, working weights)
    are bool, a quarter of the size.  The sparse term S of the model is not
    stored: it is non-zero only on cells the working mask excludes, so it never
    enters the solve, the working tensor or the noise estimate (it is
    ``(X - L)`` on the rejected cells, recoverable at any time).
    """

    def __init__(self, exp: Experiment, cfg: Config, *, dictionary: str = "tucker",
                 use_prior: bool = True, residual_stage: bool = True):
        self.exp, self.cfg = exp, cfg
        self.dictionary = dictionary
        self.use_prior = use_prior
        self.residual_stage = residual_stage
        self.L, self.A, self.T = exp.shape
        self.P = self.L * self.A
        X_std, self.mu, self.sd = standardise(exp.X_in, exp.mask_in)

        # Self-supervised validation split: a random share of the cells the
        # method may read is held out of the fit and used only to choose the
        # best iterate.  The same cells every baseline holds out.
        held = validation_split(exp, cfg)
        fit_mask = exp.mask_in & ~held

        self.Xt = self._flatten(torch.as_tensor(X_std))
        self.Mt = self._flatten(torch.as_tensor(fit_mask))          # bool
        self.Vt = self._flatten(torch.as_tensor(held))              # bool
        held_tp = self._tp_index(np.flatnonzero(held.reshape(-1)))
        self.n_val = int(held_tp.size)
        self.val_idx = torch.as_tensor(held_tp)
        self.val_target = self.Xt.reshape(-1)[self.val_idx]
        self.best_val = float("inf")
        self.best_L: Optional[torch.Tensor] = None
        self.best_iteration = -1

        # The tracker only traces here; the selection runs on the device below.
        self.tracker = Tracker(exp, cfg, X_std, self.mu, self.sd, None)
        self.trace_idx = (torch.as_tensor(self._tp_index(self.tracker.scorer.flat))
                          if self.tracker.tracing else None)
        del X_std

        self.own_flags: Optional[np.ndarray] = None
        self.stage_info: Dict[str, float] = {}
        self._cp_factors = None
        self.We = self.Mt.clone()                      # the mask minus the rejects
        self.Lt = torch.zeros_like(self.Xt)            # the model, on every cell
        self.count = self.Mt.sum(dim=1).to(torch.float32)   # observations per instant
        self.n_we = float(self.count.sum())
        self.sigma2 = 1.0                              # residual variance on We
        self.n_flagged = 0
        self._fill_mask = torch.as_tensor(fit_mask)    # used once, by the first basis

        self.r1, self.r2, self.r3 = self._ranks()
        self.tril = torch.tril_indices(self.r3, self.r3)
        self.U3 = torch.zeros(self.T, self.r3)
        K = max(1, cfg.iters)
        self.warm_iters = max(1, min(cfg.spatial_iters, max(1, K - 2)))
        self.chunk = cfg.solve_chunk or int(min(4096, max(64, 2 ** 23 // self.r3 ** 2)))

        self.directions = 2 if cfg.bidirectional else 1
        self.net: Optional[TCN] = None
        if use_prior:
            self.diurnal = diurnal_channels(self.T, torch.float32, cfg.slots_per_day)
            self.targets = torch.as_tensor(dense_instants(fit_mask, cfg.min_count_frac))
            blocks = max(1, min(cfg.tcn_blocks, int(math.log2(max(2, self.T)))))
            self.net = TCN(self.r3 + 2, self.r3, cfg.tcn_channels, blocks,
                           cfg.tcn_kernel, cfg.tcn_dropout, groups=self.directions)
            self.optimiser = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.lambda_prior = 0.0

    # -- layout ---------------------------------------------------------------

    def _flatten(self, tensor):
        """(L, A, T) -> (T, L*A)."""
        return tensor.permute(2, 0, 1).reshape(self.T, self.P).contiguous()

    def _unflatten(self, flat):
        """(T, L*A) -> (L, A, T)."""
        return flat.reshape(self.T, self.L, self.A).permute(1, 2, 0).contiguous()

    def _tp_index(self, flat: np.ndarray) -> np.ndarray:
        """C-order indices into (L, A, T) -> indices into the (T, P) layout."""
        flat = np.asarray(flat, dtype=np.int64)
        return (flat % self.T) * self.P + flat // self.T

    # -- set-up ---------------------------------------------------------------

    @staticmethod
    def _mode_rank(asked: float, size: int) -> int:
        """One mode's rank: a share of the mode when <= 1, a count when > 1,
        the whole mode when 0.  See RANKS AND ITERATIONS in the header."""
        if asked <= 0:
            return size
        wanted = asked * size if asked <= 1.0 else asked
        return max(1, min(int(round(wanted)), size))

    def _ranks(self) -> Tuple[int, int, int]:
        """Multilinear rank, with the temporal one answerable to the data volume.

        R3 is ``rank_obs_frac`` times the RANK_QUANTILE (first quartile) of the
        per-instant observation counts, so that three quarters of the instants
        carry at least 1 / rank_obs_frac observations per coefficient, and it
        is capped by ``--max-rank3`` because the solve costs T P R3^2.
        """
        cfg = self.cfg
        r1 = self._mode_rank(cfg.ranks[0], self.L)
        r2 = self._mode_rank(cfg.ranks[1], self.A)
        asked = cfg.ranks[2]
        r3 = r1 * r2 if asked <= 0 else self._mode_rank(asked, r1 * r2)
        r3 = max(1, min(r3, self.T))
        if cfg.rank_obs_frac > 0:
            live = self.count[self.count > 0]
            typical = (float(torch.quantile(live, RANK_QUANTILE)) if live.numel()
                       else float(self.P))
            budget = int(round(cfg.rank_obs_frac * typical))
            r3 = max(1, min(r3, max(cfg.min_rank3, budget)))
        if cfg.max_rank3 > 0:
            r3 = max(1, min(r3, cfg.max_rank3))
        return r1, r2, r3

    def _cache_outer(self) -> None:
        """The lower triangle of every row's outer product, phi_n phi_n^T.

        The per-instant Gram matrix is symmetric, so only its R3 (R3 + 1) / 2
        lower-triangular entries are computed: the cached table and the GEMM
        that dominates the solve both shrink by almost half, with no change
        to the result.
        """
        rows, cols = self.tril
        self.outer_tri = (self.Phi[:, rows] * self.Phi[:, cols]).contiguous()

    def _set_basis(self, Y) -> None:
        """Spatial factors, core and the dictionary Phi, from a completed tensor.

        U1 and U2 are the leading left singular vectors of the two small
        unfoldings - a HOSVD step on the current completion.  The core's mode-3
        unfolding V is the leading temporal subspace of the projected tensor and
        is the identity when R3 = R1 R2.  With ``dictionary == "cp"`` (the
        ablations) Phi is instead the Khatri-Rao product of rank-R3 CP factors.
        """
        if self.dictionary == "cp":
            self._set_cp_basis(Y)
            return
        self.U1 = top_left_singular(unfold(Y, 0), self.r1)
        self.U2 = top_left_singular(unfold(Y, 1), self.r2)
        if self.r3 >= self.r1 * self.r2:
            self.V = torch.eye(self.r1 * self.r2)
        else:
            projected = mode_product(mode_product(Y, self.U1.transpose(0, 1), 0),
                                     self.U2.transpose(0, 1), 1)
            self.V = top_left_singular(
                projected.reshape(self.r1 * self.r2, self.T), self.r3)
        core = self.V.reshape(self.r1, self.r2, self.r3)
        self.Phi = torch.einsum("li,ijk,aj->lak", self.U1, core, self.U2
                                ).reshape(self.P, self.r3).contiguous()
        self._cache_outer()

    def _set_cp_basis(self, Y) -> None:
        """CP dictionary for the ablations: Phi = normalised U1 (Khatri-Rao) U2."""
        first = self._cp_factors is None
        self._cp_factors = cp_als(Y, self.r3,
                                  self.cfg.cp_iters * (3 if first else 1),
                                  init=self._cp_factors)
        F1, F2, _ = self._cp_factors
        Phi = (F1.unsqueeze(1) * F2.unsqueeze(0)).reshape(self.P, self.r3)
        self.Phi = (Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-12)).contiguous()
        self.U1, self.U2 = F1, F2
        self._cache_outer()

    def _complete(self) -> None:
        """L = U3 Phi^T on every cell, written into the existing buffer."""
        torch.matmul(self.U3, self.Phi.transpose(0, 1), out=self.Lt)

    def _working_tensor(self):
        """Trusted cells as measured, every other cell from the model.

        (The sparse term is zero on every trusted cell, so ``X - S`` there is
        just ``X``.)
        """
        return self._unflatten(torch.where(self.We, self.Xt, self.Lt))

    # -- the steps ------------------------------------------------------------

    def _reject_step(self) -> None:
        """Move the cells that do not belong into S, and out of the fit.

        The threshold is ``spike_kappa`` robust sigmas of that attribute's own
        residual, re-decided every iteration; a rejected cell gets weight zero
        in the next solve (an iteratively reweighted least squares).  The
        residual variance on the cells that stay - the sigma^2 of the
        empirical-Bayes prior weight - falls out of the same pass and is
        cached, which saves the prior step three passes over the tensor.
        """
        residual = (self.Xt - self.Lt).masked_fill_(~self.Mt, 0.0)
        scale = torch.empty(self.A)
        for a in range(self.A):
            column = residual[:, a::self.A]
            keep = self.Mt[:, a::self.A]
            scale[a] = torch_robust_sigma(column[keep])
        threshold = (self.cfg.spike_kappa * scale).repeat(self.L)
        flagged = (residual.abs() > threshold) & self.Mt
        self.We = self.Mt & ~flagged
        self.n_flagged = int(flagged.sum())
        residual.masked_fill_(flagged, 0.0)
        flat = residual.reshape(-1)
        self.n_we = float(self.We.sum())
        self.sigma2 = float(torch.dot(flat, flat)) / max(1.0, self.n_we)

    def _solve_u3(self, prior: Optional[torch.Tensor],
                  weight: Optional[torch.Tensor]) -> None:
        """One independent ridge least squares per instant, batched over t.

        ``(Phi^T diag(w_t) Phi + D_t) u = Phi^T (w_t o x_t) + W_t p(t)``,
        ``D_t = W_t + eps I``.  The lower triangles of the Gram matrices come
        out of one GEMM against the cached outer products, are mirrored, and
        the systems are solved by batched Cholesky.  W is per instant and per
        coefficient; it is ``None`` during the warm-up, when there is no prior.
        """
        ridge = float(self.cfg.lambda_ridge)
        rows, cols = self.tril
        out = torch.empty(self.T, self.r3)
        for begin in range(0, self.T, self.chunk):
            stop = min(self.T, begin + self.chunk)
            w = self.We[begin:stop].to(self.Xt.dtype)
            lower = w @ self.outer_tri                          # (chunk, R3(R3+1)/2)
            gram = torch.zeros(stop - begin, self.r3, self.r3)
            gram[:, rows, cols] = lower
            gram[:, cols, rows] = lower
            rhs = (self.Xt[begin:stop] * w) @ self.Phi
            if prior is not None and weight is not None:
                here = weight[begin:stop]
                rhs = rhs + here * prior[begin:stop]
                gram.diagonal(dim1=-2, dim2=-1).add_(here + ridge)
            else:
                gram.diagonal(dim1=-2, dim2=-1).add_(ridge)
            factor, info = torch.linalg.cholesky_ex(gram)
            if int(info.max()) == 0:
                out[begin:stop] = torch.cholesky_solve(rhs.unsqueeze(2), factor).squeeze(2)
            else:                       # never seen, but a solve is cheap insurance
                out[begin:stop] = torch.linalg.solve(gram, rhs.unsqueeze(2)).squeeze(2)
        self.U3 = out

    def _sequences(self):
        """The normalised coefficient sequence, as the grouped network's input.

        Channels are put on a common scale before they reach the network, and
        the scale is handed back when the prior is used.  Group 1 reads the
        sequence forwards, group 2 (when bidirectional) reads it backwards.
        """
        scale = self.U3.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
        normalised = (self.U3 / scale).detach()
        forward = torch.cat([normalised.transpose(0, 1), self.diurnal], dim=0)
        parts = [forward] if self.directions == 1 else [forward, forward.flip(-1)]
        return normalised, scale, torch.cat(parts, dim=0).unsqueeze(0)

    def _train_tcn(self, steps: int) -> None:
        """Adam on the one-step-ahead prediction loss of both directions."""
        normalised, _, sequence = self._sequences()
        ahead = self.targets[self.targets >= 1]
        behind = self.targets[self.targets <= self.T - 2]
        behind_cols = (self.T - 2) - behind     # reversed position predicting t
        r3 = self.r3
        self.net.train()
        for _ in range(max(0, steps)):
            self.optimiser.zero_grad(set_to_none=True)
            out = self.net(sequence)[0]                     # (G R3, T)
            loss = torch.zeros(())
            if ahead.numel():
                predicted = out[:r3, ahead - 1].transpose(0, 1)
                loss = loss + ((normalised[ahead] - predicted) ** 2).sum(dim=1).mean()
            if self.directions == 2 and behind.numel():
                predicted = out[r3:, behind_cols].transpose(0, 1)
                loss = loss + ((normalised[behind] - predicted) ** 2).sum(dim=1).mean()
            if not loss.requires_grad:
                return
            loss.backward()
            self.optimiser.step()

    def _prior_and_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The network's estimate of u3, and how much to believe it.

        The two directions are combined by inverse variance, each one's error
        measured on the dense instants.  The weight is the empirical-Bayes
        ``sigma^2 / v`` capped at ``prior_cap`` times the observation density
        (see PRIOR WEIGHT in the header); sigma^2 comes from the reject step.
        An instant no network could predict gets no prior at all.
        """
        normalised, scale, sequence = self._sequences()
        r3 = self.r3
        self.net.eval()
        p_f = torch.zeros(self.T, r3)
        p_b = torch.zeros(self.T, r3)
        have_f = torch.zeros(self.T, 1)
        have_b = torch.zeros(self.T, 1)
        with torch.no_grad():
            out = self.net(sequence)[0]
            p_f[1:] = out[:r3, :-1].transpose(0, 1)
            have_f[1:] = 1.0
            if self.directions == 2:
                p_b[:-1] = out[r3:, :-1].transpose(0, 1).flip(0)
                have_b[:-1] = 1.0
        del out

        rows = self.targets
        ahead = rows[rows >= 1]
        behind = rows[rows <= self.T - 2]
        precision = torch.zeros(self.T, r3)
        combined = torch.zeros(self.T, r3)
        pairs = [(p_f, have_f, ahead)]
        if self.directions == 2:
            pairs.append((p_b, have_b, behind))
        for predicted, valid, index in pairs:
            if index.numel() == 0:
                continue
            error = ((normalised[index] - predicted[index]) ** 2
                     ).mean(dim=0).clamp_min(1e-12)
            share = valid / error
            precision = precision + share
            combined = combined + share * predicted
        covered = precision > 0
        prior = torch.where(covered, combined / precision.clamp_min(1e-12),
                            torch.zeros_like(combined)) * scale
        gate = covered[:, :1].to(prior.dtype)

        variance = ((self.U3[rows] - prior[rows]) ** 2).mean(dim=0).clamp_min(1e-12)
        cap = self.cfg.prior_cap * self.n_we / max(1.0, float(self.T * self.P))
        weight = (self.sigma2 / variance).clamp(0.0, max(cap, 1e-6))
        if self.cfg.lambda_prior > 0:
            weight = torch.full_like(weight, float(self.cfg.lambda_prior))
        self.lambda_prior = float(weight.mean() * gate.mean())
        return prior, weight * gate

    # -- selection, residual stage --------------------------------------------

    def model_tensor(self) -> np.ndarray:
        """L in the units of the data."""
        return destandardise(self._unflatten(self.Lt).detach().cpu().numpy(),
                             self.mu, self.sd)

    def _track_validation(self, iteration: int) -> None:
        """Keep the iterate with the lowest clipped error on the held-out cells.

        A gather of the held-out cells (3 % of the readable ones) rather than a
        pass over the whole tensor, and the selected iterate is copied into one
        buffer that is allocated once.
        """
        if self.n_val <= 0:
            return
        r = self.val_target - self.Lt.reshape(-1)[self.val_idx]
        loss = float(torch.clamp(r * r, max=self.cfg.val_clip ** 2).mean())
        if math.isfinite(loss) and loss < self.best_val:
            self.best_val, self.best_iteration = loss, iteration
            if self.best_L is None:
                self.best_L = self.Lt.clone()
            else:
                self.best_L.copy_(self.Lt)

    def _attribute_scale(self, values, keep) -> torch.Tensor:
        """Robust sigma of ``values`` per attribute over ``keep``, as a (P,) row."""
        scale = torch.empty(self.A)
        for a in range(self.A):
            here = keep[:, a::self.A]
            scale[a] = torch_robust_sigma(values[:, a::self.A][here])
        return scale.repeat(self.L)

    def _residual_stage(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """AR(1) residual completion and the leave-one-out outlier test.

        Pass 1 trusts every readable cell within --spike-kappa robust sigmas of
        the low-rank model and predicts every cell's residual from its trusted
        neighbours (:func:`ar_bridge`); for a readable cell that prediction
        never reads the cell itself.  A readable cell whose leave-one-out
        innovation exceeds --innov-kappa robust sigmas of its attribute's
        innovations is an outlier (the Hampel identifier on a model-based
        prediction).  Pass 2 re-runs the bridge on the cells that passed and
        returns L + r_hat with the outlier mask.
        """
        cfg = self.cfg
        available = self.Mt | self.Vt
        resid = self.Xt - self.Lt
        scale = self._attribute_scale(resid, available)
        trusted = available & (resid.abs() <= cfg.spike_kappa * scale)
        r_hat, _, _ = ar_bridge(resid, trusted, cfg.ar_max_phi)
        innovation = resid - r_hat
        del r_hat
        spread = self._attribute_scale(innovation, trusted)
        flags = available & (innovation.abs() > cfg.innov_kappa * spread)
        del innovation
        r_hat, phi, gamma = ar_bridge(resid, available & ~flags, cfg.ar_max_phi)
        self.stage_info = {
            "phi_median": float(phi.median()), "gamma_median": float(gamma.median()),
            "flagged_pass1": float((available & ~trusted).sum()),
            "flagged_final": float(flags.sum())}
        return r_hat.add_(self.Lt), flags

    # -- the loop -------------------------------------------------------------

    def fit(self):
        cfg = self.cfg
        K = max(1, cfg.iters)

        # Initialisation.  A hard cap at clip_sigma standardised units keeps
        # the grossest spikes out of the first HOSVD and out of the fit.
        extreme = (self.Xt.abs() > cfg.clip_sigma) & self.Mt
        self.We = self.Mt & ~extreme
        self.n_flagged = int(extreme.sum())
        del extreme
        self._set_basis(initial_fill(
            self._unflatten(self.Xt.clamp(-cfg.clip_sigma, cfg.clip_sigma)),
            self._fill_mask))
        self._fill_mask = None
        self._solve_u3(None, None)
        self._complete()
        self._track_validation(0)

        for iteration in range(1, K + 1):
            # Without a prior there is no network to keep a fixed frame for, so
            # the pure-TD / pure-CP ablations refresh the basis every iteration
            # (EM-style masked ALS), which is the classical form of each model.
            warming = iteration <= self.warm_iters or not self.use_prior
            self._reject_step()
            if warming:
                self._set_basis(self._working_tensor())
                self._solve_u3(None, None)
            else:
                first = iteration == self.warm_iters + 1
                self._train_tcn(cfg.tcn_steps * (cfg.tcn_warmup if first else 1))
                prior, weight = self._prior_and_weight()
                self._solve_u3(prior, weight)
                del prior, weight
            self._complete()
            self._track_validation(iteration)
            if self.trace_idx is not None:
                self.tracker.trace(iteration, lambda: self.Lt.reshape(-1)[
                    self.trace_idx].detach().cpu().numpy())

        # Model selection: the iterate with the lowest held-out error.  There
        # is deliberately no early exit; the budget is --iters.
        if self.best_L is not None:
            self.Lt = self.best_L
            self.best_L = None
            self.tracker.record(K, self.best_val, self.best_iteration)
        if self.residual_stage:
            completed, flags = self._residual_stage()
            self.Lt = completed
            self.own_flags = self._unflatten(flags).detach().cpu().numpy()
            del flags
        model = self.model_tensor()

        info = self.tracker.info(K)
        info.update(self.stage_info, best_iteration=self.best_iteration,
                    rank3=self.r3, dictionary=self.dictionary, prior=self.use_prior,
                    residual_stage=self.residual_stage,
                    best_x=float(max(0, self.best_iteration)),
                    val_loss=self.best_val if self.n_val > 0 else float("nan"),
                    validated=self.n_val > 0,
                    lambda_prior=self.lambda_prior, rejected=self.n_flagged)
        return model, self.tracker.points, self.tracker.seconds, {
            "flags": self.own_flags, "info": info}


def run_dltd(exp: Experiment, cfg: Config):
    """Fit Deep-TD, the full method.

    Returns the model tensor, the convergence curve, the seconds spent building
    that curve (which Result 4 subtracts), and the method's own outlier flags
    with a few facts about the fit.
    """
    return TuckerTCN(exp, cfg).fit()


def run_dltd_core(exp: Experiment, cfg: Config):
    """Ablation: TD + TCN without the residual stage."""
    return TuckerTCN(exp, cfg, residual_stage=False).fit()


def run_td(exp: Experiment, cfg: Config):
    """Ablation: pure Tucker - same ranks and robust loop, no TCN prior."""
    return TuckerTCN(exp, cfg, use_prior=False, residual_stage=False).fit()


def run_td_res(exp: Experiment, cfg: Config):
    """Ablation: Tucker + residual stage, no TCN prior."""
    return TuckerTCN(exp, cfg, use_prior=False, residual_stage=True).fit()


def run_cp(exp: Experiment, cfg: Config):
    """Ablation: pure CP - Khatri-Rao dictionary, no TCN prior."""
    return TuckerTCN(exp, cfg, dictionary="cp", use_prior=False,
                     residual_stage=False).fit()


def run_cp_tcn(exp: Experiment, cfg: Config):
    """Ablation: CP + the same TCN prior, no residual stage."""
    return TuckerTCN(exp, cfg, dictionary="cp", residual_stage=False).fit()


# ==============================================================================
# 7.  Shared pieces of the baselines
# ==============================================================================

def unit_scale(X_std: np.ndarray, mask: np.ndarray, quantile: float = 1e-3
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-attribute affine map of the standardised data into (0, 1).

    M2 (DATC) and M7 (NTC) end in a sigmoid, so both papers work on data
    normalised into [0, 1].  A plain min-max map would let a single outlier of
    the corruption protocol set the range, so the map sends the
    ``quantile`` and ``1 - quantile`` quantiles of each attribute's readable
    values to 0.1 and 0.9: the sigmoid can still express values a few tenths
    of the range beyond them, and the impulses land far outside where they
    cannot bend the fit.  Returns ``(gain, offset)``, ``u = gain x + offset``.
    """
    A = X_std.shape[1]
    gain = np.ones(A, dtype=np.float32)
    offset = np.zeros(A, dtype=np.float32)
    for a in range(A):
        values = X_std[:, a][mask[:, a]]
        if values.size < 2:
            gain[a], offset[a] = 0.4, 0.5
            continue
        lo, hi = np.quantile(values, [quantile, 1.0 - quantile])
        span = float(hi - lo) if hi > lo else 1.0
        gain[a] = 0.8 / span
        offset[a] = 0.1 - gain[a] * float(lo)
    return gain, offset


class EntryData:
    """The observed entries of a per-entry network (M4, M6, M7), on the host.

    One flat C-order index per entry (int32 when it fits) and its standardised
    value; the three mode indices are derived per mini-batch, which is a third
    of the memory of storing them.  :meth:`batches` walks a shuffled
    permutation of the entries and reshuffles when it runs out, so successive
    iterations are successive slices of ordinary epochs.
    """

    def __init__(self, X_std: np.ndarray, fit_mask: np.ndarray):
        self.shape = tuple(int(s) for s in X_std.shape)
        flat = np.flatnonzero(fit_mask.reshape(-1))
        self.flat = flat.astype(np.int32) if X_std.size < 2 ** 31 else flat
        self.values = np.asarray(X_std.reshape(-1)[flat], dtype=np.float32)
        self.n = int(flat.size)
        self._order: Optional[np.ndarray] = None
        self._cursor = 0

    def index(self, flat: np.ndarray) -> List[torch.Tensor]:
        """C-order flat indices -> the three mode indices, as long tensors."""
        L, A, T = self.shape
        flat = np.asarray(flat, dtype=np.int64)
        return [torch.as_tensor(flat // (A * T)), torch.as_tensor((flat // T) % A),
                torch.as_tensor(flat % T)]

    def batches(self, count: int, batch: int):
        """Yield ``(index, target)`` mini-batches covering ``count`` entries."""
        remaining = min(int(count), self.n)
        while remaining > 0:
            if self._order is None or self._cursor >= self.n:
                order = np.random.permutation(self.n)
                self._order = order.astype(np.int32) if self.n < 2 ** 31 else order
                self._cursor = 0
            take = min(batch, remaining, self.n - self._cursor)
            chosen = self._order[self._cursor:self._cursor + take]
            self._cursor += take
            remaining -= take
            yield self.index(self.flat[chosen]), torch.as_tensor(self.values[chosen])


def predict_entries(model_fn: Callable[[List[torch.Tensor]], torch.Tensor],
                    data: EntryData, flat: Optional[np.ndarray] = None,
                    chunk: int = 1 << 16) -> np.ndarray:
    """A per-entry network's prediction at ``flat`` (every cell when None).

    Evaluated in chunks under ``no_grad`` so that no more than ``chunk``
    entries' activations exist at once - the reason the whole-tensor
    prediction of M7, whose per-entry network is a 3-D CNN, fits in memory.
    """
    total = int(np.prod(data.shape)) if flat is None else int(np.asarray(flat).size)
    out = np.empty(total, dtype=np.float32)
    with torch.no_grad():
        for begin in range(0, total, chunk):
            stop = min(total, begin + chunk)
            part = (np.arange(begin, stop, dtype=np.int64) if flat is None
                    else np.asarray(flat[begin:stop], dtype=np.int64))
            out[begin:stop] = model_fn(data.index(part)).detach().cpu().numpy()
    return out


# ==============================================================================
# 8.  M2  DATC - Deep Adversarial Tensor Completion (Xie et al., IEEE/ACM ToN 2023)
# ==============================================================================
#
# A 3-D convolutional autoencoder generator G estimates the missing entries of a
# sliding sub-tensor X_t (L x A x w), and an encoder-decoder discriminator D
# emits one probability per entry that the entry is a real measurement (eq. 5).
# As in the paper (Sec. III-B, VI-A):
#   * the encoder has n down-sampling units, each a stride-(2,2,1) convolution
#     followed by a stride-(1,1,2) one ("same" padding), then one
#     valid-padding convolution to the latent code; the decoder is one
#     valid-padding deconvolution and n strided deconvolutions.  With n = 2 the
#     kernel counts are [4, 8, 16, 32, 64, 32, 8, 1], all kernels 3x3x3;
#   * every layer is followed by layer normalisation and ReLU, except the last,
#     which is a sigmoid (data normalised into [0, 1], see :func:`unit_scale`);
#   * G's first layer is the missing-data-aware convolution of eq. (4);
#   * G minimises eq. (9) - the masked MSE plus lambda times the mean over the
#     missing entries of log(1 - D) - and D maximises eq. (10); one D step then
#     one G step per sub-tensor (Algorithm 1), Adam, learning rate 1e-3;
#   * window w = 12, lambda = 0.1 (the paper's GEANT value; 0.5 on Abilene).
# Two adaptations, both for cost: sub-tensors are processed in mini-batches of
# --datc-batch rather than one at a time, and an epoch visits the T - w + 1
# windows of the paper at a stride of w from a random offset, so one epoch is
# one pass over the tensor and every window is visited over the epochs.
# Completion is eq. (3): measured where observed, G's output elsewhere.
# ------------------------------------------------------------------------------

def _fit_to(x, size: Sequence[int]):
    """Crop or zero-pad the trailing three axes of ``x`` to exactly ``size``."""
    out = x[..., :size[0], :size[1], :size[2]]
    pad = [0, size[2] - out.shape[-1], 0, size[1] - out.shape[-2],
           0, size[0] - out.shape[-3]]
    return F.pad(out, pad) if any(p > 0 for p in pad) else out


class MissConv3d(nn.Module):
    """DATC eq. (4): convolution normalised by the observed count in its field."""

    def __init__(self, c_out: int, kernel: int = 3, stride=(2, 2, 1), padding: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(1, c_out, kernel, stride, padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.stride, self.padding = stride, padding
        self.register_buffer("ones", torch.ones(1, 1, kernel, kernel, kernel))

    def forward(self, x, m):
        numerator = self.conv(x * m)
        count = F.conv3d(m, self.ones, stride=self.stride, padding=self.padding)
        out = torch.where(count > 0, numerator / count.clamp_min(1e-6),
                          torch.zeros_like(numerator))
        return out + self.bias.view(1, -1, 1, 1, 1)


class DATCNet(nn.Module):
    """The 3-D convolutional autoencoder shared by G and D (DATC Fig. 4)."""

    def __init__(self, shape: Sequence[int], units: int = 2, width: int = 4,
                 missing_aware: bool = True):
        super().__init__()
        units = max(1, int(units))
        self.shape = tuple(int(s) for s in shape)
        self.sizes: List[Tuple[int, int, int]] = [self.shape]
        enc: List[nn.Module] = []
        enc_norm: List[nn.Module] = []
        channels: List[int] = []
        size = list(self.shape)
        c_prev = 1
        for _ in range(units):
            for stride in ((2, 2, 1), (1, 1, 2)):
                c = width * 2 ** len(channels)
                if not channels and missing_aware:
                    enc.append(MissConv3d(c, 3, stride, 1))
                else:
                    enc.append(nn.Conv3d(c_prev, c, 3, stride, 1))
                enc_norm.append(nn.GroupNorm(1, c))          # = layer normalisation
                size = [(s + 2 - 3) // st + 1 for s, st in zip(size, stride)]
                self.sizes.append(tuple(size))
                channels.append(c)
                c_prev = c
        latent = width * 2 ** len(channels)
        self.enc = nn.ModuleList(enc)
        self.enc_norm = nn.ModuleList(enc_norm)
        self.to_code = nn.Conv3d(c_prev, latent, tuple(size))           # valid
        self.code_norm = nn.GroupNorm(1, latent)
        self.from_code = nn.ConvTranspose3d(latent, c_prev, tuple(size))  # valid
        self.from_norm = nn.GroupNorm(1, c_prev)
        up: List[nn.Module] = []
        up_norm: List[nn.Module] = []
        self.targets: List[Tuple[int, int, int]] = []
        c_in = c_prev
        for k in range(1, units + 1):
            c_out = channels[2 * (units - k) - 1] if k < units else 1
            up.append(nn.ConvTranspose3d(c_in, c_out, 3, stride=2, padding=1,
                                         output_padding=1))
            up_norm.append(nn.GroupNorm(1, c_out) if k < units else nn.Identity())
            self.targets.append(self.sizes[2 * (units - k)])
            c_in = c_out
        self.up = nn.ModuleList(up)
        self.up_norm = nn.ModuleList(up_norm)

    def forward(self, x, m=None, logits: bool = False):
        """(B, 1, L, A, w) -> (B, 1, L, A, w) in (0, 1), or the pre-sigmoid logits."""
        h = x
        for layer, norm in zip(self.enc, self.enc_norm):
            h = layer(h, m if m is not None else torch.ones_like(x)) \
                if isinstance(layer, MissConv3d) else layer(h)
            h = F.relu(norm(h))
        h = F.relu(self.code_norm(self.to_code(h)))
        h = F.relu(self.from_norm(_fit_to(self.from_code(h), self.sizes[-1])))
        last = len(self.up) - 1
        for i, (layer, norm, target) in enumerate(zip(self.up, self.up_norm, self.targets)):
            h = norm(_fit_to(layer(h), target))
            if i < last:
                h = F.relu(h)
        return h if logits else torch.sigmoid(h)


def _datc_windows(T: int, w: int, stride: int, offset: int = 0) -> np.ndarray:
    """Start indices of sliding windows, with the tail always covered."""
    last = max(0, T - w)
    starts = list(range(min(offset, last), last + 1, max(1, stride)))
    if not starts or starts[-1] != last:
        starts.append(last)
    if starts[0] != 0 and offset == 0:
        starts.insert(0, 0)
    return np.asarray(starts, dtype=np.int64)


def run_datc(exp: Experiment, cfg: Config):
    """Train DATC for ``--iters`` epochs; return the best generator by held-out cells."""
    L, A, T = exp.shape
    held = validation_split(exp, cfg) if cfg.val_frac > 0 else None
    X_std, mu, sd = standardise(exp.X_in, exp.mask_in)
    fit_mask = exp.mask_in if held is None else exp.mask_in & ~held
    tracker = Tracker(exp, cfg, X_std, mu, sd, held)
    gain, offset = unit_scale(X_std, fit_mask)
    unit = X_std * gain[None, :, None] + offset[None, :, None]
    unit[~fit_mask] = 0.0                      # the placeholder of a missing entry
    U = torch.as_tensor(unit)
    M = torch.as_tensor(fit_mask.astype(np.float32))
    del unit
    gain_t = torch.as_tensor(gain).view(1, A, 1)
    offset_t = torch.as_tensor(offset).view(1, A, 1)

    w = int(min(cfg.datc_window, T))
    stride = int(cfg.datc_stride or w)
    cover = _datc_windows(T, w, w)
    generator = DATCNet((L, A, w), cfg.datc_units, cfg.datc_width, missing_aware=True)
    discriminator = DATCNet((L, A, w), cfg.datc_units, cfg.datc_width, missing_aware=False)
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.datc_lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.datc_lr)
    lam = float(cfg.datc_lambda)

    def batch_of(index: np.ndarray):
        xs = torch.stack([U[:, :, s:s + w] for s in index]).unsqueeze(1)
        ms = torch.stack([M[:, :, s:s + w] for s in index]).unsqueeze(1)
        return xs, ms

    def lay_down() -> np.ndarray:
        """G's output on every cell, standardised: windows laid side by side."""
        total = torch.zeros(L, A, T)
        count = torch.zeros(1, 1, T)
        with torch.no_grad():
            for begin in range(0, cover.size, cfg.datc_batch):
                index = cover[begin:begin + cfg.datc_batch]
                x, m = batch_of(index)
                out = generator(x, m)[:, 0]
                for position, start in enumerate(index):
                    total[:, :, start:start + w] += out[position]
                    count[:, :, start:start + w] += 1.0
        unit_out = total / count.clamp_min(1.0)
        return ((unit_out - offset_t) / gain_t).cpu().numpy()

    best_state = None
    K = max(1, cfg.iters)
    for epoch in range(1, K + 1):
        shift = int(np.random.randint(0, max(1, min(stride, T - w + 1))))
        order = np.random.permutation(_datc_windows(T, w, stride, shift))
        for begin in range(0, order.size, cfg.datc_batch):
            x, m = batch_of(order[begin:begin + cfg.datc_batch])
            n_obs = m.sum().clamp_min(1.0)
            n_hole = (1 - m).sum().clamp_min(1.0)

            # maximisation part, eq. (10): D learns which entries are real
            with torch.no_grad():
                filled = m * x + (1 - m) * generator(x, m)
            z = discriminator(filled, logits=True)
            value_d = lam * ((m * F.logsigmoid(z)).sum() / n_obs
                             + ((1 - m) * F.logsigmoid(-z)).sum() / n_hole)
            opt_d.zero_grad(set_to_none=True)
            (-value_d).backward()
            opt_d.step()

            # minimisation part, eq. (9): masked MSE + fooling D on the holes
            discriminator.requires_grad_(False)
            estimate = generator(x, m)
            filled = m * x + (1 - m) * estimate
            z = discriminator(filled, logits=True)
            loss_g = ((m * (x - estimate) ** 2).sum() / n_obs
                      + lam * ((1 - m) * F.logsigmoid(-z)).sum() / n_hole)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            discriminator.requires_grad_(True)
        if full_checkpoint(tracker, epoch, lay_down):
            best_state = clone_state(generator)

    if best_state is not None:
        generator.load_state_dict(best_state)
    model = destandardise(lay_down(), mu, sd)
    return model, tracker.points, tracker.seconds, {"info": tracker.info(K)}


# ==============================================================================
# 9.  M3  DeepTensor - low-rank decomposition with deep network priors
#     (Saragadam et al., IEEE TPAMI 2024)
# ==============================================================================
#
# Eq. (8): X ~ f_U(z_U) o f_V(z_V) o f_W(z_W), a PARAFAC decomposition whose
# three factor matrices are the outputs of three untrained 1-D networks fed
# fixed uniform-noise codes; the optimisation runs over the network weights
# with Adam (learning rate 1e-3) on the l2 loss over the observed entries.  As
# the paper prescribes for imputation and denoising, the networks are the
# over-parameterised Deep-Image-Prior architecture with skip connections,
# changed from 2-D to 1-D with the output channels equal to the rank (Sec.
# IV, Fig. 5): an encoder of strided convolutions, a decoder of linear
# up-sampling, a 1x1 skip branch at every scale, batch normalisation and
# LeakyReLU, reflection padding.  The deep prior carries no penalty term; it
# regularises because it fits signal before noise, so the stopping point is
# the regulariser (Fig. 8) - which is what the held-out selection picks.
# --dt-loss l1 is the variant the paper reports for spatially sparse noise.
# ------------------------------------------------------------------------------

def _dip_conv(c_in: int, c_out: int, kernel: int, stride: int, length: int) -> nn.Conv1d:
    pad = (kernel - 1) // 2
    mode = "reflect" if pad > 0 and length > pad else "zeros"
    return nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, padding_mode=mode)


def _dip_norm(channels: int, length: int) -> nn.Module:
    return nn.BatchNorm1d(channels) if length > 1 else nn.Identity()


class _DIPLevel(nn.Module):
    """One scale of the DIP skip network: down, recurse, up, concat skip."""

    def __init__(self, c_in: int, width: int, skip: int, depth: int, length: int):
        super().__init__()
        half = max(1, length // 2)
        self.skip = (nn.Sequential(_dip_conv(c_in, skip, 1, 1, length),
                                   _dip_norm(skip, length), nn.LeakyReLU(0.2))
                     if skip > 0 else None)
        self.down = nn.Sequential(
            _dip_conv(c_in, width, 3, 2, length), _dip_norm(width, half), nn.LeakyReLU(0.2),
            _dip_conv(width, width, 3, 1, half), _dip_norm(width, half), nn.LeakyReLU(0.2))
        self.inner = _DIPLevel(width, width, skip, depth - 1, half) if depth > 1 else None
        joined = width + max(0, skip)
        self.up = nn.Sequential(
            _dip_norm(joined, length),
            _dip_conv(joined, width, 3, 1, length), _dip_norm(width, length), nn.LeakyReLU(0.2),
            _dip_conv(width, width, 1, 1, length), _dip_norm(width, length), nn.LeakyReLU(0.2))

    def forward(self, x):
        deeper = self.down(x)
        if self.inner is not None:
            deeper = self.inner(deeper)
        deeper = F.interpolate(deeper, size=x.shape[-1], mode="linear", align_corners=False)
        parts = [deeper] if self.skip is None else [self.skip(x), deeper]
        return self.up(torch.cat(parts, dim=1))


class FactorNet(nn.Module):
    """1-D DIP network emitting one ``(n, rank)`` factor matrix."""

    def __init__(self, n: int, rank: int, width: int = 64, levels: int = 5,
                 skip: int = 4, in_channels: int = 32):
        super().__init__()
        self.n = int(n)
        levels = max(0, min(int(levels), int(math.floor(math.log2(max(1, n)))) - 1))
        step = 2 ** levels
        self.size = int(math.ceil(n / step) * step)
        self.register_buffer("code", torch.rand(1, in_channels, self.size) * 0.1)
        if levels > 0:
            self.body = _DIPLevel(in_channels, width, skip, levels, self.size)
        else:
            self.body = nn.Sequential(
                _dip_conv(in_channels, width, 3, 1, self.size), _dip_norm(width, self.size),
                nn.LeakyReLU(0.2),
                _dip_conv(width, width, 3, 1, self.size), _dip_norm(width, self.size),
                nn.LeakyReLU(0.2))
        self.head = nn.Conv1d(width, rank, 1)

    def forward(self):
        out = self.head(self.body(self.code))                    # (1, rank, size)
        return out[0, :, :self.n].transpose(0, 1)                # (n, rank)


def run_deeptensor(exp: Experiment, cfg: Config):
    """Fit the deep-prior CP model; one iteration = --dt-steps-per-iter Adam steps."""
    L, A, T = exp.shape
    held = validation_split(exp, cfg) if cfg.val_frac > 0 else None
    X_std, mu, sd = standardise(exp.X_in, exp.mask_in)
    fit_mask = exp.mask_in if held is None else exp.mask_in & ~held
    tracker = Tracker(exp, cfg, X_std, mu, sd, held)
    X = torch.as_tensor(X_std)
    M = torch.as_tensor(fit_mask)
    n_obs = max(1.0, float(fit_mask.sum()))
    rank = max(1, cfg.dt_rank)
    nets = nn.ModuleList([FactorNet(n, rank, cfg.dt_width, cfg.dt_levels, cfg.dt_skip,
                                    cfg.dt_input) for n in (L, A, T)])
    optimiser = torch.optim.Adam(nets.parameters(), lr=cfg.dt_lr)
    l1 = cfg.dt_loss == "l1"

    def reconstruct():
        U, V, W = (net() for net in nets)
        kr = (U.unsqueeze(1) * V.unsqueeze(0)).reshape(L * A, rank)
        return (kr @ W.transpose(0, 1)).reshape(L, A, T)

    def current() -> np.ndarray:
        # the batch-norm layers stay in training mode: they are part of the
        # prior, and switching to running statistics would change the factors.
        with torch.no_grad():
            return reconstruct().cpu().numpy()

    best_state = None
    K = max(1, cfg.iters)
    for iteration in range(1, K + 1):
        for _ in range(max(1, cfg.dt_steps_per_iter)):
            optimiser.zero_grad(set_to_none=True)
            diff = (X - reconstruct()).masked_fill(~M, 0.0)
            loss = (diff.abs().sum() if l1 else (diff * diff).sum()) / n_obs
            loss.backward()
            optimiser.step()
        if full_checkpoint(tracker, iteration, current):
            best_state = clone_state(nets)

    if best_state is not None:
        nets.load_state_dict(best_state)
    model = destandardise(current(), mu, sd)
    return model, tracker.points, tracker.seconds, {"info": tracker.info(K)}


# ==============================================================================
# 10.  M4  NeuralCP - Bayesian multiway analysis with neural tensor decomposition
#      (Liu et al., Cognitive Computation 2018)
# ==============================================================================
#
# Each row of each factor matrix has a Gaussian variational posterior
# N(mu, diag(lambda)^-1) (eq. 5) under a shared, learnt Gaussian prior; the
# sampled rows (reparameterisation trick) are concatenated and pass through one
# shared tanh layer h = tanh(W u + b) (eq. 4) whose two linear read-outs give
# the mean and the log-variance of the entry (eq. 3).  The loss is the negative
# ELBO of eq. (9): the Gaussian negative log-likelihood of the observed entries
# plus the KL divergence of *every* factor row from the prior, estimated per
# mini-batch as  mean NLL + KL_total / n_observed.  Adam on mini-batches of
# observed entries (Algorithm 1).  R = 50 and K = 100 hidden units are the
# paper's real-data settings.  Adaptations for scale: mini-batches of
# --ncp-batch (the paper uses 30) with learning rate 1e-3.
# ------------------------------------------------------------------------------

class NeuralCP(nn.Module):

    def __init__(self, shape: Sequence[int], rank: int, hidden: int):
        super().__init__()
        self.rank = rank
        self.mu = nn.ParameterList(
            [nn.Parameter(torch.randn(int(n), rank) * 0.1) for n in shape])
        self.log_var = nn.ParameterList(
            [nn.Parameter(torch.full((int(n), rank), -4.0)) for n in shape])
        self.prior_mu = nn.Parameter(torch.zeros(rank))
        self.prior_log_var = nn.Parameter(torch.zeros(rank))
        self.W = nn.Linear(len(shape) * rank, hidden)
        self.head_mu = nn.Linear(hidden, 1)
        self.head_log_var = nn.Linear(hidden, 1)

    def latents(self, index, sample: bool):
        out = []
        for mode, rows in enumerate(index):
            mean = self.mu[mode][rows]
            if not sample:
                out.append(mean)
                continue
            std = (0.5 * self.log_var[mode][rows]).exp()
            out.append(mean + std * torch.randn_like(std))
        return out

    def forward(self, index, sample: bool = True):
        h = torch.tanh(self.W(torch.cat(self.latents(index, sample), dim=1)))
        return self.head_mu(h).squeeze(1), self.head_log_var(h).squeeze(1).clamp(-8, 8)

    def kl(self):
        """KL(q(U) || p(U)) summed over every row of every factor, eq. (9)."""
        total = 0.0
        prior_var = self.prior_log_var.exp()
        for mean, log_var in zip(self.mu, self.log_var):
            delta = mean - self.prior_mu
            total = total + 0.5 * (self.prior_log_var - log_var
                                   + (log_var.exp() + delta ** 2) / prior_var - 1.0).sum()
        return total


def run_neuralcp(exp: Experiment, cfg: Config):
    """Fit NeuralCP; one iteration = one epoch, capped at --ncp-entries-per-iter."""
    L, A, T = exp.shape
    held = validation_split(exp, cfg) if cfg.val_frac > 0 else None
    X_std, mu, sd = standardise(exp.X_in, exp.mask_in)
    fit_mask = exp.mask_in if held is None else exp.mask_in & ~held
    tracker = Tracker(exp, cfg, X_std, mu, sd, held)
    data = EntryData(X_std, fit_mask)
    del X_std
    model = NeuralCP((L, A, T), cfg.ncp_rank, cfg.ncp_hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.ncp_lr)
    per_iter = min(data.n, max(1, cfg.ncp_entries_per_iter))

    def mean_of(index):
        return model(index, sample=False)[0]

    def predict_at(flat: np.ndarray) -> np.ndarray:
        model.eval()
        out = predict_entries(mean_of, data, flat)
        model.train()
        return out

    best_state = None
    K = max(1, cfg.iters)
    model.train()
    for iteration in range(1, K + 1):
        for index, target in data.batches(per_iter, cfg.ncp_batch):
            mean, log_var = model(index, sample=True)
            nll = 0.5 * (log_var + (target - mean) ** 2 / log_var.exp()).mean()
            loss = nll + cfg.ncp_kl * model.kl() / max(1, data.n)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        if entry_checkpoint(tracker, iteration, predict_at):
            best_state = clone_state(model)

    if best_state is not None:
        model.load_state_dict(best_state)
    full = predict_at(None).reshape(L, A, T)
    return (destandardise(full, mu, sd), tracker.points, tracker.seconds,
            {"info": tracker.info(K)})


# ==============================================================================
# 11.  M5  TRPCA-TNN - tensor robust PCA with the t-SVD nuclear norm
#      (Lu et al., IEEE TPAMI 2020)
# ==============================================================================
#
# min ||L||_* + lambda ||E||_1  s.t.  X = L + E, with the tensor nuclear norm of
# the t-product (eq. 5), solved by the ADMM of Algorithm 4 with its published
# constants (mu0 = 1e-3, rho = 1.1, mu_max = 1e10, epsilon = 1e-8 on the three
# max-norm changes) and lambda = 1 / sqrt(max(n1, n2) n3) (Theorem 4.1).  The
# L step is the t-SVT of Algorithm 3: an FFT along the tubes, a soft-threshold
# of the singular values of the first ceil((n3+1)/2) frontal slices, conjugate
# symmetry for the rest (rfft / irfft), an inverse FFT.
#
# TRPCA assumes a fully observed X.  Missing entries are handled the only way
# that keeps both closed forms: the constraint is imposed on the observed set,
# i.e. the unmodified ADMM runs on a working tensor equal to the data where it
# was measured and to L + E elsewhere, and E is zero off the data.
# ------------------------------------------------------------------------------

def batched_svt(A, tau: float):
    """Soft-threshold the singular values of every matrix in a batch.

    The slim side here is A (5 or 6), so this goes through the Gram matrix: a
    small Hermitian eigendecomposition batches well where a batched SVD of
    47 x 5 matrices does not, and ``U S_t V^H = A V diag(S_t / S) V^H``.
    """
    wide = A.shape[-1] > A.shape[-2]
    M = A.mH if wide else A
    evals, V = torch.linalg.eigh(M.mH @ M)
    S = evals.clamp_min(0.0).sqrt()
    floor = torch.finfo(S.dtype).eps * S.amax(dim=-1, keepdim=True)
    scale = torch.where(S > floor, (S - tau).clamp_min(0.0) / S.clamp_min(1e-30),
                        torch.zeros_like(S))
    out = ((M @ V) * scale.unsqueeze(-2)) @ V.mH
    return out.mH if wide else out


def tsvt(Y, tau: float):
    """Algorithm 3: tensor singular-value thresholding, the proximal map of TNN."""
    n3 = Y.shape[2]
    spectrum = torch.fft.rfft(Y, dim=2).permute(2, 0, 1)
    thresholded = batched_svt(spectrum, tau)
    return torch.fft.irfft(thresholded.permute(1, 2, 0), n=n3, dim=2)


def run_trpca(exp: Experiment, cfg: Config):
    """Run the ADMM; one iteration = --trpca-steps-per-iter ADMM steps.

    It stops at the paper's tolerance or after --iters iterations; nothing is
    held out (the problem is convex and its own stopping rule needs none).
    """
    L, A, T = exp.shape
    X_std, mu, sd = standardise(exp.X_in, exp.mask_in)
    tracker = Tracker(exp, cfg, X_std, mu, sd, None)
    X = torch.as_tensor(X_std)
    observed = torch.as_tensor(exp.mask_in)
    lam = 1.0 / math.sqrt(max(L, A) * T)
    Low = torch.zeros_like(X)
    Sparse = torch.zeros_like(X)
    Dual = torch.zeros_like(X)
    mu_k = cfg.trpca_mu0
    K = max(1, cfg.iters)
    steps = 0
    used = 0
    converged = False
    for iteration in range(1, K + 1):
        for _ in range(max(1, cfg.trpca_steps_per_iter)):
            data = torch.where(observed, X, Low + Sparse)
            Low_new = tsvt(data - Sparse - Dual / mu_k, 1.0 / mu_k)
            raw = data - Low_new - Dual / mu_k
            Sparse_new = (torch.sign(raw) * (raw.abs() - lam / mu_k).clamp_min(0.0)
                          ).masked_fill_(~observed, 0.0)
            residual = Low_new + Sparse_new - data
            change = float(torch.maximum(
                torch.maximum((Low_new - Low).abs().max(),
                              (Sparse_new - Sparse).abs().max()),
                residual.abs().max()))
            Dual.add_(residual, alpha=mu_k)
            Low, Sparse = Low_new, Sparse_new
            del data, raw, residual, Low_new, Sparse_new
            mu_k = min(cfg.trpca_rho * mu_k, cfg.trpca_mu_max)
            steps += 1
            if change < cfg.trpca_tol:
                converged = True
                break
        used = iteration
        tracker.trace(iteration, lambda: Low.reshape(-1)[tracker.scorer.flat].cpu().numpy())
        if converged:
            break

    info = tracker.info(used)
    info.update(stopped="tolerance" if converged else "budget", admm_steps=steps)
    return (destandardise(Low.cpu().numpy(), mu, sd), tracker.points,
            tracker.seconds, {"info": info})


# ==============================================================================
# 12.  M6  CoSTCo - convolutional sparse tensor completion (Liu et al., KDD 2019)
# ==============================================================================
#
# Embedding module (eq. 4): the rows U^1_{i1}, ..., U^N_{iN} of N rank-r
# factor matrices, stacked into a 1 x r x N map.  Nonlinear mapping module
# (eq. 5): two 2-D convolutions with ReLU - the first with an (1, N) filter, so
# each of its outputs mixes the N modes of one rank component (the alignment
# the paper's parameter-efficiency argument rests on, Sec. 4.3), the second
# with an (r, 1) filter that mixes the rank components.  Aggregation module
# (eq. 6): an MLP with nc hidden units and one output (Appendix A.1), nc = 2r
# as the paper sets for its sensor tensors (TR, EC).  Loss: the squared error
# on the observed entries (eq. 7), Adam.  The paper trains with batch 32,
# learning rate 1e-4, 50 epochs, early stopping on 10 % held-out entries; here
# the batch is --costco-batch with learning rate 1e-3, and the held-out cells
# are the shared 3 % with best-iterate selection.  The output unit is linear
# rather than ReLU, because the standardised data are signed.
# ------------------------------------------------------------------------------

class CoSTCo(nn.Module):

    def __init__(self, shape: Sequence[int], rank: int, channels: int):
        super().__init__()
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.empty(int(n), rank).uniform_(-0.05, 0.05)) for n in shape])
        modes = len(shape)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=(1, modes))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(rank, 1))
        self.hidden = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, 1)

    def forward(self, index):
        h = torch.stack([f[i] for f, i in zip(self.factors, index)], dim=2)  # (B, r, N)
        h = F.relu(self.conv1(h.unsqueeze(1)))          # (B, C, r, 1)
        h = F.relu(self.conv2(h))                       # (B, C, 1, 1)
        h = F.relu(self.hidden(h.flatten(1)))
        return self.out(h).squeeze(1)


def run_costco(exp: Experiment, cfg: Config):
    """Fit CoSTCo; one iteration = one epoch, capped at --costco-entries-per-iter."""
    L, A, T = exp.shape
    held = validation_split(exp, cfg) if cfg.val_frac > 0 else None
    X_std, mu, sd = standardise(exp.X_in, exp.mask_in)
    fit_mask = exp.mask_in if held is None else exp.mask_in & ~held
    tracker = Tracker(exp, cfg, X_std, mu, sd, held)
    data = EntryData(X_std, fit_mask)
    del X_std
    channels = cfg.costco_channels or 2 * cfg.costco_rank
    model = CoSTCo((L, A, T), cfg.costco_rank, channels)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.costco_lr)
    per_iter = min(data.n, max(1, cfg.costco_entries_per_iter))

    def predict_at(flat: Optional[np.ndarray]) -> np.ndarray:
        return predict_entries(model, data, flat)

    best_state = None
    K = max(1, cfg.iters)
    for iteration in range(1, K + 1):
        for index, target in data.batches(per_iter, cfg.costco_batch):
            loss = ((model(index) - target) ** 2).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        if entry_checkpoint(tracker, iteration, predict_at):
            best_state = clone_state(model)

    if best_state is not None:
        model.load_state_dict(best_state)
    full = predict_at(None).reshape(L, A, T)
    return (destandardise(full, mu, sd), tracker.points, tracker.seconds,
            {"info": tracker.info(K)})


# ==============================================================================
# 13.  M7  NTC - Neural Tensor Completion (Xie et al., IEEE INFOCOM 2021)
# ==============================================================================
#
# Input and embedding layer (eq. 5): one-hot indices times embedding matrices
# A, B, C give the R-dimensional feature vectors a_i, b_j, c_k.  Interaction
# map layer (eq. 6): their outer product E = a_i o b_j o c_k, an R x R x R
# tensor that holds every cross-dimension correlation, not only the diagonal an
# inner product (CP) keeps.  Feature extraction layer (eqs. 7-9): a 3-D CNN in
# the paper's "half-size tower" form - L layers of T kernels of size 2x2x2,
# stride 2, ReLU - which reduces R^3 to 1 x 1 x 1 x T after log2(R) layers.
# Inferring layer (eq. 10): x = sigmoid(w^T s + b).  Loss (eq. 4): squared
# error on the observed entries plus alpha ||A||^2 + beta ||B||^2 +
# gamma ||C||^2, minimised by mini-batch Adam with the observed entries
# shuffled every epoch.  Data are normalised into [0, 1] for the sigmoid
# (:func:`unit_scale`).
#
# The paper's configurations are R = 16, T = 32 (Abilene, 12 nodes) and R = 32,
# T = 64 (WS-DREAM).  The network runs once per *entry*, and its cost grows as
# R^3 T: at R = 16 it is about 0.7 M multiply-adds per entry, which on the
# 2 x 10^7 cells of Dataset II is hours per fit on a CPU.  The default here is
# therefore R = 8 (3 layers, T = 32, 0.09 M per entry); --ntc-rank 16 gives
# the paper's Abilene network.  An iteration is one epoch capped at
# --ntc-entries-per-iter entries for the same reason.
# ------------------------------------------------------------------------------

class NTC(nn.Module):

    def __init__(self, shape: Sequence[int], rank: int = 8, kernels: int = 32):
        super().__init__()
        if len(shape) != 3:
            raise ValueError("NTC models a 3-way tensor")
        layers = int(round(math.log2(max(2, rank))))
        if 2 ** layers != rank:
            raise SystemExit("--ntc-rank must be a power of two (the half-size "
                             "tower halves it down to 1)")
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.rand(int(n), rank)) for n in shape])
        self.convs = nn.ModuleList(
            [nn.Conv3d(1 if i == 0 else kernels, kernels, 2, stride=2) for i in range(layers)])
        self.out = nn.Linear(kernels, 1)

    def embed(self, index):
        return [f[i] for f, i in zip(self.factors, index)]

    def forward(self, index, embedded=None):
        a, b, c = embedded if embedded is not None else self.embed(index)
        h = (a[:, :, None, None] * b[:, None, :, None] * c[:, None, None, :]).unsqueeze(1)
        for conv in self.convs:
            h = F.relu(conv(h))
        return torch.sigmoid(self.out(h.flatten(1))).squeeze(1)


def run_ntc(exp: Experiment, cfg: Config):
    """Fit NTC; one iteration = one epoch, capped at --ntc-entries-per-iter."""
    L, A, T = exp.shape
    held = validation_split(exp, cfg) if cfg.val_frac > 0 else None
    X_std, mu, sd = standardise(exp.X_in, exp.mask_in)
    fit_mask = exp.mask_in if held is None else exp.mask_in & ~held
    tracker = Tracker(exp, cfg, X_std, mu, sd, held)
    gain, offset = unit_scale(X_std, fit_mask)
    data = EntryData(X_std, fit_mask)
    del X_std
    attr = (data.flat.astype(np.int64) // T) % A
    data.values = (data.values * gain[attr] + offset[attr]).astype(np.float32)
    del attr
    gain_t, offset_t = torch.as_tensor(gain), torch.as_tensor(offset)
    model = NTC((L, A, T), cfg.ntc_rank, cfg.ntc_kernels)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.ntc_lr)
    per_iter = min(data.n, max(1, cfg.ntc_entries_per_iter))

    def standardised(index):
        a = index[1]
        return (model(index) - offset_t[a]) / gain_t[a]

    def predict_at(flat: Optional[np.ndarray]) -> np.ndarray:
        return predict_entries(standardised, data, flat, chunk=cfg.ntc_batch * 8)

    best_state = None
    K = max(1, cfg.iters)
    for iteration in range(1, K + 1):
        for index, target in data.batches(per_iter, cfg.ntc_batch):
            embedded = model.embed(index)
            loss = ((model(index, embedded) - target) ** 2).mean()
            if cfg.ntc_reg > 0:
                loss = loss + cfg.ntc_reg * sum((e * e).sum(dim=1).mean() for e in embedded)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        if entry_checkpoint(tracker, iteration, predict_at):
            best_state = clone_state(model)

    if best_state is not None:
        model.load_state_dict(best_state)
    full = predict_at(None).reshape(L, A, T)
    return (destandardise(full, mu, sd), tracker.points, tracker.seconds,
            {"info": tracker.info(K)})


# ==============================================================================
# 14.  The runner: one job per (method, missing ratio, outlier bound, noise)
# ==============================================================================

RUNNERS: Dict[str, Callable[[Experiment, Config], object]] = {
    "dltd": run_dltd,
    "datc": run_datc,
    "deeptensor": run_deeptensor,
    "neuralcp": run_neuralcp,
    "trpca": run_trpca,
    "costco": run_costco,
    "ntc": run_ntc,
    "dltd_core": run_dltd_core,
    "td": run_td,
    "td_res": run_td_res,
    "cp": run_cp,
    "cp_tcn": run_cp_tcn,
}

#: a rough cost order, so the long jobs are started first and the pool drains evenly
COST: Dict[str, int] = {"datc": 5, "ntc": 5, "dltd": 4, "dltd_core": 4, "cp_tcn": 4,
                        "deeptensor": 3, "trpca": 3, "td": 3, "td_res": 3, "cp": 3,
                        "costco": 2, "neuralcp": 2}

#: bumped whenever a cached row stops meaning what it used to; ``--resume`` then
#: recomputes instead of mixing two protocols in one figure.  9: seven methods,
#: one common iteration budget, the curve on a fixed sample of cells.
SCHEMA: int = 9

_SHARED: Dict[str, object] = {}

Job = Tuple[str, float, float, float]          # method, ratio, outlier bound, noise


def job_key(method: str, ratio: float, bound: float = 0.0, noise: float = 0.0) -> str:
    return f"{method}_r{pct(ratio):03d}_{scenario_tag(bound, noise)}"


def _worker_init(threads: int, data: str, cfg: Config) -> None:
    configure_device(cfg)
    set_threads(threads)
    if mp.parent_process() is not None:
        logging.basicConfig(level=logging.WARNING, force=True)
    apply_attributes(cfg)
    _SHARED["X"] = prepare_tensor(Path(data), cfg)
    _SHARED["cfg"] = cfg
    warm_up_device()


def _run_job(job: Job) -> Dict[str, object]:
    """Corrupt the tensor, run one method on it, and score the result."""
    method, ratio, bound, noise = job
    cfg: Config = _SHARED["cfg"]                     # type: ignore[assignment]
    X: np.ndarray = _SHARED["X"]                     # type: ignore[assignment]
    key = job_key(method, ratio, bound, noise)
    print(f"    -> {key}", file=sys.stderr, flush=True)
    seed = job_seed(cfg.seed, ratio)
    row: Dict[str, object] = {"method": method, "ratio": ratio,
                              "outlier_bound": float(bound),
                              "noise_level": float(noise),
                              "outlier_ratio": float(cfg.outlier_ratio if bound > 0 else 0.0),
                              "attributes": list(ATTRS),
                              "dataset": cfg.dataset, "shape": list(X.shape),
                              "traced": is_traced(cfg, method, ratio, bound, noise),
                              "iters": int(cfg.iters),
                              "key": key, "seed": seed, "schema": SCHEMA}
    try:
        set_seed(seed)
        exp = build_experiment(X, ratio, seed, bound, cfg.outlier_ratio,
                               cfg.outlier_spread, noise)
        exp.trace = bool(row["traced"])
        row["observed_in"] = int(exp.mask_in.sum())
        row["outliers_in"] = int(exp.outliers.sum()) if exp.outliers is not None else 0
        set_seed(seed + 17)

        # Both ends of the measured window are synchronised on a GPU, so the
        # clock does not stop while kernels are still queued.
        def attempt(use_gpu: bool):
            if use_gpu:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            with PeakMemory() as peak:
                clock = time.perf_counter()
                result = RUNNERS[method](exp, cfg)
                if use_gpu:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - clock
            device_peak = (round(torch.cuda.max_memory_allocated() / 2 ** 20, 2)
                           if use_gpu else 0.0)
            return result, elapsed, peak, device_peak

        on_gpu = cfg.device != "cpu" and torch.cuda.is_available()
        row["device_used"] = "cuda" if on_gpu else "cpu"
        try:
            output, seconds, memory, gpu_peak = attempt(on_gpu)
        except Exception as error:
            # A job the device had no room for is re-run on the CPU and marked,
            # and Result 4 leaves it out.  See WHEN THE DEVICE IS FULL.
            if not (on_gpu and cfg.cpu_fallback and is_allocator_failure(error)):
                raise
            print(f"    !! {row['key']}: device allocation failed, retrying on "
                  f"the CPU ({type(error).__name__})", file=sys.stderr, flush=True)
            row["device_fallback"] = True
            row["device_error"] = f"{type(error).__name__}: {error}"
            row["device_used"] = "cpu"
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
            torch.set_default_device("cpu")
            try:
                set_seed(seed + 17)
                output, seconds, memory, gpu_peak = attempt(False)
            finally:
                torch.set_default_device(f"cuda:{cfg.gpu_index}")
        row["gpu_peak_mb"] = gpu_peak

        # A runner returns (model, curve, diagnostic seconds, extras).  The
        # seconds are time spent drawing the curve, and come off the clock.
        model, curve, excluded, extras = output
        extras = extras or {}
        seconds = max(0.0, seconds - float(excluded))

        X_hat, flagged = robust_estimate(exp.X_in, exp.mask_in, model,
                                         cfg.flag_kappa, extras.get("flags"))
        row["flagged"] = int(flagged.sum())
        row["fit_info"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                           for k, v in (extras.get("info") or {}).items()}
        sets = {"observed": exp.obs0, "hidden": exp.hidden,
                "outliers": (exp.outliers if exp.outliers is not None
                             else np.zeros_like(exp.obs0))}
        row["metrics"] = {name: metrics_per_attribute(exp.X_true, X_hat, mask)
                          for name, mask in sets.items()}
        row["pooled"] = {name: (nrmse_pooled(exp.X_true, X_hat, mask)
                                if mask.any() else float("nan"))
                         for name, mask in sets.items()}
        row["detection"] = detection_scores(flagged, exp.outliers)
        row["nrmse_mean"] = {name: mean_nrmse(exp.X_true, X_hat, mask)
                             for name, mask in sets.items()}
        row["seconds"] = seconds
        row["diagnostic_seconds"] = float(excluded)
        row["memory_mb"] = memory.delta_mb
        row["peak_rss_mb"] = memory.peak_mb
        used = float(row["fit_info"].get("budget_used", cfg.iters) or cfg.iters)
        row["budget_used"] = used
        row["iteration_unit"] = ITERATION_UNIT.get(method, "iteration")
        row["convergence"] = [{"x": float(p["x"]), "nrmse": float(p["nrmse"])}
                              for p in (curve or [])
                              if math.isfinite(float(p["nrmse"]))]
        row["final"] = {"x": used, "nrmse": row["nrmse_mean"][cfg.eval_set]}
        row["ok"] = True
    except Exception as error:                       # keep a long sweep alive
        import traceback
        row["ok"] = False
        row["error"] = f"{type(error).__name__}: {error}"
        row["traceback"] = traceback.format_exc()
    finally:
        if cfg.device != "cpu" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
    return row


def _job_summary(row: Dict[str, object]) -> str:
    """One line per finished job: cost, error, and what the finishing rule did."""
    if not row.get("ok"):
        return str(row.get("error"))
    mean = row.get("nrmse_mean") or {}
    f1 = float((row.get("detection") or {}).get("f1", float("nan")))
    info = row.get("fit_info") or {}
    return ("%7.1f s  %6.0f MB  NRMSE obs %.4f hid %.4f  %g %ss (best %s, %s)  "
            "%d replaced%s"
            % (row["seconds"], float(row.get("memory_mb", 0.0)),
               float(mean.get("observed", float("nan"))),
               float(mean.get("hidden", float("nan"))),
               float(row.get("budget_used", 0.0)),
               row.get("iteration_unit", "iteration"),
               info.get("best_x", "-"), info.get("stopped", "budget"),
               row.get("flagged", 0),
               f"  F1 {f1:.3f}" if math.isfinite(f1) else ""))


def _row_is_current(row: Dict[str, object], cfg: Config) -> bool:
    """Was this cached row written by this protocol?"""
    return (int(row.get("schema", 0)) == SCHEMA
            and list(row.get("attributes", [])) == list(ATTRS)
            and list(row.get("shape", [])) == list(cfg.shape)
            and int(row.get("iters", -1)) == int(cfg.iters))


def run_experiments(cfg: Config, data: Path, out: Path) -> List[Dict[str, object]]:
    """Fan the jobs out over the machine and cache every one as it lands."""
    cache = out / "raw"
    cache.mkdir(parents=True, exist_ok=True)

    jobs: List[Job] = [
        (method, ratio, bound, noise) for method in cfg.methods
        for ratio in cfg.ratios for bound, noise in scenarios(cfg)]
    jobs.sort(key=lambda j: (-COST.get(j[0], 0), j[2], j[3], j[1]))

    done: List[Dict[str, object]] = []
    todo: List[Job] = []
    stale = 0
    for job in jobs:
        path = cache / (job_key(*job) + ".json")
        if cfg.resume and path.exists():
            try:
                row = json.loads(path.read_text())
                if (row.get("ok") and _row_is_current(row, cfg)
                        and bool(row.get("traced")) == is_traced(cfg, *job)):
                    done.append(row)
                    continue
                stale += int(bool(row.get("ok")))
            except Exception:
                pass
        todo.append(job)

    info = machine(cfg.gpu_index)
    LOG.info("machine: %s, %s logical cores (%s physical), %.1f GB RAM "
             "(%.1f GB available), %.1f GB free disk",
             info["cpu_model"], info["logical_cpus"], info["physical_cores"],
             info["ram_gb"], info["ram_available_gb"], info["disk_free_gb"])
    if cfg.device != "cpu" and info.get("cuda_available"):
        LOG.info("cuda: %s, %.2f of %.2f GB free, torch %s (cuda %s)",
                 info["cuda_devices"][cfg.gpu_index] if info.get("cuda_devices") else "?",
                 info["cuda_free_gb"], info["cuda_total_gb"],
                 info.get("torch", "?"), info.get("torch_cuda", "?"))
        free_gb = float(info.get("cuda_free_gb", -1.0))
        total_gb = float(info.get("cuda_total_gb", -1.0))
        if 0.0 <= free_gb < max(2.0, 0.15 * max(total_gb, 0.0)):
            LOG.warning(
                "only %.2f of %.2f GB of the device is free - another process "
                "has most of it.  Jobs that run out of room are re-run on the "
                "CPU and left out of Result 4%s", free_gb, total_gb,
                "" if cfg.cpu_fallback else " - except that --no-cpu-fallback "
                "was passed, so they will simply fail")
    elif cfg.device == "auto":
        LOG.info("cuda: not visible to torch, running on the CPU")
    if not todo:
        LOG.info("every job is already cached under %s", cache)
        return done

    workers, threads = plan_pool(cfg, len(todo), info)
    if stale:
        LOG.info("%d cached job(s) were written by another protocol (schema "
                 "!= %d, or other attributes, shape or --iters) and will be "
                 "recomputed", stale, SCHEMA)
    LOG.info("%d job(s) to run, %d cached; %d worker process(es) x %d torch "
             "thread(s) on %s", len(todo), len(done), workers, threads,
             f"cuda:{cfg.gpu_index}"
             if cfg.device != "cpu" and info.get("cuda_available") else "cpu")

    started = time.perf_counter()
    finished = 0
    # "spawn", not "fork" (torch's OpenMP runtime does not survive a fork), and
    # a fresh worker per job, so that the RSS-based memory of Result 4 starts
    # from a clean process every time.
    context = mp.get_context("spawn")
    with context.Pool(processes=workers, initializer=_worker_init,
                      initargs=(threads, str(data), cfg),
                      maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_run_job, todo):
            (cache / (row["key"] + ".json")).write_text(json.dumps(row, indent=1))
            done.append(row)
            finished += 1
            LOG.info("[%d/%d] %-28s %s", finished, len(todo), row["key"],
                     _job_summary(row))

    failed = [r for r in done if not r.get("ok")]
    LOG.info("sweep finished in %.1f min; %d ok, %d failed",
             (time.perf_counter() - started) / 60, len(done) - len(failed), len(failed))
    for row in failed:
        LOG.error("%s failed: %s", row["key"], row.get("error"))
    return done


def load_results(out: Path, cfg: Config) -> List[Dict[str, object]]:
    """Every cached row of one dataset that this run's protocol produced."""
    rows, skipped = [], 0
    for path in sorted((out / "raw").glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except Exception:
            LOG.warning("ignoring unreadable cache file %s", path)
            continue
        if not _row_is_current(row, cfg):
            skipped += 1
            continue
        rows.append(row)
    if skipped:
        LOG.warning("%d cached job(s) under %s were written by another protocol "
                    "and were left out of the results", skipped, out / "raw")
    return rows


def index_results(rows: Sequence[Dict[str, object]]) -> Dict[Tuple[str, int], Dict]:
    """Finished rows by (method, missing ratio in whole percent)."""
    table: Dict[Tuple[str, int], Dict] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        table[(str(row["method"]), pct(float(row["ratio"])))] = row
    return table


def missing_cells(rows: Sequence[Dict[str, object]], cfg: Config) -> List[Job]:
    """Configurations the results have no usable row for."""
    holes: List[Job] = []
    for bound, noise in scenarios(cfg):
        table = index_results(rows_for(rows, bound, noise))
        holes.extend((method, ratio, bound, noise)
                     for method in cfg.methods for ratio in cfg.ratios
                     if (method, pct(ratio)) not in table)
    return holes


def _report_missing(rows: Sequence[Dict[str, object]], cfg: Config,
                    label: str) -> int:
    """Name every hole of one dataset on stdout and in the log; return the count."""
    holes = missing_cells(rows, cfg)
    if not holes:
        return 0
    errors = {str(r["key"]): str(r.get("error", "")) for r in rows
              if not r.get("ok")}
    expected = len(cfg.methods) * len(cfg.ratios) * len(scenarios(cfg))
    _banner(f"INCOMPLETE  {label}: {len(holes)} of {expected} configuration(s) "
            f"are missing from the results")
    for hole in holes:
        key = job_key(*hole)
        print(f"  {key:<30} {errors.get(key, 'no cached row at all')[:120]}")
    print("\n  Result 1 prints a missing cell as '-', Results 2 and 3 omit the "
          "curve and\n  Result 4 averages over the jobs that finished.  "
          "Re-running writes only the\n  missing jobs: finished ones are cached.")
    LOG.error("%s: %d of %d configuration(s) missing: %s", label, len(holes),
              expected, ", ".join(job_key(*h) for h in holes))
    return len(holes)


# ==============================================================================
# 15.  The four results.  Nothing else is drawn or printed.
# ==============================================================================
#
#   Result 1   result1_nrmse_tables.md          NRMSE averaged over the
#                                               attributes, outlier bound x
#                                               missing ratio by method
#   Result 2   result2_convergence_all.png      NRMSE against iterations, every
#                                               method; dataset x missing ratio
#   Result 3   result3_convergence_deeptd.png   NRMSE against iterations,
#                                               Deep-TD, one line per missing
#                                               ratio; dataset x outlier bound
#   Result 4   result4_runtime_memory.png       runtime and memory by method
# ------------------------------------------------------------------------------

def _mpl():
    """matplotlib, configured once, imported only when a result is drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
        "savefig.facecolor": C_SURFACE, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 8.0,
        "axes.edgecolor": C_GRID, "axes.linewidth": 0.7,
        "axes.labelcolor": C_INK, "axes.titlecolor": C_INK,
        "text.color": C_INK, "xtick.color": C_INK_2, "ytick.color": C_INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "grid.color": C_GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "hatch.linewidth": 0.5,
    })
    return plt


def _banner(title: str) -> None:
    rule = "=" * 78
    print(f"\n{rule}\n{title}\n{rule}")


def _method_names(cfg: Config) -> List[Tuple[str, str, str]]:
    """``(key, "M1", "Deep-TD")`` for every method run, in the order given."""
    return [(m, f"M{i + 1}", METHOD_LABEL[m]) for i, m in enumerate(cfg.methods)]


def _slot(method: str) -> int:
    """A method's fixed palette slot: its position in METHODS, whatever was run."""
    return METHODS.index(method) % len(C_SERIES)


def _scenario_text(bound: float, noise: float) -> str:
    parts = [f"outlier bound k = {bound:g}" if bound > 0 else "no outliers"]
    parts.append(f"noise level η = {noise:g}" if noise > 0 else "no noise")
    return ", ".join(parts)


def _nrmse_of(row: Optional[Dict[str, object]], cfg: Config) -> float:
    """The attribute-averaged NRMSE of one finished job on --eval-set."""
    if not row or not row.get("ok"):
        return float("nan")
    value = (row.get("nrmse_mean") or {}).get(cfg.eval_set)
    return float(value) if value is not None else float("nan")


EVAL_TEXT: Dict[str, str] = {
    "observed": "every cell the raw tensor observed",
    "hidden": "the cells hidden by the missing ratio",
    "outliers": "the cells that received an outlier",
}


# ---- Result 1: the NRMSE tables -----------------------------------------------

def _bold_best(values: Sequence[float]) -> List[str]:
    """Format one table row, lowest NRMSE (as printed) in bold, '-' if missing."""
    texts = [f"{v:.4f}" if math.isfinite(v) else "-" for v in values]
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return texts
    best = f"{min(finite):.4f}"
    return [f"**{t}**" if t == best else t for t in texts]


def result1_nrmse_tables(datasets: Sequence[Dataset], cfg: Config, out: Path) -> Path:
    """Result 1: one table per dataset (and per noise level) in the supplied format."""
    names = _method_names(cfg)
    header = ["Outlier Bound", "Missing Ratio"] + [short for _, short, _ in names]
    several = len(cfg.noise_levels) > 1
    lines: List[str] = []
    number = 0
    for noise in cfg.noise_levels:
        if several:
            lines += [f"## Noise level η = {noise:g}", ""]
        for dataset in datasets:
            numeral = ROMAN[number] if number < len(ROMAN) else str(number + 1)
            number += 1
            suffix = f" (noise level η = {noise:g})" if several else ""
            lines += [f"**Table {numeral} : NRMSE performance on {dataset.label} "
                      f"for various missing ratio{suffix}**", "",
                      "| " + " | ".join(header) + " |",
                      "| " + " | ".join(["-----"] * len(header)) + " |"]
            for bound in cfg.outlier_bounds:
                table = index_results(rows_for(dataset.rows, bound, noise))
                for ratio in cfg.ratios:
                    values = [_nrmse_of(table.get((m, pct(ratio))), cfg)
                              for m, _, _ in names]
                    lines.append(f"| {bound:g} | {pct(ratio)}% | "
                                 + " | ".join(_bold_best(values)) + " |")
            lines.append("")

    lines += ["**Legend.** " + ", ".join(f"{short} = {label}" for _, short, label in names)
              + ".  ", "  \n".join(d.describe() for d in datasets) + ".  ",
              f"Each cell is the NRMSE, ||x̂ − x||₂ / ||x||₂ per "
              f"attribute, averaged over the attributes and scored on "
              f"{EVAL_TEXT[cfg.eval_set]} against the clean values.  Lower is "
              f"better; the best value of every row is in bold.  Every method "
              f"ran for {cfg.iters} iterations.  ",
              f"Outlier bound k: a share ε = {cfg.outlier_ratio:g} of the readable "
              f"cells receives an impulse of k to {1 + cfg.outlier_spread:g}k "
              f"standard deviations of its attribute."
              + ("" if several else
                 f"  Noise level η = {cfg.noise_levels[0]:g}."), ""]
    path = out / "result1_nrmse_tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    _banner("RESULT 1  NRMSE averaged over the attributes "
            f"(scored on {EVAL_TEXT[cfg.eval_set]})")
    print("\n".join(lines))
    print(f"  table -> {path}")
    return path


# ---- Results 2 and 3: convergence ---------------------------------------------

def _curve(row: Optional[Dict[str, object]]) -> Tuple[List[float], List[float]]:
    """The checkpoints of one traced job that a log y axis can show."""
    points = [(float(p["x"]), float(p["nrmse"]))
              for p in ((row.get("convergence") if row else None) or [])
              if float(p["nrmse"]) > 0]
    points.sort()
    return [x for x, _ in points], [y for _, y in points]


def _draw_curve(ax, row: Optional[Dict[str, object]], cfg: Config, colour: str,
                dash, marker: str, width: float, zorder: int) -> Tuple[bool, float]:
    """One convergence line and the marker of the estimate the fit returned."""
    x, y = _curve(row)
    if x:
        ax.plot(x, y, color=colour, linestyle=dash, linewidth=width, zorder=zorder,
                marker=marker, markersize=3.2, markevery=max(1, len(x) // 6),
                markeredgecolor=C_SURFACE, markeredgewidth=0.5)
    final = _nrmse_of(row, cfg)
    fx = float(((row or {}).get("final") or {}).get("x", 0.0) or 0.0)
    if x and fx < x[-1]:
        fx = x[-1]
    if fx > 0 and math.isfinite(final) and final > 0:
        if x:
            ax.plot([x[-1], fx], [y[-1], final], color=colour, linewidth=0.8,
                    linestyle=(0, (1, 1.5)), zorder=zorder)
        ax.plot([fx], [final], marker=marker, markersize=6.5, color=colour,
                markeredgecolor=C_INK, markeredgewidth=0.6, linestyle="none",
                zorder=zorder + 2)
    return bool(x) or math.isfinite(final), final


def _style_convergence_axis(ax, cfg: Config, title: str) -> None:
    from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator
    ax.set_yscale("log")
    ax.set_xlim(0, cfg.iters * 1.03 + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_minor_locator(LogLocator(subs=(2.0, 5.0)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.0)
    ax.set_title(title, fontsize=8.6, pad=5)
    ax.set_xlabel("Iteration", fontsize=8.0)
    ax.set_ylabel("NRMSE (mean over attributes)", fontsize=8.0)


def _convergence_figure(plt, n_rows: int, n_cols: int):
    return plt.subplots(n_rows, n_cols, squeeze=False,
                        figsize=(3.05 * n_cols + 0.5, 2.75 * n_rows + 1.4))


def _finish_convergence_figure(fig, handles, title: str, note: str, ncol: int) -> None:
    fig.legend(handles=handles, loc="upper center", ncol=ncol, fontsize=7.6,
               handlelength=3.0, columnspacing=1.4,
               bbox_to_anchor=(0.5, 1.0 - 0.32 / fig.get_figheight()))
    fig.suptitle(title, fontsize=9.6, y=1.0 - 0.04 / fig.get_figheight(), va="top")
    import textwrap
    note = textwrap.fill(note, width=int(28 * fig.get_figwidth()))
    fig.text(0.5, 0.04 / fig.get_figheight(), note, ha="center", va="bottom",
             fontsize=7.0, color=C_INK_2)
    height, width = fig.get_figheight(), fig.get_figwidth()
    legend_rows = 1 + (len(handles) - 1) // ncol
    fig.subplots_adjust(left=0.75 / width, right=1.0 - 0.12 / width,
                        bottom=(0.7 + 0.13 * note.count("\n")) / height,
                        top=1.0 - (0.8 + 0.2 * legend_rows) / height,
                        wspace=0.42, hspace=0.62)


def _iteration_note(cfg: Config) -> str:
    units = "; ".join(f"{short} {ITERATION_UNIT.get(m, 'iteration')}"
                      for m, short, _ in _method_names(cfg))
    return ("Every method runs exactly %d iterations.  One iteration: %s.  Line: the "
            "model at the end of each iteration (on %s sampled cells).  Large "
            "marker: the estimate the fit returned (its entry in Result 1)."
            % (cfg.iters, units, f"{cfg.trace_cells:,}" if cfg.trace_cells else "all"))


def result2_convergence_all(datasets: Sequence[Dataset], cfg: Config, out: Path) -> Path:
    """Result 2: every method's NRMSE against iterations, one panel per
    (dataset, missing ratio), at --conv-bound and --conv-noise."""
    from matplotlib.lines import Line2D
    plt = _mpl()
    names = _method_names(cfg)
    bound, noise = convergence_scenario(cfg)
    fig, axes = _convergence_figure(plt, len(datasets), len(cfg.ratios))
    summary: List[List[str]] = []
    lead = cfg.methods[0]
    for r, dataset in enumerate(datasets):
        table = index_results(rows_for(dataset.rows, bound, noise))
        for c, ratio in enumerate(cfg.ratios):
            ax = axes[r][c]
            drawn = 0
            for method, short, label in names:
                row = table.get((method, pct(ratio)))
                slot = _slot(method)
                ok, final = _draw_curve(ax, row, cfg, C_SERIES[slot],
                                        DASHES[slot % len(DASHES)],
                                        MARKERS[slot % len(MARKERS)],
                                        2.0 if method == lead else 1.2,
                                        4 if method == lead else 3)
                drawn += int(ok)
                if row and row.get("ok"):
                    _, y = _curve(row)
                    info = row.get("fit_info") or {}
                    summary.append([dataset.label, f"{pct(ratio)}%", f"{short} {label}",
                                    f"{float(row.get('budget_used', 0.0)):g}",
                                    str(info.get("best_x", "-")),
                                    f"{y[0]:.4f}" if y else "-",
                                    f"{final:.4f}" if math.isfinite(final) else "-"])
            if not drawn:
                ax.text(0.5, 0.5, "no curve cached", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7.5, color=C_INK_2)
            _style_convergence_axis(ax, cfg, f"{dataset.label}, {pct(ratio)}% missing")

    handles = [Line2D([0], [0], color=C_SERIES[_slot(m)],
                      linestyle=DASHES[_slot(m) % len(DASHES)],
                      linewidth=2.0 if m == lead else 1.2,
                      marker=MARKERS[_slot(m) % len(MARKERS)], markersize=5.0,
                      markeredgecolor=C_INK, markeredgewidth=0.5,
                      label=f"{short} {label}") for m, short, label in names]
    _finish_convergence_figure(
        fig, handles,
        f"Convergence of all methods: NRMSE against iterations "
        f"({_scenario_text(bound, noise)})", _iteration_note(cfg), min(len(handles), 7))
    path = out / "result2_convergence_all.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    _banner(f"RESULT 2  convergence of all methods ({_scenario_text(bound, noise)})")
    if summary:
        head = ["dataset", "missing", "method", "iterations", "best iteration",
                "NRMSE at it. 1", "final NRMSE"]
        widths = [max(len(head[i]), *(len(line[i]) for line in summary))
                  for i in range(len(head))]
        print("  " + "  ".join(h.ljust(w) for h, w in zip(head, widths)))
        for line in summary:
            print("  " + "  ".join(v.ljust(w) for v, w in zip(line, widths)))
    else:
        print("  no traced job is cached for this outlier bound and noise level")
    print(f"\n  figure -> {path}")
    return path


def result3_convergence_deeptd(datasets: Sequence[Dataset], cfg: Config,
                               out: Path) -> Path:
    """Result 3: Deep-TD alone, one panel per (dataset, outlier bound), one line
    per missing ratio, at --conv-noise."""
    from matplotlib.lines import Line2D
    plt = _mpl()
    lead = "dltd" if "dltd" in cfg.methods else cfg.methods[0]
    _, noise = convergence_scenario(cfg)
    bounds = list(cfg.outlier_bounds)
    fig, axes = _convergence_figure(plt, len(datasets), len(bounds))
    summary: List[List[str]] = []
    for r, dataset in enumerate(datasets):
        for c, bound in enumerate(bounds):
            ax = axes[r][c]
            table = index_results(rows_for(dataset.rows, bound, noise))
            drawn = 0
            for i, ratio in enumerate(cfg.ratios):
                row = table.get((lead, pct(ratio)))
                ok, final = _draw_curve(ax, row, cfg, C_SERIES[i % len(C_SERIES)],
                                        DASHES[i % len(DASHES)], MARKERS[i % len(MARKERS)],
                                        1.5, 3)
                drawn += int(ok)
                if row and row.get("ok"):
                    _, y = _curve(row)
                    summary.append([dataset.label, f"{bound:g}", f"{pct(ratio)}%",
                                    str((row.get("fit_info") or {}).get("best_x", "-")),
                                    f"{y[0]:.4f}" if y else "-",
                                    f"{y[-1]:.4f}" if y else "-",
                                    f"{final:.4f}" if math.isfinite(final) else "-"])
            if not drawn:
                ax.text(0.5, 0.5, "no curve cached", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7.5, color=C_INK_2)
            what = f"outlier bound k = {bound:g}" if bound > 0 else "no outliers"
            _style_convergence_axis(ax, cfg, f"{dataset.label}, {what}")

    handles = [Line2D([0], [0], color=C_SERIES[i % len(C_SERIES)],
                      linestyle=DASHES[i % len(DASHES)], linewidth=1.5,
                      marker=MARKERS[i % len(MARKERS)], markersize=5.0,
                      markeredgecolor=C_INK, markeredgewidth=0.5,
                      label=f"{pct(ratio)}% missing")
               for i, ratio in enumerate(cfg.ratios)]
    noise_text = f"noise level η = {noise:g}" if noise > 0 else "no noise"
    _finish_convergence_figure(
        fig, handles,
        f"Convergence of {METHOD_LABEL[lead]}: NRMSE against iterations for every "
        f"missing ratio and outlier bound ({noise_text})",
        f"One iteration = one {ITERATION_UNIT.get(lead, 'iteration')}; {cfg.iters} "
        f"iterations.  Line: the low-rank model at each iteration.  Large marker: the "
        f"returned estimate (selected iterate after the residual stage, its entry in "
        f"Result 1).", min(len(handles), 7))
    path = out / "result3_convergence_deeptd.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    _banner(f"RESULT 3  convergence of {METHOD_LABEL[lead]} ({noise_text})")
    if summary:
        head = ["dataset", "bound", "missing", "best iteration", "NRMSE at it. 1",
                f"NRMSE at it. {cfg.iters}", "final NRMSE"]
        widths = [max(len(head[i]), *(len(line[i]) for line in summary))
                  for i in range(len(head))]
        print("  " + "  ".join(h.ljust(w) for h, w in zip(head, widths)))
        for line in summary:
            print("  " + "  ".join(v.ljust(w) for v, w in zip(line, widths)))
    else:
        print("  no traced Deep-TD job is cached for this noise level")
    print(f"\n  figure -> {path}")
    return path


# ---- Result 4: runtime and memory -----------------------------------------------

def _cost_rows(rows: Sequence[Dict[str, object]], method: str
               ) -> List[Dict[str, object]]:
    """The jobs of one method whose cost is the method's own.

    Traced jobs are left out when there are others (the curve's time is off
    their clock, but its buffers are in their peak memory), and so are jobs
    re-run on the CPU after the device ran out of room.
    """
    mine = [r for r in rows if r.get("ok") and r.get("method") == method]
    clean = [r for r in mine if not r.get("traced") and not r.get("device_fallback")]
    return clean or [r for r in mine if not r.get("device_fallback")] or mine


def _nice(value: float) -> str:
    """A bar's printed value: three significant figures, no exponent."""
    if not math.isfinite(value):
        return ""
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def result4_runtime_memory(datasets: Sequence[Dataset], cfg: Config, out: Path) -> Path:
    """Result 4: runtime and memory of one fit, by method, one bar per dataset.

    Runtime is the wall clock of the fit with the diagnostic curve's time
    subtracted.  CPU memory is the peak resident set size above where the job
    started (a fresh process per job, sampled every 20 ms), which includes the
    method's copies of the tensor.  When the fits ran on a GPU a third chart
    gives the peak device allocation.  Each bar is the mean over the
    configurations of the sweep.
    """
    plt = _mpl()
    names = _method_names(cfg)
    shape = (len(datasets), len(names))
    seconds, memory, device = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    count = np.zeros(shape, dtype=int)
    for d, dataset in enumerate(datasets):
        for m, (method, _, _) in enumerate(names):
            rows = _cost_rows(dataset.rows, method)
            if rows:
                seconds[d, m] = float(np.mean([float(r["seconds"]) for r in rows]))
                memory[d, m] = float(np.mean([float(r["memory_mb"]) for r in rows]))
                device[d, m] = float(np.mean([float(r.get("gpu_peak_mb", 0.0)) for r in rows]))
                count[d, m] = len(rows)
    on_gpu = any(r.get("device_used") == "cuda" for d in datasets for r in d.rows)

    in_minutes = bool(np.isfinite(seconds).any() and np.nanmax(seconds) >= 120.0)
    in_gb = bool(np.isfinite(memory).any() and np.nanmax(memory) >= 2048.0)
    panels = [("(a) Runtime of one fit",
               f"Runtime ({'minutes' if in_minutes else 'seconds'})",
               seconds / 60.0 if in_minutes else seconds),
              ("(b) Peak CPU memory of one fit (RSS above the job's start)",
               f"CPU memory ({'GB' if in_gb else 'MB'})",
               memory / 1024.0 if in_gb else memory)]
    if on_gpu:
        panels.append(("(c) Peak GPU memory of one fit", "GPU memory (MB)", device))

    fig, axes = plt.subplots(len(panels), 1, squeeze=False,
                             figsize=(max(8.0, 1.3 * len(names) + 2.0), 3.3 * len(panels) + 0.6))
    centre = np.arange(len(names), dtype=float)
    width = 0.8 / max(1, len(datasets))
    for ax, (title, ylabel, values) in zip(axes[:, 0], panels):
        for d, dataset in enumerate(datasets):
            shift = (d - (len(datasets) - 1) / 2.0) * width
            bars = ax.bar(centre + shift, np.nan_to_num(values[d]), width * 0.92,
                          color=C_SERIES[d % len(C_SERIES)],
                          edgecolor=C_EDGE[d % len(C_EDGE)], linewidth=0.6,
                          hatch=HATCHES[d % len(HATCHES)],
                          label=f"{dataset.label}: {dataset.title}", zorder=3)
            for bar, value in zip(bars, values[d]):
                if math.isfinite(value):
                    ax.annotate(_nice(value), (bar.get_x() + bar.get_width() / 2.0,
                                               bar.get_height()),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=6.4, color=C_INK)
        ax.set_xticks(centre)
        ax.set_xticklabels([f"{short}\n{label}" for _, short, label in names], fontsize=7.4)
        ax.set_xlabel("Method", fontsize=8.4)
        ax.set_ylabel(ylabel, fontsize=8.4)
        ax.set_title(title, loc="left", fontsize=9.0, pad=6)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.margins(y=0.16)
        ax.set_ylim(bottom=0.0)
        ax.tick_params(axis="x", length=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)),
                   fontsize=7.8, bbox_to_anchor=(0.5, 1.0), handlelength=1.8)
    fig.text(0.5, 0.0,
             f"Mean over the configurations of the sweep ({len(cfg.ratios)} missing "
             f"ratios × {len(cfg.outlier_bounds)} outlier bounds × "
             f"{len(cfg.noise_levels)} noise levels), {cfg.iters} iterations per fit; "
             f"fits ran on the {'GPU' if on_gpu else 'CPU'}.",
             ha="center", va="top", fontsize=7.0, color=C_INK_2)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    path = out / "result4_runtime_memory.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    _banner("RESULT 4  runtime and memory of one fit, mean over the sweep")
    for d, dataset in enumerate(datasets):
        print(f"\n  {dataset.describe()}")
        print(f"  {'':4}{'method':<28}{'runtime':>12}{'CPU memory':>14}"
              f"{'GPU memory' if on_gpu else '':>14}{'jobs':>7}")
        for m, (_, short, label) in enumerate(names):
            if not count[d, m]:
                print(f"  {short:<4}{label:<28}{'-':>12}{'-':>14}{'':>14}{0:>7}")
                continue
            gpu = f"{device[d, m]:>11.0f} MB" if on_gpu else ""
            print(f"  {short:<4}{label:<28}{seconds[d, m]:>10.1f} s"
                  f"{memory[d, m]:>11.0f} MB{gpu:>14}{count[d, m]:>7}")
    print(f"\n  figure -> {path}")
    return path


# ==============================================================================
# 16.  Entry point
# ==============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = config_from_args(argv)
    out = Path(cfg.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "run.log")

    datasets = resolve_datasets(cfg)
    for dataset in datasets:
        LOG.info("%s", dataset.describe())
        if dataset.shape != dataset.full_shape:
            LOG.info("    cut from the stored %s", " x ".join(map(str, dataset.full_shape)))
    LOG.info("methods: %s", ", ".join(f"{short} = {label}"
                                      for _, short, label in _method_names(cfg)))
    LOG.info("missing ratios: %s", ", ".join(f"{r:g}" for r in cfg.ratios))
    LOG.info("outlier bounds: %s (epsilon %g, spread %g); noise levels: %s",
             ", ".join(f"{k:g}" for k in cfg.outlier_bounds), cfg.outlier_ratio,
             cfg.outlier_spread, ", ".join(f"{v:g}" for v in cfg.noise_levels))
    LOG.info("budget: %d iterations for every method; best iterate by %g held-out "
             "cells; all-method curves at %s, Deep-TD curves at every bound",
             cfg.iters, cfg.val_frac, _scenario_text(*convergence_scenario(cfg)))

    for dataset in datasets:
        cfg_d = dataset_config(cfg, dataset)
        apply_attributes(cfg_d)
        out_d = out / dataset.tag
        if cfg.stage in ("all", "run"):
            LOG.info("---- %s: %d job(s) in all ----", dataset.label,
                     len(cfg.methods) * len(cfg.ratios) * len(scenarios(cfg)))
            run_experiments(cfg_d, dataset.path, out_d)
        dataset.rows = load_results(out_d, cfg_d)

    if not any(r.get("ok") for d in datasets for r in d.rows):
        LOG.error("no finished job under %s; nothing to draw", out)
        return 1

    (out / "environment.json").write_text(json.dumps(
        {"machine": machine(cfg.gpu_index),
         "datasets": [{"label": d.label, "tag": d.tag, "path": str(d.path),
                       "title": d.title, "shape": list(d.shape),
                       "stored_shape": list(d.full_shape),
                       "attributes": list(d.kept),
                       "slots_per_day": d.slots_per_day} for d in datasets],
         "config": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in vars(cfg).items()}}, indent=1))

    result1_nrmse_tables(datasets, cfg, out)
    result2_convergence_all(datasets, cfg, out)
    result3_convergence_deeptd(datasets, cfg, out)
    result4_runtime_memory(datasets, cfg, out)
    holes = sum(_report_missing(d.rows, cfg, d.label) for d in datasets)
    print()
    return 2 if holes else 0


if __name__ == "__main__":
    raise SystemExit(main())
