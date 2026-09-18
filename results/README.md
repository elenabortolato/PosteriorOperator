# Captured experiment output

The full console output of every example, so the numbers quoted in the top-level
README can be traced to a run rather than taken on trust. Regenerate with:

```bash
bash results/run_all.sh
```

Each file is the verbatim stdout of one script. They are checked in because the
scripts take a while to run (see the table) and because a reader should be able
to see the evidence without waiting for it.

| file | script | what to look for |
| --- | --- | --- |
| `01_gaussian_operator_exact.txt` | `examples/lfi/01_...` | CCA reconstruction to 1e-15; the exact Hermite spectrum; the rank needed for exactness |
| `02_amortized_functionals.txt` | `examples/lfi/02_...` | 11 functionals from one fit, each against quadrature |
| `03_prior_retargeting.txt` | `examples/lfi/03_...` | four target priors, ESS, and retargeting vs refitting |
| `04_identifiability.txt` | `examples/lfi/04_...` | sigma_1 vs its closed form; v_1 vs the identified direction; rank vs concentration |
| `05_spectrum_decay.txt` | `examples/lfi/05_...` | geometric vs polynomial decay on MA(2), AR(2), g-and-k |
| `06_mechanistic_amortization.txt` | `examples/lfi/06_...` | **the NPE comparison**: NCP vs NPE vs direct regression on SIR, accuracy and cost |
| `07_functional_confidence_intervals.txt` | `examples/lfi/07_...` | sampling distribution of T_hat_f(y_0); bootstrap coverage |
| `08_gandk_npe.txt` | `examples/lfi/08_...` | **NPE on a genuinely intractable model**: g-and-k against a large-budget ABC reference, marginal accuracy, calibration, query cost |
| `09_conjugate_exact_truth.txt` | `examples/lfi/09_...` | **exact truth, no reference error**: NCP vs NPE on a conjugate model, swept over parameter dimension and posterior concentration |
| `10_self_certification.txt` | `examples/lfi/10_...` | **the central claim**: the reported spectral tail against the true truncation error, and what the accuracy looks like as the dimension grows |
| `11_simulate_more.txt` | `examples/lfi/11_...` | when more simulations help and when they do not; a plausible rank self-check shown to FAIL |
| `12_gandk_budget.txt` | `examples/lfi/12_...` | the same budget-versus-rank grid on g-and-k, where no closed-form ratio exists |
| `14_multimodal_budget.txt` | `examples/lfi/14_...` | rank vs budget on that defect, and the discovery that embedding width caps every rank sweep |
| `13_multimodal_exact.txt` | `examples/lfi/13_...` | multimodality against an exact posterior; the operator smooths across modes |

Runs are seeded, so re-running reproduces these files up to floating-point
non-determinism in the threaded BLAS calls. Timings in the captured output are
machine-dependent and will differ.
