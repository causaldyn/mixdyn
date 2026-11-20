# The readers on null panels

Regenerated on 2026-10-31 by

```
JAX_PLATFORMS=cpu uv run --frozen python -m validation.aa_audit --jobs 4
```

from the repository's root (`--draws` sets the draws a cell, 200 by default; the table is the same
at any `--jobs`). The harness, its panels, its designs and its seeds are `validation/aa_audit.py`'s
docstring; the decision it informs is ADR 0058. `tests/test_aa_audit.py` reads 3 of these cells
over their first 60 draws.

Each cell is 200 null panels of 104 weeks: no treatment effect, the noise `ar1` (a first-order
autoregression of 0.5), `trend` (the same and a linear trend of its own a geo) or `shift` (the
same and a level shift all geos share from a random week), at 10 or 30 geos. FPR is the share of
draws whose p-value is below 0.05, SE its Monte Carlo standard error at 0.05, `sqrt(0.05 x 0.95 /
200) = 0.0154`, and coverage the share whose 95 % interval holds 0.

## The band

A cell passes where its FPR is within [0.03, 0.08]. The top is the nominal 0.05 plus two Monte
Carlo standard errors, 0.0808, rounded to the hundredth: a reader rejecting a true null more
often than that is overconfident. The floor is set tighter than the 0.0192 two standard errors
give, so that a reader rejecting well under its level, an interval wider than it need be and
power spent, fails too. A cell passes on 6 to 16 rejections of 200: a reader exactly at 0.05
passes with probability 0.914 (so about 3 of 36 such cells fail by chance), one at 0.10 with
0.207, one at 0.025 with 0.384.

## The readers at their shipped defaults

| reader | noise | geos | FPR | SE | coverage | band |
| --- | --- | --- | --- | --- | --- | --- |
| did | ar1 | 10 | 0.025 | 0.0154 | 0.975 | FAIL |
| did | ar1 | 30 | 0.055 | 0.0154 | 0.945 | pass |
| did | trend | 10 | 0.065 | 0.0154 | 0.935 | pass |
| did | trend | 30 | 0.065 | 0.0154 | 0.935 | pass |
| did | shift | 10 | 0.065 | 0.0154 | 0.935 | pass |
| did | shift | 30 | 0.045 | 0.0154 | 0.955 | pass |
| tbr | ar1 | 10 | 0.010 | 0.0154 | 0.990 | FAIL |
| tbr | ar1 | 30 | 0.015 | 0.0154 | 0.985 | FAIL |
| tbr | trend | 10 | 0.010 | 0.0154 | 0.990 | FAIL |
| tbr | trend | 30 | 0.020 | 0.0154 | 0.980 | FAIL |
| tbr | shift | 10 | 0.015 | 0.0154 | 0.985 | FAIL |
| tbr | shift | 30 | 0.025 | 0.0154 | 0.975 | FAIL |
| synthetic_control | ar1 | 10 | 0.035 | 0.0154 | 0.965 | pass |
| synthetic_control | ar1 | 30 | 0.050 | 0.0154 | 0.950 | pass |
| synthetic_control | trend | 10 | 0.045 | 0.0154 | 0.955 | pass |
| synthetic_control | trend | 30 | 0.055 | 0.0154 | 0.945 | pass |
| synthetic_control | shift | 10 | 0.040 | 0.0154 | 0.960 | pass |
| synthetic_control | shift | 30 | 0.030 | 0.0154 | 0.970 | pass |
| sdid | ar1 | 10 | 0.055 | 0.0154 | 0.945 | pass |
| sdid | ar1 | 30 | 0.050 | 0.0154 | 0.950 | pass |
| sdid | trend | 10 | 0.020 | 0.0154 | 0.980 | FAIL |
| sdid | trend | 30 | 0.035 | 0.0154 | 0.965 | pass |
| sdid | shift | 10 | 0.030 | 0.0154 | 0.970 | pass |
| sdid | shift | 30 | 0.030 | 0.0154 | 0.970 | pass |

## The switchback, at every method it offers (audited only)

The national KPI summed over the geos, 26 blocks of 4 weeks, a washout of 1, half the blocks on,
read by each method `mixdyn.experiments.switchback.Method` names at `origin/main` on 2026-10-31.
The `switchback:jackknife` rows were added on 2026-11-06 with the method (ADR 0056), on the same
panels and designs, by `python -m validation.aa_audit --readers switchback:jackknife`; the other
rows were not rerun.

| reader | noise | geos | FPR | SE | coverage | band |
| --- | --- | --- | --- | --- | --- | --- |
| switchback:randomization | ar1 | 10 | 0.065 | 0.0154 | 0.935 | pass |
| switchback:randomization | ar1 | 30 | 0.030 | 0.0154 | 0.970 | pass |
| switchback:randomization | trend | 10 | 0.030 | 0.0154 | 0.970 | pass |
| switchback:randomization | trend | 30 | 0.040 | 0.0154 | 0.960 | pass |
| switchback:randomization | shift | 10 | 0.060 | 0.0154 | 0.940 | pass |
| switchback:randomization | shift | 30 | 0.045 | 0.0154 | 0.955 | pass |
| switchback:hac | ar1 | 10 | 0.125 | 0.0154 | 0.875 | FAIL |
| switchback:hac | ar1 | 30 | 0.090 | 0.0154 | 0.910 | FAIL |
| switchback:hac | trend | 10 | 0.100 | 0.0154 | 0.900 | FAIL |
| switchback:hac | trend | 30 | 0.090 | 0.0154 | 0.910 | FAIL |
| switchback:hac | shift | 10 | 0.105 | 0.0154 | 0.895 | FAIL |
| switchback:hac | shift | 30 | 0.080 | 0.0154 | 0.920 | pass |
| switchback:jackknife | ar1 | 10 | 0.065 | 0.0154 | 0.935 | pass |
| switchback:jackknife | ar1 | 30 | 0.030 | 0.0154 | 0.970 | pass |
| switchback:jackknife | trend | 10 | 0.030 | 0.0154 | 0.970 | pass |
| switchback:jackknife | trend | 30 | 0.055 | 0.0154 | 0.945 | pass |
| switchback:jackknife | shift | 10 | 0.055 | 0.0154 | 0.945 | pass |
| switchback:jackknife | shift | 30 | 0.075 | 0.0154 | 0.925 | pass |

## The opt-in alternatives, on the same panels and designs

`did:unit_placebo` and `tbr:unit_placebo` are `read_did` and `read_tbr` at
`variance="unit_placebo"` over the design's pairs; `pairs:sign_flip` is `read_pairs`;
`sdid:unit_placebo` is `read_sdid` at `variance="unit_placebo"`. Every reader reads the same
200 panels and the same design a draw, so a reader and its alternative differ draw by draw only
by how they read.

| reader | noise | geos | FPR | SE | coverage | band |
| --- | --- | --- | --- | --- | --- | --- |
| did:unit_placebo | ar1 | 10 | 0.030 | 0.0154 | 0.970 | pass |
| did:unit_placebo | ar1 | 30 | 0.070 | 0.0154 | 0.930 | pass |
| did:unit_placebo | trend | 10 | 0.030 | 0.0154 | 0.970 | pass |
| did:unit_placebo | trend | 30 | 0.045 | 0.0154 | 0.955 | pass |
| did:unit_placebo | shift | 10 | 0.035 | 0.0154 | 0.965 | pass |
| did:unit_placebo | shift | 30 | 0.040 | 0.0154 | 0.960 | pass |
| tbr:unit_placebo | ar1 | 10 | 0.015 | 0.0154 | 0.985 | FAIL |
| tbr:unit_placebo | ar1 | 30 | 0.085 | 0.0154 | 0.915 | FAIL |
| tbr:unit_placebo | trend | 10 | 0.030 | 0.0154 | 0.970 | pass |
| tbr:unit_placebo | trend | 30 | 0.050 | 0.0154 | 0.950 | pass |
| tbr:unit_placebo | shift | 10 | 0.045 | 0.0154 | 0.955 | pass |
| tbr:unit_placebo | shift | 30 | 0.045 | 0.0154 | 0.955 | pass |
| pairs:sign_flip | ar1 | 10 | nan | nan | nan | FAIL (200 refused) |
| pairs:sign_flip | ar1 | 30 | 0.070 | 0.0154 | 0.930 | pass |
| pairs:sign_flip | trend | 10 | nan | nan | nan | FAIL (200 refused) |
| pairs:sign_flip | trend | 30 | 0.030 | 0.0154 | 0.970 | pass |
| pairs:sign_flip | shift | 10 | nan | nan | nan | FAIL (200 refused) |
| pairs:sign_flip | shift | 30 | 0.045 | 0.0154 | 0.955 | pass |
| sdid:unit_placebo | ar1 | 10 | 0.040 | 0.0154 | 0.960 | pass |
| sdid:unit_placebo | ar1 | 30 | 0.030 | 0.0154 | 0.970 | pass |
| sdid:unit_placebo | trend | 10 | 0.045 | 0.0154 | 0.955 | pass |
| sdid:unit_placebo | trend | 30 | 0.045 | 0.0154 | 0.955 | pass |
| sdid:unit_placebo | shift | 10 | 0.040 | 0.0154 | 0.960 | pass |
| sdid:unit_placebo | shift | 30 | 0.050 | 0.0154 | 0.950 | pass |

`read_pairs` refuses five pairs: its 32 sign assignments read each sign and its reverse alike, so
its least p-value is 2 / 32 and no test at 5 % is possible. The randomisation readings over 5
pairs reject at most 1 / 32 = 0.031 of the time by construction, at the band's floor.

## Cells read over 1000 draws

The cells that passed or failed nearest the band's edges, read again over their first 1000 draws
(SE 0.0069):

| reader | noise | geos | FPR over 1000 | coverage |
| --- | --- | --- | --- | --- |
| did | ar1 | 10 | 0.036 | 0.964 |
| tbr | ar1 | 30 | 0.015 | 0.985 |
| did:unit_placebo | ar1 | 30 | 0.052 | 0.948 |
| tbr:unit_placebo | ar1 | 10 | 0.029 | 0.971 |
| tbr:unit_placebo | ar1 | 30 | 0.055 | 0.945 |
| pairs:sign_flip | ar1 | 30 | 0.046 | 0.954 |
