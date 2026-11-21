# 0096. The tanh curve's constant is a correctly rounded literal

Status: accepted, 2026-11-21

## Context

The tanh saturation curve is `tanh(atanh(1/2) a / K)`, half its ceiling at `K`. Three modules held
its constant as `math.atanh(0.5)`: `models.bayes_mmm`, `transforms.saturation` and
`inference.sensitivity`. Python's `math.atanh` is the platform libm's, and libm need not round
correctly. glibc 2.39, Ubuntu 24.04's, returns `0x1.193ea7aad030ap-1`; glibc 2.43 returns
`0x1.193ea7aad030bp-1`. Maxima's 50-digit `log(3)/2` puts the second 0.41 ulp from the exact value
and the first 0.59 ulp, so 2.43's is the correctly rounded one.

One ulp of a constant does not stay one ulp. On the same AVX2 machine, Ubuntu 24.04 and 26.04
gave other digests of the tanh world's density and gradient and of its short seeded fit, and no
other world moved. NUTS carries a last bit into every later draw, and some parameter's draws then
differ by more than six times the spread of its own draws, so a seeded tanh fit was a fact of the
platform's libm. The CI digests of these tests were recorded as per-CPU classes, and the class
that GitHub's ubuntu-24.04 runners drew on 2026-11-01 was this libm, not their CPU.

## Decision

- **The constant is the literal `0.5493061443340549`**, `ln(3)/2` correctly rounded, in each of
  the three modules, with a docstring that says why it is not `math.atanh(0.5)`.
- **A test computes `ln(3)/2` in decimal arithmetic**, which no libm answers, and holds each
  module's constant to the nearest double, `0x1.193ea7aad030bp-1`. Under glibc 2.39 it failed on
  the tree before; under glibc 2.43 that tree passed it, since its `math.atanh(0.5)` was already
  the nearest double.
- **The pinned digests are held to the machine's dispatch** (`tests/dispatch.py`), whose key names
  the libc beside numpy's loops, the OpenBLAS cores and XLA's target. glibc 2.39 now draws glibc
  2.43's digests; its key stays recorded, so a later libm difference shows as a failure under a
  known key rather than as a new CPU class.

## Consequences

- Under glibc 2.43 no number changes: the literal is the value `math.atanh(0.5)` gave there.
- Under glibc 2.39 the constant moves up one ulp, and every number of a tanh curve moves with it:
  a seeded tanh fit draws what it draws under glibc 2.43. A result recorded under glibc 2.39 with
  a tanh curve before this change does not reproduce bit for bit after it.
- The tests and test worlds that write the curve again with `math.atanh(0.5)`, as an oracle or to
  draw data, keep the platform's value. Under glibc 2.39 they hold against the literal: the eleven
  test files that build a tanh curve passed there, but for two Hypothesis tests and one Maxima
  script that a read-only checkout and Ubuntu 24.04's Maxima 5.46 stop, whatever the constant.

## Alternatives

- **`math.log(3) / 2`.** Halving is exact, so it is correct wherever libm's `log(3)` is; it moves
  the dependence on libm to another function rather than removing it.
- **Record the digests per libc and leave the constant.** The tests would pass, and the model would
  still answer differently by platform; the digests are a symptom of that, not the defect.
