# ava

[![Build Status](https://travis-ci.org/ggf84/ava.svg?branch=master)](https://travis-ci.org/ggf84/ava)
[![Crates.io](https://img.shields.io/crates/v/ava.svg)](https://crates.io/crates/ava)
[![Docs](https://docs.rs/ava/badge.svg)](https://docs.rs/ava)
[![Minimum rustc version](https://img.shields.io/badge/rustc-nightly-red.svg)](https://github.com/ggf84/ava#minimum-rustc-version)
[![License](https://img.shields.io/crates/l/ava.svg)](https://github.com/ggf84/ava#license)
[![Lines of Code](https://tokei.rs/b1/github/ggf84/ava?category=code)](https://github.com/ggf84/ava)
[![Code size in bytes](https://img.shields.io/github/languages/code-size/ggf84/ava.svg)](https://github.com/ggf84/ava)
[![dependency status](https://deps.rs/repo/github/ggf84/ava/status.svg)](https://deps.rs/repo/github/ggf84/ava)

A Rust toolkit for gravitational N-body simulations.


## Status

**Work in Progress:** This project is not ready for general usage. The API may still change
a lot and the documentation is incomplete. Please check back later if you are curious.


## Algorithms

Gravitational force calculations were carefully implemented in order to explore Newton's
third-law. This improves the performance by a factor of 2. The overall force calculation
is parallelized with `rayon` and the innermost loop is auto-vectorized by `rustc`. However,
it still scales as O(N^2). I have no plan to use Tree- [O(NlogN)] or FMM- [O(N)] algorithms,
since they unavoidably introduce errors in the force. My focus here is on correctness and
simplicity rather than performance at all costs. A GPU/OpenCL implementation is being planned.

Besides that, here are other algorithms already implemented:
- Initial conditions:
    - Initial mass functions
        - Equal masses
        - Maschberger (2013)
    - Spherical density profiles
        - Plummer (1911)
        - Dehnen (1993), gamma = [0, 1/2, 1, 3/2, 2]
- Methods of integration:
    - Hermite integrators (4th-, 6th- and 8th-order)
        - constant time-steps
        - variable (shared) time-steps
        - individual (block) time-steps


## How to Contribute

Any kind of contribution is welcome. Please contact the main author or open an issue
to discuss improvements. If you want to help, there are several ways to go:
- adding new algorithms
- improving the documentation
- improving the code implementation and/or performance
- testing the code on your systems to find bugs
- helping with language issues
- providing feature requests
- etc.


## Cargo Features

- `f32`: single precision mode
- `f64`: double precision mode [default]
- `serde1`: support for (de-)serialization


## Minimum `rustc` version

The minimum supported `rustc` version is `nightly`.


## License

This project is distributed under the terms of the MIT License.

See [LICENSE](LICENSE) for details.
