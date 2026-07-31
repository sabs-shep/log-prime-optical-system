**Archived DOI:** https://doi.org/10.5281/zenodo.21724070

Log-Prime Optical Encoding and Single-Detector Decoding System
Author: Sebastian Shepherd
Original disclosure date: 30 July 2026
Repository status: Defensive technical disclosure and software/theory proof of concept

This repository documents a log-prime optical frequency/code generation and single-detector decoding system. It includes a defensive technical disclosure, calculated wavelength-to-pixel mapping software, simulated mapped-lane optical NOR behaviour, decoder proof output, and source code.

The purpose of this repository is to establish a timestamped public authorship record and to make the disclosed subject matter discoverable as public technical prior art.

Core Idea
The disclosed method does not merely assign prime labels to existing wavelength lanes. Instead, intended optical frequency or calibrated frequency-code positions are generated from prime numbers using a relation of the form:

C(P) = A * log(P) + B
where:

P is a prime number;
A and B are fixed tuned constants for the chosen code scale or calibration model;
C(P) is the generated log-prime frequency/code coordinate.

In an optical implementation, the generated code coordinate can be used to define an intended optical frequency:
f(P) = A_f * log(P) + B_f
where f(P) is the intended optical frequency. The photon-energy equivalent of that frequency is:
E = h * f(P)
and the corresponding energy-per-unit-charge scale is:
E / e = h * f(P) / e
where:

h is Planck’s constant;
f(P) is the optical frequency generated from the prime P;
e is the elementary charge.

For detector readout, the measured voltage is treated as a calibrated voltage-code quantity rather than as a raw direct measurement of photon energy. In the constant-intensity case, the voltage-code readout is interpreted as a mean log-prime code state. In the variable-intensity case, the voltage-code readout is interpreted as an amplitude-weighted mean, optionally combined with a separate current or amplitude measurement.
