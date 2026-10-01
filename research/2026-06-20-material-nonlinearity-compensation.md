# Material Nonlinearity as Computation and Compensation

Date: 2026-06-20

## Prompt

The project note captured that nonlinearity may sometimes be addressed in the
material itself, by deliberately introducing a second compensating nonlinearity
rather than correcting everything electronically.

## Interpretation

This is an important refinement. In optical AI hardware, material nonlinearity
should not be treated only as an impairment. It can play at least three roles:

1. Error term to be compensated.
2. Activation function needed by neural networks.
3. Designed physical transfer function that partially cancels or reshapes
   another unwanted nonlinearity.

The third case is especially relevant to the bulk-medium idea: the physical
medium may be co-designed so that propagation, loss, phase shift, saturation,
or coupling produce the target input-output map directly.

## Mechanisms Worth Tracking

- Kerr nonlinearity: intensity-dependent refractive index.
- Saturable absorption: transmission changes with optical intensity.
- Reverse saturable absorption: absorption increases with intensity.
- Two-photon absorption: nonlinear absorption in materials such as silicon and
  germanium platforms.
- Free-carrier dispersion and absorption: carrier effects induced by optical
  intensity.
- Stimulated Brillouin scattering: optoacoustic interaction that can create
  nonlinear transfer functions.
- Semiconductor optical amplifiers and electro-absorption modulators.
- Phase-change, graphene, and other integrated nonlinear materials.

## Design Idea

Instead of forcing the optical substrate to behave like an ideal digital linear
layer plus separately implemented activation, train or design the substrate as
a physical nonlinear operator:

- Let the first material/process introduce a useful but imperfect nonlinear
  response.
- Add a second material/process/geometric stage whose nonlinearity reshapes the
  total transfer curve.
- Train the full cascade with physics-aware or hardware-aware training.
- Treat the fabricated object as a learned physical function, not merely as a
  fragile implementation of abstract matrix math.

This turns a liability into a degree of freedom.

## Constraints

- Useful nonlinear effects often require optical power, resonant enhancement, or
  long interaction lengths.
- Strong nonlinearities can introduce loss, heat, bandwidth limits, hysteresis,
  drift, or noise.
- Compensation may be narrow-band, temperature-sensitive, or amplitude-range
  limited.
- A second nonlinearity can cancel one error while making calibration harder.
- The correct target may not be "compensate to linear"; for neural hardware it
  may be "shape to a stable activation-like function."

## Implication for Toy Workload

When building the first simulation, do not model the optical block as merely a
linear transform plus additive noise. Include at least one optional nonlinear
transfer curve:

- saturating transmission;
- phase shift proportional to intensity;
- detector saturation;
- a two-stage compensating nonlinearity;
- learned activation-like transfer curves.

Then compare:

1. purely linear optical block;
2. linear block plus unwanted nonlinearity;
3. linear block plus trained compensating nonlinearity;
4. physical nonlinear block trained end-to-end.

This will show whether material nonlinearity is mostly a nuisance, an activation
candidate, or part of the actual computational advantage.

## Follow-Up Result

The 2026-06-21 follow-up selected concrete material-inspired candidates and
tested them on the transformer-style MLP-up target. See
`research/2026-06-21-material-nonlinearity-candidates.md`.
