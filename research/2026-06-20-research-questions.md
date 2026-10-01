# Research Questions

Date: 2026-06-20

## Q1: Cheap inference only, or cheap training too?

Answer:

- Cheap inference is mandatory.
- Training is not the initial target.
- The intended concept is transfer: start with an existing trained open model
  and encode some of its weights or functions into optics.

Implication:

The research target is not "train a frontier model optically." It is:

1. train or obtain a model digitally;
2. choose a model component or subgraph that can be physically represented;
3. compile or transfer that component into an optical substrate;
4. calibrate and fine-tune around hardware imperfections;
5. run inference with optics performing part of the computation.

Open follow-up:

Which component of a trained model should be transferred first: a projection
matrix, a convolution/filter bank, an embedding-like transform, or a small
end-to-end toy classifier?

## Q2: What should the optical block do first?

Answer:

- The first optical block should replace a fixed matrix multiply / projection.
- Material nonlinearities should be included as simulation variants, not as the
  primary first target.
- The purpose is to test whether trained model weights can transfer into optics
  in a measurable, bounded way.

Implication:

The first model-to-optics transfer target should be a single fixed linear
operation. The simulation should compare:

1. ideal digital baseline;
2. optical matrix multiply with quantization/loss/noise;
3. optical matrix multiply with unwanted material nonlinearity;
4. optical matrix multiply with trained compensating nonlinearity.

Do not begin with a full nonlinear physical operator or a pure optical front
end. Those remain useful later directions, but a fixed projection is the
cleanest first bridge from trained weights to optical hardware.

Open follow-up:

What should matter most in the success metric: energy, latency, fabrication
cost, scalability, or proof that the physics can preserve useful transforms?

## Q3: What should matter most as the success metric?

Answer:

Priority order:

1. Prove that the optical implementation can preserve useful trained transforms
   under realistic noise, drift, and calibration error.
2. Then measure energy per inference.
3. Then measure latency.

Implication:

The first experiment must not optimize for a flattering energy or latency claim
before it proves that the optical block still computes something useful. The
primary pass/fail metric should be functional preservation under realistic
hardware constraints.

Initial measurement categories:

- accuracy or task-performance drop vs digital baseline;
- equivalent useful precision of the optical block;
- robustness to fabrication variation, detector noise, optical loss, and drift;
- calibration frequency and calibration cost;
- improvement from hardware-aware fine-tuning or compensating nonlinearities;
- energy and latency estimates after functional preservation is established.

Open follow-up:

Do we care more about fixed fabricated weights, tunable photonic meshes, or both?

## Q4: Fixed fabricated weights, tunable photonic meshes, or both?

Answer:

Both matter, but sequence them:

1. Simulate both fixed fabricated weights and tunable photonic meshes.
2. Prototype or emulate tunable photonic meshes first because they are easier to
   calibrate, debug, and iterate.
3. Keep fixed fabricated weights as the long-term cheap replicated inference
   target because they best match the original bulk-medium / etched-weights
   concept.

Implication:

The first simulation plan should compare at least two deployment modes:

- Tunable mode: optical weights can be adjusted, calibrated, and corrected.
- Fixed mode: optical weights are compiled into a static fabricated structure.

The tunable path is the research/debug path. The fixed path is the eventual
cost-scaling hypothesis.

Open follow-up:

What error/noise level can the model tolerate before the optical idea stops
being useful?

## Q5: What error/noise level is tolerable?

Answer:

Do not pick a single guessed threshold up front. Define tolerance empirically by
mapping the performance cliff under injected optical-style errors.

The first experiment should sweep:

- weight quantization;
- phase noise;
- optical loss;
- crosstalk;
- detector noise;
- thermal drift;
- calibration error;
- unwanted material nonlinearity;
- trained compensating nonlinearity.

Provisional toy-workload viability target:

- Less than 1-2% absolute accuracy loss after hardware-aware adaptation on a
  toy classifier.
- Stable behavior across the expected drift/noise range.
- Measurable recovery from calibration or compensating nonlinearity.

Implication:

The output should be a tolerance curve, not just a pass/fail number. The useful
research artifact is the map showing where the optical block degrades smoothly,
where adaptation recovers performance, and where it falls off a cliff.

Initial question pass status:

Complete enough to select the first toy workload.
