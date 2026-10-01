# From Trained Model to Optical Implementation

Date: 2026-06-20

## Question

If we already have a trained open model, how do we produce an optical
implementation from it?

## Short Answer

Treat it as a compiler and calibration problem, not as a direct file conversion.

The pipeline is:

1. Select the part of the model that optics should implement.
2. Convert that part into an optical-friendly mathematical form.
3. Map the mathematical form onto a physical optical architecture.
4. Simulate the optical hardware with realistic imperfections.
5. Fine-tune or distill the model around the hardware model.
6. Fabricate or configure the optical device.
7. Calibrate the fabricated device.
8. Wrap it in electronics for input, output, control, and fallback.

## Why a Whole Model Is the Wrong First Target

An existing trained model contains more than weights:

- tokenization and embeddings;
- matrix multiplies;
- attention state and softmax;
- nonlinear activations;
- normalization layers;
- residual connections;
- memory movement;
- sampling, decoding, and control flow.

Optics is most naturally good at propagation, mixing, interference, weighting,
parallel transforms, and sometimes nonlinear transfer functions. It is not a
drop-in implementation of a full transformer runtime.

So the practical move is to transfer a subgraph.

## Candidate Transfer Targets

### 1. Single fixed linear layer

Examples:

- projection matrix;
- classifier head;
- image patch projection;
- learned compression/projection layer.

Why it is first:

- It maps cleanly to optical matrix-vector multiplication.
- It makes error and precision losses easy to measure.
- MZI meshes, microring weight banks, and diffractive systems can all be
  evaluated against the same mathematical target.

### 2. Convolution or filter bank

Examples:

- early vision model filters;
- optical preprocessing front end;
- spectral/spatial feature extractor.

Why it is attractive:

- Optics naturally handles spatial transforms and parallel filtering.
- Fixed filters can be physically encoded.
- Errors may be tolerable if downstream digital layers adapt.

### 3. Embedding-like transform

Examples:

- fixed random projection;
- learned projection into lower-dimensional latent space;
- analog feature map before a digital classifier.

Why it is useful:

- It is a good bridge between toy vision tasks and more general model
  acceleration.
- It tests whether optics can cheaply produce useful representations.

### 4. Tiny end-to-end optical classifier

Examples:

- MNIST or Fashion-MNIST passive diffractive classifier;
- small optical convolutional classifier;
- hybrid optical front end plus digital classifier.

Why it is useful:

- It is measurable.
- It exercises a complete inference path.
- It avoids frontier-model theatre while still testing the physics.

## Transfer Pipeline

### Step 1: Extract

Choose a trained open model and extract a submodule:

- a weight matrix `W`;
- a convolution kernel bank;
- a learned projection;
- a small stack of layers.

Freeze it first. Do not start with full-model retraining.

### Step 2: Approximate

Convert the target into a form the optical substrate can represent:

- decompose a matrix with singular value decomposition;
- factor a large matrix into smaller optical tiles;
- quantize weights into achievable phase, attenuation, or coupling levels;
- constrain weights to non-negative, signed-balanced, unitary, low-rank, sparse,
  or complex-valued forms depending on the optical architecture.

For MZI meshes, arbitrary matrix mappings often go through decompositions into
unitary transforms plus scaling. For diffractive systems, the target becomes a
set of phase/amplitude masks or volumetric refractive-index features.

### Step 3: Map

Map the approximated representation to hardware parameters:

- phase shifts;
- attenuation levels;
- coupler ratios;
- microring resonance weights;
- waveguide path lengths;
- diffractive layer geometry;
- material nonlinear transfer curves.

This is where a "weight" stops being a number in a tensor and becomes a
manufacturing/control parameter.

### Step 4: Simulate

Simulate the physical implementation, including:

- optical loss;
- crosstalk;
- shot noise and detector noise;
- finite precision;
- fabrication variation;
- alignment error;
- thermal drift;
- nonlinear response;
- calibration error.

This produces a hardware-realistic surrogate model.

### Step 5: Adapt

Adapt the trained model to the hardware surrogate:

- hardware-aware fine-tuning;
- distillation from the original digital model;
- retraining only surrounding digital layers;
- robustness training against noise, drift, and fabrication variation;
- learned compensation for nonlinearities.

This is still digital training, but the target is an optical implementation.

### Step 6: Fabricate or Configure

Depending on architecture:

- fixed diffractive/printed optics: fabricate masks, layers, or volumetric
  structures;
- tunable integrated photonics: set MZI phases, ring resonances, attenuators,
  and control voltages;
- bulk waveguides: fabricate waveguide paths, couplers, and material regions.

### Step 7: Calibrate

Measure the actual device:

- feed known input patterns;
- estimate the realized transfer function;
- compare expected vs observed outputs;
- adjust tunable parameters if available;
- update the digital wrapper or compensation model.

Calibration is not optional. Analog hardware lies politely and continuously.

### Step 8: Integrate

Wrap the optical block with:

- input encoding;
- light source and modulation;
- detectors;
- analog-to-digital conversion;
- digital post-processing;
- health checks;
- recalibration path;
- fallback to digital compute.

## First Practical Experiment

Start with a trained small digital classifier and transfer one component:

1. Train a tiny MNIST/Fashion-MNIST model digitally.
2. Extract the first projection or classifier-head matrix.
3. Map it to a simulated optical matrix multiply.
4. Add quantization, loss, crosstalk, noise, and nonlinear transfer variants.
5. Fine-tune the digital layers around that simulated optical block.
6. Compare against the original digital baseline.

Success metrics:

- accuracy drop under optical constraints;
- equivalent bits of useful precision;
- sensitivity to drift/noise;
- calibration frequency needed;
- estimated latency/energy advantage;
- whether compensating nonlinearity improves robustness.

## What This Means for Frontier Models

If this works on the toy path, scale the idea gradually:

1. one fixed matrix;
2. one block in a small model;
3. repeated optical tiles;
4. attention-adjacent projections;
5. narrow LLM-adjacent benchmark.

Do not begin with "compile Llama into glass." That phrase sounds good right up
until the first layer norm walks into the room carrying a clipboard.

