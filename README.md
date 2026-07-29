# Finite Element Teaching Repository

This repository is written in Julia and is designed for teaching purposes. It introduces
the fundamentals of the finite element method (FEM) and explores both assembled and
matrix-free approaches. It has two parts:

- `src/` — **FESolid2D**, a 2D solid mechanics finite element package (matrix-free,
  libCEED/Ratel-style), with a `test/` suite. Hyperelasticity uses a hand-derived
  Jacobian; plasticity's Jacobian is instead obtained via
  [Enzyme.jl](https://enzyme.mit.edu/) automatic differentiation.
- `notebooks/` — the teaching/exploratory notebooks the package is developed from.

## Running

**Package + tests:**
```julia
using Pkg
Pkg.develop(path=".")     # or path="/path/to/fe-demo" from elsewhere
Pkg.test("FESolid2D")
```

**Notebooks:** notebooks are not tied to the package's own `Project.toml` — they run in
your notebook kernel's own Julia environment, so install their dependencies there
(`FESolid2D` itself, plus whichever of `NLsolve`/`Plots` a given notebook uses):
```julia
using Pkg
Pkg.develop(path="/path/to/fe-demo")
Pkg.add(["NLsolve", "Plots"])
```
then open any notebook under `notebooks/` in Jupyter/IJulia.

## FEM1D
- Begins with Ciarlet’s (1978) definition of finite elements.  
- Discusses the ill-conditioning problem of evenly spaced nodes for higher-order polynomials.  
- Shows how using **Lobatto points** resolves this issue.  
- Implements **L² projection** and **Poisson problems** using assembled and matrix-free methods for both linear and nonlinear constitutive equations.

## FEM2D
- Extends 1D elements to 2D quadrilateral elements using tensor products.  
- Solves **L² projection** and **Poisson problems** in 2D.  

## Benchmark2D
- Includes the benchmark problem from **libCEED**.  
- Compares the performance of assembled vs. matrix-free approaches.  

## Elasticity2D
- Solves **linear elasticity** problems (manufactured solutions).  
- Demonstrates convergence orders for different element types using assembled and matrix-free approaches.  
- Includes traction (Neumann) BCs via a face element restriction.

## MixedElasticity2D
- Solves **incompressible linear elasticity** (manufactured solutions).  
- Explores different mixed elements with continuous and discontinuous pressure spaces.  
- Performs the **inf-sup test** to examine the stability of mixed elements.  

## Hyperelasticity2D
- Solves **Neo-Hookean hyperelasticity** (isochoric-volumetric split, plane strain), matrix-free.
- Demo: a beam clamped on one edge and pulled by a traction on the opposite edge, solved
  with a load-stepped Newton solve, with a deformed-vs-undeformed mesh visualization.

## Plasticity2D
- Solves **multiplicative Hencky plasticity** (von Mises/J2 yield, linear + Voce hardening,
  plane strain), matrix-free, with the Jacobian obtained via Enzyme.jl automatic
  differentiation.
- Demo: the same clamped-beam problem as Hyperelasticity2D, run through both a purely
  elastic reference material and a yielding one, showing the load-displacement response
  diverge once yielding starts.
