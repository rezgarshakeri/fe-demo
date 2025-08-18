# Finite Element Teaching Repository

This repository, written in Julia (Jupyter notebooks), is designed for teaching purposes. It introduces the fundamentals of the finite element method (FEM) and explores both assembled and matrix-free approaches.

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

## MixedElasticity2D
- Solves **incompressible linear elasticity** (manufactured solutions).  
- Explores different mixed elements with continuous and discontinuous pressure spaces.  
- Performs the **inf-sup test** to examine the stability of mixed elements.  
