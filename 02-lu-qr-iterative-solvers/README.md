# LU, QR and Iterative Solvers

This project implements several numerical methods for solving systems of linear algebraic equations in C++.
It is part of the numerical-methods repository and focuses on both direct and iterative approaches to solving linear systems.

## Implemented methods
### Direct methods

- LU decomposition with partial pivoting

- QR decomposition using Householder reflections

### Iterative methods

- Simple Iteration Method (MPI)

- Gauss–Seidel method

Each method is tested on different types of matrices, including: diagonally dominant matrices, indefinite matrices, ill-conditioned matrices, specially constructed test matrices.
All results are compared with exact solutions obtained using the Eigen library.
