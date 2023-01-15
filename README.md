# SCF_MRJD_RS
In this project I aim to learn the basics of Rust and computation within Rust 
by implementing a progamme that can analyze the molecule geometry and run standard
HF SCF (maybe DFT) calculations. At the beginning I will follow the Crawford's group
tutorials on the subject. Later on I will try to implement the code myself.

## Dependencies
- Rust
-  `OpenBLAS` installation with the `LAPACK` and `BLAS` libraries, for Arch Linux or Manjaro Linux users, install the `openblas-lapack` package from the AUR.
```bash
yay -S openblas-lapack
```
- `libgls-dev` for the `gsl` library, for Arch Linux or Manjaro Linux users, install the `gsl` package from the AUR.
```bash
yay -S gsl
```
