# SCF_MRJD_RS
This project started out of curiosity to understand the 
fundamentals of Hartree-Fock (HF) calculations. As we only had an "overview" lecture which 
did not include the necessary details for a broader understanding of the subject, I decided
to combine my interest in Rust and computational chemistry and implement a "simple" HF 
programme.

As a starting point I used the tutorials from the Crawford's group, which can be found
at the following link:
[C++ Programming Tutorial in Chemistry](https://github.com/CrawfordGroup/ProgrammingProjects/)

Although the tutorials are written in C++, I decided to implement the code in Rust. I did also start a C++ version of the code, but I decided to focus on the Rust version.

My Rust implementation is not a direct translation of the C++ code, but I tried to keep the same structure and the same naming conventions. 

The code for the Crawford's group tutorials can be found in `src/Crawford_projects`. I am going to use part of the code from the tutorials to implement the Hartree-Fock SCF algorithm.

# Dependencies
- `Rust`
-  `OpenBLAS` (including the LAPACK implementation)
    ```bash
    # Arch based Linux distro:
    yay -S openblas-lapack

    # Ubuntu based Linux distro:
    sudo apt install libopenblas-dev
    ```
- `libgls-dev` 
    ```bash
    # Arch based Linux distro:
    yay -S gsl

    # Ubuntu based Linux distro:
    sudo apt install libgsl-dev
    ```
