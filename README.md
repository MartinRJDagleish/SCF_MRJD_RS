[![](https://github.com/MartinRJDagleish/SCF_MRJD_RS/actions/workflows/build.yml/badge.svg)](https://github.com/MartinRJDagleish/SCF_MRJD_RS) 
[![](https://www.aschey.tech/tokei/github/MartinRJDagleish/SCF_MRJD_RS)](https://github.com/MartinRJDagleish/SCF_MRJD_RS) 
[![](https://www.aschey.tech/tokei/github/MartinRJDagleish/SCF_MRJD_RS?category=code)](https://github.com/MartinRJDagleish/SCF_MRJD_RS) 
[![](https://www.aschey.tech/tokei/github/MartinRJDagleish/SCF_MRJD_RS?category=comments)](https://github.com/MartinRJDagleish/SCF_MRJD_RS) 
[![](https://www.aschey.tech/tokei/github/MartinRJDagleish/SCF_MRJD_RS?category=files)](https://github.com/MartinRJDagleish/SCF_MRJD_RS) 

# SCF_MRJD_RS
<details>
<summary> Background info about the project </summary>
This project started out of curiosity to understand the 
fundamentals of Hartree-Fock (HF) calculations. As I only had an "overview" lecture in my 
Bachelor's degree which did not include the necessary details for a broader understanding of the subject, I decided
to combine my interest in Rust and computational chemistry and implement a "simple" HF 
programme.

As a starting point I used the tutorials from the Crawford's group, which can be found
at the following link:
[C++ Programming Tutorial in Chemistry](https://github.com/CrawfordGroup/ProgrammingProjects/)

Although the tutorials are written in C++, I decided to implement the code in Rust. I did also start a C++ version of the code, but I decided to focus on the Rust version.

My Rust implementation is not a direct translation of the C++ code, but I tried to keep the same structure and the same naming conventions. 

The code for the Crawford's group tutorials can be found in `src/Crawford_projects`. I am going to use part of the code from the tutorials to implement the Hartree-Fock SCF algorithm.
</details>


<details>
<summary> Parts that are implemented </summary>

- Molecular integrals ($T_\text{el}, V_\text{eN}, V_\text{NN}, V_\text{ee}$) which are needed for RHF SCF using McMurchie-Davidson algorithm

- Dipole integrals (Mu-Tensor ($3\times N \times N$))

- Bad $N^8$ scaling MP2 (better: $N^5$ soon)

- Basic geometry analysis (Angles, OOP, Dihedrals, Inertia tensor, Analysis of geometry, Rotor classification)
</details>

<details>
<summary> PLANNED:</summary>

- Better geometry analysis (using vdW radii for better classification of which bond distances are necessary + which angles are necessary)

- Transform integrals from cartesian (Hermite) basis functions to pure regular solid harmonics 

</details>

# Dependencies
- `Rust`
-  `OpenBLAS` (including the LAPACK implementation)

    [![](https://img.shields.io/badge/Arch_Linux-1793D1?style=for-the-badge&logo=arch-linux&logoColor=white)](https://aur.archlinux.org/packages/openblas-lapack)
    ```bash
    # Arch based Linux distro:
    yay -S openblas-lapack

    # Ubuntu based Linux distro:
    sudo apt install libopenblas-dev
    ```
- `libgsl-dev` 

    [![](https://img.shields.io/badge/Arch_Linux-1793D1?style=for-the-badge&logo=arch-linux&logoColor=white)](https://archlinux.org/packages/extra/x86_64/gsl/)
    ```bash
    # Arch based Linux distro:
    yay -S gsl

    # Ubuntu based Linux distro:
    sudo apt install libgsl-dev
    ```

# Sources / Thank you
This project would not have been possible without the great introductory **[PDF](https://joshuagoings.com/assets/integrals.pdf)** by [Joshua Goings](https://github.com/jjgoings) (@jjgoings) and his Python / Cython implementation of **[McMurchie-Davidson](https://github.com/jjgoings/McMurchie-Davidson)**. 

When starting this project the integral routines were quite a lot to learn, so in the beginning I translated his Python code to idiomatic Rust (to my understanding) and then tried to simplify things to my understanding. 

Although the integral routines are mostly a copy, the structure of the code and the "building blocks" are made by myself. 

Another thanks I would like to give to **[PySCF](https://github.com/pyscf/pyscf)**, which gave me the initial idea for a parser of `.gbs` files of basis set data, provided by **[BasisSetExchange](https://www.basissetexchange.org)**
