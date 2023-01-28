use crate::molecule::Molecule;
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt};
use std::fs;

pub fn run_project3_1(mut mol: Molecule, is_run_project4: bool) {
    println!("\nProject 3 implementation:\n");
    //* Project 3: SCF (with data provided)
    // ! THIS IS A QUICK FIX AND NOT A GOOD SOLUTION
    let no_basis_funcs: usize = 7;
    //* Step 1: Read Nuclear Repulsion Energy (enuc) from file
    let E_nn_val: f64 = fs::read_to_string("inp/Project3_1/STO-3G/enuc.dat")
        .expect("Failed to open enuc data!")
        .parse()
        .expect("Failed to parse enuc data file!");
    println!("Nuclear Repulsion Energy: {}\n", E_nn_val);

    //* Step 2.1: Read the overlap matrix
    let mut S_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));
    let S_matr_file_contents =
        fs::read_to_string("inp/Project3_1/STO-3G/s.dat").expect("Failed to open S matrix data!");

    for line in S_matr_file_contents.lines() {
        let line_split: Vec<&str> = line.trim().split_whitespace().collect();
        let row: usize = line_split[0].parse::<usize>().unwrap() - 1;
        let col: usize = line_split[1].parse::<usize>().unwrap() - 1;
        let val: f64 = line_split[2].parse().unwrap();
        S_matr[(row, col)] = val;
        S_matr[(col, row)] = val;
    }
    println!("Overlap matrix S:\n{:1.5}\n", &S_matr);

    //* Step 2.2: Read the kinetic energy matrix
    let mut T_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));
    let T_matr_file_contents =
        fs::read_to_string("inp/Project3_1/STO-3G/t.dat").expect("Failed to open T matrix data!");

    for line in T_matr_file_contents.lines() {
        let line_split: Vec<&str> = line.trim().split_whitespace().collect();
        let row: usize = line_split[0].parse::<usize>().unwrap() - 1;
        let col: usize = line_split[1].parse::<usize>().unwrap() - 1;
        let val: f64 = line_split[2].parse().unwrap();
        T_matr[(row, col)] = val;
        T_matr[(col, row)] = val;
    }
    println!("Kinetic energy matrix T:\n{:1.5}\n", &T_matr);

    //* Step 2.3: Read the nuclear attraction matrix
    let mut V_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));
    let V_matr_file_contents =
        fs::read_to_string("inp/Project3_1/STO-3G/v.dat").expect("Failed to open V matrix data!");

    for line in V_matr_file_contents.lines() {
        let line_split: Vec<&str> = line.trim().split_whitespace().collect();
        let row: usize = line_split[0].parse::<usize>().unwrap() - 1;
        let col: usize = line_split[1].parse::<usize>().unwrap() - 1;
        let val: f64 = line_split[2].parse().unwrap();
        V_matr[(row, col)] = val;
        V_matr[(col, row)] = val;
    }
    println!("Nuclear attraction:\n{:1.5}\n", &V_matr);

    //* Step 2.4: Form core Hamiltonian H_core = T + V
    let H_core_matr: Array2<f64> = &T_matr + &V_matr;
    println!("Core Hamiltonian:\n{:^1.5}\n", &H_core_matr);

    //* Step 3: Read the 2-electron integrals -> ERI tensor
    //* Test with Psi4
    // let ERI_file_contents = fs::read_to_string("inp/Project3_1/STO-3G/eri_Psi4_test.dat")
    //     .expect("Failed to open ERI matrix data!");
    //*Original
    let ERI_file_contents = fs::read_to_string("inp/Project3_1/STO-3G/eri.dat")
        .expect("Failed to open ERI matrix data!");

    //* 2nd try but with using a vec!
    let mut ERI_vec: Vec<f64> = Vec::new();

    for line in ERI_file_contents.lines() {
        //* Just a test with the correct data from Psi4
        // let mut line_split: Vec<&str> = line.trim().split_whitespace().collect();
        // let i: usize = line_split[0].parse::<usize>().unwrap();
        // let j: usize = line_split[1].parse::<usize>().unwrap();
        // let k: usize = line_split[2].parse::<usize>().unwrap();
        // let l: usize = line_split[3].parse::<usize>().unwrap();
        // let val: f64 = line_split[4].parse::<f64>().unwrap();
        // let idx: usize = calc_ijkl_idx(i, j, k, l);
        // Debugging:
        // println!("{} {} {} {} {} {}", i, j, k, l, val, idx);

        //* Original code:
        let line_split: Vec<&str> = line.trim().split_whitespace().collect();
        let i: usize = line_split[0].parse::<usize>().unwrap() - 1;
        let j: usize = line_split[1].parse::<usize>().unwrap() - 1;
        let k: usize = line_split[2].parse::<usize>().unwrap() - 1;
        let l: usize = line_split[3].parse::<usize>().unwrap() - 1;
        let val: f64 = line_split[4].parse::<f64>().unwrap();
        let idx: usize = calc_ijkl_idx(i, j, k, l);
        // Debugging:
        // println!("{} {} {} {} {} {}", i, j, k, l, val, idx);

        while ERI_vec.len() <= idx {
            ERI_vec.push(0.0);
        }

        // ! THIS FUCKED ME UP FOR 3 DAYS !!!!!!!
        // ! Insert: inserts the value at a given position and then shifts
        // ! the rest of the vector to the right
        // ! I wanted to overwrite the value…
        // ! Moral of the story: Don't use insert() when you want to overwrite
        // ERI_vec.insert(idx, val);

        ERI_vec[idx] = val;
    }
    let ERI_array: Array1<f64> = Array1::from_vec(ERI_vec);
    println!("Electron-electron repulsion array:\n{:}\n", &ERI_array);

    //* Step 4: Build the orthogonalization matrix
    let S_matr_sqrt: Array2<f64> = S_matr.ssqrt(ndarray_linalg::UPLO::Upper).unwrap(); // S^1/2 matrix
    let S_matr_inv_sqrt = S_matr_sqrt.inv().unwrap(); // S^-1/2 matrix

    println!("S^-1/2:\n{:^1.5}\n", S_matr_inv_sqrt);

    //* Step 5: Build the inital guess density matrix
    // let S_matr_inv_sqrt_T: Array2<f64> = S_matr_inv_sqrt.reversed_axes();

    //? Intial guess the fock matrix
    let F_matr_init_prime: Array2<f64> = S_matr_inv_sqrt
        .clone()
        .reversed_axes()
        .dot(&H_core_matr)
        .dot(&S_matr_inv_sqrt.clone());

    println!("F_matr:\n{:1.5}\n", F_matr_init_prime);

    // ! WRONG FIRST TRY (my misunderstanding)
    // //* Read the initials MO coefficients (file has non-orthogonals AO basis C0 matrix)
    // let mut C_matr_AO_basis: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));
    // let C_matr_file_contents = fs::read_to_string("inp/Project3_1/STO-3G/c0.dat")
    //     .expect("Failed to open C0 matrix data!");
    // for (row_idx, line) in C_matr_file_contents.lines().enumerate() {
    //     let line_split: Vec<&str> = line.trim().split_whitespace().collect();
    //     for col_idx in 0..no_basis_funcs {
    //         let val: f64 = line_split[col_idx + 1].parse().unwrap();
    //         C_matr_AO_basis[(row_idx, col_idx)] = val;
    //     }
    // }
    // println!("Initial coeff matrix C0:\n{:^.5}\n", &C_matr_AO_basis);
    // let C_matr_AO_basis_inv: Array2<f64> = C_matr_AO_basis.clone().inv().unwrap();
    // let C_matr_MO_basis: Array2<f64> = S_matr_sqrt.dot(&C_matr_AO_basis);
    // let C_matr_MO_basis_inv: Array2<f64> = C_matr_MO_basis.clone().inv().unwrap();

    let (orb_energy_list, C_matr_MO_basis_from_F): (Array1<f64>, Array2<f64>) =
        F_matr_init_prime.eigh(ndarray_linalg::UPLO::Upper).unwrap();
    // let mut orb_energy_matr = C_matr_MO_basis_from_F
    println!("Orbital energy matrix:\n{:^.5}\n", &orb_energy_list);
    let C_matr_AO_basis_from_F: Array2<f64> = S_matr_inv_sqrt.dot(&C_matr_MO_basis_from_F);
    println!(
        "Initial coeff matrix C0:\n{:^.5}\n",
        &C_matr_AO_basis_from_F
    );

    // ! THIS IS ONLY VALID FOR RHF -> QUICK FIX
    // ! QUICK FIX FOR WATER
    //TODO: Add a calculation of the number of occupied orbitals
    let no_occ_orb: usize = (8 + 1 + 1) / 2; //* only correct for STO-3G -> 1 CGTO per orbital

    let mut D_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));

    for mu in 0..no_basis_funcs {
        for nu in 0..no_basis_funcs {
            for m in 0..no_occ_orb {
                D_matr[(mu, nu)] +=
                    C_matr_AO_basis_from_F[(mu, m)] * C_matr_AO_basis_from_F[(nu, m)];
            }
        }
    }
    println!("Initial density matrix:\n{:^.5}\n", &D_matr);

    //* Step 6: Compute the initial SCF energy
    // let F_matr: Array2<f64> =
    let mut E_scf_vec: Vec<f64> = Vec::new();
    let mut E_total_vec: Vec<f64> = Vec::new();
    let mut E_scf: f64 = 0.0;
    for mu in 0..no_basis_funcs {
        for nu in 0..no_basis_funcs {
            // E_scf += D_matr[(mu, nu)] * (H_matr[(mu, nu)] + F_matr_prime[(mu, nu)]);
            //* test: (yes this is what Crawford does -> because the Fock matrix has to be transfomred to the AO basis)
            E_scf += D_matr[(mu, nu)] * (H_core_matr[(mu, nu)] + H_core_matr[(mu, nu)]);
        }
    }

    let E_total: f64 = E_scf + E_nn_val;
    E_total_vec.push(E_total);
    E_scf_vec.push(E_scf);

    println!("E_nuc from file: {:^1.5}", &E_nn_val);
    println!("Initial SCF energy: {:^1.5}", &E_scf_vec[0]);
    println!("Initial total energy: {:^1.5}", &E_total_vec[0]);

    //* Step 7: Iterate the SCF procedure -> 7.1 compute the new Fock matrix
    let scf_maxiter: usize = 20;
    //? THE SCF ITERATIONS START HERE
    for scf_iter in 0..scf_maxiter {
        // println!("Previous F matrix: \n{:^1.5}\n", &F_matr_init_prime);
        let mut F_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));
        for mu in 0..no_basis_funcs {
            for nu in 0..no_basis_funcs {
                F_matr[(mu, nu)] = H_core_matr[(mu, nu)];
                for lambda in 0..no_basis_funcs {
                    for sigma in 0..no_basis_funcs {
                        let J_idx: usize = calc_ijkl_idx(mu, nu, lambda, sigma);
                        let K_idx: usize = calc_ijkl_idx(mu, lambda, nu, sigma);
                        F_matr[(mu, nu)] +=
                            D_matr[(lambda, sigma)] * (2.0 * ERI_array[J_idx] - ERI_array[K_idx]);
                    }
                }
            }
        }
        // println!("New F_matr:\n{:1.5}\n", &F_matr);

        //* Step 7.2: Build the new density matrix
        let F_matr_prime: Array2<f64> = S_matr_inv_sqrt
            .clone()
            .reversed_axes()
            .dot(&F_matr.clone())
            .dot(&S_matr_inv_sqrt.clone());
        // println!("New F_matr_prime:\n{:1.5}\n", &F_matr_prime);

        let (orb_energy_list, C_matr_MO_basis_from_F): (Array1<f64>, Array2<f64>) =
            F_matr_prime.eigh(ndarray_linalg::UPLO::Upper).unwrap();
        // Debugging:
        // println!(
        //     "Orbital energy matrix after 1st iter:\n{:^.5}\n",
        //     &orb_energy_list
        // );
        let C_matr_AO_basis_from_F: Array2<f64> = S_matr_inv_sqrt.dot(&C_matr_MO_basis_from_F);
        // Debugging:
        // println!(
        //     "Matrix C0 after 1st iter:\n{:^.5}\n",
        //     &C_matr_AO_basis_from_F
        // );

        let D_matr_prev: Array2<f64> = D_matr.clone();

        for mu in 0..no_basis_funcs {
            for nu in 0..no_basis_funcs {
                D_matr[(mu, nu)] = 0.0;
                for m in 0..no_occ_orb {
                    D_matr[(mu, nu)] +=
                        C_matr_AO_basis_from_F[(mu, m)] * C_matr_AO_basis_from_F[(nu, m)];
                }
            }
        }

        E_scf = 0.0;
        for mu in 0..no_basis_funcs {
            for nu in 0..no_basis_funcs {
                E_scf += D_matr[(mu, nu)] * (H_core_matr[(mu, nu)] + F_matr[(mu, nu)]);
            }
        }
        // println!("New SCF energy (iter): {:^1.5} ({})", &E_scf, &scf_iter);
        E_scf_vec.push(E_scf);
        E_total_vec.push(E_scf + E_nn_val);

        //* Step 10: Calc rms density matrix difference
        let mut rms_d_val: f64 = 0.0;
        for mu in 0..no_basis_funcs {
            for nu in 0..no_basis_funcs {
                rms_d_val += (D_matr[(mu, nu)] - D_matr_prev[(mu, nu)]).powi(2);
            }
        }
        rms_d_val = rms_d_val.sqrt();
        println!("Iter  E_scf           E_total      RMS D");
        println!(
            " {}  {:^5.8} {:^5.8} {:^1.8}",
            &scf_iter, &E_scf, &E_total, &rms_d_val
        );
    }
    // println!("SCF energy after {} iterations: {:?}", scf_maxiter, &E_scf_vec);


    if is_run_project4 {
        //* Project 4: Second oder Moller-Plesset Perturbation Theory

        println!("\n\nMP2 PROJECT STARTS HERE:\n");
        //* THIS PROJECT NEEDS PROJECT 3 TO RUN ASWELL!

        //* Step 1: Read in the ERI array -> DONE IN PROJECT 3
        //* Step 2: Obtain the MO cooefficients and energies -> DONE IN PROJECT 3
        let MP2_C_matr_AO_basis: Array2<f64> = C_matr_AO_basis_from_F.clone();
        let C_matr_occ: Array2<f64> = MP2_C_matr_AO_basis.slice(s![.., ..no_occ_orb]).to_owned();
        let C_matr_virt: Array2<f64> = MP2_C_matr_AO_basis.slice(s![.., no_occ_orb..]).to_owned();

        let orb_energy_list_MP2 = orb_energy_list.clone();
        let orb_energy_list_MP2_occ: Array1<f64> =
            orb_energy_list_MP2.slice(s![..no_occ_orb]).to_owned();
        let orb_energy_list_MP2_virt: Array1<f64> =
            orb_energy_list_MP2.slice(s![no_occ_orb..]).to_owned();

        let no_virt_orb: usize = no_basis_funcs - no_occ_orb;

        //* Step 3: Transform the ERI array to the MO basis
        let mut noddy_matr: Array4<f64> = Array4::zeros((
            no_basis_funcs,
            no_basis_funcs,
            no_basis_funcs,
            no_basis_funcs,
        ));

        // for p in 0..no_basis_funcs {
        //     for q in 0..p {
        //         for r in 0..p {
        //             let upper_lim_s: usize = if p == r { q } else { r};
        //             for s in 0..upper_lim_s {
        //                 for mu in 0..no_basis_funcs {
        //                     for nu in 0..no_basis_funcs {
        //                         for lambda in 0..no_basis_funcs {
        //                             for sigma in 0..no_basis_funcs {
        //                                 let idx: usize = calc_ijkl_idx(mu, nu, lambda, sigma);
        //                                 noddy_matr[(p,q,r,s)] += C_matr_occ[(mu, p)] * C_matr_virt[(nu, q)] * ERI_array[idx] * C_matr_occ[(lambda, r)] * C_matr_virt[(sigma, s)];
        //                                 // noddy_matr[(mu, nu, lambda, sigma)] += C_matr_occ[(mu, nu)]
        //                             }
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
        println!("\nNoddy matr: {:^1.5}\n", &noddy_matr);

        //* Step 3.2: Better version:
        let mut smarter_matr: Array4<f64> = Array4::zeros((
            no_basis_funcs,
            no_basis_funcs,
            no_basis_funcs,
            no_basis_funcs,
        ));

        for s in 0..no_virt_orb {
            for mu in 0..no_occ_orb {
                for nu in 0..no_occ_orb {
                    for lambda in 0..no_occ_orb {
                        for sigma in 0..no_occ_orb {
                            let idx: usize = calc_ijkl_idx(mu, nu, lambda, sigma);
                            smarter_matr[(mu, nu, lambda, sigma)] +=
                                C_matr_virt[(sigma, s)] * ERI_array[idx];
                        }
                    }
                }
            }
        }

        for r in 0..no_virt_orb {
            for mu in 0..no_occ_orb {
                for nu in 0..no_occ_orb {
                    for lambda in 0..no_occ_orb {
                        for sigma in 0..no_occ_orb {
                            smarter_matr[(mu, nu, lambda, sigma)] +=
                                C_matr_occ[(lambda, r)] * smarter_matr[(mu, nu, lambda, sigma)];
                        }
                    }
                }
            }
        }

        for q in 0..no_virt_orb {
            for mu in 0..no_occ_orb {
                for nu in 0..no_occ_orb {
                    for lambda in 0..no_occ_orb {
                        for sigma in 0..no_occ_orb {
                            smarter_matr[(mu, nu, lambda, sigma)] +=
                                C_matr_virt[(nu, q)] * smarter_matr[(mu, nu, lambda, sigma)];
                        }
                    }
                }
            }
        }

        for p in 0..no_virt_orb {
            for mu in 0..no_occ_orb {
                for nu in 0..no_occ_orb {
                    for lambda in 0..no_occ_orb {
                        for sigma in 0..no_occ_orb {
                            smarter_matr[(mu, nu, lambda, sigma)] +=
                                C_matr_occ[(mu, p)] * smarter_matr[(mu, nu, lambda, sigma)];
                        }
                    }
                }
            }
        }
        println!("Smarter matr:\n{:^1.5}\n", &smarter_matr);

        //* Step 4: Calculate the MP2 energy
        let mut MP2_energy: f64 = 0.0;
        // for mu in 0..no_occ_orb {
        //     for nu in 0..no_occ_orb {
        //         for lambda in 0..no_occ_orb {
        //             for sigma in 0..no_occ_orb {
        //                 let idx: usize = calc_ijkl_idx(mu, nu, lambda, sigma);
        //                 MP2_energy += (2.0 * ERI_array[idx] - ERI_array[idx])
        //                     * (1.0 / (orb_energy_list_MP2_occ[mu] + orb_energy_list_MP2_occ[nu] - orb_energy_list_MP2_virt[lambda] - orb_energy_list_MP2_virt[sigma]));
        //             }
        //         }
        //     }
        // }
        // MP2_energy *= 0.25;

        // println!("\nMP2 energy: {}", MP2_energy);
    }
}

pub fn calc_ijkl_idx(i: usize, j: usize, k: usize, l: usize) -> usize {
    let ij: usize = if i > j {
        calc_cmp_idx(i, j)
    } else {
        calc_cmp_idx(j, i)
    };
    let kl: usize = if k > l {
        calc_cmp_idx(k, l)
    } else {
        calc_cmp_idx(l, k)
    };
    let ijkl: usize = if ij > kl {
        calc_cmp_idx(ij, kl)
    } else {
        calc_cmp_idx(kl, ij)
    };
    ijkl
}

pub fn calc_cmp_idx(idx1: usize, idx2: usize) -> usize {
    (idx1 * (idx1 + 1)) / 2 + idx2
}