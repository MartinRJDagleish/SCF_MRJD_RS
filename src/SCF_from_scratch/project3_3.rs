use ndarray::prelude::*;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt};

use crate::molecule::{
    self,
    wfn::{ints::*, CGTO, PGTO},
};

pub fn run_project3_3() {
    //***************************************************************************************/
    // ! Supplying mol_6_311g object for TESTING
    // TODO: FIX THIS LATER
    //* Define the primitive gaussians
    //* 6-311G here
    println!("Defining the primitive gaussians");
    println!("Test molecule: H2 (6-311G)");
    let mut mol_6_311g = molecule::Molecule::new("inp/Project3_2/geom/h2.xyz", 0);
    //* The first H atom -> H1
    // H1_contr_gauss_1s
    mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos = vec![CGTO::new(vec![PGTO::new(
        33.86500,
        0.0254938,
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0, 0, 0]),
    )])];
    mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[0]
        .pgto_vec
        .push(PGTO::new(
            5.094790,
            0.190373,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[0]
        .pgto_vec
        .push(PGTO::new(
            1.158790,
            0.852161,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    // H1_contr_gauss_2s
    mol_6_311g
        .wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.325840,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    // H1_contr_gauss_3s
    mol_6_311g
        .wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.102741,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    //* The second H atom -> H2
    // H2_contr_gauss_1s
    mol_6_311g
        .wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            33.86500,
            0.0254938,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));
    mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[3]
        .pgto_vec
        .push(PGTO::new(
            5.094790,
            0.190373,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[3]
        .pgto_vec
        .push(PGTO::new(
            1.158790,
            0.852161,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    // H2_contr_gauss_2s
    mol_6_311g
        .wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.325840,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    // H2_contr_gauss_3s
    mol_6_311g
        .wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.102741,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    // mol_6_311g.wfn_total.update_no_of_contr_gauss();

    for cgto in &mut mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos {
        cgto.update_no_pgtos();
    }

    mol_6_311g.wfn_total.basis_set_total.update_no_cgtos();

    //***************************************************************************************/
    // * Trying to use my Basisset parser to build molecule from gbs file info
    // let mol_6_311g_from_gbs = molecule::Molecule::new("inp/Project3_2/geom/h2.xyz", 0);


    println!("\nSCF from scratch:\n");
    //* Project 3: SCF from scratch
    //* Step 1: Read Nuclear Repulsion Energy (enuc) from file
    mol_6_311g.wfn_total.HFMatrices.V_nn_val = calc_E_nn_val(&mol_6_311g.geom_obj.geom_matr);

    //* Step 2.1: Calculate the overlap matrix S
    mol_6_311g.wfn_total.HFMatrices.S_matr = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
    ));
    for i in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
        for j in 0..=i {
            if i == j {
                mol_6_311g.wfn_total.HFMatrices.S_matr[(i, j)] = 1.0;
                continue;
            } else {
                mol_6_311g.wfn_total.HFMatrices.S_matr[(i, j)] = calc_overlap_int_cgto(
                    &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[i],
                    &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[j],
                );
                mol_6_311g.wfn_total.HFMatrices.S_matr[(j, i)] =
                    mol_6_311g.wfn_total.HFMatrices.S_matr[(i, j)];
            }
        }
    }

    println!(
        "Overlap matrix S:\n{:1.5}\n",
        &mol_6_311g.wfn_total.HFMatrices.S_matr
    );

    //* Step 2.2: Calculate the kinetic energy matrix T
    mol_6_311g.wfn_total.HFMatrices.T_matr = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
    ));

    for i in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
        for j in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            mol_6_311g.wfn_total.HFMatrices.T_matr[(i, j)] = calc_kin_energy_int_cgto(
                &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[i],
                &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[j],
            );
            // mol_6_311g.wfn_total.HFMatrices.T_matr[(j, i)] =
            //     mol_6_311g.wfn_total.HFMatrices.T_matr[(i, j)].clone();
        }
    }
    println!(
        "Kinetic energy matrix T:\n{:1.5}\n",
        &mol_6_311g.wfn_total.HFMatrices.T_matr
    );

    //* Step 2.3: Calculate the nuclear attraction matrix V_ne
    mol_6_311g.wfn_total.HFMatrices.V_ne_matr = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
    ));

    for i in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
        for j in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            for (idx, atom_pos) in mol_6_311g
                .geom_obj
                .geom_matr
                .axis_iter(ndarray::Axis(0))
                .enumerate()
            {
                mol_6_311g.wfn_total.HFMatrices.V_ne_matr[(i, j)] -=
                    (mol_6_311g.geom_obj.Z_vals[idx] as f64)
                        * calc_nuc_attr_int_cgto(
                            &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[i],
                            &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[j],
                            &atom_pos.to_owned(),
                        );
            }
        }
    }

    println!(
        "Nuclear attraction:\n{:1.5}\n",
        &mol_6_311g.wfn_total.HFMatrices.V_ne_matr
    );
    //* Step 2.4: Form the core Hamiltonian matrix H_core
    mol_6_311g.wfn_total.HFMatrices.H_core_matr = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
    ));
    mol_6_311g.wfn_total.HFMatrices.H_core_matr =
        &mol_6_311g.wfn_total.HFMatrices.T_matr + &mol_6_311g.wfn_total.HFMatrices.V_ne_matr;

    println!(
        "Core Hamiltonian matrix H_core:\n{:1.5}\n",
        &mol_6_311g.wfn_total.HFMatrices.H_core_matr
    );

    //* Step 3: Calculate the 2-electron integrals (ERI)

    println!("Electron-electron repulsion integrals (V_ee / ERI matrix):");
    mol_6_311g.wfn_total.HFMatrices.ERI_tensor = Array4::<f64>::zeros((
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
    ));
    for i in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
        for j in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            for k in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
                for l in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
                    mol_6_311g.wfn_total.HFMatrices.ERI_tensor[(i, j, k, l)] =
                        calc_elec_elec_repul_cgto(
                            &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[i],
                            &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[j],
                            &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[k],
                            &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[l],
                        );
                }
            }
        }
    }

    println!("{:^5.6}\n", &mol_6_311g.wfn_total.HFMatrices.ERI_tensor);

    //* Step 4: Build the orthogonalization matrix S^(-1/2)
    let S_matr_sqrt: Array2<f64> = mol_6_311g
        .wfn_total
        .HFMatrices
        .S_matr
        .ssqrt(ndarray_linalg::UPLO::Upper)
        .unwrap();
    let S_matr_sqrt_inv: Array2<f64> = S_matr_sqrt.inv().unwrap();

    println!("S^(-1/2):\n{:^5.6}\n", &S_matr_sqrt_inv);

    //* Step 5: Form the initial guess density matrix D_0
    //* Step 5.1: Form the initial guess Fock matrix F_0 in the AO basis
    let F_matr_0_pr: Array2<f64> = S_matr_sqrt_inv
        .clone()
        .reversed_axes()
        .dot(&mol_6_311g.wfn_total.HFMatrices.H_core_matr)
        .dot(&S_matr_sqrt_inv.clone());

    println!("F_matr_0_pr:\n{:^5.6}\n", &F_matr_0_pr);

    //* Step 5.2: Get the coefficients of the initial guess Fock matrix F_0 in the MO basis
    let (orb_energy_arr, C_matr_MO_basis) = F_matr_0_pr.eigh(ndarray_linalg::UPLO::Upper).unwrap();
    let C_matr_AO_basis: Array2<f64> = S_matr_sqrt_inv.dot(&C_matr_MO_basis);
    println!("C_matr_AO_basis:\n{:^5.6}\n", &C_matr_AO_basis);

    //* Step 5.3: Form the initial guess density matrix D_0
    // ! THIS WORKS ONLY FOR CLOSED SHELL SYSTEMS (RHF)
    // TODO: CHANGE THIS TO WORK FOR OPEN SHELL SYSTEMS (UHF)
    /*
    ? How do I get the correct number of occupied orbitals, when I have multiple CGTOs per orbital per atom?
    */
    let no_occ_orb: usize = 3; // * QUICK FIX: 3 CGTO describe 1sσ orbital

    let mut D_matr: Array2<f64> = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
    ));

    for mu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
        for nu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            for m in 0..no_occ_orb {
                D_matr[(mu, nu)] += C_matr_AO_basis[(mu, m)] * C_matr_AO_basis[(nu, m)];
            }
        }
    }
    println!("Initial density matrix:\n{:^.5}\n", &D_matr);

    let mut E_scf: f64 = 0.0;
    let mut E_scf_vec: Vec<f64> = Vec::new();
    let mut E_tot_vec: Vec<f64> = Vec::new();

    //* Here the Fock matrix is guessed to be the core Hamiltonian matrix
    //* That's why the initial SCF energy differs from the other SCF energy calcs
    for mu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
        for nu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            E_scf +=
                D_matr[(mu, nu)] * 2.0 * (mol_6_311g.wfn_total.HFMatrices.H_core_matr[(mu, nu)]);
        }
    }

    E_scf_vec.push(E_scf);
    let mut E_tot = E_scf + mol_6_311g.wfn_total.HFMatrices.V_nn_val;
    E_tot_vec.push(E_tot);

    //* Step 7: Iterate the SCF procedure until convergence
    let scf_maxiter: usize = 50;
    // ! THE SCF ITERATIONS START HERE
    for scf_iter in 0..scf_maxiter {
        //* Step 6: Form the Fock matrix F in the AO basis
        let mut F_matr: Array2<f64> = Array2::<f64>::zeros((
            mol_6_311g.wfn_total.basis_set_total.no_cgtos,
            mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        ));

        for mu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            for nu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
                for lambda in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
                    for sigma in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
                        F_matr[(mu, nu)] += D_matr[(lambda, sigma)]
                            * (2.0
                                * mol_6_311g.wfn_total.HFMatrices.ERI_tensor
                                    [(mu, nu, lambda, sigma)]
                                - mol_6_311g.wfn_total.HFMatrices.ERI_tensor
                                    [(mu, lambda, nu, sigma)]);
                    }
                }
            }
        }

        //* Step 7: Form the Fock matrix F in the MO basis
        let F_matr_pr: Array2<f64> = S_matr_sqrt_inv
            .clone()
            .reversed_axes()
            .dot(&F_matr)
            .dot(&S_matr_sqrt_inv.clone());

        //* Step 8: Get the coefficients of the Fock matrix F in the MO basis
        let (orb_energy_arr, C_matr_MO_basis) =
            F_matr_pr.eigh(ndarray_linalg::UPLO::Upper).unwrap();
        let C_matr_AO_basis: Array2<f64> = S_matr_sqrt_inv.dot(&C_matr_MO_basis);
        let D_matr_prev: Array2<f64> = D_matr.clone();

        for mu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            for nu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
                D_matr[(mu, nu)] = 0.0;
                for m in 0..no_occ_orb {
                    D_matr[(mu, nu)] += C_matr_AO_basis[(mu, m)] * C_matr_AO_basis[(nu, m)];
                }
            }
        }

        E_scf = 0.0;
        for mu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            for nu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
                E_scf += D_matr[(mu, nu)]
                    * (mol_6_311g.wfn_total.HFMatrices.H_core_matr[(mu, nu)] + F_matr[(mu, nu)]);
            }
        }

        E_scf_vec.push(E_scf);
        E_tot = E_scf + mol_6_311g.wfn_total.HFMatrices.V_nn_val;
        E_tot_vec.push(E_tot);

        let mut rms_d_val: f64 = 0.0;
        for mu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            for nu in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
                rms_d_val += (D_matr[(mu, nu)] - D_matr_prev[(mu, nu)]).powi(2);
            }
        }
        rms_d_val = rms_d_val.sqrt();

        println!("Iter  E_scf           E_total      RMS D");
        println!(
            " {}  {:^5.8} {:^5.8} {:^1.8}",
            &scf_iter, &E_scf, &E_tot, &rms_d_val
        );
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
