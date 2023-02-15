use ndarray::prelude::*;
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt};

use crate::molecule::{
    wfn::{ints::*, CGTO, PGTO, BasisSetTotal},
    Molecule,
};

use crate::molecule::wfn::basisset::create_basis_set_total;
use crate::molecule::wfn::basisset::parse_basis_set_file_gaussian;

pub fn run_project3_3_h2o() {
    let mut mol: Molecule = Molecule::new("inp/Project3_1/STO-3G/h2o_v2.xyz", Some(0));

    //* Create basis for mol object */
    let basis_set_name = "STO-3G";
    let basis_set_total: BasisSetTotal = create_basis_set_total(
        parse_basis_set_file_gaussian(basis_set_name),
        mol.geom_obj.geom_matr.clone(),
        mol.geom_obj.Z_vals.clone()
    );

    //TODO: QUICK FIX FOR NOW, NEED TO FIX THIS -> Manual creation

    // O_contr_gauss_1s
    let O_pos: Array1<f64> =
        Array1::from_vec(vec![0.000000000000, -0.143225816552, 0.000000000000]);
    mol.wfn_total.basis_set_total.basis_set_cgtos = vec![CGTO::new(vec![PGTO::new(
        0.1307093214E+03,
        0.1543289673E+00,
        O_pos.clone(),
        Array1::from_vec(vec![0, 0, 0]),
    )])];
    mol.wfn_total.basis_set_total.basis_set_cgtos[0]
        .pgto_vec
        .push(PGTO::new(
            0.2380886605E+02,
            0.5353281423E+00,
            O_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol.wfn_total.basis_set_total.basis_set_cgtos[0]
        .pgto_vec
        .push(PGTO::new(
            0.6443608313E+01,
            0.4446345422E+00,
            O_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    // O_contr_gauss_2s
    mol.wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.5033151319E+01,
            -0.9996722919E-01,
            O_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        )]));
    mol.wfn_total.basis_set_total.basis_set_cgtos[1]
        .pgto_vec
        .push(PGTO::new(
            0.1169596125E+01,
            0.3995128261E+00,
            O_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol.wfn_total.basis_set_total.basis_set_cgtos[1]
        .pgto_vec
        .push(PGTO::new(
            0.3803889600E+00,
            0.7001154689E+00,
            O_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    // O_contr_gauss_2p -> (1,0,0)
    mol.wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.5033151319E+01,
            0.1559162750E+00,
            O_pos.clone(),
            Array1::from_vec(vec![1, 0, 0]),
        )]));
    mol.wfn_total.basis_set_total.basis_set_cgtos[2]
        .pgto_vec
        .push(PGTO::new(
            0.1169596125E+01,
            0.6076837186E+00,
            O_pos.clone(),
            Array1::from_vec(vec![1, 0, 0]),
        ));
    mol.wfn_total.basis_set_total.basis_set_cgtos[2]
        .pgto_vec
        .push(PGTO::new(
            0.3803889600E+00,
            0.3919573931E+00,
            O_pos.clone(),
            Array1::from_vec(vec![1, 0, 0]),
        ));

    // O_contr_gauss_2p -> (0,1,0)
    mol.wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.5033151319E+01,
            0.1559162750E+00,
            O_pos.clone(),
            Array1::from_vec(vec![0, 1, 0]),
        )]));
    mol.wfn_total.basis_set_total.basis_set_cgtos[3]
        .pgto_vec
        .push(PGTO::new(
            0.1169596125E+01,
            0.6076837186E+00,
            O_pos.clone(),
            Array1::from_vec(vec![0, 1, 0]),
        ));
    mol.wfn_total.basis_set_total.basis_set_cgtos[3]
        .pgto_vec
        .push(PGTO::new(
            0.3803889600E+00,
            0.3919573931E+00,
            O_pos.clone(),
            Array1::from_vec(vec![0, 1, 0]),
        ));

    // O_contr_gauss_2p -> (0,0,1)
    mol.wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.5033151319E+01,
            0.1559162750E+00,
            O_pos.clone(),
            Array1::from_vec(vec![0, 0, 1]),
        )]));
    mol.wfn_total.basis_set_total.basis_set_cgtos[4]
        .pgto_vec
        .push(PGTO::new(
            0.1169596125E+01,
            0.6076837186E+00,
            O_pos.clone(),
            Array1::from_vec(vec![0, 0, 1]),
        ));
    mol.wfn_total.basis_set_total.basis_set_cgtos[4]
        .pgto_vec
        .push(PGTO::new(
            0.3803889600E+00,
            0.3919573931E+00,
            O_pos.clone(),
            Array1::from_vec(vec![0, 0, 1]),
        ));

    // H1_contr_gauss_1s
    let H1_pos: Array1<f64> =
        Array1::from_vec(vec![1.638036840407, 1.136548822547, -0.000000000000]);
    mol.wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.3425250914E+01,
            0.1543289673E+00,
            H1_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        )]));
    mol.wfn_total.basis_set_total.basis_set_cgtos[5]
        .pgto_vec
        .push(PGTO::new(
            0.6239137298E+00,
            0.5353281423E+00,
            H1_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol.wfn_total.basis_set_total.basis_set_cgtos[5]
        .pgto_vec
        .push(PGTO::new(
            0.1688554040E+00,
            0.4446345422E+00,
            H1_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    // H2_contr_gauss_1s
    let H2_pos: Array1<f64> =
        Array1::from_vec(vec![-1.638036840407, 1.136548822547, -0.000000000001]);
    mol.wfn_total
        .basis_set_total
        .basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.3425250914E+01,
            0.1543289673E+00,
            H2_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        )]));
    mol.wfn_total.basis_set_total.basis_set_cgtos[6]
        .pgto_vec
        .push(PGTO::new(
            0.6239137298E+00,
            0.5353281423E+00,
            H2_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol.wfn_total.basis_set_total.basis_set_cgtos[6]
        .pgto_vec
        .push(PGTO::new(
            0.1688554040E+00,
            0.4446345422E+00,
            H2_pos.clone(),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    for cgto in &mut mol.wfn_total.basis_set_total.basis_set_cgtos {
        cgto.update_no_pgtos();
    }

    mol.wfn_total.basis_set_total.update_no_cgtos();

    println!("\nSCF from scratch:\n");
    //* Project 3: SCF from scratch
    //* Step 1: Read Nuclear Repulsion Energy (enuc) from file
    mol.wfn_total.HFMatrices.V_nn_val =
        calc_V_nn_val(&mol.geom_obj.geom_matr, &mol.geom_obj.Z_vals);

    //* Step 2.1: Calculate the overlap matrix S
    mol.wfn_total.HFMatrices.S_matr = Array2::<f64>::zeros((
        mol.wfn_total.basis_set_total.no_cgtos,
        mol.wfn_total.basis_set_total.no_cgtos,
    ));
    for i in 0..mol.wfn_total.basis_set_total.no_cgtos {
        for j in 0..=i {
            if i == j {
                mol.wfn_total.HFMatrices.S_matr[(i, j)] = 1.0;
                continue;
            } else {
                mol.wfn_total.HFMatrices.S_matr[(i, j)] = calc_overlap_int_cgto(
                    &mol.wfn_total.basis_set_total.basis_set_cgtos[i],
                    &mol.wfn_total.basis_set_total.basis_set_cgtos[j],
                );
                mol.wfn_total.HFMatrices.S_matr[(j, i)] = mol.wfn_total.HFMatrices.S_matr[(i, j)];
            }
        }
    }

    println!(
        "Overlap matrix S:\n{:1.5}\n",
        &mol.wfn_total.HFMatrices.S_matr
    );

    //* Step 2.2: Calculate the kinetic energy matrix T
    mol.wfn_total.HFMatrices.T_matr = Array2::<f64>::zeros((
        mol.wfn_total.basis_set_total.no_cgtos,
        mol.wfn_total.basis_set_total.no_cgtos,
    ));

    for i in 0..mol.wfn_total.basis_set_total.no_cgtos {
        for j in 0..=i {
            mol.wfn_total.HFMatrices.T_matr[(i, j)] = calc_kin_energy_int_cgto(
                &mol.wfn_total.basis_set_total.basis_set_cgtos[i],
                &mol.wfn_total.basis_set_total.basis_set_cgtos[j],
            );
            mol.wfn_total.HFMatrices.T_matr[(j, i)] =
                mol.wfn_total.HFMatrices.T_matr[(i, j)];
        }
    }
    println!(
        "Kinetic energy matrix T:\n{:1.5}\n",
        &mol.wfn_total.HFMatrices.T_matr
    );

    //* Step 2.3: Calculate the nuclear attraction matrix V_ne
    mol.wfn_total.HFMatrices.V_ne_matr = Array2::<f64>::zeros((
        mol.wfn_total.basis_set_total.no_cgtos,
        mol.wfn_total.basis_set_total.no_cgtos,
    ));

    for i in 0..mol.wfn_total.basis_set_total.no_cgtos {
        for j in 0..=i {
            for (idx, atom_pos) in mol
                .geom_obj
                .geom_matr
                .axis_iter(ndarray::Axis(0))
                .enumerate()
            {
                mol.wfn_total.HFMatrices.V_ne_matr[(i, j)] -= (mol.geom_obj.Z_vals[idx] as f64)
                    * calc_nuc_attr_int_cgto(
                        &mol.wfn_total.basis_set_total.basis_set_cgtos[i],
                        &mol.wfn_total.basis_set_total.basis_set_cgtos[j],
                        &atom_pos.to_owned(),
                    );
            }
            mol.wfn_total.HFMatrices.V_ne_matr[(j, i)] =
                mol.wfn_total.HFMatrices.V_ne_matr[(i, j)];
        }
    }

    println!(
        "Nuclear attraction:\n{:1.5}\n",
        &mol.wfn_total.HFMatrices.V_ne_matr
    );
    //* Step 2.4: Form the core Hamiltonian matrix H_core
    mol.wfn_total.HFMatrices.H_core_matr = Array2::<f64>::zeros((
        mol.wfn_total.basis_set_total.no_cgtos,
        mol.wfn_total.basis_set_total.no_cgtos,
    ));
    mol.wfn_total.HFMatrices.H_core_matr =
        &mol.wfn_total.HFMatrices.T_matr + &mol.wfn_total.HFMatrices.V_ne_matr;

    println!(
        "Core Hamiltonian matrix H_core:\n{:1.5}\n",
        &mol.wfn_total.HFMatrices.H_core_matr
    );

    //* Step 3: Calculate the 2-electron integrals (ERI)

    println!("Electron-electron repulsion integrals (V_ee / ERI matrix):");
    mol.wfn_total.HFMatrices.ERI_tensor = Array4::<f64>::zeros((
        mol.wfn_total.basis_set_total.no_cgtos,
        mol.wfn_total.basis_set_total.no_cgtos,
        mol.wfn_total.basis_set_total.no_cgtos,
        mol.wfn_total.basis_set_total.no_cgtos,
    ));
    for i in 0..mol.wfn_total.basis_set_total.no_cgtos {
        for j in 0..mol.wfn_total.basis_set_total.no_cgtos {
            for k in 0..mol.wfn_total.basis_set_total.no_cgtos {
                for l in 0..mol.wfn_total.basis_set_total.no_cgtos {
                    mol.wfn_total.HFMatrices.ERI_tensor[(i, j, k, l)] = calc_elec_elec_repul_cgto(
                        &mol.wfn_total.basis_set_total.basis_set_cgtos[i],
                        &mol.wfn_total.basis_set_total.basis_set_cgtos[j],
                        &mol.wfn_total.basis_set_total.basis_set_cgtos[k],
                        &mol.wfn_total.basis_set_total.basis_set_cgtos[l],
                    );
                }
            }
        }
    }

    // println!("{:^5.6}\n", &mol.wfn_total.HFMatrices.ERI_tensor);

    //* Step 4: Build the orthogonalization matrix S^(-1/2)
    let S_matr_sqrt: Array2<f64> = mol
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
        .dot(&mol.wfn_total.HFMatrices.H_core_matr)
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
    let no_occ_orb: usize = 5; // * 1 CGTO per orbital per atom

    let mut D_matr: Array2<f64> = Array2::<f64>::zeros((
        mol.wfn_total.basis_set_total.no_cgtos,
        mol.wfn_total.basis_set_total.no_cgtos,
    ));

    for mu in 0..mol.wfn_total.basis_set_total.no_cgtos {
        for nu in 0..mol.wfn_total.basis_set_total.no_cgtos {
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
    for mu in 0..mol.wfn_total.basis_set_total.no_cgtos {
        for nu in 0..mol.wfn_total.basis_set_total.no_cgtos {
            E_scf += D_matr[(mu, nu)] * 2.0 * (mol.wfn_total.HFMatrices.H_core_matr[(mu, nu)]);
        }
    }

    E_scf_vec.push(E_scf);
    let mut E_tot = E_scf + mol.wfn_total.HFMatrices.V_nn_val;
    E_tot_vec.push(E_tot);

    //* Step 7: Iterate the SCF procedure until convergence
    let scf_maxiter: usize = 50;
    // ! THE SCF ITERATIONS START HERE
    for scf_iter in 0..scf_maxiter {
        //* Step 6: Form the Fock matrix F in the AO basis
        let mut F_matr: Array2<f64> = Array2::<f64>::zeros((
            mol.wfn_total.basis_set_total.no_cgtos,
            mol.wfn_total.basis_set_total.no_cgtos,
        ));

        for mu in 0..mol.wfn_total.basis_set_total.no_cgtos {
            for nu in 0..mol.wfn_total.basis_set_total.no_cgtos {
                for lambda in 0..mol.wfn_total.basis_set_total.no_cgtos {
                    for sigma in 0..mol.wfn_total.basis_set_total.no_cgtos {
                        F_matr[(mu, nu)] += D_matr[(lambda, sigma)]
                            * (2.0 * mol.wfn_total.HFMatrices.ERI_tensor[(mu, nu, lambda, sigma)]
                                - mol.wfn_total.HFMatrices.ERI_tensor[(mu, lambda, nu, sigma)]);
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

        for mu in 0..mol.wfn_total.basis_set_total.no_cgtos {
            for nu in 0..mol.wfn_total.basis_set_total.no_cgtos {
                D_matr[(mu, nu)] = 0.0;
                for m in 0..no_occ_orb {
                    D_matr[(mu, nu)] += C_matr_AO_basis[(mu, m)] * C_matr_AO_basis[(nu, m)];
                }
            }
        }

        E_scf = 0.0;
        for mu in 0..mol.wfn_total.basis_set_total.no_cgtos {
            for nu in 0..mol.wfn_total.basis_set_total.no_cgtos {
                E_scf += D_matr[(mu, nu)]
                    * (mol.wfn_total.HFMatrices.H_core_matr[(mu, nu)] + F_matr[(mu, nu)]);
            }
        }

        E_scf_vec.push(E_scf);
        E_tot = E_scf + mol.wfn_total.HFMatrices.V_nn_val;
        E_tot_vec.push(E_tot);

        let mut rms_d_val: f64 = 0.0;
        for mu in 0..mol.wfn_total.basis_set_total.no_cgtos {
            for nu in 0..mol.wfn_total.basis_set_total.no_cgtos {
                rms_d_val += (D_matr[(mu, nu)] - D_matr_prev[(mu, nu)]).powi(2);
            }
        }
        rms_d_val = rms_d_val.sqrt();

        println!("Iter  E_scf      E_total   RMS D");
        println!(
            " {}  {:^5.8} {:^5.8} {:^1.8}",
            &scf_iter, &E_scf, &E_tot, &rms_d_val
        );
    }

    println!("\n\nFinal E_total = {:^5.8}", &E_tot);
}
