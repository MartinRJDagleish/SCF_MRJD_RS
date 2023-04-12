use std::collections::VecDeque;

use ndarray::{concatenate, prelude::*, Zip};
use ndarray_linalg::{Eigh, Inverse, SolveH, SymmetricSqrt, UPLO};

use ndarray::parallel::prelude::*;

use crate::molecule::{
    wfn::{
        basisset::{create_basis_set_total, parse_basis_set_file_gaussian},
        integrals::{
            calc_V_nn_val, calc_elec_elec_repul_cgto, calc_kin_energy_int_cgto,
            calc_nuc_attr_int_cgto, calc_overlap_int_cgto,
        },
    },
    Molecule, geometry::calc_r_ij_general,
};

#[derive(Debug)]
pub struct SCF {
    pub mol: Molecule,
    pub E_scf_final: f64,
    pub E_tot_final: f64,
    pub C_matr_final: Array2<f64>,
    pub D_matr_final: Array2<f64>,
    pub orb_energies_final: Array1<f64>,
    pub F_matr_set: VecDeque<Array2<f64>>, // TODO: Impl ADIIS for Fock matrix
    pub error_matr_set: VecDeque<Array2<f64>>, //TODO: Impl ADIIS for error matrix
}

impl SCF {
    pub fn new(mol: Molecule) -> Self {
        let C_matr_final = Array::default((0, 0));
        let D_matr_final = Array::default((0, 0));
        let orb_energies_final = Array::default(0);
        let F_matr_set = VecDeque::new();
        let error_matr_set = VecDeque::new();
        SCF {
            mol,
            E_scf_final: 0.0,
            E_tot_final: 0.0,
            C_matr_final,
            D_matr_final,
            orb_energies_final,
            F_matr_set,
            error_matr_set,
        }
    }

    pub fn geom_analysis_pre_calc(&self) {

        let no_atoms = self.mol.no_atoms;
    //* Step 2: Bond lengths
    println!("\nInteratomic distances (in bohr):");
        for i in 0..no_atoms{
            for j in 0..i {
                    let bond_length: f64 = self.mol.geom_obj.calc_r_ij(i, j);

                    println!("Distance between {}-{} is: {:3.5}", i, j, bond_length);
            }
        }

    // //* Step 3: Bond angles
    // println!("\nBond angles (in degrees):");
    // for i in 0..mol.no_atoms {
    //     for j in 0..i {
    //         for k in 0..j {
    //             if mol.geom_obj.calc_r_ij(i, j) < 4.0 && mol.geom_obj.calc_r_ij(j, k) < 4.0 {
    //                 let bond_angle: f64 = mol.geom_obj.calc_bond_angle(i, j, k);
    //                 println!("Angle for {}-{}-{} is: {:.5}", i, j, k, bond_angle);
    //             }
    //         }
    //     }
    // }

    // //* Step 4: OOP angles
    // println!("\nOut-of-plane angles (in degrees):\n");
    // for i in 0..mol.no_atoms {
    //     for j in 0..mol.no_atoms {
    //         for k in 0..mol.no_atoms {
    //             for l in 0..mol.no_atoms {
    //                 let bond_dist_jk: f64 = mol.geom_obj.calc_r_ij(j, k);
    //                 let bond_dist_kl: f64 = mol.geom_obj.calc_r_ij(k, l);
    //                 let bond_dist_ik: f64 = mol.geom_obj.calc_r_ij(i, k);
    //                 if i != j
    //                     && i != k
    //                     && i != l
    //                     && j != k
    //                     && k != l
    //                     && j != l
    //                     && bond_dist_jk < 4.0
    //                     && bond_dist_kl < 4.0
    //                     && bond_dist_ik < 4.0
    //                 {
    //                     let oop_angle: f64 = mol.geom_obj.calc_oop_angle(i, j, k, l);

    //                     println!("OOP angle for {}-{}-{}-{} is: {:.5}", i, j, k, l, oop_angle);
    //                 }
    //             }
    //         }
    //     }
    // }

    // // * Step 5: Torsion / dihedral angles
    // println!("\nTorsion angles (in degrees):\n");
    // for i in 0..mol.no_atoms {
    //     for j in 0..i {
    //         for k in 0..j {
    //             for l in 0..k {
    //                 let bond_dist_ij: f64 = mol.geom_obj.calc_r_ij(i, j);
    //                 let bond_dist_jk: f64 = mol.geom_obj.calc_r_ij(j, k);
    //                 let bond_dist_kl: f64 = mol.geom_obj.calc_r_ij(k, l);
    //                 if bond_dist_ij < 4.0 && bond_dist_jk < 4.0 && bond_dist_kl < 4.0 {
    //                     let dihedral_angle: f64 = mol.geom_obj.calc_dihedral_angle(i, j, k, l);
    //                     println!(
    //                         "Dihedral angle for {}-{}-{}-{} is: {:.5}",
    //                         i, j, k, l, dihedral_angle
    //                     );
    //                 }
    //             }
    //         }
    //     }
    // }

    // //* Step 6: Center of mass
    // println!("\nCenter of mass: {:^.6}", &mol.geom_obj.calc_center_mass());

    // //* Step 6.5: Translate molecule such that center of mass is in middle of coordinate system
    // println!("\nTranslate molecule such that center of mass is in middle of coordinate system");
    // println!("\nBefore translation:");
    // mol.geom_obj.print_geom_input();

    // println!("\nAfter translation:");
    // mol.geom_obj.translate_mol_to_center_mass();
    // mol.geom_obj.print_geom_input();

    // //* Step 7: Inertia tensor
    // println!("\nPrinting the moment of inertia tensor:");
    // let inertia_tensor: Array2<f64> = mol.geom_obj.calc_inertia_tensor();
    // println!("Inertia tensor: \n{:^.5}\n", inertia_tensor);

    // //* Step 7.1 : Get eigenvalues and eigenvectors of inertia tensor
    // let eigenvals: Array1<f64> = inertia_tensor
    //     .eigvalsh(ndarray_linalg::UPLO::Upper)
    //     .unwrap();

    // println!(
    //     "Principal moments of inertia (amu * bohr^2): \n{:^.5}\n",
    //     &eigenvals
    // );
    // println!(
    //     "Principal moments of inertia (amu * Angstrom^2): \n{:^.5}\n",
    //     &eigenvals * (1.0e10 * physical_constants::BOHR_RADIUS).powi(2) //* prefactor but not exponent for conversion
    // );
    // println!(
    //     "Principal moments of inertia (g * cm^2): \n{:^.5e}\n",
    //     &eigenvals
    //         * physical_constants::ATOMIC_MASS_CONSTANT
    //         * (100. * physical_constants::BOHR_RADIUS).powi(2)
    // );

    // //* Step 8: Rotational constants
    // let conv_factor_recip_cm: f64 = (100.0
    //     * physical_constants::ATOMIC_MASS_CONSTANT
    //     * physical_constants::BOHR_RADIUS.powi(2))
    // .recip();
    // // println!("\nConversion factor: {}\n", conv_factor_recip_cm);
    // let rot_const_A_per_cm: f64 = conv_factor_recip_cm
    //     * physical_constants::PLANCK_CONSTANT
    //     * (8.0 * PI.powi(2) * physical_constants::SPEED_OF_LIGHT_IN_VACUUM * &eigenvals[0]).recip();
    // let rot_const_B_per_cm: f64 = conv_factor_recip_cm
    //     * physical_constants::PLANCK_CONSTANT
    //     * (8.0 * PI.powi(2) * physical_constants::SPEED_OF_LIGHT_IN_VACUUM * &eigenvals[1]).recip();
    // let rot_const_C_per_cm: f64 = conv_factor_recip_cm
    //     * physical_constants::PLANCK_CONSTANT
    //     * (8.0 * PI.powi(2) * physical_constants::SPEED_OF_LIGHT_IN_VACUUM * &eigenvals[2]).recip();
    // println!(
    //     "Rotational constants (cm^-1): \nA: {:.4}\nB: {:.4}\nC: {:.4}\n",
    //     &rot_const_A_per_cm, &rot_const_B_per_cm, &rot_const_C_per_cm
    // );

    // //* Step 8.1: Classify the type of rotor for molecule
    // println!("Classifying the type of rotor for molecule...");
    // if mol.no_atoms == 2 {
    //     println!("Molecule is linear and diatomic!");
    // } else if &eigenvals[0] < &1.0e-4 {
    //     println!("Molecule is linear!");
    // } else if (&eigenvals[0] - &eigenvals[1]).abs() < 1.0e-4
    //     && (&eigenvals[1] - &eigenvals[2]).abs() < 1.0e-4
    // {
    //     println!("Molecule is symmetric top!");
    // } else if (&eigenvals[0] - &eigenvals[1]).abs() < 1.0e-4
    //     && (&eigenvals[1] - &eigenvals[2]).abs() > 1.0e-4
    // {
    //     println!("Molecule is oblate symmetric top!")
    // } else if (&eigenvals[0] - &eigenvals[1]).abs() > 1.0e-4
    //     && (&eigenvals[1] - &eigenvals[2]).abs() < 1.0e-4
    // {
    //     println!("Molecule is a prolate symmetric top!")
    // } else {
    //     println!("Molecule is an asymmetric top!");
    // }
    }

    pub fn RHF_par(&mut self, is_debug: bool, basis_set_name: &str) {
        // * Step 1: Create basis set for molecule -> mol object gets passed to SCF object

        //* Create basis for mol object */
        self.mol.wfn_total.basis_set_total = create_basis_set_total(
            parse_basis_set_file_gaussian(basis_set_name),
            self.mol.geom_obj.geom_matr.clone(),
            &self.mol.geom_obj.Z_vals,
        );

        self.mol.update_no_occ_orb_rhf();

        for cgto in self
            .mol
            .wfn_total
            .basis_set_total
            .basis_set_cgtos
            .iter_mut()
        //TODO: Can I do this in parallel? -> par_iter_mut needs more impl work
        {
            cgto.calc_cart_norm_const_cgto();
        }

        // * CONSTANTS from mol object
        let no_occ_orb = self.mol.wfn_total.basis_set_total.no_occ_orb;
        let no_cgtos = self.mol.wfn_total.basis_set_total.no_cgtos;

        // * Step 2: Calculate the 1e- integrals (S, T, V_ne, H_core) and V_nn
        self.calc_1e_ints_par(is_debug);

        //* Step 3: Calc the 2e-ints (V_ee) */
        self.calc_2e_ints_par(is_debug);

        //* Step 4: Build the orthogonalization matrix S^(-1/2)
        let S_matr_sqrt_inv = self.calc_S_mart_inv_sqrt();

        if is_debug {
            println!("S^(-1/2):\n{:>11.6}\n", &S_matr_sqrt_inv);
        }

        // ! THE SCF ITERATIONS START HERE
        // *************************************************************************
        //*                               DIIS                                    */
        // *************************************************************************
        const SCF_MAXITER: usize = 100;
        const DIIS_MAX_FOCK_NO: usize = 6;
        const MIN_FOCK_NO_DIIS: usize = 2;

        // * Matrices for SCF iterations
        let mut F_matr = Array2::<f64>::zeros((no_cgtos, no_cgtos));
        let mut D_matr = Array2::<f64>::zeros((no_cgtos, no_cgtos));
        let mut D_matr_prev = Array2::<f64>::zeros((no_cgtos, no_cgtos));
        let mut C_matr_AO_basis = Array2::<f64>::zeros((no_cgtos, no_cgtos));
        let mut C_matr_MO_basis = Array2::<f64>::zeros((no_cgtos, no_cgtos));
        let mut orb_energy_arr = Array1::<f64>::zeros(no_cgtos);

        //* mutable floats for SCF */
        let mut E_scf = 0.0_f64;
        let mut E_scf_old = 0.0_f64;
        let mut E_tot = 0.0_f64;

        //* Give initial F_matr the H_core  */
        F_matr.assign(&self.mol.wfn_total.HF_Matrices.H_core_matr);

        for scf_iter in 0..SCF_MAXITER {
            if scf_iter == 0 {
                //* Step 5.1: Orthogonalize the F_matr */
                let F_matr_pr = self.ortho_F_matr(&F_matr, &S_matr_sqrt_inv);
                if is_debug {
                    println!("F_matr_pr:\n{:>11.6}\n", &F_matr_pr);
                }

                //* Step 5.2: Solve eigenvalue problem */
                (orb_energy_arr, C_matr_MO_basis) = F_matr_pr.eigh(UPLO::Upper).unwrap();
                C_matr_AO_basis = S_matr_sqrt_inv.dot(&C_matr_MO_basis);
                if is_debug {
                    println!("orb_energy_arr:\n{:>11.6}\n", &orb_energy_arr);
                    println!("C_matr_MO_basis:\n{:>11.6}\n", &C_matr_MO_basis);
                    println!("C_matr_AO_basis:\n{:>11.6}\n", &C_matr_AO_basis);
                }

                self.build_D_matr(&mut D_matr, &C_matr_AO_basis, no_occ_orb);
                for i in 0..no_cgtos - 1 {
                    let slice = D_matr.slice(s![i + 1..no_cgtos, i]).to_shared();
                    D_matr.slice_mut(s![i, i + 1..no_cgtos]).assign(&slice);
                }
                if is_debug {
                    println!("D^0_matr (inital density matrix):\n{:>11.6}\n", &D_matr);
                }

                E_scf = Zip::from(&D_matr)
                    .and(&self.mol.wfn_total.HF_Matrices.H_core_matr)
                    .into_par_iter()
                    .map(|(d_matr_val, h_core_val)| d_matr_val * 2.0 * h_core_val)
                    .sum();
                E_tot = E_scf + self.mol.wfn_total.HF_Matrices.V_nn_val;
                E_scf_old = E_scf;

                // ? Printing for SCF
                let header_scf_str = format!(
                    "{:^3} {:^16}{:^16}{:^16}{:^12}",
                    "Iter", "E_scf", "E_total", "ΔE", "RMS D"
                );
                println!("{header_scf_str}");
                // First iteration separately
                println!(
                    "{:>3} {: >16.8}{: >16.8}{: >16.8}{: >12.8}",
                    "0", &E_scf, &E_tot, " ", " "
                );
            } else {
                //* Step 5.5: Build F_matr and add to Fock set */
                F_matr = self.calc_F_matr_AO_par(&D_matr);
                self.F_matr_set.push_back(F_matr.clone()); //* copy the new F_matr to fock_set */
                if is_debug {
                    println!("F_matr:\n{:>11.6}\n", &F_matr);
                }

                //* Step 6: Calc error matrix + add error matrix to subspace */
                let error_matr = self.calc_DIIS_error_matr(&D_matr, &F_matr, &S_matr_sqrt_inv);
                self.error_matr_set.push_back(error_matr);

                let mut error_set_len = self.error_matr_set.len();
                debug_assert_eq!(error_set_len, self.F_matr_set.len());
                
                if error_set_len > DIIS_MAX_FOCK_NO {
                    self.F_matr_set.pop_front(); //* remove oldest */
                    self.error_matr_set.pop_front();
                    error_set_len -= 1;
                }

                //* Step 8: Run DIIS with F_matr_DIIS */
                if scf_iter >= MIN_FOCK_NO_DIIS {
                    F_matr = self.run_DIIS(error_set_len, is_debug);
                }

                //* Step 9: Transform F_matr to F_matr_pr */
                let F_matr_pr = S_matr_sqrt_inv.dot(&F_matr).dot(&S_matr_sqrt_inv);

                //* Step 10: Get the coefficients of the Fock matrix F in the MO basis
                (orb_energy_arr, C_matr_MO_basis) = F_matr_pr.eigh(UPLO::Lower).unwrap();
                C_matr_AO_basis = S_matr_sqrt_inv.dot(&C_matr_MO_basis);
                D_matr_prev = D_matr.clone();

                //* Step 11: Build D_matr for next Fock, but DO NOT add F_matr to Fock_set */
                self.build_D_matr(&mut D_matr, &C_matr_AO_basis, no_occ_orb);
                for i in 0..no_cgtos - 1 {
                    let slice = D_matr.slice(s![i + 1..no_cgtos, i]).to_shared();
                    D_matr.slice_mut(s![i, i + 1..no_cgtos]).assign(&slice);
                }

                //* Step 7: Compute E_scf + E_tot */
                E_scf = Zip::from(&D_matr)
                    .and(&self.mol.wfn_total.HF_Matrices.H_core_matr)
                    .and(&F_matr)
                    .into_par_iter()
                    .map(|(d_matr_val, h_core_val, f_matr_val)| {
                        d_matr_val * (h_core_val + f_matr_val)
                    })
                    .sum();

                E_tot = E_scf + self.mol.wfn_total.HF_Matrices.V_nn_val;

                let rms_d_val = Zip::from(&D_matr)
                    .and(&D_matr_prev)
                    .into_par_iter()
                    .map(|(d_matr_val, d_matr_prev_val)| (d_matr_val - d_matr_prev_val).powi(2))
                    .sum::<f64>()
                    .sqrt();

                let delta_E_scf = &E_scf - &E_scf_old;

                // * Printing the SCF results
                let line = format!(
                    "{:>3} {: >16.8}{: >16.8}{: >16.8}{: >12.8}",
                    &scf_iter, &E_scf, &E_tot, &delta_E_scf, &rms_d_val
                );
                println!("{line}");

                self.E_scf_final = E_scf;
                self.E_tot_final = E_tot;

                if rms_d_val < 1e-8 && delta_E_scf < 1e-8 {
                    break;
                }
                E_scf_old = E_scf;

                if scf_iter == SCF_MAXITER {
                    println!("SCF did not converge in {} iterations", SCF_MAXITER);
                }
            }
        }

        // *************************************************************************
        //*                               NO DIIS                                 */
        // *************************************************************************
        //* Step 7: Iterate the SCF procedure until convergence

        // for scf_iter in 0..scf_maxiter {
        //     //* Step 6: Form the Fock matrix F in the AO basis
        //     let mut F_matr = Array2::<f64>::zeros((no_cgtos, no_cgtos));
        //     F_matr.assign(&self.mol.wfn_total.HF_Matrices.H_core_matr);

        //     Zip::indexed(&mut F_matr).par_for_each(|(mu, nu), f_val| {
        //         for lambda in 0..no_cgtos {
        //             for sigma in 0..no_cgtos {
        //                 // let mu_nu_lambda_sigma =
        //                 //     calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
        //                 // let mu_lambda_nu_sigma =
        //                 //     calc_ijkl_idx(mu + 1, lambda + 1, nu + 1, sigma + 1);
        //                 *f_val += D_matr[(lambda, sigma)]
        //                     * (2.0
        //                         * self.mol.wfn_total.HF_Matrices.ERI_tensor
        //                             [(mu, nu, lambda, sigma)]
        //                         - self.mol.wfn_total.HF_Matrices.ERI_tensor
        //                             [(mu, lambda, nu, sigma)]);
        //             }
        //         }
        //     });

        //     // * Paralll code V3 (par_map_assign_into)
        //     // Zip::indexed(&mut F_matr_temp).par_map_assign_into(|(mu, nu), f| {
        //     //     // *f = self.mol.wfn_total.HF_Matrices.H_core_matr[(mu, nu)];
        //     //     for lambda in 0..no_cgtos {
        //     //         for sigma in 0..no_cgtos {
        //     //             let mu_nu_lambda_sigma =
        //     //                 calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
        //     //             let mu_lambda_nu_sigma =
        //     //                 calc_ijkl_idx(mu + 1, lambda + 1, nu + 1, sigma + 1);
        //     //             *f += D_matr[(lambda, sigma)]
        //     //                 * (2.0 * self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_nu_lambda_sigma]
        //     //                     - self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_lambda_nu_sigma]);
        //     //         }
        //     //     }
        //     // });
        //     // // fn calc_F_matr() -> Array1<f64> {

        //     // }

        //     //* Parallel code V2
        //     // let F_matr_mutex = Mutex::new(F_matr);

        //     // (0..no_cgtos).into_par_iter().for_each(|mu| {
        //     //     (0..no_cgtos).into_par_iter().for_each(|nu| {
        //     //         (0..no_cgtos).into_par_iter().for_each(|lambda| {
        //     //             (0..no_cgtos).into_par_iter().for_each(|sigma| {
        //     //                 // let mu_nu_lambda_sigma =
        //     //                 //     calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
        //     //                 // let mu_lambda_nu_sigma =
        //     //                 //     calc_ijkl_idx(mu + 1, lambda + 1, nu + 1, sigma + 1);
        //     //                 let mut f = F_matr_mutex.lock().unwrap();
        //     //                 f[(mu, nu)] += D_matr[(lambda, sigma)]
        //     //                     * (2.0
        //     //                         * self.mol.wfn_total.HF_Matrices.ERI_tensor
        //     //                             [(mu, nu, lambda, sigma)]
        //     //                         - self.mol.wfn_total.HF_Matrices.ERI_tensor
        //     //                             [(mu, lambda, nu, sigma)]);
        //     //             });
        //     //         });
        //     //     });
        //     // });

        //     // let F_matr = F_matr_mutex.into_inner().unwrap();

        //     // //* Parallel code V1 (still wrong)
        //     // let F_matr = Mutex::new(F_matr);
        //     // for mu in 0..no_cgtos {
        //     //     for nu in 0..no_cgtos {
        //     //         Zip::indexed(&D_matr).par_for_each(|(lambda, sigma), d_matr_val| {
        //     //             let mu_nu_lambda_sigma = calc_ijkl_idx(mu, nu, lambda, sigma);
        //     //             let mu_lambda_nu_sigma = calc_ijkl_idx(mu, lambda, nu, sigma);
        //     //             let mut f = F_matr.lock().unwrap();
        //     //             f[(mu, nu)] += d_matr_val
        //     //                 * (2.0 * self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_nu_lambda_sigma]
        //     //                     - self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_lambda_nu_sigma]);
        //     //         });
        //     //     }
        //     // }
        //     // let mut F_matr = F_matr.into_inner().unwrap();

        //     // //* Serial code
        //     // for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     //     for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     //         F_matr[(mu, nu)] = self.mol.wfn_total.HF_Matrices.H_core_matr[(mu, nu)];
        //     //         for lambda in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     //             for sigma in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     //                 let mu_nu_lambda_sigma =
        //     //                     calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
        //     //                 let mu_lambda_nu_sigma =
        //     //                     calc_ijkl_idx(mu + 1, lambda + 1, nu + 1, sigma + 1);
        //     //                 F_matr[(mu, nu)] += D_matr[(lambda, sigma)]
        //     //                     * (2.0
        //     //                         * self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_nu_lambda_sigma]
        //     //                         - self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_lambda_nu_sigma]);
        //     //             }
        //     //         }
        //     //     }
        //     // }

        //     //* Step 7: Form the Fock matrix F in the MO basis
        //     let F_matr_pr: Array2<f64> = S_matr_sqrt_inv
        //         .clone()
        //         .reversed_axes()
        //         .dot(&F_matr)
        //         .dot(&S_matr_sqrt_inv.clone());

        //     //* Step 8: Get the coefficients of the Fock matrix F in the MO basis
        //     (orb_energy_arr, C_matr_MO_basis) = F_matr_pr.eigh(UPLO::Lower).unwrap();
        //     C_matr_AO_basis = S_matr_sqrt_inv.dot(&C_matr_MO_basis);
        //     let D_matr_prev: Array2<f64> = D_matr.clone();

        //     //* D_matr indexed with par_for_each

        //     Zip::indexed(&mut D_matr).par_for_each(|(mu, nu), d_val| {
        //         if mu >= nu {
        //             let slice1 = C_matr_AO_basis.slice(s![mu, ..no_occ_orb]);
        //             let slice2 = C_matr_AO_basis.slice(s![nu, ..no_occ_orb]);
        //             *d_val = slice1.dot(&slice2);
        //         } else {
        //             *d_val = 0.0;
        //         }
        //     });

        //     // * Assign lower triangle to upper triangle with slices -> larger chunks
        //     for i in 0..no_cgtos - 1 {
        //         let slice = D_matr.slice(s![i + 1..no_cgtos, i]).to_shared();
        //         D_matr.slice_mut(s![i, i + 1..no_cgtos]).assign(&slice);
        //     }

        //     //* Parallel code
        //     // let D_matr_mutex = Mutex::new(Array2::<f64>::zeros((no_cgtos, no_cgtos)));
        //     // Zip::indexed(C_matr_AO_basis.outer_iter()).par_for_each(|mu, row1| {
        //     //     Zip::indexed(C_matr_AO_basis.outer_iter()).par_for_each(|nu, row2| {
        //     //         let mut d = D_matr_mutex.lock().unwrap();
        //     //         let slice1 = row1.slice(s![..no_occ_orb]);
        //     //         let slice2 = row2.slice(s![..no_occ_orb]);
        //     //         d[(mu, nu)] = slice1.dot(&slice2);
        //     //     });
        //     // });
        //     // D_matr = D_matr_mutex.into_inner().unwrap(); //* DO NOT SHADOW vars in Rust -> D_matr from outer scoop will be used otherwise */
        //     //                                              //* Serial code
        //     //                                              // for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     //                                              //     for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     //                                              //         D_matr[(mu, nu)] = 0.0;
        //     //                                              //         for m in 0..self.mol.wfn_total.basis_set_total.no_occ_orb {
        //     //                                              //             D_matr[(mu, nu)] += C_matr_AO_basis[(mu, m)] * C_matr_AO_basis[(nu, m)];
        //     //                                              //         }
        //     //                                              //     }
        //     //                                              // }

        //     //* Parallel code
        //     E_scf = Zip::from(&D_matr)
        //         .and(&self.mol.wfn_total.HF_Matrices.H_core_matr)
        //         .and(&F_matr)
        //         .into_par_iter()
        //         .map(|(d_matr_val, h_core_val, f_matr_val)| d_matr_val * (h_core_val + f_matr_val))
        //         .sum();

        //     //* Serial code
        //     // for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     //     for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     //         E_scf += D_matr[(mu, nu)]
        //     //             * (self.mol.wfn_total.HF_Matrices.H_core_matr[(mu, nu)] + F_matr[(mu, nu)]);
        //     //     }
        //     // }

        //     E_scf_vec.push(E_scf);
        //     E_tot = E_scf + self.mol.wfn_total.HF_Matrices.V_nn_val;
        //     E_tot_vec.push(E_tot);

        //     //* Parllel code
        //     let rms_d_val = Zip::from(&D_matr)
        //         .and(&D_matr_prev)
        //         .into_par_iter()
        //         .map(|(d_matr_val, d_matr_prev_val)| (d_matr_val - d_matr_prev_val).powi(2))
        //         .sum::<f64>()
        //         .sqrt();

        //     //* Serial code
        //     // let mut rms_d_val: f64 = 0.0;
        //     // for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     //     for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     //         rms_d_val += (D_matr[(mu, nu)] - D_matr_prev[(mu, nu)]).powi(2);
        //     //     }
        //     // }
        //     // rms_d_val = rms_d_val.sqrt();
        //     rms_d_vec.push(rms_d_val);

        self.C_matr_final = C_matr_AO_basis;
        self.D_matr_final = D_matr;
        self.orb_energies_final = orb_energy_arr;
    }

    pub fn RHF_ser(&mut self, is_debug: bool, basis_set_name: &str) {
        // * Step 1: Create basis set for molecule -> mol object gets passed to SCF object

        //* Create basis for mol object */
        self.mol.wfn_total.basis_set_total = create_basis_set_total(
            parse_basis_set_file_gaussian(basis_set_name),
            self.mol.geom_obj.geom_matr.clone(),
            &self.mol.geom_obj.Z_vals,
        );

        self.mol.update_no_occ_orb_rhf();
        for cgto in self
            .mol
            .wfn_total
            .basis_set_total
            .basis_set_cgtos
            .iter_mut()
        {
            cgto.calc_cart_norm_const_cgto();
        }

        // * Step 2: Calculate the 1e- integrals (S, T, V_ne, H_core) and V_nn
        self.calc_1e_ints_ser(is_debug);

        //* Step 3: Calc the 2e-ints (V_ee) */
        self.calc_2e_ints_ser(is_debug);

        //* Step 4: Build the orthogonalization matrix S^(-1/2)
        let S_matr_sqrt: Array2<f64> = self
            .mol
            .wfn_total
            .HF_Matrices
            .S_matr
            .ssqrt(ndarray_linalg::UPLO::Upper)
            .unwrap();
        let S_matr_sqrt_inv: Array2<f64> = S_matr_sqrt.inv().unwrap();

        if is_debug {
            println!("S^(-1/2):\n{:>11.6}\n", &S_matr_sqrt_inv);
        }

        //* Step 5: Form the initial guess density matrix D_0
        //* Step 5.1: Form the initial guess Fock matrix F_0 in the AO basis
        let F_matr_0_pr: Array2<f64> = S_matr_sqrt_inv
            .clone()
            .reversed_axes()
            .dot(&self.mol.wfn_total.HF_Matrices.H_core_matr)
            .dot(&S_matr_sqrt_inv.clone());

        if is_debug {
            println!("F_matr_0_pr:\n{:>11.6}\n", &F_matr_0_pr);
        }

        //* Step 5.2: Get the coefficients of the initial guess Fock matrix F_0 in the MO basis
        let (mut orb_energy_arr, mut C_matr_MO_basis) =
            F_matr_0_pr.eigh(ndarray_linalg::UPLO::Upper).unwrap();
        let mut C_matr_AO_basis: Array2<f64> = S_matr_sqrt_inv.dot(&C_matr_MO_basis);

        if is_debug {
            println!("orb_energy_arr:\n{:>11.6}\n", &orb_energy_arr);
            println!("C_matr_MO_basis:\n{:>11.6}\n", &C_matr_MO_basis);
            println!("C_matr_AO_basis:\n{:>11.6}\n", &C_matr_AO_basis);
        }

        //* Step 5.3: Form the initial guess density matrix D_0
        let mut D_matr: Array2<f64> = Array2::<f64>::zeros((
            self.mol.wfn_total.basis_set_total.no_cgtos,
            self.mol.wfn_total.basis_set_total.no_cgtos,
        ));

        for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
            for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                for m in 0..self.mol.wfn_total.basis_set_total.no_occ_orb {
                    D_matr[(mu, nu)] += C_matr_AO_basis[(mu, m)] * C_matr_AO_basis[(nu, m)];
                }
            }
        }

        if is_debug {
            println!("D^0_matr (inital density matrix):\n{:>11.6}\n", &D_matr);
        }

        let mut E_scf: f64 = 0.0;
        let mut E_scf_vec: Vec<f64> = Vec::new();
        let mut E_tot_vec: Vec<f64> = Vec::new();
        let mut rms_d_vec: Vec<f64> = Vec::new();

        //* Here the Fock matrix is guessed to be the core Hamiltonian matrix
        //* That's why the initial SCF energy differs from the other SCF energy calcs
        for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
            for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                E_scf +=
                    D_matr[(mu, nu)] * 2.0 * (self.mol.wfn_total.HF_Matrices.H_core_matr[(mu, nu)]);
            }
        }

        E_scf_vec.push(E_scf);
        let mut E_tot = E_scf + self.mol.wfn_total.HF_Matrices.V_nn_val;
        E_tot_vec.push(E_tot);

        //* Step 7: Iterate the SCF procedure until convergence
        let scf_maxiter: usize = 50;

        // ! THE SCF ITERATIONS START HERE
        for scf_iter in 0..scf_maxiter {
            //* Step 6: Form the Fock matrix F in the AO basis
            let n = self.mol.wfn_total.basis_set_total.no_cgtos;
            let mut F_matr = Array2::<f64>::zeros((n, n));

            F_matr.assign(&self.mol.wfn_total.HF_Matrices.H_core_matr);

            for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                    // F_matr[(mu, nu)] = self.mol.wfn_total.HF_Matrices.H_core_matr[(mu, nu)];
                    for lambda in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                        for sigma in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                            let mu_nu_lambda_sigma =
                                calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
                            let mu_lambda_nu_sigma =
                                calc_ijkl_idx(mu + 1, lambda + 1, nu + 1, sigma + 1);

                            F_matr[(mu, nu)] += D_matr[(lambda, sigma)]
                                * (2.0
                                    * self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_nu_lambda_sigma]
                                    - self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_lambda_nu_sigma]);
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
            (orb_energy_arr, C_matr_MO_basis) =
                F_matr_pr.eigh(ndarray_linalg::UPLO::Upper).unwrap();
            C_matr_AO_basis = S_matr_sqrt_inv.dot(&C_matr_MO_basis);
            let D_matr_prev: Array2<f64> = D_matr.clone();

            for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                    D_matr[(mu, nu)] = 0.0;
                    for m in 0..self.mol.wfn_total.basis_set_total.no_occ_orb {
                        D_matr[(mu, nu)] += C_matr_AO_basis[(mu, m)] * C_matr_AO_basis[(nu, m)];
                    }
                }
            }

            E_scf = 0.0;
            for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                    E_scf += D_matr[(mu, nu)]
                        * (self.mol.wfn_total.HF_Matrices.H_core_matr[(mu, nu)] + F_matr[(mu, nu)]);
                }
            }

            E_scf_vec.push(E_scf);
            E_tot = E_scf + self.mol.wfn_total.HF_Matrices.V_nn_val;
            E_tot_vec.push(E_tot);

            let mut rms_d_val: f64 = 0.0;
            for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                    rms_d_val += (D_matr[(mu, nu)] - D_matr_prev[(mu, nu)]).powi(2);
                }
            }
            rms_d_val = rms_d_val.sqrt();
            rms_d_vec.push(rms_d_val);

            if rms_d_val < 1e-6 {
                break;
            }
        }

        self.C_matr_final = C_matr_AO_basis;
        self.D_matr_final = D_matr;
        self.orb_energies_final = orb_energy_arr;

        let header_scf_str = format!(
            "{:^3} {:^16}{:^16}{:^12}",
            "Iter", "E_scf", "E_total", "RMS D"
        );
        println!("{header_scf_str}");

        // * First line extra
        let line = format!(
            "{:>3} {: >16.8}{: >16.8}{: >12.8}",
            "0", &E_scf_vec[0], &E_tot_vec[0], " "
        );
        println!("{line}");

        for idx in 1..E_scf_vec.len() {
            let line = format!(
                "{:>3} {: >16.8}{: >16.8}{: >12.8}",
                &idx,
                &E_scf_vec[idx],
                &E_tot_vec[idx],
                &rms_d_vec[idx - 1]
            );
            println!("{line}");
        }

        // * Safe final values in SCF object
        self.E_scf_final = E_scf_vec[E_scf_vec.len() - 1];
        self.E_tot_final = E_tot_vec[E_tot_vec.len() - 1];
    }

    fn calc_1e_ints_par(&mut self, is_debug: bool) {
        let n = self.mol.wfn_total.basis_set_total.no_cgtos;
        // * Build matrices for SCF */
        self.mol.wfn_total.HF_Matrices.V_nn_val =
            calc_V_nn_val(&self.mol.geom_obj.geom_matr, &self.mol.geom_obj.Z_vals);

        //* Step 2.1: Calculate the overlap matrix S
        //* New version: par_for_each and indexed */
        self.mol.wfn_total.HF_Matrices.S_matr = Array2::<f64>::zeros((n, n));

        Zip::indexed(&mut self.mol.wfn_total.HF_Matrices.S_matr).par_for_each(|(i, j), s_val| {
            if i == j {
                *s_val = 1.0;
            } else if i >= j {
                //* Calculate only lower triangle matrix */
                *s_val = calc_overlap_int_cgto(
                    &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
                    &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
                )
            } else {
                *s_val = 0.0;
            }
        });

        // * Assign lower triangle to upper triangle with slices -> larger chunks
        for i in 0..n - 1 {
            let slice = self
                .mol
                .wfn_total
                .HF_Matrices
                .S_matr
                .slice(s![i + 1..n, i])
                .to_shared();
            self.mol
                .wfn_total
                .HF_Matrices
                .S_matr
                .slice_mut(s![i, i + 1..n])
                .assign(&slice);
        }

        // let S_matr_lower = self.mol.wfn_total.HF_Matrices.S_matr.t().to_owned();
        // Zip::indexed(&mut self.mol.wfn_total.HF_Matrices.S_matr)
        //     .and(&S_matr_lower)
        //     .par_for_each(|(i, j), s_val, s_val_lower| {
        //         if i < j {
        //             *s_val = *s_val_lower;
        //         }
        //     });

        // ************************************************
        // let S_matr_tmp = Array2::<f64>::zeros((n, n));
        // let S_matr_mutex = Mutex::new(S_matr_tmp);
        // let overlap_val_mutex = Mutex::new(0.0);

        // (0..n).into_par_iter().for_each(|i| {
        //     (0..=i).into_par_iter().for_each(|j| {
        //         let mut s = S_matr_mutex.lock().unwrap();
        //         // let mut s = overlap_val_mutex.lock().unwrap();
        //         if i == j {
        //             s[(i, j)] = 1.0;
        //             // s = 1.0;
        //         } else {
        //             s[(i, j)] = calc_overlap_int_cgto(
        //                 &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
        //                 &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
        //             );
        //             s[(j, i)] = s[(i, j)];
        //         }
        //     })
        // });

        // self.mol.wfn_total.HF_Matrices.S_matr = S_matr_mutex.into_inner().unwrap();

        // for i in 0..n {
        //     for j in 0..=i {
        //         if i == j {
        //             self.mol.wfn_total.HF_Matrices.S_matr[(i, j)] = 1.0;
        //         } else {
        //             self.mol.wfn_total.HF_Matrices.S_matr[(i, j)] = calc_overlap_int_cgto(
        //                 &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
        //                 &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
        //             );
        //             self.mol.wfn_total.HF_Matrices.S_matr[(j, i)] =
        //                 self.mol.wfn_total.HF_Matrices.S_matr[(i, j)];
        //         }
        //     }
        // }

        //* Step 2.2: Calculate the kinetic energy matrix T
        //* New version: par_map_assign_into */
        self.mol.wfn_total.HF_Matrices.T_matr = Array2::<f64>::zeros((n, n));
        Zip::indexed(&mut self.mol.wfn_total.HF_Matrices.T_matr).par_for_each(
            |(idx1, idx2), t_val| {
                if idx1 >= idx2 {
                    *t_val = calc_kin_energy_int_cgto(
                        &self.mol.wfn_total.basis_set_total.basis_set_cgtos[idx1],
                        &self.mol.wfn_total.basis_set_total.basis_set_cgtos[idx2],
                    )
                } else {
                    *t_val = 0.0;
                }
            },
        );

        for i in 0..n - 1 {
            let slice = self
                .mol
                .wfn_total
                .HF_Matrices
                .T_matr
                .slice(s![i + 1..n, i])
                .to_shared();
            self.mol
                .wfn_total
                .HF_Matrices
                .T_matr
                .slice_mut(s![i, i + 1..n])
                .assign(&slice);
        }

        // let T_matr_lower = self.mol.wfn_total.HF_Matrices.T_matr.t().to_owned();
        // Zip::indexed(&mut self.mol.wfn_total.HF_Matrices.T_matr)
        //     .and(&T_matr_lower)
        //     .par_for_each(|(i, j), t_val, t_val_lower| {
        //         if i < j {
        //             *t_val = *t_val_lower;
        //         }
        //     });

        // let T_matr_tmp = Array2::<f64>::zeros((n, n));
        // let T_matr_mutex = Mutex::new(T_matr_tmp);

        // (0..n).into_par_iter().for_each(|i| {
        //     (0..=i).into_par_iter().for_each(|j| {
        //         let mut t = T_matr_mutex.lock().unwrap();
        //         t[(i, j)] = calc_kin_energy_int_cgto(
        //             &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
        //             &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
        //         );

        //         t[(j, i)] = t[(i, j)];
        //     })
        // });

        // self.mol.wfn_total.HF_Matrices.T_matr = T_matr_mutex.into_inner().unwrap();

        // for i in 0..n {
        //     for j in 0..=i {
        //         self.mol.wfn_total.HF_Matrices.T_matr[(i, j)] = calc_kin_energy_int_cgto(
        //             &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
        //             &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
        //         );
        //         self.mol.wfn_total.HF_Matrices.T_matr[(j, i)] =
        //             self.mol.wfn_total.HF_Matrices.T_matr[(i, j)];
        //     }
        // }

        //* Step 2.3: Calculate the nuclear attraction matrix V_ne

        // * New version: par_for_each */
        self.mol.wfn_total.HF_Matrices.V_ne_matr = Array2::<f64>::zeros((n, n));
        Zip::indexed(&mut self.mol.wfn_total.HF_Matrices.V_ne_matr).par_for_each(
            |(idx1, idx2), v_ne_val| {
                if idx1 >= idx2 {
                    // let mut v_ne_val = 0.0_f64;
                    for (atom_idx, atom_pos) in self
                        .mol
                        .geom_obj
                        .geom_matr
                        .axis_iter(ndarray::Axis(0))
                        .enumerate()
                    {
                        *v_ne_val += (-self.mol.geom_obj.Z_vals[atom_idx] as f64)
                            * calc_nuc_attr_int_cgto(
                                &self.mol.wfn_total.basis_set_total.basis_set_cgtos[idx1],
                                &self.mol.wfn_total.basis_set_total.basis_set_cgtos[idx2],
                                &atom_pos.to_owned(),
                            );
                    }
                    // self.mol.wfn_total.HF_Matrices.V_ne_matr[(idx1, idx2)] = *v_ne_val;
                } else {
                    *v_ne_val = 0.0;
                }
            },
        );

        // * copy lower triangle to upper triangle
        // * assign version
        for i in 0..n - 1 {
            let slice = self
                .mol
                .wfn_total
                .HF_Matrices
                .V_ne_matr
                .slice(s![i + 1..n, i])
                .to_shared();
            self.mol
                .wfn_total
                .HF_Matrices
                .V_ne_matr
                .slice_mut(s![i, i + 1..n])
                .assign(&slice);
        }

        // * parallel version
        // let V_matr_lower = self.mol.wfn_total.HF_Matrices.V_ne_matr.t().to_owned();

        // Zip::indexed(&mut self.mol.wfn_total.HF_Matrices.V_ne_matr)
        //     .and(&V_matr_lower)
        //     .par_for_each(|(idx1, idx2), v_ne_val, v_ne_val_t| {
        //         if idx1 < idx2 {
        //             *v_ne_val = *v_ne_val_t;
        //         }
        //     });

        // * serial version
        // for i in 0..n {
        //     for j in 0..i {
        //         self.mol.wfn_total.HF_Matrices.V_ne_matr[(j, i)] =
        //             self.mol.wfn_total.HF_Matrices.V_ne_matr[(i, j)];
        //     }
        // }

        // let V_ne_matr_tmp = Array2::<f64>::zeros((n, n));
        // let V_ne_matr_mutex = Mutex::new(V_ne_matr_tmp);
        // // self.mol.wfn_total.HF_Matrices.V_ne_matr = Array2::<f64>::zeros((n, n));

        // (0..n).into_par_iter().for_each(|i| {
        //     (0..=i).into_par_iter().for_each(|j| {
        //         let mut v = V_ne_matr_mutex.lock().unwrap();
        //         for (idx, atom_pos) in self
        //             .mol
        //             .geom_obj
        //             .geom_matr
        //             .axis_iter(ndarray::Axis(0))
        //             .enumerate()
        //         {
        //             v[(i, j)] += (-self.mol.geom_obj.Z_vals[idx] as f64)
        //                 * calc_nuc_attr_int_cgto(
        //                     &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
        //                     &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
        //                     &atom_pos.to_owned(),
        //                 );
        //         }
        //         v[(j, i)] = v[(i, j)];
        //     })
        // });
        // self.mol.wfn_total.HF_Matrices.V_ne_matr = V_ne_matr_mutex.into_inner().unwrap();

        // for i in 0..n {
        //     for j in 0..=i {
        //         for (idx, atom_pos) in self
        //             .mol
        //             .geom_obj
        //             .geom_matr
        //             .axis_iter(ndarray::Axis(0))
        //             .enumerate()
        //         {
        //             self.mol.wfn_total.HF_Matrices.V_ne_matr[(i, j)] +=
        //                 (-self.mol.geom_obj.Z_vals[idx] as f64)
        //                     * calc_nuc_attr_int_cgto(
        //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
        //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
        //                         &atom_pos.to_owned(),
        //                     );
        //         }
        //         self.mol.wfn_total.HF_Matrices.V_ne_matr[(j, i)] =
        //             self.mol.wfn_total.HF_Matrices.V_ne_matr[(i, j)];
        //     }
        // }

        //* Step 2.4: Form the core Hamiltonian matrix H_core
        // self.mol.wfn_total.HF_Matrices.H_core_matr = Array2::<f64>::zeros((n, n));
        self.mol.wfn_total.HF_Matrices.H_core_matr =
            &self.mol.wfn_total.HF_Matrices.T_matr + &self.mol.wfn_total.HF_Matrices.V_ne_matr;

        if is_debug {
            println!(
                "Overlap matrix S:\n{:>8.5}\n",
                &self.mol.wfn_total.HF_Matrices.S_matr
            );
            println!(
                "Kinetic energy matrix T:\n{:>8.5}\n",
                &self.mol.wfn_total.HF_Matrices.T_matr
            );
            println!(
                "Nuclear attraction matrix V_ne:\n{:>8.5}\n",
                &self.mol.wfn_total.HF_Matrices.V_ne_matr
            );
            println!(
                "Core Hamiltonian matrix H_core:\n{:>8.5}\n",
                &self.mol.wfn_total.HF_Matrices.H_core_matr
            );
        }
    }

    fn calc_1e_ints_ser(&mut self, is_debug: bool) {
        let n = self.mol.wfn_total.basis_set_total.no_cgtos;
        // * Build matrices for SCF */
        self.mol.wfn_total.HF_Matrices.V_nn_val =
            calc_V_nn_val(&self.mol.geom_obj.geom_matr, &self.mol.geom_obj.Z_vals);

        //* Step 2.1: Calculate the overlap matrix S
        self.mol.wfn_total.HF_Matrices.S_matr = Array2::<f64>::zeros((n, n));
        for i in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
            for j in 0..=i {
                if i == j {
                    self.mol.wfn_total.HF_Matrices.S_matr[(i, j)] = 1.0;
                } else {
                    self.mol.wfn_total.HF_Matrices.S_matr[(i, j)] = calc_overlap_int_cgto(
                        &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
                        &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
                    );
                    self.mol.wfn_total.HF_Matrices.S_matr[(j, i)] =
                        self.mol.wfn_total.HF_Matrices.S_matr[(i, j)];
                }
            }
        }

        //* Step 2.2: Calculate the kinetic energy matrix T
        self.mol.wfn_total.HF_Matrices.T_matr = Array2::<f64>::zeros((n, n));

        for i in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
            for j in 0..=i {
                self.mol.wfn_total.HF_Matrices.T_matr[(i, j)] = calc_kin_energy_int_cgto(
                    &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
                    &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
                );
                self.mol.wfn_total.HF_Matrices.T_matr[(j, i)] =
                    self.mol.wfn_total.HF_Matrices.T_matr[(i, j)];
            }
        }

        //* Step 2.3: Calculate the nuclear attraction matrix V_ne
        self.mol.wfn_total.HF_Matrices.V_ne_matr = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                for (idx, atom_pos) in self
                    .mol
                    .geom_obj
                    .geom_matr
                    .axis_iter(ndarray::Axis(0))
                    .enumerate()
                {
                    self.mol.wfn_total.HF_Matrices.V_ne_matr[(i, j)] +=
                        (-self.mol.geom_obj.Z_vals[idx] as f64)
                            * calc_nuc_attr_int_cgto(
                                &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
                                &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
                                &atom_pos.to_owned(),
                            );
                }
                self.mol.wfn_total.HF_Matrices.V_ne_matr[(j, i)] =
                    self.mol.wfn_total.HF_Matrices.V_ne_matr[(i, j)];
            }
        }

        //* Step 2.4: Form the core Hamiltonian matrix H_core
        self.mol.wfn_total.HF_Matrices.H_core_matr = Array2::<f64>::zeros((n, n));
        self.mol.wfn_total.HF_Matrices.H_core_matr =
            &self.mol.wfn_total.HF_Matrices.T_matr + &self.mol.wfn_total.HF_Matrices.V_ne_matr;

        if is_debug {
            println!(
                "Overlap matrix S:\n{:>8.5}\n",
                &self.mol.wfn_total.HF_Matrices.S_matr
            );
            println!(
                "Kinetic energy matrix T:\n{:>8.5}\n",
                &self.mol.wfn_total.HF_Matrices.T_matr
            );
            println!(
                "Nuclear attraction matrix V_ne:\n{:>8.5}\n",
                &self.mol.wfn_total.HF_Matrices.V_ne_matr
            );
            println!(
                "Core Hamiltonian matrix H_core:\n{:>8.5}\n",
                &self.mol.wfn_total.HF_Matrices.H_core_matr
            );
        }
    }

    fn calc_2e_ints_par(&mut self, is_debug: bool) {
        // * Array variant (1D-Tensor == Array)

        let n = self.mol.wfn_total.basis_set_total.no_cgtos;
        // let n_idx = calc_cmp_idx(n, n);
        // let ERI_arr1_max_idx = calc_cmp_idx(n_idx, n_idx);

        // self.mol.wfn_total.HF_Matrices.ERI_arr1 = Array1::<f64>::zeros(ERI_arr1_max_idx + 1);

        // // (0..n).into_par_iter().for_each(|i| {
        // //     (0..=i).into_par_iter().for_each(|j| {
        // //     });
        // // });

        // for i in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     for j in 0..=i {
        //         // for j in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //         let ij = calc_cmp_idx(i, j);
        //         for k in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //             for l in 0..=k {
        //                 // for l in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //                 let kl = calc_cmp_idx(k, l);
        //                 if ij >= kl {
        //                     // let ijkl = calc_cmp_idx(ij, kl);
        //                     let ijkl = calc_ijkl_idx(i + 1, j + 1, k + 1, l + 1);
        //                     self.mol.wfn_total.HF_Matrices.ERI_arr1[ijkl] =
        //                         calc_elec_elec_repul_cgto(
        //                             &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
        //                             &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
        //                             &self.mol.wfn_total.basis_set_total.basis_set_cgtos[k],
        //                             &self.mol.wfn_total.basis_set_total.basis_set_cgtos[l],
        //                         );
        //                 }
        //             }
        //         }
        //     }
        // }

        // * Tensor variant (4D-Tensor), but in parallel?
        self.mol.wfn_total.HF_Matrices.ERI_tensor = Array4::<f64>::zeros((n, n, n, n));

        Zip::indexed(&mut self.mol.wfn_total.HF_Matrices.ERI_tensor).par_for_each(
            |(i, j, k, l), eri_val| {
                let ij = calc_cmp_idx(i, j);
                let kl = calc_cmp_idx(k, l);
                if ij >= kl {
                    *eri_val = calc_elec_elec_repul_cgto(
                        &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
                        &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
                        &self.mol.wfn_total.basis_set_total.basis_set_cgtos[k],
                        &self.mol.wfn_total.basis_set_total.basis_set_cgtos[l],
                    );
                }
            },
        );

        //TODO: Assign the values to the other 7 permutations with ndarray assign method

        // * Serial to copy ERI_val to all 8 permutations
        for i in 0..n {
            for j in 0..=i {
                let ij = calc_cmp_idx(i, j);
                for k in 0..n {
                    for l in 0..=k {
                        let kl = calc_cmp_idx(k, l);
                        if ij >= kl {
                            let ERI_val = self.mol.wfn_total.HF_Matrices.ERI_tensor[[i, j, k, l]];
                            self.mol.wfn_total.HF_Matrices.ERI_tensor[(j, i, k, l)] = ERI_val;
                            self.mol.wfn_total.HF_Matrices.ERI_tensor[(i, j, l, k)] = ERI_val;
                            self.mol.wfn_total.HF_Matrices.ERI_tensor[(j, i, l, k)] = ERI_val;
                            self.mol.wfn_total.HF_Matrices.ERI_tensor[(k, l, i, j)] = ERI_val;
                            self.mol.wfn_total.HF_Matrices.ERI_tensor[(l, k, i, j)] = ERI_val;
                            self.mol.wfn_total.HF_Matrices.ERI_tensor[(k, l, j, i)] = ERI_val;
                            self.mol.wfn_total.HF_Matrices.ERI_tensor[(l, k, j, i)] = ERI_val;
                        }
                    }
                }
            }
        }

        // self.mol.wfn_total.HF_Matrices.ERI_tensor = Array4::<f64>::zeros((
        //     self.mol.wfn_total.basis_set_total.no_cgtos,
        //     self.mol.wfn_total.basis_set_total.no_cgtos,
        //     self.mol.wfn_total.basis_set_total.no_cgtos,
        //     self.mol.wfn_total.basis_set_total.no_cgtos,
        // ));
        // for i in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     for j in 0..=i {
        //         let ij = calc_cmp_idx(i, j);
        //         for k in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //             for l in 0..=k {
        //                 let kl = calc_cmp_idx(k, l);
        //                 if ij >= kl {
        //                     let ERI_val = calc_elec_elec_repul_cgto(
        //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
        //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
        //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[k],
        //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[l],
        //                     );
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(i, j, k, l)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(j, i, k, l)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(i, j, l, k)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(j, i, l, k)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(k, l, i, j)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(l, k, i, j)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(k, l, j, i)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(l, k, j, i)] = ERI_val;
        //                 }
        //             }
        //         }
        //     }
        // }

        if is_debug {
            println!("V_ee tensor (ERI vals):");
            println!("{:>8.5}\n", &self.mol.wfn_total.HF_Matrices.ERI_tensor);
        }
    }

    fn calc_2e_ints_ser(&mut self, is_debug: bool) {
        // * Array variant (1D-Tensor == Array)

        let mut n = self.mol.wfn_total.basis_set_total.no_cgtos;
        n = calc_cmp_idx(n, n);
        let ERI_arr1_max_idx = calc_cmp_idx(n, n);

        self.mol.wfn_total.HF_Matrices.ERI_arr1 = Array1::<f64>::zeros(ERI_arr1_max_idx + 1);

        for i in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
            for j in 0..=i {
                // for j in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                let ij = calc_cmp_idx(i, j);
                for k in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                    for l in 0..=k {
                        // for l in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                        let kl = calc_cmp_idx(k, l);
                        if ij >= kl {
                            // let ijkl = calc_cmp_idx(ij, kl);
                            let ijkl = calc_ijkl_idx(i + 1, j + 1, k + 1, l + 1);
                            self.mol.wfn_total.HF_Matrices.ERI_arr1[ijkl] =
                                calc_elec_elec_repul_cgto(
                                    &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
                                    &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
                                    &self.mol.wfn_total.basis_set_total.basis_set_cgtos[k],
                                    &self.mol.wfn_total.basis_set_total.basis_set_cgtos[l],
                                );
                        }
                    }
                }
            }
        }

        // * Tensor variant (4D-Tensor)
        // self.mol.wfn_total.HF_Matrices.ERI_tensor = Array4::<f64>::zeros((
        //     self.mol.wfn_total.basis_set_total.no_cgtos,
        //     self.mol.wfn_total.basis_set_total.no_cgtos,
        //     self.mol.wfn_total.basis_set_total.no_cgtos,
        //     self.mol.wfn_total.basis_set_total.no_cgtos,
        // ));
        // for i in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     for j in 0..=i {
        //         let ij = calc_cmp_idx(i, j);
        //         for k in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //             for l in 0..=k {
        //                 let kl = calc_cmp_idx(k, l);
        //                 if ij >= kl {
        //                     let ERI_val = calc_elec_elec_repul_cgto(
        //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
        //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
        //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[k],
        //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[l],
        //                     );
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(i, j, k, l)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(j, i, k, l)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(i, j, l, k)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(j, i, l, k)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(k, l, i, j)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(l, k, i, j)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(k, l, j, i)] = ERI_val;
        //                     self.mol.wfn_total.HF_Matrices.ERI_tensor[(l, k, j, i)] = ERI_val;
        //                 }
        //             }
        //         }
        //     }
        // }

        if is_debug {
            println!("V_ee tensor (ERI vals):");
            println!("{:>8.5}\n", &self.mol.wfn_total.HF_Matrices.ERI_arr1);
        }
    }

    #[inline]
    fn calc_DIIS_error_matr(
        &self,
        _D_matr: &Array2<f64>,
        _F_matr: &Array2<f64>,
        _S_matr_sqrt_inv: &Array2<f64>,
    ) -> Array2<f64> {
        let mut error_matr = _F_matr
            .dot(_D_matr)
            .dot(&self.mol.wfn_total.HF_Matrices.S_matr)
            - &self
                .mol
                .wfn_total
                .HF_Matrices
                .S_matr
                .dot(_D_matr)
                .dot(_F_matr);
        error_matr = _S_matr_sqrt_inv.dot(&error_matr).dot(_S_matr_sqrt_inv);

        error_matr
    }

    #[inline]
    fn run_DIIS(&mut self, error_set_len: usize, is_debug: bool) -> Array2<f64> {
        let no_cgtos = self.mol.wfn_total.basis_set_total.no_cgtos;

        let mut B_matr = Array2::<f64>::zeros((error_set_len, error_set_len));
        let mut sol_vec = Array1::<f64>::zeros(error_set_len + 1);
        sol_vec[error_set_len] = -1.0;

        // * ACTUALLY: Frobenius inner product of matrices (B_ij = error_matr_i * error_matr_j)
        // * OR: flatten error_matr and do dot product
        Zip::indexed(&mut B_matr).par_for_each(|(idx1, idx2), b_val| {
            if idx1 >= idx2 {
                *b_val = Zip::from(&self.error_matr_set[idx1])
                    .and(&self.error_matr_set[idx2])
                    .into_par_iter()
                    .map(|(error_matr_val1, error_matr_val2)| error_matr_val1 * error_matr_val2)
                    .sum();
            }
        });

        for i in 0..error_set_len - 1 {
            let slice = B_matr.slice(s![i + 1..error_set_len, i]).to_shared();
            B_matr.slice_mut(s![i, i + 1..error_set_len]).assign(&slice);
        }

        // * Add langrange multiplier to B_matr_extended
        let new_axis_extension_1 = Array2::from_elem((error_set_len, 1), -1.0_f64);
        let mut new_axis_extension_2 = Array2::from_elem((1, error_set_len + 1), -1.0_f64);
        new_axis_extension_2[[0, error_set_len]] = 0.0_f64;
        let mut B_matr_extended = concatenate![Axis(1), B_matr, new_axis_extension_1];
        B_matr_extended = concatenate![Axis(0), B_matr_extended, new_axis_extension_2];

        // * Calculate the coefficients c_vec
        let c_vec = B_matr_extended.solveh(&sol_vec).unwrap();
        if is_debug {
            println!("c_vec: {:>8.5}", c_vec);
        }

        // * Calculate the new DIIS Fock matrix for new D_matr
        let mut _F_matr_DIIS = Array2::<f64>::zeros((no_cgtos, no_cgtos));
        for i in 0..error_set_len {
            _F_matr_DIIS = _F_matr_DIIS + c_vec[i] * &self.F_matr_set[i];
        }

        _F_matr_DIIS
    }

    #[inline]
    fn build_D_matr(
        &mut self,
        D_matr: &mut Array2<f64>,
        C_matr_AO_basis: &Array2<f64>,
        no_occ_orb: usize,
    ) {
        Zip::indexed(D_matr).par_for_each(|(mu, nu), D_matr_val| {
            if mu >= nu {
                let slice1 = C_matr_AO_basis.slice(s![mu, ..no_occ_orb]);
                let slice2 = C_matr_AO_basis.slice(s![nu, ..no_occ_orb]);
                *D_matr_val = slice1.dot(&slice2);
            } else {
                *D_matr_val = 0.0;
            }
        });
    }

    #[inline]
    fn ortho_F_matr(&mut self, F_matr: &Array2<f64>, S_matr_sqrt_inv: &Array2<f64>) -> Array2<f64> {
        S_matr_sqrt_inv
            .clone()
            .reversed_axes()
            .dot(F_matr)
            .dot(S_matr_sqrt_inv)
    }

    #[inline]
    fn calc_F_matr_AO_par(&self, D_matr: &Array2<f64>) -> Array2<f64> {
        let no_cgtos = self.mol.wfn_total.basis_set_total.no_cgtos;
        let mut _F_matr = Array2::<f64>::zeros((no_cgtos, no_cgtos));
        _F_matr.assign(&self.mol.wfn_total.HF_Matrices.H_core_matr);

        Zip::indexed(&mut _F_matr).par_for_each(|(mu, nu), f_val| {
            for lambda in 0..no_cgtos {
                for sigma in 0..no_cgtos {
                    *f_val += D_matr[(lambda, sigma)]
                        * (2.0
                            * self.mol.wfn_total.HF_Matrices.ERI_tensor[(mu, nu, lambda, sigma)]
                            - self.mol.wfn_total.HF_Matrices.ERI_tensor[(mu, lambda, nu, sigma)]);
                }
            }
        });

        _F_matr
    }

    #[inline]
    fn calc_S_mart_inv_sqrt(&mut self) -> Array2<f64> {
        let S_matr_sqrt = self
            .mol
            .wfn_total
            .HF_Matrices
            .S_matr
            .ssqrt(ndarray_linalg::UPLO::Upper)
            .unwrap();
        S_matr_sqrt.inv().unwrap()
    }

    pub fn MP2(&mut self, is_debug: bool, basis_set_name: &str) {
        // * only rerun HF, if it has not been run before
        if self.mol.wfn_total.HF_Matrices.ERI_arr1.is_empty() {
            Self::RHF_par(self, is_debug, basis_set_name);
        }
        println!("\nThis is the end of RHF");
        println!("Starting MP2...\n");
        let no_cgtos = self.mol.wfn_total.basis_set_total.no_cgtos;
        let no_occ_orb = self.mol.wfn_total.basis_set_total.no_occ_orb;

        let orb_energies_ground_state = self.orb_energies_final.slice(s![..no_occ_orb]).to_owned();
        let orb_energies_exc_state = self.orb_energies_final.slice(s![no_occ_orb..]).to_owned();

        let C_occ = self.C_matr_final.slice(s![.., ..no_occ_orb]).to_owned();
        let C_virt = self.C_matr_final.slice(s![.., no_occ_orb..]).to_owned();

        let orb_energies = self.orb_energies_final.to_owned();
        let C_matr = self.C_matr_final.to_owned();
        // * Calc 2e ints if not already done
        if self.mol.wfn_total.HF_Matrices.ERI_arr1.is_empty() {
            Self::calc_2e_ints_par(self, false);
        }

        let is_naive_mp2 = true;
        let is_smarter_mp2 = false;

        let mut n = self.mol.wfn_total.basis_set_total.no_cgtos;
        n = calc_cmp_idx(n, n);
        let ERI_arr1_max_idx = calc_cmp_idx(n, n);
        let mut ERI_MO_MP2 = Array1::<f64>::zeros(ERI_arr1_max_idx + 1);

        if is_naive_mp2 {
            // * Naive implementation with nested for loops
            for i in 0..no_cgtos {
                for j in 0..=i {
                    let ij = calc_cmp_idx(i, j);
                    for k in 0..no_cgtos {
                        for l in 0..=k {
                            let kl = calc_cmp_idx(k, l);
                            if ij >= kl {
                                let ijkl = calc_ijkl_idx(i + 1, j + 1, k + 1, l + 1);
                                for mu in 0..no_cgtos {
                                    for nu in 0..no_cgtos {
                                        for lambda in 0..no_cgtos {
                                            for sigma in 0..no_cgtos {
                                                let mu_nu_lambda_sigma = calc_ijkl_idx(
                                                    mu + 1,
                                                    nu + 1,
                                                    lambda + 1,
                                                    sigma + 1,
                                                );
                                                ERI_MO_MP2[ijkl] += C_matr[[mu, i]]
                                                    * C_matr[[nu, j]]
                                                    * C_matr[[lambda, k]]
                                                    * C_matr[[sigma, l]]
                                                    * self.mol.wfn_total.HF_Matrices.ERI_arr1
                                                        [mu_nu_lambda_sigma];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // TODO: FIX smarter MP2 -> not yet working
        if is_smarter_mp2 {
            // * 4 for-loops

            let mut tmp_matr = Array2::<f64>::zeros((no_cgtos, no_cgtos));

            // * 1st for-loop */
            for l in 0..no_cgtos - no_occ_orb {
                for mu in 0..no_cgtos {
                    for nu in 0..no_cgtos {
                        for lambda in 0..no_cgtos {
                            for sigma in 0..no_cgtos {
                                // let ijkl = calc_ijkl_idx(i + 1, j + 1, k + 1, l + 1);
                                let mu_nu_lambda_sigma =
                                    calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
                                tmp_matr[(l, mu_nu_lambda_sigma)] += C_matr[(mu, l)]
                                    * self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_nu_lambda_sigma];
                                tmp_matr[(mu_nu_lambda_sigma, l)] =
                                    tmp_matr[(l, mu_nu_lambda_sigma)];
                            }
                        }
                    }
                }
            }

            //* 2nd for-loop */
            for k in no_occ_orb..no_cgtos {
                for mu in 0..no_cgtos {
                    for nu in 0..no_cgtos {
                        for lambda in 0..no_cgtos {
                            for sigma in 0..no_cgtos {
                                // let ijkl = calc_ijkl_idx(i + 1, j + 1, k + 1, l + 1);
                                let mu_nu_lambda_sigma =
                                    calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
                                tmp_matr[(k, mu_nu_lambda_sigma)] += C_matr[(nu, k)]
                                    * self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_nu_lambda_sigma];
                                tmp_matr[(mu_nu_lambda_sigma, k)] =
                                    tmp_matr[(k, mu_nu_lambda_sigma)];
                            }
                        }
                    }
                }
            }

            //* 3rd for-loop */
            for j in 0..no_cgtos - no_occ_orb {
                for mu in 0..no_cgtos {
                    for nu in 0..no_cgtos {
                        for lambda in 0..no_cgtos {
                            for sigma in 0..no_cgtos {
                                // let ijkl = calc_ijkl_idx(i + 1, j + 1, k + 1, l + 1);
                                let mu_nu_lambda_sigma =
                                    calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
                                tmp_matr[(j, mu_nu_lambda_sigma)] += C_matr[(lambda, j)]
                                    * self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_nu_lambda_sigma];
                                tmp_matr[(mu_nu_lambda_sigma, j)] =
                                    tmp_matr[(j, mu_nu_lambda_sigma)];
                            }
                        }
                    }
                }
            }

            //* 4th for-loop */
            for i in no_occ_orb..no_cgtos {
                for mu in 0..no_cgtos {
                    for nu in 0..no_cgtos {
                        for lambda in 0..no_cgtos {
                            for sigma in 0..no_cgtos {
                                // let ijkl = calc_ijkl_idx(i + 1, j + 1, k + 1, l + 1);
                                let mu_nu_lambda_sigma =
                                    calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
                                tmp_matr[(i, mu_nu_lambda_sigma)] += C_matr[(sigma, i)]
                                    * self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_nu_lambda_sigma];
                                tmp_matr[(mu_nu_lambda_sigma, i)] =
                                    tmp_matr[(i, mu_nu_lambda_sigma)];
                            }
                        }
                    }
                }
            }

            println!("tmp_matr:");
            println!("{:>8.5}\n", &tmp_matr);
            // ERI_MO_MP2
            // //* Save result in ERI_MO_MP2 */
            // for i in 0..no_cgtos {
            //     for j in 0..=i {
            //         let ij = calc_cmp_idx(i, j);
            //         for k in 0..no_cgtos {
            //             for l in 0..=k {
            //                 let kl = calc_cmp_idx(k, l);
            //                 let ijkl = calc_ijkl_idx(i + 1, j + 1, k + 1, l + 1);
            //                 ERI_MO_MP2[ijkl] = tmp_matr[(i, j)] * tmp_matr[(k, l)];
            //             }
            //         }
            //     }
            // }
        }

        if is_debug {
            println!("ERI_MO_MP2 tensor (ERI vals):");
            println!("{:>8.5}\n", &ERI_MO_MP2);
        }

        // * Calc MP2 energy
        let mut MP2_E = 0.0;
        for i in 0..no_occ_orb {
            for a in no_occ_orb..no_cgtos {
                for j in 0..no_occ_orb {
                    for b in no_occ_orb..no_cgtos {
                        let iajb = calc_ijkl_idx(i + 1, a + 1, j + 1, b + 1);
                        let ibja = calc_ijkl_idx(i + 1, b + 1, j + 1, a + 1);
                        MP2_E += ERI_MO_MP2[iajb] * (2.0 * ERI_MO_MP2[iajb] - ERI_MO_MP2[ibja])
                            / (orb_energies[i] + orb_energies[j]
                                - orb_energies[a]
                                - orb_energies[b]);
                    }
                }
            }
        }

        println!("MP2 energy: {:>10.5}", MP2_E);
        println!("New total energy: {:>10.5}", self.E_tot_final + MP2_E);
    }
}

#[inline]
pub fn calc_ijkl_idx(i: usize, j: usize, k: usize, l: usize) -> usize {
    let ij: usize = if i >= j {
        calc_cmp_idx(i, j)
    } else {
        calc_cmp_idx(j, i)
    };
    let kl: usize = if k >= l {
        calc_cmp_idx(k, l)
    } else {
        calc_cmp_idx(l, k)
    };
    let ijkl: usize = if ij >= kl {
        calc_cmp_idx(ij, kl)
    } else {
        calc_cmp_idx(kl, ij)
    };
    ijkl
}

#[inline(always)]
pub fn calc_cmp_idx(idx1: usize, idx2: usize) -> usize {
    (idx1 * (idx1 + 1)) / 2 + idx2
}
