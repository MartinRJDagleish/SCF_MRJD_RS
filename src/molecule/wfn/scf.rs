use ndarray::{prelude::*, Zip};
use ndarray_linalg::{Eigh, Inverse, SymmetricSqrt};

use ndarray::parallel::prelude::*;

use std::sync::Mutex;

use crate::molecule::{
    wfn::{
        basisset::{create_basis_set_total, parse_basis_set_file_gaussian},
        integrals::{
            calc_V_nn_val, calc_elec_elec_repul_cgto, calc_kin_energy_int_cgto,
            calc_nuc_attr_int_cgto, calc_overlap_int_cgto,
        },
    },
    Molecule,
};

#[derive(Debug)]
pub struct SCF {
    pub mol: Molecule,
    pub E_scf_final: f64,
    pub E_tot_final: f64,
    pub C_matr_final: Array2<f64>,
    pub D_matr_final: Array2<f64>,
    pub orb_energies_final: Array1<f64>,
}

impl SCF {
    pub fn new(mol: Molecule) -> Self {
        let C_matr_final = Array::default((0, 0));
        let D_matr_final = Array::default((0, 0));
        let orb_energies_final = Array::default(0);
        SCF {
            mol,
            E_scf_final: 0.0,
            E_tot_final: 0.0,
            C_matr_final,
            D_matr_final,
            orb_energies_final,
        }
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
        {
            cgto.calc_cart_norm_const_cgto();
        }

        // * Step 2: Calculate the 1e- integrals (S, T, V_ne, H_core) and V_nn
        self.calc_1e_ints(is_debug);

        //* Step 3: Calc the 2e-ints (V_ee) */
        self.calc_2e_ints(is_debug);

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

        // * For testing
        let mut D_matr = Mutex::new(D_matr);
        let no_occ_orb = self.mol.wfn_total.basis_set_total.no_occ_orb;

        //* D^0 matrix */
        //* Trying to parallelize it */
        Zip::indexed(C_matr_AO_basis.axis_iter(Axis(0)))
            .par_for_each(|mu, row1| {
            Zip::indexed(C_matr_AO_basis.outer_iter())
                .par_for_each(|nu, row2| {
                    let mut d = D_matr.lock().unwrap();
                    println!(
                        "mu = {}, nu = {}, row1 = {:>11.6}, row2 = {:>11.6}\n",
                        mu, nu, row1, row2);
                    let slice1 = row1.slice(s![..no_occ_orb]);
                    let slice2 = row2.slice(s![..no_occ_orb]);
                    d[(mu,nu)] = slice1.dot(&row2);
            });
        });

        let mut D_matr = D_matr.into_inner().unwrap();

        // D_matr.outer_iter_mut()
        //     .into_par_iter()
        //     .enumerate()
        //     .for_each(|(mu, mut row_D)| {
        //         C_matr_AO_basis.outer_iter()
        //             .into_par_iter()
        //             .enumerate()
        //             .for_each(|(nu, row_C)| {
        //                 let mut sum = 0.0;
        //                 for m in 0..self.mol.wfn_total.basis_set_total.no_occ_orb {
        //                     sum += row_C[m] * C_matr_AO_basis[(nu, m)];
        //                 }
        //                 row_D[mu] += sum * row_C[nu];
        //             });
        //     });

        // * Original code:
        // * Serial code 
        // for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //     for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
        //         for m in 0..self.mol.wfn_total.basis_set_total.no_occ_orb {
        //             D_matr[(mu, nu)] += C_matr_AO_basis[(mu, m)] * C_matr_AO_basis[(nu, m)];
        //         }
        //     }
        // }


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

            let idx_pairs: Vec<(usize, usize)> = (0..n)
                .flat_map(|mu| (0..mu).map(move |nu| (mu, nu)))
                .collect();

            // println!("Before: {:>8.5}", &F_matr);

            // Zip::indexed(&mut F_matr)
            //     .par_for_each(|(i,j), | {
            //         for lambda in 0..n {
            //             for sigma in 0..=lambda {
            //                 let mu_nu_lambda_sigma =
            //                     calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
            //                 let mu_lambda_nu_sigma =
            //                     calc_ijkl_idx(mu + 1, lambda + 1, nu + 1, sigma + 1);

            //                 *F_matr_val += D_matr[(lambda, sigma)]
            //                     * (self.mol.wfn_total.HF_Matrices.G_matr[(mu_nu_lambda_sigma - 1)]
            //                         - 0.5
            //                             * self.mol.wfn_total.HF_Matrices.G_matr
            //                                 [(mu_lambda_nu_sigma - 1)]);
            //             }
            //         }
            //     });

            // let F_matr_mutex = Arc::new(Mutex::new(F_matr));
            // let F_matr_mutex = Mutex::new(&mut F_matr);

            // let par_iter = idx_pairs.par_iter();

            // par_iter.for_each(|(mu, nu)| {
            //     let mut F_matr_val: f64 = 0.0;
            //     for lambda in 0..n {
            //         for sigma in 0..=lambda {
            //             let mu_nu_lambda_sigma =
            //                 calc_ijkl_idx(mu + 1, nu + 1, lambda + 1, sigma + 1);
            //             let mu_lambda_nu_sigma =
            //                 calc_ijkl_idx(mu + 1, lambda + 1, nu + 1, sigma + 1);

            //             F_matr_val += D_matr[(lambda, sigma)]
            //                 * (2.0 * self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_nu_lambda_sigma]
            //                     - self.mol.wfn_total.HF_Matrices.ERI_arr1[mu_lambda_nu_sigma]);
            //         }
            //     }
            //     let mut F_matr_guard = F_matr_mutex.lock().unwrap();
            //     F_matr_guard[(*mu,*nu)] += F_matr_val;
            //     F_matr_guard[(*nu,*mu)] += F_matr_val;
            // });
            // let F_matr = F_matr_mutex.into_inner().unwrap().to_owned();

            // println!("After: {:>8.5}", &F_matr);

            for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                    F_matr[(mu, nu)] = self.mol.wfn_total.HF_Matrices.H_core_matr[(mu, nu)];
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
        self.calc_1e_ints(is_debug);

        //* Step 3: Calc the 2e-ints (V_ee) */
        self.calc_2e_ints(is_debug);

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

            let idx_pairs: Vec<(usize, usize)> = (0..n)
                .flat_map(|mu| (0..mu).map(move |nu| (mu, nu)))
                .collect();

            for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
                    F_matr[(mu, nu)] = self.mol.wfn_total.HF_Matrices.H_core_matr[(mu, nu)];
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

    fn calc_1e_ints(&mut self, is_debug: bool) {
        // * Build matrices for SCF */
        self.mol.wfn_total.HF_Matrices.V_nn_val =
            calc_V_nn_val(&self.mol.geom_obj.geom_matr, &self.mol.geom_obj.Z_vals);

        //* Step 2.1: Calculate the overlap matrix S
        self.mol.wfn_total.HF_Matrices.S_matr = Array2::<f64>::zeros((
            self.mol.wfn_total.basis_set_total.no_cgtos,
            self.mol.wfn_total.basis_set_total.no_cgtos,
        ));
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
        self.mol.wfn_total.HF_Matrices.T_matr = Array2::<f64>::zeros((
            self.mol.wfn_total.basis_set_total.no_cgtos,
            self.mol.wfn_total.basis_set_total.no_cgtos,
        ));

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
        self.mol.wfn_total.HF_Matrices.V_ne_matr = Array2::<f64>::zeros((
            self.mol.wfn_total.basis_set_total.no_cgtos,
            self.mol.wfn_total.basis_set_total.no_cgtos,
        ));

        for i in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
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
        self.mol.wfn_total.HF_Matrices.H_core_matr = Array2::<f64>::zeros((
            self.mol.wfn_total.basis_set_total.no_cgtos,
            self.mol.wfn_total.basis_set_total.no_cgtos,
        ));
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

    fn calc_2e_ints(&mut self, is_debug: bool) {
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

    pub fn MP2(&mut self, is_debug: bool, basis_set_name: &str) {
        // * only rerun HF, if it has not been run before
        if self.mol.wfn_total.HF_Matrices.ERI_arr1.is_empty() {
            Self::RHF_ser(self, is_debug, basis_set_name);
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
            Self::calc_2e_ints(self, false);
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

#[inline]
pub fn calc_cmp_idx(idx1: usize, idx2: usize) -> usize {
    (idx1 * (idx1 + 1)) / 2 + idx2
}
