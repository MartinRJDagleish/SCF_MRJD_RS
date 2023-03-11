use ndarray::prelude::*;

use crate::molecule::{Molecule, wfn::basisset::{create_basis_set_total, parse_basis_set_file_gaussian}};

#[derive(Debug)]
struct SCF {
    pub mol: Molecule,
    pub E_scf_final: f64,
    pub E_tot_final: f64,
}

impl SCF {
    pub fn new(mut mol: Molecule) -> Self {
        SCF {
            mol,
            E_scf_final: 0.0,
            E_tot_final: 0.0,
        }
    }

    pub fn RHF(&self) {
        todo!();
        // let mut mol: Molecule = Molecule::new("inp/Project3_1/STO-3G/h2o_v2.xyz", 0);

        //* Create basis for mol object */
        let basis_set_name = "STO-3G";
        self.mol.wfn_total.basis_set_total = create_basis_set_total(
            parse_basis_set_file_gaussian(basis_set_name),
            self.mol.geom_obj.geom_matr.clone(),
            &self.mol.geom_obj.Z_vals,
        );

    //     // * Build matrices for SCF */

    //     self.mol.wfn_total.HFMatrices.V_nn_val =
    //         calc_V_nn_val(&self.mol.geom_obj.geom_matr, &self.mol.geom_obj.Z_vals);

    //     //* Step 2.1: Calculate the overlap matrix S
    //     self.mol.wfn_total.HFMatrices.S_matr = Array2::<f64>::zeros((
    //         self.mol.wfn_total.basis_set_total.no_cgtos,
    //         self.mol.wfn_total.basis_set_total.no_cgtos,
    //     ));
    //     for i in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //         for j in 0..=i {
    //             if i == j {
    //                 self.mol.wfn_total.HFMatrices.S_matr[(i, j)] = 1.0;
    //                 continue;
    //             } else {
    //                 self.mol.wfn_total.HFMatrices.S_matr[(i, j)] = calc_overlap_int_cgto(
    //                     &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
    //                     &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
    //                 );
    //                 self.mol.wfn_total.HFMatrices.S_matr[(j, i)] =
    //                     self.mol.wfn_total.HFMatrices.S_matr[(i, j)];
    //             }
    //         }
    //     }

    //     println!(
    //         "Overlap matrix S:\n{:1.5}\n",
    //         &self.mol.wfn_total.HFMatrices.S_matr
    //     );

    //     //* Step 2.2: Calculate the kinetic energy matrix T
    //     self.mol.wfn_total.HFMatrices.T_matr = Array2::<f64>::zeros((
    //         self.mol.wfn_total.basis_set_total.no_cgtos,
    //         self.mol.wfn_total.basis_set_total.no_cgtos,
    //     ));

    //     for i in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //         for j in 0..=i {
    //             self.mol.wfn_total.HFMatrices.T_matr[(i, j)] = calc_kin_energy_int_cgto(
    //                 &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
    //                 &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
    //             );
    //             self.mol.wfn_total.HFMatrices.T_matr[(j, i)] = self.mol.wfn_total.HFMatrices.T_matr[(i, j)];
    //         }
    //     }
    //     println!(
    //         "Kinetic energy matrix T:\n{:1.5}\n",
    //         &self.mol.wfn_total.HFMatrices.T_matr
    //     );

    //     //* Step 2.3: Calculate the nuclear attraction matrix V_ne
    //     self.mol.wfn_total.HFMatrices.V_ne_matr = Array2::<f64>::zeros((
    //         self.mol.wfn_total.basis_set_total.no_cgtos,
    //         self.mol.wfn_total.basis_set_total.no_cgtos,
    //     ));

    //     for i in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //         for j in 0..=i {
    //             for (idx, atom_pos) in mol
    //                 .geom_obj
    //                 .geom_matr
    //                 .axis_iter(ndarray::Axis(0))
    //                 .enumerate()
    //             {
    //                 self.mol.wfn_total.HFMatrices.V_ne_matr[(i, j)] += (-self.mol.geom_obj.Z_vals[idx]
    //                     as f64)
    //                     * calc_nuc_attr_int_cgto(
    //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[i],
    //                         &self.mol.wfn_total.basis_set_total.basis_set_cgtos[j],
    //                         &atom_pos.to_owned(),
    //                     );
    //             }
    //             self.mol.wfn_total.HFMatrices.V_ne_matr[(j, i)] =
    //                 self.mol.wfn_total.HFMatrices.V_ne_matr[(i, j)];
    //         }
    //     }

    //     println!(
    //         "Nuclear attraction matrix V_ne:\n{:1.5}\n",
    //         &self.mol.wfn_total.HFMatrices.V_ne_matr
    //     );
    //     //* Step 2.4: Form the core Hamiltonian matrix H_core
    //     self.mol.wfn_total.HFMatrices.H_core_matr = Array2::<f64>::zeros((
    //         self.mol.wfn_total.basis_set_total.no_cgtos,
    //         self.mol.wfn_total.basis_set_total.no_cgtos,
    //     ));
    //     self.mol.wfn_total.HFMatrices.H_core_matr =
    //         &self.mol.wfn_total.HFMatrices.T_matr + &self.mol.wfn_total.HFMatrices.V_ne_matr;

    //     println!(
    //         "Core Hamiltonian matrix H_core:\n{:1.5}\n",
    //         &self.mol.wfn_total.HFMatrices.H_core_matr
    //     );

    //     // //* Step 3: Calculate the 2-electron integrals (ERI)

    // println!("Electron-electron repulsion integrals (V_ee / ERI matrix):");
    // mol.wfn_total.HFMatrices.ERI_tensor = Array4::<f64>::zeros((
    //     mol.wfn_total.basis_set_total.no_cgtos,
    //     mol.wfn_total.basis_set_total.no_cgtos,
    //     mol.wfn_total.basis_set_total.no_cgtos,
    //     mol.wfn_total.basis_set_total.no_cgtos,
    // ));
    // for i in 0..mol.wfn_total.basis_set_total.no_cgtos {
    //     for j in 0..=i {
    //         let ij = calc_cmp_idx(i, j);
    //         for k in 0..mol.wfn_total.basis_set_total.no_cgtos {
    //             for l in 0..=k {
    //                 let kl = calc_cmp_idx(k, l);
    //                 if ij >= kl {
    //                     let ERI_val = calc_elec_elec_repul_cgto(
    //                             &mol.wfn_total.basis_set_total.basis_set_cgtos[i],
    //                             &mol.wfn_total.basis_set_total.basis_set_cgtos[j],
    //                             &mol.wfn_total.basis_set_total.basis_set_cgtos[k],
    //                             &mol.wfn_total.basis_set_total.basis_set_cgtos[l],
    //                         );
    //                     mol.wfn_total.HFMatrices.ERI_tensor[(i, j, k, l)] = ERI_val;
    //                     mol.wfn_total.HFMatrices.ERI_tensor[(j, i, k, l)] = ERI_val;
    //                     mol.wfn_total.HFMatrices.ERI_tensor[(i, j, l, k)] = ERI_val;
    //                     mol.wfn_total.HFMatrices.ERI_tensor[(j, i, l, k)] = ERI_val;
    //                     mol.wfn_total.HFMatrices.ERI_tensor[(k, l, i, j)] = ERI_val;
    //                     mol.wfn_total.HFMatrices.ERI_tensor[(l, k, i, j)] = ERI_val;
    //                     mol.wfn_total.HFMatrices.ERI_tensor[(k, l, j, i)] = ERI_val;
    //                     mol.wfn_total.HFMatrices.ERI_tensor[(l, k, j, i)] = ERI_val;
    //                 }
    //             }
    //         }
    //     }
    // }

    //     println!("{:^5.6}\n", &self.mol.wfn_total.HFMatrices.ERI_tensor);

    //     //* Step 4: Build the orthogonalization matrix S^(-1/2)
    //     let S_matr_sqrt: Array2<f64> = mol
    //         .wfn_total
    //         .HFMatrices
    //         .S_matr
    //         .ssqrt(ndarray_linalg::UPLO::Upper)
    //         .unwrap();
    //     let S_matr_sqrt_inv: Array2<f64> = S_matr_sqrt.inv().unwrap();

    //     println!("S^(-1/2):\n{:^5.6}\n", &S_matr_sqrt_inv);

    //     //* Step 5: Form the initial guess density matrix D_0
    //     //* Step 5.1: Form the initial guess Fock matrix F_0 in the AO basis
    //     let F_matr_0_pr: Array2<f64> = S_matr_sqrt_inv
    //         .clone()
    //         .reversed_axes()
    //         .dot(&self.mol.wfn_total.HFMatrices.H_core_matr)
    //         .dot(&S_matr_sqrt_inv.clone());

    //     println!("F_matr_0_pr:\n{:^5.6}\n", &F_matr_0_pr);

    //     //* Step 5.2: Get the coefficients of the initial guess Fock matrix F_0 in the MO basis
    //     let (orb_energy_arr, C_matr_MO_basis) =
    //         F_matr_0_pr.eigh(ndarray_linalg::UPLO::Upper).unwrap();
    //     println!("C_matr_MO_basis:\n{:^5.6}\n", &C_matr_MO_basis);
    //     let C_matr_AO_basis: Array2<f64> = S_matr_sqrt_inv.dot(&C_matr_MO_basis);
    //     println!("C_matr_AO_basis:\n{:^5.6}\n", &C_matr_AO_basis);

    //     //* Step 5.3: Form the initial guess density matrix D_0
    //     // ! THIS WORKS ONLY FOR CLOSED SHELL SYSTEMS (RHF)
    //     // TODO: CHANGE THIS TO WORK FOR OPEN SHELL SYSTEMS (UHF)
    //     /*
    //     ? How do I get the correct number of occupied orbitals, when I have multiple CGTOs per orbital per atom?
    //     */
    //     let no_occ_orb: usize = 5; // * 1 CGTO per orbital per atom
    //                                // let no_occ_orb: usize = 21; // * 1 CGTO per orbital per atom

    //     let mut D_matr: Array2<f64> = Array2::<f64>::zeros((
    //         self.mol.wfn_total.basis_set_total.no_cgtos,
    //         self.mol.wfn_total.basis_set_total.no_cgtos,
    //     ));

    //     for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //         for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //             for m in 0..no_occ_orb {
    //                 D_matr[(mu, nu)] += C_matr_AO_basis[(mu, m)] * C_matr_AO_basis[(nu, m)];
    //             }
    //         }
    //     }
    //     println!("Initial density matrix:\n{:^.5}\n", &D_matr);

    //     let mut E_scf: f64 = 0.0;
    //     let mut E_scf_vec: Vec<f64> = Vec::new();
    //     let mut E_tot_vec: Vec<f64> = Vec::new();
    //     let mut rms_d_vec: Vec<f64> = Vec::new();

    //     //* Here the Fock matrix is guessed to be the core Hamiltonian matrix
    //     //* That's why the initial SCF energy differs from the other SCF energy calcs
    //     for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //         for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //             E_scf += D_matr[(mu, nu)] * 2.0 * (self.mol.wfn_total.HFMatrices.H_core_matr[(mu, nu)]);
    //         }
    //     }

    //     E_scf_vec.push(E_scf);
    //     let mut E_tot = E_scf + self.mol.wfn_total.HFMatrices.V_nn_val;
    //     E_tot_vec.push(E_tot);

    //     //* Step 7: Iterate the SCF procedure until convergence
    //     let scf_maxiter: usize = 50;
    //     // ! THE SCF ITERATIONS START HERE
    //     for scf_iter in 0..scf_maxiter {
    //         //* Step 6: Form the Fock matrix F in the AO basis
    //         let mut F_matr: Array2<f64> = Array2::<f64>::zeros((
    //             self.mol.wfn_total.basis_set_total.no_cgtos,
    //             self.mol.wfn_total.basis_set_total.no_cgtos,
    //         ));

    //         for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //             for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //                 F_matr[(mu, nu)] = self.mol.wfn_total.HFMatrices.H_core_matr[(mu, nu)];
    //                 for lambda in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //                     for sigma in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //                         F_matr[(mu, nu)] += D_matr[(lambda, sigma)]
    //                             * (2.0
    //                                 * self.mol.wfn_total.HFMatrices.ERI_tensor[(mu, nu, lambda, sigma)]
    //                                 - self.mol.wfn_total.HFMatrices.ERI_tensor[(mu, lambda, nu, sigma)]);
    //                     }
    //                 }
    //             }
    //         }

    //         //* Step 7: Form the Fock matrix F in the MO basis
    //         let F_matr_pr: Array2<f64> = S_matr_sqrt_inv
    //             .clone()
    //             .reversed_axes()
    //             .dot(&F_matr)
    //             .dot(&S_matr_sqrt_inv.clone());

    //         //* Step 8: Get the coefficients of the Fock matrix F in the MO basis
    //         let (orb_energy_arr, C_matr_MO_basis) =
    //             F_matr_pr.eigh(ndarray_linalg::UPLO::Upper).unwrap();
    //         let C_matr_AO_basis: Array2<f64> = S_matr_sqrt_inv.dot(&C_matr_MO_basis);
    //         let D_matr_prev: Array2<f64> = D_matr.clone();

    //         for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //             for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //                 D_matr[(mu, nu)] = 0.0;
    //                 for m in 0..no_occ_orb {
    //                     D_matr[(mu, nu)] += C_matr_AO_basis[(mu, m)] * C_matr_AO_basis[(nu, m)];
    //                 }
    //             }
    //         }

    //         E_scf = 0.0;
    //         for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //             for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //                 E_scf += D_matr[(mu, nu)]
    //                     * (self.mol.wfn_total.HFMatrices.H_core_matr[(mu, nu)] + F_matr[(mu, nu)]);
    //             }
    //         }

    //         E_scf_vec.push(E_scf);
    //         E_tot = E_scf + self.mol.wfn_total.HFMatrices.V_nn_val;
    //         E_tot_vec.push(E_tot);

    //         let mut rms_d_val: f64 = 0.0;
    //         for mu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //             for nu in 0..self.mol.wfn_total.basis_set_total.no_cgtos {
    //                 rms_d_val += (D_matr[(mu, nu)] - D_matr_prev[(mu, nu)]).powi(2);
    //             }
    //         }
    //         rms_d_val = rms_d_val.sqrt();
    //         rms_d_vec.push(rms_d_val);

    //         if rms_d_val < 1e-6 {
    //             break;
    //         }
    //     }

    //     let header_scf_str = format!(
    //         "{:^3} {:^16}{:^16}{:^12}",
    //         "Iter", "E_scf", "E_total", "RMS D"
    //     );
    //     println!("{header_scf_str}");

    //     // * First line extra
    //     let line = format!(
    //         "{:>3} {: >16.8}{: >16.8}{: >12.8}",
    //         "0", &E_scf_vec[0], &E_tot_vec[0], " "
    //     );
    //     println!("{line}");

    //     for idx in 1..E_scf_vec.len() {
    //         let line = format!(
    //             "{:>3} {: >16.8}{: >16.8}{: >12.8}",
    //             &idx,
    //             &E_scf_vec[idx],
    //             &E_tot_vec[idx],
    //             &rms_d_vec[idx - 1]
    //         );
    //         println!("{line}");
    //     }
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