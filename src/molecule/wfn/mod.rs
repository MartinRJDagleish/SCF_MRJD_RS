use boys;
use ndarray::prelude::*;
use ndarray_linalg::Scalar;
use std::f64::consts::PI;

pub mod basisset;
pub mod integrals;
pub mod scf;

#[derive(Clone, Debug)]
pub struct PGTO {
    pub alpha: f64,
    pub cgto_coeff: f64,
    pub gauss_center_pos: Array1<f64>,
    pub ang_mom_vec: Array1<i32>,
    pub norm_const: f64,
}

#[derive(Debug)]
pub struct CGTO {
    pub pgto_vec: Vec<PGTO>,
    pub no_pgtos: usize,
}

#[derive(Debug, Default)]
pub struct BasisSetTotal {
    pub basis_set_cgtos: Vec<CGTO>,
    pub no_cgtos: usize,
    pub no_occ_orb: usize,
    pub center_charge: Array1<f64>,
    pub dipole_moment_total: Array1<f64>,
}

#[derive(Debug, Default)]
pub struct HF_Matrices {
    pub S_matr: Array2<f64>,
    pub T_matr: Array2<f64>,
    pub V_ne_matr: Array2<f64>,
    pub H_core_matr: Array2<f64>,
    pub ERI_arr1: Array1<f64>,
    pub ERI_tensor: Array4<f64>,
    pub V_nn_val: f64,
    pub Mu_tensor: Array3<f64>,
}

#[derive(Debug, Default)]
pub struct WfnTotal {
    pub basis_set_total: BasisSetTotal,
    pub HF_Matrices: HF_Matrices,
}

// impl Iterator for CGTO {
//     type Item = PGTO;

//     fn next(&mut self) -> Option<Self::Item> {
//         self.pgto_vec.pop()
//     }
// }

#[allow(non_snake_case)]
impl PGTO {
    pub fn new(
        alpha: f64,
        cgto_coeff: f64,
        gauss_center_pos: Array1<f64>,
        ang_mom_vec: Array1<i32>,
    ) -> Self {
        let norm_const: f64 = Self::calc_cart_norm_const_pgto(&alpha, &ang_mom_vec);

        PGTO {
            alpha,
            cgto_coeff,
            gauss_center_pos,
            ang_mom_vec,
            norm_const,
        }
    }

    fn calc_cart_norm_const_pgto(alpha: &f64, ang_mom_vec: &Array1<i32>) -> f64 {
        let numerator: f64 = (2.0 * alpha / PI).powf(1.5) * (4.0 * alpha).powi(ang_mom_vec.sum());
        let denom: i32 = ang_mom_vec.mapv(|x| double_factorial(2 * x - 1)).product();

        (numerator / denom as f64).sqrt()
    }
}

impl CGTO {
    pub fn new(pgto_vec: Vec<PGTO>) -> Self {
        let pgto_vec = pgto_vec;
        let no_pgtos: usize = pgto_vec.len();

        CGTO { pgto_vec, no_pgtos }
    }

    fn calc_cart_norm_const_cgto(&mut self) {
        let mut norm_const_cgto = 0.0_f64;

        let L = self.pgto_vec[0].ang_mom_vec.sum();
        let pi_factor = PI.powf(1.5) / (2.0.powi(L))
            * (self.pgto_vec[0]
                .ang_mom_vec
                .map(|x| double_factorial(2 * x - 1)))
            .product() as f64;

        for pgto1 in &self.pgto_vec {
            for pgto2 in &self.pgto_vec {
                norm_const_cgto +=
                    pgto1.cgto_coeff * pgto2.cgto_coeff * pgto1.norm_const * pgto2.norm_const
                        / (pgto1.alpha + pgto2.alpha).powf(L as f64 + 1.5);
            }
        }

        norm_const_cgto *= pi_factor;
        norm_const_cgto = norm_const_cgto.powf(-0.5);

        for pgto in self.pgto_vec.iter_mut() {
            pgto.norm_const *= norm_const_cgto;
        }
    }

    pub fn update_no_pgtos(&mut self) {
        self.no_pgtos = self.pgto_vec.len();
    }
}

impl BasisSetTotal {
    fn new() -> Self {
        let basis_set_cgtos: Vec<CGTO> = Vec::new();
        let no_cgtos: usize = 0;
        let no_occ_orb: usize = 0;
        let center_charge = Array1::zeros(3);
        let dipole_moment_total = Array1::zeros(3);

        BasisSetTotal {
            basis_set_cgtos,
            no_cgtos,
            no_occ_orb,
            center_charge,
            dipole_moment_total,
        }
    }

    pub fn update_no_cgtos(&mut self) {
        self.no_cgtos = self.basis_set_cgtos.len();
    }
}

impl WfnTotal {
    pub fn new() -> Self {
        let HFMatrices = HF_Matrices::default();
        let basis_set_total = BasisSetTotal::new();

        WfnTotal {
            HF_Matrices: HFMatrices,
            basis_set_total,
        }
    }

    pub fn calc_S_matr_l_eq_0(&self) -> Array2<f64> {
        let no_basis_funcs: usize = self.basis_set_total.no_cgtos;
        let mut S_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));

        for i in 0..no_basis_funcs {
            for j in 0..=i {
                let no_pgto_i: usize = self.basis_set_total.basis_set_cgtos[i].no_pgtos;
                let no_pgto_j: usize = self.basis_set_total.basis_set_cgtos[j].no_pgtos;
                //* Skips over diagonal elements -> reduces computation time
                //* COMMENT OUT IF YOU WANT TO CALCULATE DIAGONAL ELEMENTS
                if i == j {
                    S_matr[(i, j)] = 1.0;
                    continue;
                }
                for k in 0..no_pgto_i {
                    for l in 0..no_pgto_j {
                        let norm_const: f64 = self.basis_set_total.basis_set_cgtos[i].pgto_vec[k]
                            .norm_const
                            * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].norm_const; //* This is N
                        let sum_alphas_recip: f64 =
                            (self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].alpha
                                + self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].alpha)
                                .recip(); //* This is p^-1
                        let prod_alphas_div_sum: f64 =
                            self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].alpha
                                * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].alpha
                                * sum_alphas_recip; //* This is q
                        let diff_pos: Array1<f64> = &self.basis_set_total.basis_set_cgtos[i]
                            .pgto_vec[k]
                            .gauss_center_pos
                            - &self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].gauss_center_pos; //* This is Q
                        let diff_pos_squ: f64 = diff_pos.dot(&diff_pos); //* This is Q^2

                        S_matr[(i, j)] += norm_const
                            * self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].cgto_coeff
                            * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].cgto_coeff
                            * (PI * sum_alphas_recip).powf(1.5)
                            * (-prod_alphas_div_sum * diff_pos_squ).exp();
                    }
                }
                S_matr[(j, i)] = S_matr[(i, j)];
            }
        }

        S_matr
    }

    pub fn calc_T_matr_l_eq_0(&self) -> Array2<f64> {
        let no_basis_funcs: usize = self.basis_set_total.no_cgtos;
        let mut T_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));

        for i in 0..no_basis_funcs {
            for j in 0..=i {
                let no_prim_gauss_i: usize = self.basis_set_total.basis_set_cgtos[i].no_pgtos;
                let no_prim_gauss_j: usize = self.basis_set_total.basis_set_cgtos[j].no_pgtos;

                for k in 0..no_prim_gauss_i {
                    for l in 0..no_prim_gauss_j {
                        let norm_const: f64 = self.basis_set_total.basis_set_cgtos[i].pgto_vec[k]
                            .norm_const
                            * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].norm_const; //* This is N
                        let prod_coeffs: f64 = self.basis_set_total.basis_set_cgtos[i].pgto_vec[k]
                            .cgto_coeff
                            * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].cgto_coeff; //* This is c_i * c_j

                        let sum_alphas_recip: f64 =
                            (self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].alpha
                                + self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].alpha)
                                .recip(); //* This is p^-1
                        let prod_alphas_div_sum: f64 =
                            self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].alpha
                                * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].alpha
                                * sum_alphas_recip; //* This is q
                        let diff_pos: Array1<f64> = &self.basis_set_total.basis_set_cgtos[i]
                            .pgto_vec[k]
                            .gauss_center_pos
                            - &self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].gauss_center_pos; //* This is Q
                        let diff_pos_squ: f64 = diff_pos.dot(&diff_pos); //* This is Q^2

                        let new_center_pos: Array1<f64> = &self.basis_set_total.basis_set_cgtos[i]
                            .pgto_vec[k]
                            .gauss_center_pos
                            * self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].alpha
                            + &self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].gauss_center_pos
                                * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].alpha; //* This is P
                        let new_center_pos: Array1<f64> = new_center_pos * sum_alphas_recip; //* This is Pp
                        let new_center_pos_diff_2nd: Array1<f64> = &new_center_pos
                            - &self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].gauss_center_pos; //* This is PG = Pp - Pi
                        let new_center_pos_diff_2nd_elem_squ: Array1<f64> =
                            new_center_pos_diff_2nd.mapv(|x| x.powi(2)); //* This is PG^2

                        let mini_S: f64 = norm_const
                            * prod_coeffs
                            * (PI * sum_alphas_recip).powf(1.5)
                            * (-prod_alphas_div_sum * diff_pos_squ).exp();

                        T_matr[(i, j)] += 3.0
                            * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].alpha
                            * mini_S;
                        for cart_coord in 0..3 {
                            T_matr[(i, j)] -= 2.0
                                * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l]
                                    .alpha
                                    .powi(2)
                                * mini_S
                                * (new_center_pos_diff_2nd_elem_squ[cart_coord]
                                    + 0.5 * sum_alphas_recip);
                        }
                    }
                }
                T_matr[(j, i)] = T_matr[(i, j)];
            }
        }

        T_matr
    }

    pub fn calc_V_ne_matr_l_eq_0(&self) -> Array2<f64> {
        // let no_atoms: usize = self.ContrGauss_vec.len(); //TODO: fix this for right code
        // ↓ This was a quick fix!
        // let no_atoms: usize = 2; //TODO: fix this for right code
        // let no_contr_gauss: usize = Self.no_of_contr_gauss(&self.ContrGauss_vec);
        let no_atoms: usize = 2; //TODO: fix this for right code
        let no_basis_funcs: usize = self.basis_set_total.basis_set_cgtos.len();
        //* QUICK FIX:
        let Z_val_list = [1, 1]; //TODO: change this to be read from input file
        let mut V_ne_matr: Array2<f64> = Array2::zeros((
            self.basis_set_total.basis_set_cgtos.len(),
            self.basis_set_total.basis_set_cgtos.len(),
        ));

        for i in 0..no_basis_funcs {
            for j in 0..=i {
                let no_prim_gauss_i: usize = self.basis_set_total.basis_set_cgtos[i].no_pgtos;
                let no_prim_gauss_j: usize = self.basis_set_total.basis_set_cgtos[j].no_pgtos;

                for k in 0..no_prim_gauss_i {
                    for l in 0..no_prim_gauss_j {
                        let norm_const: f64 = self.basis_set_total.basis_set_cgtos[i].pgto_vec[k]
                            .norm_const
                            * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].norm_const; //* This is N
                        let prod_coeffs: f64 = self.basis_set_total.basis_set_cgtos[i].pgto_vec[k]
                            .cgto_coeff
                            * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].cgto_coeff; //* This is c_i * c_j
                        let sum_alphas = self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].alpha
                            + self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].alpha; //* This is p
                        let sum_alphas_recip: f64 =
                            (self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].alpha
                                + self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].alpha)
                                .recip(); //* This is p^-1
                        let prod_alphas_div_sum: f64 =
                            self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].alpha
                                * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].alpha
                                * sum_alphas_recip; //* This is q
                        let diff_pos: Array1<f64> = &self.basis_set_total.basis_set_cgtos[i]
                            .pgto_vec[k]
                            .gauss_center_pos
                            - &self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].gauss_center_pos; //* This is Q
                        let diff_pos_squ: f64 = diff_pos.dot(&diff_pos); //* This is Q^2

                        let mut new_center_pos: Array1<f64> =
                            &self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].gauss_center_pos
                                * self.basis_set_total.basis_set_cgtos[i].pgto_vec[k].alpha
                                + &self.basis_set_total.basis_set_cgtos[j].pgto_vec[l]
                                    .gauss_center_pos
                                    * self.basis_set_total.basis_set_cgtos[j].pgto_vec[l].alpha; //* This is P
                        new_center_pos *= sum_alphas_recip; //* This is Pp

                        (0..no_atoms).for_each(|atom| {
                            let diff_pos_atom: Array1<f64> = &new_center_pos
                                - &self.basis_set_total.basis_set_cgtos[atom].pgto_vec[0]
                                    .gauss_center_pos; //* This is PA
                                                       //TODO: ↑ this is not correct -> only if one CGTO per atom
                                                       //TODO: -> fix this for right code
                                                       //TODO: this is why STO-3G is working, but not 6-311G
                                                       //* the atom index is not correct
                            let diff_pos_atom_squ: f64 = diff_pos_atom.dot(&diff_pos_atom); //* This is PA^2
                            V_ne_matr[(i, j)] += norm_const
                                * (-Z_val_list[atom] as f64)
                                * (2.0 * PI * sum_alphas_recip)
                                * prod_coeffs
                                * (-prod_alphas_div_sum * diff_pos_squ).exp()
                                * boys::micb25::boys(0, sum_alphas * diff_pos_atom_squ);
                        });
                    }
                }
                V_ne_matr[(j, i)] = V_ne_matr[(i, j)];
            }
        }

        V_ne_matr
    }

    pub fn calc_V_ee_matr_l_eq_0(&self) -> Array4<f64> {
        let no_basis_funcs: usize = self.basis_set_total.basis_set_cgtos.len();
        let mut V_ee_matr: Array4<f64> = Array4::zeros((
            self.basis_set_total.basis_set_cgtos.len(),
            self.basis_set_total.basis_set_cgtos.len(),
            self.basis_set_total.basis_set_cgtos.len(),
            self.basis_set_total.basis_set_cgtos.len(),
        ));
        for i in 0..no_basis_funcs {
            for j in 0..no_basis_funcs {
                for k in 0..no_basis_funcs {
                    for l in 0..no_basis_funcs {
                        let no_prim_gauss_i: usize =
                            self.basis_set_total.basis_set_cgtos[i].no_pgtos;
                        let no_prim_gauss_j: usize =
                            self.basis_set_total.basis_set_cgtos[j].no_pgtos;
                        let no_prim_gauss_k: usize =
                            self.basis_set_total.basis_set_cgtos[k].no_pgtos;
                        let no_prim_gauss_l: usize =
                            self.basis_set_total.basis_set_cgtos[l].no_pgtos;

                        for m in 0..no_prim_gauss_i {
                            for n in 0..no_prim_gauss_j {
                                for o in 0..no_prim_gauss_k {
                                    for p in 0..no_prim_gauss_l {
                                        let norm_const: f64 = self.basis_set_total.basis_set_cgtos
                                            [i]
                                            .pgto_vec[m]
                                            .norm_const
                                            * self.basis_set_total.basis_set_cgtos[j].pgto_vec[n]
                                                .norm_const
                                            * self.basis_set_total.basis_set_cgtos[k].pgto_vec[o]
                                                .norm_const
                                            * self.basis_set_total.basis_set_cgtos[l].pgto_vec[p]
                                                .norm_const; //* This is N
                                        let prod_coeffs: f64 = self.basis_set_total.basis_set_cgtos
                                            [i]
                                            .pgto_vec[m]
                                            .cgto_coeff
                                            * self.basis_set_total.basis_set_cgtos[j].pgto_vec[n]
                                                .cgto_coeff
                                            * self.basis_set_total.basis_set_cgtos[k].pgto_vec[o]
                                                .cgto_coeff
                                            * self.basis_set_total.basis_set_cgtos[l].pgto_vec[p]
                                                .cgto_coeff; //* This is c_i * c_j

                                        let sum_alphas_ij = self.basis_set_total.basis_set_cgtos[i]
                                            .pgto_vec[m]
                                            .alpha
                                            + self.basis_set_total.basis_set_cgtos[j].pgto_vec[n]
                                                .alpha; //* This is p_ij
                                        let sum_alphas_recip_ij: f64 = sum_alphas_ij.recip(); //* This is p_ij^-1
                                        let sum_alphas_kl = self.basis_set_total.basis_set_cgtos[k]
                                            .pgto_vec[o]
                                            .alpha
                                            + self.basis_set_total.basis_set_cgtos[l].pgto_vec[p]
                                                .alpha; //* This is p_kl
                                        let sum_alphas_recip_kl: f64 = sum_alphas_kl.recip(); //* This is p_kl^-1

                                        let prod_alphas_div_sum_ij: f64 =
                                            self.basis_set_total.basis_set_cgtos[i].pgto_vec[m]
                                                .alpha
                                                * self.basis_set_total.basis_set_cgtos[j].pgto_vec
                                                    [n]
                                                    .alpha
                                                * sum_alphas_recip_ij; //* This is q_ij
                                        let prod_alphas_div_sum_kl: f64 =
                                            self.basis_set_total.basis_set_cgtos[k].pgto_vec[o]
                                                .alpha
                                                * self.basis_set_total.basis_set_cgtos[l].pgto_vec
                                                    [p]
                                                    .alpha
                                                * sum_alphas_recip_kl; //* This is q_kl

                                        let mut new_center_pos_ij: Array1<f64> = &self
                                            .basis_set_total
                                            .basis_set_cgtos[i]
                                            .pgto_vec[m]
                                            .gauss_center_pos
                                            * self.basis_set_total.basis_set_cgtos[i].pgto_vec[m]
                                                .alpha
                                            + &self.basis_set_total.basis_set_cgtos[j].pgto_vec[n]
                                                .gauss_center_pos
                                                * self.basis_set_total.basis_set_cgtos[j].pgto_vec
                                                    [n]
                                                    .alpha; //* This is P_ij
                                        new_center_pos_ij *= sum_alphas_recip_ij; //* This is Pp_ij

                                        let mut new_center_pos_kl: Array1<f64> = &self
                                            .basis_set_total
                                            .basis_set_cgtos[k]
                                            .pgto_vec[o]
                                            .gauss_center_pos
                                            * self.basis_set_total.basis_set_cgtos[k].pgto_vec[o]
                                                .alpha
                                            + &self.basis_set_total.basis_set_cgtos[l].pgto_vec[p]
                                                .gauss_center_pos
                                                * self.basis_set_total.basis_set_cgtos[l].pgto_vec
                                                    [p]
                                                    .alpha; //* This is P_kl
                                        new_center_pos_kl *= sum_alphas_recip_kl; //* This is Pp_kl

                                        let diff_new_center_pos_ijkl: Array1<f64> =
                                            &new_center_pos_ij - &new_center_pos_kl; //* This is Pp_ijkl = Pp_ij - Pp_kl
                                        let diff_new_center_pos_ijkl_sq: f64 =
                                            diff_new_center_pos_ijkl.dot(&diff_new_center_pos_ijkl); //* This is Pp_ijkl^2 = (Pp_ij - Pp_kl)^2

                                        let sum_alphas_ijkl_div_prod_alphas_ijkl: f64 =
                                            (sum_alphas_ij + sum_alphas_kl)
                                                * (sum_alphas_ij * sum_alphas_kl).recip(); //* This is (p_kl + p_ij) / (p_ij * p_kl)

                                        let diff_pos_ij: Array1<f64> =
                                            &self.basis_set_total.basis_set_cgtos[i].pgto_vec[m]
                                                .gauss_center_pos
                                                - &self.basis_set_total.basis_set_cgtos[j].pgto_vec
                                                    [n]
                                                    .gauss_center_pos; //* This is Q_ij
                                        let diff_pos_kl: Array1<f64> =
                                            &self.basis_set_total.basis_set_cgtos[k].pgto_vec[o]
                                                .gauss_center_pos
                                                - &self.basis_set_total.basis_set_cgtos[l].pgto_vec
                                                    [p]
                                                    .gauss_center_pos; //* This is Q_kl
                                        let diff_pos_ij_squ: f64 = diff_pos_ij.dot(&diff_pos_ij); //* This is Q_ij^2
                                        let diff_pos_kl_squ: f64 = diff_pos_kl.dot(&diff_pos_kl); //* This is Q_kl^2

                                        V_ee_matr[(i, j, k, l)] += norm_const
                                            * prod_coeffs
                                            * 2.0
                                            * PI.powi(2)
                                            * (sum_alphas_ij * sum_alphas_kl).recip()
                                            * (PI * (sum_alphas_ij + sum_alphas_kl).recip()).sqrt()
                                            * (-prod_alphas_div_sum_ij * diff_pos_ij_squ).exp()
                                            * (-prod_alphas_div_sum_kl * diff_pos_kl_squ).exp()
                                            * boys::micb25::boys(
                                                0,
                                                diff_new_center_pos_ijkl_sq
                                                    * sum_alphas_ijkl_div_prod_alphas_ijkl.recip(),
                                            );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        V_ee_matr
    }
}

// pub struct Cache<T>
// where
//     T: Fn(i32) -> i32,
// {
//     calculation: T,
//     internal: HashMap<i32, i32>,
// }

// impl<T> Cache<T>
// where
//     T: Fn(i32) -> i32,
// {
//     pub fn new(calculation: T) -> Cache<T> {
//         Cache {
//             calculation,
//             internal: HashMap::new(),
//         }
//     }

//     pub fn value(&mut self, n: i32) -> i32 {
//         if let Some(&result) = self.internal.get(&n) {
//             return result;
//         }

//         let result = (self.calculation)(n);
//         self.internal.insert(n, result);
//         result
//     }
// }

// pub fn double_factorial(n: i32) -> i32 {
//     let mut cache = Cache::new(|n| {
//         match n {
//             -1 => 1,
//             0 => 1,
//             1 => 1,
//             _ => n * double_factorial(n - 2),
//         }
//     });

//     cache.value(n)
// }

pub fn double_factorial(n: i32) -> i32 {
    match n {
        -1 => 1,
        0 => 1,
        1 => 1,
        _ => n * double_factorial(n - 2),
    }
}
