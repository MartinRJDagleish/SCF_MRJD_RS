use crate::molecule::wfn::CGTO;
use boys::micb25::boys;
// use boys::exact::boys;
use ndarray::{parallel::prelude::*, prelude::*};
use ndarray_linalg::Norm;
use std::f64::consts::PI;

// use std::sync::atomic::{atomic}

// use crate::molecule::geometry::calc_r_ij_general;

#[inline]
pub fn calc_E_herm_gauss_coeff(
    l1: i32,
    l2: i32,
    no_nodes: i32,
    gauss_dist: f64,
    alpha1: &f64,
    alpha2: &f64,
) -> f64 {
    // Calculate the expansion coefficient for the overlap integral
    // between two contracted Gaussian functions.
    //
    // # Arguments
    // ----------
    // l1 : i32
    //    Angular momentum of the first Gaussian function.
    // l2 : i32
    //   Angular momentum of the second Gaussian function.
    // no_nodes : i32
    //   Number of nodes in Hermite (depends on type of int, e.g. always zero for overlap).
    // gauss_dist : f64
    //   Distance between the two Gaussian functions (from the origin)
    // alpha1 : f64
    //   Exponent of the first Gaussian function.
    // alpha2 : f64
    //   Exponent of the second Gaussian function.
    //

    let p_recip = (alpha1 + alpha2).recip();
    let q = alpha1 * alpha2 * p_recip;
    match (no_nodes, l1, l2) {
        (t, _, _) if t < 0 || t > (l1 + l2) => 0.0,
        (0, 0, 0) => (-q * gauss_dist * gauss_dist).exp(),
        (_, _, 0) => {
            //* decrement index l1
            0.5 * p_recip
                * calc_E_herm_gauss_coeff(l1 - 1, l2, no_nodes - 1, gauss_dist, alpha1, alpha2)
                - (q * gauss_dist / alpha1)
                    * calc_E_herm_gauss_coeff(l1 - 1, l2, no_nodes, gauss_dist, alpha1, alpha2)
                + (no_nodes + 1) as f64
                    * calc_E_herm_gauss_coeff(l1 - 1, l2, no_nodes + 1, gauss_dist, alpha1, alpha2)
        }
        (_, _, _) => {
            //* decrement index l2
            0.5 * p_recip
                * calc_E_herm_gauss_coeff(l1, l2 - 1, no_nodes - 1, gauss_dist, alpha1, alpha2)
                + (q * gauss_dist / alpha2) //* 2 bugs here: 1. alpha1 instead of alpha2 -> fixed; 2. - instead of + -> fixed
                    * calc_E_herm_gauss_coeff(
                        l1,
                        l2 - 1,
                        no_nodes,
                        gauss_dist,
                        alpha1,
                        alpha2,
                    )
                + (no_nodes + 1) as f64
                    * calc_E_herm_gauss_coeff(l1, l2 - 1, no_nodes + 1, gauss_dist, alpha1, alpha2)
        }
    }
}

#[inline]
pub fn calc_overlap_int_pgto(
    alpha1: &f64,
    alpha2: &f64,
    ang_mom_vec1: &Array1<i32>,
    ang_mom_vec2: &Array1<i32>,
    gauss1_center_pos: &Array1<f64>,
    gauss2_center_pos: &Array1<f64>,
) -> f64 {
    // Calculate the overlap integral between two Gaussian functions.
    //
    // # Arguments
    // ----------
    // alpha1 : f64
    //   Exponent of the first Gaussian function.
    // alpha2 : f64
    //   Exponent of the second Gaussian function.
    // ang_mom_vec1 : Array1<i32>
    //   Angular momentum vector of the first Gaussian function.
    // ang_mom_vec2 : Array1<i32>
    //   Angular momentum vector of the second Gaussian function.
    // gauss1_center_pos : Array1<f64>
    //   Position of the first Gaussian function. (Center of the Gaussian)
    // gauss2_center_pos : Array1<f64>
    //   Position of the second Gaussian function. (Center of the Gaussian)
    //
    // # Returns
    // --------
    // overlap_int : f64
    //   The overlap integral between the two Gaussian functions.
    //

    let mut S_cart_total: f64 = 1.0; //* Start with 1, otherwise it will be 0
    for cart_coord in 0..3 {
        S_cart_total *= calc_E_herm_gauss_coeff(
            ang_mom_vec1[cart_coord],
            ang_mom_vec2[cart_coord],
            0,
            gauss1_center_pos[cart_coord] - gauss2_center_pos[cart_coord],
            alpha1,
            alpha2,
        );
    }
    S_cart_total * PI.powf(1.5) * (alpha1 + alpha2).recip().powf(1.5)
}

pub fn calc_overlap_int_cgto(cgto1: &CGTO, cgto2: &CGTO) -> f64 {
    // Calculate the overlap integral between two contracted Gaussian functions.
    //
    // # Arguments
    // ----------
    // ContrGaus1 : ContractedGaussian
    //   The first contracted Gaussian function.
    // ContrGaus2 : ContractedGaussian
    //   The second contracted Gaussian function.
    //
    // # Returns
    // --------
    // overlap_int : f64
    //   The overlap integral between the two contracted Gaussian functions.
    //

    //* Parallel code */
    // * double par_iter with map and sum -> produces overlap_int
    cgto1
        .pgto_vec
        .par_iter()
        .map(|pgto1| {
            cgto2
                .pgto_vec
                .par_iter()
                .map(|pgto2| {
                    pgto1.norm_const
                        * pgto2.norm_const
                        * pgto1.cgto_coeff
                        * pgto2.cgto_coeff
                        * calc_overlap_int_pgto(
                            &pgto1.alpha,
                            &pgto2.alpha,
                            &pgto1.ang_mom_vec,
                            &pgto2.ang_mom_vec,
                            &pgto1.gauss_center_pos,
                            &pgto2.gauss_center_pos,
                        )
                })
                .sum::<f64>()
        })
        .sum::<f64>()

    //* Serial code */
    // let mut overlap_int_val: f64 = 0.0;
    // for pgto1 in cgto1.pgto_vec.iter() {
    //     for pgto2 in cgto2.pgto_vec.iter() {
    //         overlap_int_val += pgto1.norm_const
    //             * pgto2.norm_const
    //             * pgto1.cgto_coeff
    //             * pgto2.cgto_coeff
    //             * calc_overlap_int_prim(
    //                 &pgto1.alpha,
    //                 &pgto2.alpha,
    //                 &pgto1.ang_mom_vec,
    //                 &pgto2.ang_mom_vec,
    //                 &pgto1.gauss_center_pos,
    //                 &pgto2.gauss_center_pos,
    //             );
    //     }
    // }
    // overlap_int_val
}

pub fn calc_kin_energy_int_pgto(
    alpha1: &f64,
    alpha2: &f64,
    ang_mom_vec1: &Array1<i32>,
    ang_mom_vec2: &Array1<i32>,
    gauss1_center_pos: &Array1<f64>,
    gauss2_center_pos: &Array1<f64>,
) -> f64 {
    // Calculate the overlap integral between two Gaussian functions.
    //
    // # Arguments
    // ----------
    // alpha1 : f64
    //   Exponent of the first Gaussian function.
    // alpha2 : f64
    //   Exponent of the second Gaussian function.
    // angular_momentum_vec1 : Array1<i32>
    //   Angular momentum vector of the first Gaussian function.
    // angular_momentum_vec2 : Array1<i32>
    //   Angular momentum vector of the second Gaussian function.
    // gauss1_center_pos : Array1<f64>
    //   Position of the first Gaussian function. (Center of the Gaussian)
    // gauss2_center_pos : Array1<f64>
    //   Position of the second Gaussian function. (Center of the Gaussian)
    //
    // # Returns
    // --------
    // overlap_int : f64
    //   The overlap integral between the two Gaussian functions.
    //

    //* This is the clone in the outside scoop.
    let mut ang_mom_vec2_tmp = ang_mom_vec2.clone();
    let part1: f64 = alpha2
        * (2.0 * ang_mom_vec2_tmp.sum() as f64 + 3.0)
        * calc_overlap_int_pgto(
            alpha1,
            alpha2,
            ang_mom_vec1,
            ang_mom_vec2,
            gauss1_center_pos,
            gauss2_center_pos,
        );

    let mut part2: f64 = 0.0;
    (0..3).for_each(|i| {
        ang_mom_vec2_tmp[i] += 2;
        part2 += calc_overlap_int_pgto(
            alpha1,
            alpha2,
            ang_mom_vec1,
            &ang_mom_vec2_tmp,
            gauss1_center_pos,
            gauss2_center_pos,
        );
        ang_mom_vec2_tmp[i] -= 2;
    });
    part2 *= -2.0 * alpha2.powi(2);

    let mut part3: f64 = 0.0;
    (0..3).for_each(|i| {
        ang_mom_vec2_tmp[i] -= 2;
        part3 += ((ang_mom_vec2_tmp[i] + 1) as f64) //* this is l * (l-1) effectively 
            * ((ang_mom_vec2_tmp[i] + 2) as f64)
            * calc_overlap_int_pgto(
                alpha1,
                alpha2,
                ang_mom_vec1,
                &ang_mom_vec2_tmp,
                gauss1_center_pos,
                gauss2_center_pos,
            );
        ang_mom_vec2_tmp[i] += 2;
    });
    part3 *= -0.5;

    part1 + part2 + part3
}

pub fn calc_kin_energy_int_cgto(cgto1: &CGTO, cgto2: &CGTO) -> f64 {
    // Calculate the kinetic energy integral between two contracted Gaussian functions.
    //
    // # Arguments
    // ----------
    // ContrGaus1 : ContractedGaussian
    //   The first contracted Gaussian function.
    // ContrGaus2 : ContractedGaussian
    //   The second contracted Gaussian function.
    //
    // # Returns
    // --------
    // kinetic_energy_int : f64
    //   The kinetic energy integral between the two contracted Gaussian functions.
    //

    //* Parallel code */
    cgto1
        .pgto_vec
        .par_iter()
        .map(|pgto1| {
            cgto2
                .pgto_vec
                .par_iter()
                .map(|pgto2| {
                    pgto1.norm_const
                        * pgto2.norm_const
                        * pgto1.cgto_coeff
                        * pgto2.cgto_coeff
                        * calc_kin_energy_int_pgto(
                            &pgto1.alpha,
                            &pgto2.alpha,
                            &pgto1.ang_mom_vec,
                            &pgto2.ang_mom_vec,
                            &pgto1.gauss_center_pos,
                            &pgto2.gauss_center_pos,
                        )
                })
                .sum::<f64>()
        })
        .sum::<f64>()

    //* Serial code */
    // let mut kinetic_energy_int_val: f64 = 0.0;
    // for pgto1 in cgto1.pgto_vec.iter() {
    //     for pgto2 in cgto2.pgto_vec.iter() {
    //         kinetic_energy_int_val += pgto1.norm_const
    //             * pgto2.norm_const
    //             * pgto1.cgto_coeff
    //             * pgto2.cgto_coeff
    //             * calc_kin_energy_int_prim(
    //                 &pgto1.alpha,
    //                 &pgto2.alpha,
    //                 &pgto1.ang_mom_vec,
    //                 &pgto2.ang_mom_vec,
    //                 &pgto1.gauss_center_pos,
    //                 &pgto2.gauss_center_pos,
    //             );
    //     }
    // }

    // kinetic_energy_int_val
}

#[inline]
pub fn calc_R_coulomb_aux_herm(
    t: i32,
    u: i32,
    v: i32,
    order_boys: u64,
    p: &f64,
    // alpha1: &f64,
    // alpha2: &f64,
    P_C_vec: &Array1<f64>,
    dist_P_C: f64,
) -> f64 {
    // Calculate the expansion coefficient of the attraction integral.
    //
    // # Arguments
    // ----------
    // t : i32
    //   Order of Coulomb Hermite derivative in x direction (see Helgaker and Taylor)
    // u : i32
    //   Order of Coulomb Hermite derivative in y direction (see Helgaker and Taylor)
    // v : i32
    //   Order of Coulomb Hermite derivative in z direction (see Helgaker and Taylor)
    // order_boys : i32
    //   Order of boys function
    // P_C_vec : Array1<f64>
    //   Cartesian vec from Gaussian composite center P to nucleus center C.
    // dist_P_C : f64
    //   Distance from Gaussian composite center P to nucleus center C.
    // alpha1 : f64
    //   Exponent of the first Gaussian function.
    // alpha2 : f64
    //   Exponent of the second Gaussian function.

    // let p: f64 = alpha1 + alpha2;
    let boys_arg: f64 = p * dist_P_C * dist_P_C;
    let mut result: f64 = 0.0;

    // if (t == 0) && (u == 0) && (v == 0) {
    //     result += (-2.0 * p).powi(order_boys as i32) * boys(order_boys, boys_arg);
    // } else if (t == 0) && (u == 0) {
    //     if v > 1 {
    //         result += (v - 1) as f64
    //             * calc_R_coulomb_aux_herm_int(
    //                 t,
    //                 u,
    //                 v - 2,
    //                 order_boys + 1,
    //                 alpha1,
    //                 alpha2,
    //                 P_C_vec,
    //                 dist_P_C,
    //             );
    //     }
    //     result += P_C_vec[2]
    //         * calc_R_coulomb_aux_herm_int(
    //             t,
    //             u,
    //             v - 1,
    //             order_boys + 1,
    //             alpha1,
    //             alpha2,
    //             P_C_vec,
    //             dist_P_C,
    //         );
    // } else if t == 0 {
    //     if u > 1 {
    //         result += (u - 1) as f64
    //             * calc_R_coulomb_aux_herm_int(
    //                 t,
    //                 u - 2,
    //                 v,
    //                 order_boys + 1,
    //                 alpha1,
    //                 alpha2,
    //                 P_C_vec,
    //                 dist_P_C,
    //             );
    //     }
    //     result += P_C_vec[1]
    //         * calc_R_coulomb_aux_herm_int(
    //             t,
    //             u - 1,
    //             v,
    //             order_boys + 1,
    //             alpha1,
    //             alpha2,
    //             P_C_vec,
    //             dist_P_C,
    //         );
    // } else {
    //     if t > 1 {
    //         result += (t - 1) as f64
    //             * calc_R_coulomb_aux_herm_int(
    //                 t - 2,
    //                 u,
    //                 v,
    //                 order_boys + 1,
    //                 alpha1,
    //                 alpha2,
    //                 P_C_vec,
    //                 dist_P_C,
    //             );
    //     }
    //     result += P_C_vec[0]
    //         * calc_R_coulomb_aux_herm_int(
    //             t - 1,
    //             u,
    //             v,
    //             order_boys + 1,
    //             alpha1,
    //             alpha2,
    //             P_C_vec,
    //             dist_P_C,
    //         );
    // }

    // V1: contains error?
    match (t, u, v) {
        (0, 0, 0) => {
            result += (-2.0 * p).powi(order_boys as i32) * boys(order_boys, boys_arg);
        }
        (0, 0, _) => {
            if v > 1 {
                result += (v - 1) as f64
                    * calc_R_coulomb_aux_herm(t, u, v - 2, order_boys + 1, p, P_C_vec, dist_P_C);
            }
            result += P_C_vec[2]
                * calc_R_coulomb_aux_herm(t, u, v - 1, order_boys + 1, p, P_C_vec, dist_P_C);
        }
        (0, _, _) => {
            if u > 1 {
                result += (u - 1) as f64
                    * calc_R_coulomb_aux_herm(t, u - 2, v, order_boys + 1, p, P_C_vec, dist_P_C);
            }
            result += P_C_vec[1]
                * calc_R_coulomb_aux_herm(t, u - 1, v, order_boys + 1, p, P_C_vec, dist_P_C);
        }
        (_, _, _) => {
            if t > 1 {
                result += (t - 1) as f64
                    * calc_R_coulomb_aux_herm(t - 2, u, v, order_boys + 1, p, P_C_vec, dist_P_C);
            }
            result += P_C_vec[0]
                * calc_R_coulomb_aux_herm(t - 1, u, v, order_boys + 1, p, P_C_vec, dist_P_C);
        }
    }

    result
}

#[inline(always)]
pub fn calc_gaussian_prod_center(
    alpha1: f64,
    alpha2: f64,
    gauss1_center_pos: &Array1<f64>,
    gauss2_center_pos: &Array1<f64>,
) -> Array1<f64> {
    // Calculate the center of the Gaussian product of two Gaussian functions.
    //
    // # Arguments
    // ----------
    // alpha1 : f64
    //   Exponent of the first Gaussian function.
    // alpha2 : f64
    //   Exponent of the second Gaussian function.
    // gauss1_center_pos : Array1<f64>
    //   Cartesian vec of the first Gaussian function.
    // gauss2_center_pos : Array1<f64>
    //   Cartesian vec of the second Gaussian function.
    //
    // # Returns
    // --------
    // gaussian_prod_center : Array1<f64>
    //   Cartesian vec of the center of the Gaussian product of the two Gaussian functions.
    //

    let p_recip: f64 = (alpha1 + alpha2).recip();
    p_recip * (alpha1 * gauss1_center_pos + alpha2 * gauss2_center_pos)
}

pub fn calc_nuc_attr_int_pgto(
    alpha1: &f64,
    alpha2: &f64,
    ang_mom_vec1: &Array1<i32>,
    ang_mom_vec2: &Array1<i32>,
    gauss1_center_pos: &Array1<f64>,
    gauss2_center_pos: &Array1<f64>,
    nuc_center: &Array1<f64>,
) -> f64 {
    let gaussian_prod_center: Array1<f64> =
        calc_gaussian_prod_center(*alpha1, *alpha2, gauss1_center_pos, gauss2_center_pos);
    let p = alpha1 + alpha2;
    let p_recip = p.recip();
    let P_C_vec = &gaussian_prod_center - nuc_center;
    let dist_P_C = P_C_vec.norm();

    let mut result_V_ne_prim: f64 = 0.0;

    for t in 0..(ang_mom_vec1[0] + ang_mom_vec2[0] + 1) {
        for u in 0..(ang_mom_vec1[1] + ang_mom_vec2[1] + 1) {
            for v in 0..(ang_mom_vec1[2] + ang_mom_vec2[2] + 1) {
                // * Same result as the commented out code below
                // result += calc_E_herm_gauss_coeff(
                //     ang_mom_vec1[0],
                //     ang_mom_vec2[0],
                //     t,
                //     gauss1_center_pos[0] - gauss2_center_pos[0],
                //     alpha1,
                //     alpha2,
                // ) * calc_E_herm_gauss_coeff(
                //     ang_mom_vec1[1],
                //     ang_mom_vec2[1],
                //     u,
                //     gauss1_center_pos[1] - gauss2_center_pos[1],
                //     alpha1,
                //     alpha2,
                // ) * calc_E_herm_gauss_coeff(
                //     ang_mom_vec1[2],
                //     ang_mom_vec2[2],
                //     v,
                //     gauss1_center_pos[2] - gauss2_center_pos[2],
                //     alpha1,
                //     alpha2,
                // ) * calc_R_coulomb_aux_herm_int(
                //     t,
                //     u,
                //     v,
                //     0,
                //     alpha1,
                //     alpha2,
                //     &gaussian_prod_center,
                //     dist_P_C,
                // );

                // * Previous version of code → same result as above
                let tuv = [t, u, v];
                let mut result_tmp: f64 = 1.0;
                for cart_coord in 0..3 {
                    result_tmp *= calc_E_herm_gauss_coeff(
                        ang_mom_vec1[cart_coord],
                        ang_mom_vec2[cart_coord],
                        tuv[cart_coord], //* This is t, u, v depending on the cartesian coordinate
                        gauss1_center_pos[cart_coord] - gauss2_center_pos[cart_coord],
                        alpha1,
                        alpha2,
                    )
                }

                result_V_ne_prim += result_tmp
                    * calc_R_coulomb_aux_herm(
                        t, u, v, 0, &p,
                        &P_C_vec, // wrong: &gaussian_prod_center, here should be P-C vec
                        dist_P_C,
                    );
            }
        }
    }

    result_V_ne_prim * 2.0 * PI * p_recip
}

pub fn calc_nuc_attr_int_cgto(cgto1: &CGTO, cgto2: &CGTO, nuc_center: &Array1<f64>) -> f64 {
    //* Parallel code */
    cgto1
        .pgto_vec
        .par_iter()
        .map(|pgto1| {
            cgto2
                .pgto_vec
                .par_iter()
                .map(|pgto2| {
                    pgto1.norm_const
                        * pgto2.norm_const
                        * pgto1.cgto_coeff
                        * pgto2.cgto_coeff
                        * calc_nuc_attr_int_pgto(
                            &pgto1.alpha,
                            &pgto2.alpha,
                            &pgto1.ang_mom_vec,
                            &pgto2.ang_mom_vec,
                            &pgto1.gauss_center_pos,
                            &pgto2.gauss_center_pos,
                            nuc_center,
                        )
                })
                .sum::<f64>()
        })
        .sum::<f64>()

    //* Serial code */
    // let mut nuc_attr_int_val: f64 = 0.0;
    // for pgto1 in cgto1.pgto_vec.iter() {
    //     for pgto2 in cgto2.pgto_vec.iter() {
    //         nuc_attr_int_val += pgto1.norm_const
    //             * pgto2.norm_const
    //             * pgto1.cgto_coeff
    //             * pgto2.cgto_coeff
    //             * calc_nuc_attr_int_prim(
    //                 &pgto1.alpha,
    //                 &pgto2.alpha,
    //                 &pgto1.ang_mom_vec,
    //                 &pgto2.ang_mom_vec,
    //                 &pgto1.gauss_center_pos,
    //                 &pgto2.gauss_center_pos,
    //                 nuc_center,
    //             );
    //     }
    // }

    // nuc_attr_int_val
}

#[allow(clippy::too_many_arguments)]
pub fn calc_elec_elec_repul_pgto(
    alpha1: &f64,
    alpha2: &f64,
    alpha3: &f64,
    alpha4: &f64,
    ang_mom_vec1: &Array1<i32>,
    ang_mom_vec2: &Array1<i32>,
    ang_mom_vec3: &Array1<i32>,
    ang_mom_vec4: &Array1<i32>,
    gauss1_center_pos: &Array1<f64>,
    gauss2_center_pos: &Array1<f64>,
    gauss3_center_pos: &Array1<f64>,
    gauss4_center_pos: &Array1<f64>,
) -> f64 {
    let p = alpha1 + alpha2;
    let q = alpha3 + alpha4;
    let alpha_ERI = p * q / (p + q);
    let P: Array1<f64> =
        calc_gaussian_prod_center(*alpha1, *alpha2, gauss1_center_pos, gauss2_center_pos);
    let Q: Array1<f64> =
        calc_gaussian_prod_center(*alpha3, *alpha4, gauss3_center_pos, gauss4_center_pos);
    let P_Q_vec: Array1<f64> = &P - &Q;
    let dist_P_Q: f64 = P_Q_vec.norm();

    let mut ERI_result: f64 = 0.0;

    for t in 0..(ang_mom_vec1[0] + ang_mom_vec2[0] + 1) {
        for u in 0..(ang_mom_vec1[1] + ang_mom_vec2[1] + 1) {
            for v in 0..(ang_mom_vec1[2] + ang_mom_vec2[2] + 1) {
                let tuv = [t, u, v];
                let mut result_tmp1: f64 = 1.0;
                for cart_coord in 0..3 {
                    result_tmp1 *= calc_E_herm_gauss_coeff(
                        ang_mom_vec1[cart_coord],
                        ang_mom_vec2[cart_coord],
                        tuv[cart_coord], //* This is t, u, v depending on the cartesian coordinate
                        gauss1_center_pos[cart_coord] - gauss2_center_pos[cart_coord],
                        alpha1,
                        alpha2,
                    )
                }

                for tau in 0..(ang_mom_vec3[0] + ang_mom_vec4[0] + 1) {
                    for nu in 0..(ang_mom_vec3[1] + ang_mom_vec4[1] + 1) {
                        for phi in 0..(ang_mom_vec3[2] + ang_mom_vec4[2] + 1) {
                            let tau_nu_phi = [tau, nu, phi];
                            let mut result_tmp2: f64 = 1.0; //* added result_tmp2 to make it work */
                            for cart_coord in 0..3 {
                                result_tmp2 *= calc_E_herm_gauss_coeff(
                                    ang_mom_vec3[cart_coord],
                                    ang_mom_vec4[cart_coord],
                                    tau_nu_phi[cart_coord], //* This is tau, nu, phi depending on the cartesian coordinate
                                    gauss3_center_pos[cart_coord] - gauss4_center_pos[cart_coord],
                                    alpha3,
                                    alpha4,
                                )
                            }

                            ERI_result += result_tmp1
                                * result_tmp2
                                * (-1.0_f64).powi(tau + nu + phi)
                                * calc_R_coulomb_aux_herm(
                                    t + tau,
                                    u + nu,
                                    v + phi,
                                    0,
                                    &alpha_ERI,
                                    &P_Q_vec,
                                    dist_P_Q,
                                )

                            // ERI_result += (-1.0).powi(tau + nu + phi)
                            //     * calc_E_herm_gauss_coeff(
                            //         ang_mom_vec1[0],
                            //         ang_mom_vec2[0],
                            //         t,
                            //         gauss1_center_pos[0] - gauss2_center_pos[0],
                            //         alpha1,
                            //         alpha2,
                            //     )
                            //     * calc_E_herm_gauss_coeff(
                            //         ang_mom_vec1[1],
                            //         ang_mom_vec2[1],
                            //         u,
                            //         gauss1_center_pos[1] - gauss2_center_pos[1],
                            //         alpha1,
                            //         alpha2,
                            //     )
                            //     * calc_E_herm_gauss_coeff(
                            //         ang_mom_vec1[2],
                            //         ang_mom_vec2[2],
                            //         v,
                            //         gauss1_center_pos[2] - gauss2_center_pos[2],
                            //         alpha1,
                            //         alpha2,
                            //     )
                            //     * calc_E_herm_gauss_coeff(
                            //         ang_mom_vec3[0],
                            //         ang_mom_vec4[0],
                            //         tau,
                            //         gauss3_center_pos[0] - gauss4_center_pos[0],
                            //         alpha3,
                            //         alpha4,
                            //     )
                            //     * calc_E_herm_gauss_coeff(
                            //         ang_mom_vec3[1],
                            //         ang_mom_vec4[1],
                            //         nu,
                            //         gauss3_center_pos[1] - gauss4_center_pos[1],
                            //         alpha3,
                            //         alpha4,
                            //     )
                            //     * calc_E_herm_gauss_coeff(
                            //         ang_mom_vec3[2],
                            //         ang_mom_vec4[2],
                            //         phi,
                            //         gauss3_center_pos[2] - gauss4_center_pos[2],
                            //         alpha3,
                            //         alpha4,
                            //     )
                            //     * calc_R_coulomb_aux_herm_int(
                            //         t + tau,
                            //         u + nu,
                            //         v + phi,
                            //         0,
                            //         &p,
                            //         &q,
                            //         &P_Q_vec,
                            //         dist_P_Q,
                            //     );
                        }
                    }
                }
            }
        }
    }

    ERI_result * 2.0 * PI.powf(2.5) * (p * q * (p + q).sqrt()).recip()
}

pub fn calc_elec_elec_repul_cgto(cgto1: &CGTO, cgto2: &CGTO, cgto3: &CGTO, cgto4: &CGTO) -> f64 {
    // * Parallel code
    cgto1
        .pgto_vec
        .par_iter()
        .map(|pgto1| {
            cgto2
                .pgto_vec
                .par_iter()
                .map(|pgto2| {
                    cgto3
                        .pgto_vec
                        .par_iter()
                        .map(|pgto3| {
                            cgto4
                                .pgto_vec
                                .par_iter()
                                .map(|pgto4| {
                                    pgto1.norm_const
                                        * pgto2.norm_const
                                        * pgto3.norm_const
                                        * pgto4.norm_const
                                        * pgto1.cgto_coeff
                                        * pgto2.cgto_coeff
                                        * pgto3.cgto_coeff
                                        * pgto4.cgto_coeff
                                        * calc_elec_elec_repul_pgto(
                                            &pgto1.alpha,
                                            &pgto2.alpha,
                                            &pgto3.alpha,
                                            &pgto4.alpha,
                                            &pgto1.ang_mom_vec,
                                            &pgto2.ang_mom_vec,
                                            &pgto3.ang_mom_vec,
                                            &pgto4.ang_mom_vec,
                                            &pgto1.gauss_center_pos,
                                            &pgto2.gauss_center_pos,
                                            &pgto3.gauss_center_pos,
                                            &pgto4.gauss_center_pos,
                                        )
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .sum::<f64>()

    // * Serial code
    // let mut ERI_val: f64 = 0.0;
    // for pgto1 in cgto1.pgto_vec.iter() {
    //     for pgto2 in cgto2.pgto_vec.iter() {
    //         for pgto3 in cgto3.pgto_vec.iter() {
    //             for pgto4 in cgto4.pgto_vec.iter() {
    //                 ERI_val += pgto1.norm_const
    //                     * pgto2.norm_const
    //                     * pgto3.norm_const
    //                     * pgto4.norm_const
    //                     * pgto1.cgto_coeff
    //                     * pgto2.cgto_coeff
    //                     * pgto3.cgto_coeff
    //                     * pgto4.cgto_coeff
    //                     * calc_elec_elec_repul_prim(
    //                         &pgto1.alpha,
    //                         &pgto2.alpha,
    //                         &pgto3.alpha,
    //                         &pgto4.alpha,
    //                         &pgto1.ang_mom_vec,
    //                         &pgto2.ang_mom_vec,
    //                         &pgto3.ang_mom_vec,
    //                         &pgto4.ang_mom_vec,
    //                         &pgto1.gauss_center_pos,
    //                         &pgto2.gauss_center_pos,
    //                         &pgto3.gauss_center_pos,
    //                         &pgto4.gauss_center_pos,
    //                     );
    //             }
    //         }
    //     }
    // }

    // ERI_val
}

pub fn calc_cart_mu_val_pgto(
    alpha1: &f64,
    alpha2: &f64,
    ang_mom_vec1: &Array1<i32>,
    ang_mom_vec2: &Array1<i32>,
    gauss1_center_pos: &Array1<f64>,
    gauss2_center_pos: &Array1<f64>,
    charge_center: &Array1<f64>,
    cart_coord: usize,
) -> f64 {
    assert!(cart_coord < 3); //* Checks for valid input */
    let gauss_prod_cent =
        calc_gaussian_prod_center(*alpha1, *alpha2, gauss1_center_pos, gauss2_center_pos);
    let PC_vec = &gauss_prod_cent - charge_center;

    // * Calculate the overlap integral over the other cart_coord directions */
    let mut overlap_other_two: f64 = 1.0_f64;
    for cart_coord_other in 0..3 {
        if cart_coord_other != cart_coord {
            overlap_other_two *= calc_E_herm_gauss_coeff(
                ang_mom_vec1[cart_coord_other],
                ang_mom_vec2[cart_coord_other],
                0,
                gauss1_center_pos[cart_coord_other] - gauss2_center_pos[cart_coord_other],
                alpha1,
                alpha2,
            );
        }
    }
    //* Calculate the dipole part */
    let mu_int: f64 = calc_E_herm_gauss_coeff(
        ang_mom_vec1[cart_coord],
        ang_mom_vec2[cart_coord],
        1,
        gauss1_center_pos[cart_coord] - gauss2_center_pos[cart_coord],
        alpha1,
        alpha2,
    ) + PC_vec[cart_coord]
        * calc_E_herm_gauss_coeff(
            ang_mom_vec1[cart_coord],
            ang_mom_vec2[cart_coord],
            0,
            gauss1_center_pos[cart_coord] - gauss2_center_pos[cart_coord],
            alpha1,
            alpha2,
        );

    mu_int * overlap_other_two * PI.powf(1.5) * (alpha1 + alpha2).powf(-1.5)
}

pub fn calc_cart_mu_val_cgto(
    cgto1: &CGTO,
    cgto2: &CGTO,
    charge_center: &Array1<f64>,
    cart_coord: usize,
) -> f64 {
    cgto1
        .pgto_vec
        .par_iter()
        .map(|pgto1| {
            cgto2
                .pgto_vec
                .par_iter()
                .map(|pgto2| {
                    pgto1.norm_const
                        * pgto2.norm_const
                        * pgto1.cgto_coeff
                        * pgto2.cgto_coeff
                        * calc_cart_mu_val_pgto(
                            &pgto1.alpha,
                            &pgto2.alpha,
                            &pgto1.ang_mom_vec,
                            &pgto2.ang_mom_vec,
                            &pgto1.gauss_center_pos,
                            &pgto2.gauss_center_pos,
                            charge_center,
                            cart_coord,
                        )
                })
                .sum::<f64>()
        })
        .sum::<f64>()
}

pub fn calc_V_nn_val(geom_matr: &Array2<f64>, Z_vals: &[i32]) -> f64 {
    let mut V_nn_val: f64 = 0.0;
    for (i, atom1_pos) in geom_matr.axis_iter(ndarray::Axis(0)).enumerate() {
        for (j, atom2_pos) in geom_matr.axis_iter(ndarray::Axis(0)).enumerate() {
            if i == j {
                continue;
            }
            if i < j {
                let Z_val1 = *Z_vals.get(i).unwrap_or(&0);
                let Z_val2 = *Z_vals.get(j).unwrap_or(&0);
                V_nn_val += ((Z_val1 * Z_val2) as f64)
                    / (atom2_pos.to_shared() - atom1_pos.to_shared()).norm();
            }
        }
    }

    V_nn_val
}

// #[inline(always)]
// fn other_two_idxs(n: usize) -> [usize; 2] {
//     assert!(n < 3);
//     let arr: [usize; 3] = [0, 1, 2];
//     let mut result: [usize; 2] = [0; 2];
//     let mut j = 0;
//     for i in 0..3 {
//         if i != n {
//             result[j] = arr[i];
//             j += 1;
//         }
//     }
//     result
// }
