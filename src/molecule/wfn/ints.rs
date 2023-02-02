use crate::molecule::wfn::CGTO;
use boys::micb25::boys;
// use boys::exact::boys;
use ndarray::{Array1, Array2};
use ndarray_linalg::Scalar;
use std::f64::consts::PI;
// use rayon::prelude::*;

use crate::molecule::geometry::calc_r_ij_general;

#[doc(hidden)]
pub fn calc_expansion_coeff_overlap_int(
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
        (x, _, _) if x < 0 || x > (l1 + l2) => 0.0,
        (0, 0, 0) => (-q * gauss_dist.powi(2)).exp(),
        (_, _, 0) => {
            //* decrement index l1
            0.5 * p_recip
                * calc_expansion_coeff_overlap_int(
                    l1 - 1,
                    l2,
                    no_nodes - 1,
                    gauss_dist,
                    alpha1,
                    alpha2,
                )
                - (q * gauss_dist / alpha1)
                    * calc_expansion_coeff_overlap_int(
                        l1 - 1,
                        l2,
                        no_nodes,
                        gauss_dist,
                        alpha1,
                        alpha2,
                    )
                + (no_nodes + 1) as f64
                    * calc_expansion_coeff_overlap_int(
                        l1 - 1,
                        l2,
                        no_nodes + 1,
                        gauss_dist,
                        alpha1,
                        alpha2,
                    )
        }
        (_, _, _) => {
            //* decrement index l2
            0.5 * p_recip
                * calc_expansion_coeff_overlap_int(
                    l1,
                    l2 - 1,
                    no_nodes - 1,
                    gauss_dist,
                    alpha1,
                    alpha2,
                )
                - (q * gauss_dist / alpha1)
                    * calc_expansion_coeff_overlap_int(
                        l1,
                        l2 - 1,
                        no_nodes,
                        gauss_dist,
                        alpha1,
                        alpha2,
                    )
                + (no_nodes + 1) as f64
                    * calc_expansion_coeff_overlap_int(
                        l1,
                        l2 - 1,
                        no_nodes + 1,
                        gauss_dist,
                        alpha1,
                        alpha2,
                    )
        }
    }
}

pub fn calc_overlap_int_prim(
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
    (0..3).for_each(|cart_coord| {
        S_cart_total *= calc_expansion_coeff_overlap_int(
            ang_mom_vec1[cart_coord],
            ang_mom_vec2[cart_coord],
            0,
            &gauss1_center_pos[cart_coord] - &gauss2_center_pos[cart_coord], //* abs does not fix the problem
            alpha1,
            alpha2,
        );
    });

    S_cart_total * PI.powf(1.5) * (alpha1 + alpha2).recip().powf(1.5)
}

pub fn calc_overlap_int_cgto(ContrGauss1: &CGTO, ContrGauss2: &CGTO) -> f64 {
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

    let mut overlap_int_val: f64 = 0.0;
    for prim1 in ContrGauss1.pgto_vec.iter() {
        for prim2 in ContrGauss2.pgto_vec.iter() {
            overlap_int_val += prim1.norm_const
                * prim2.norm_const
                * prim1.cgto_coeff
                * prim2.cgto_coeff
                * calc_overlap_int_prim(
                    &prim1.alpha,
                    &prim2.alpha,
                    &prim1.ang_mom_vec,
                    &prim2.ang_mom_vec,
                    &prim1.gauss_center_pos,
                    &prim2.gauss_center_pos,
                );
        }
    }
    overlap_int_val
}

pub fn calc_kin_energy_int_prim(
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
        * calc_overlap_int_prim(
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
        part2 += calc_overlap_int_prim(
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
            * calc_overlap_int_prim(
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

pub fn calc_kin_energy_int_cgto(ContrGauss1: &CGTO, ContrGauss2: &CGTO) -> f64 {
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

    let mut kinetic_energy_int_val: f64 = 0.0;
    for prim1 in ContrGauss1.pgto_vec.iter() {
        for prim2 in ContrGauss2.pgto_vec.iter() {
            kinetic_energy_int_val += prim1.norm_const
                * prim2.norm_const
                * prim1.cgto_coeff
                * prim2.cgto_coeff
                * calc_kin_energy_int_prim(
                    &prim1.alpha,
                    &prim2.alpha,
                    &prim1.ang_mom_vec,
                    &prim2.ang_mom_vec,
                    &prim1.gauss_center_pos,
                    &prim2.gauss_center_pos,
                );
        }
    }

    kinetic_energy_int_val
}

pub fn calc_expansion_coeff_attr_int(
    t: i32,
    u: i32,
    v: i32,
    order_boys: u64,
    alpha1: &f64,
    alpha2: &f64,
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

    let p: f64 = alpha1 + alpha2;
    let val1: f64 = p * dist_P_C.powi(2);
    let mut result: f64 = 0.0;

    match (t, u, v) {
        (0, 0, 0) => {
            result += (-2.0 * p).powi(order_boys as i32) * boys(order_boys, val1);
        }
        (0, 0, _) => {
            if v > 1 {
                result += (v - 1) as f64
                    * calc_expansion_coeff_attr_int(
                        t,
                        u,
                        v - 2,
                        order_boys + 1,
                        alpha1,
                        alpha2,
                        P_C_vec,
                        dist_P_C,
                    );
            }
            result += P_C_vec[2]
                * calc_expansion_coeff_attr_int(
                    t,
                    u,
                    v - 1,
                    order_boys + 1,
                    alpha1,
                    alpha2,
                    P_C_vec,
                    dist_P_C,
                )
        }
        (0, _, _) => {
            if u > 1 {
                result += (u - 1) as f64
                    * calc_expansion_coeff_attr_int(
                        t,
                        u - 2,
                        v,
                        order_boys + 1,
                        alpha1,
                        alpha2,
                        P_C_vec,
                        dist_P_C,
                    );
            }
            result += P_C_vec[1]
                * calc_expansion_coeff_attr_int(
                    t,
                    u - 1,
                    v,
                    order_boys + 1,
                    alpha1,
                    alpha2,
                    P_C_vec,
                    dist_P_C,
                )
        }
        (_, _, _) => {
            if t > 1 {
                result += (t - 1) as f64
                    * calc_expansion_coeff_attr_int(
                        t - 2,
                        u,
                        v,
                        order_boys + 1,
                        alpha1,
                        alpha2,
                        P_C_vec,
                        dist_P_C,
                    );
            }
            result += P_C_vec[0]
                * calc_expansion_coeff_attr_int(
                    t - 1,
                    u,
                    v,
                    order_boys + 1,
                    alpha1,
                    alpha2,
                    P_C_vec,
                    dist_P_C,
                )
        }
    }

    result
}

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
    // origin_vec1 : Array1<f64>
    //   Cartesian vec of the first Gaussian function.
    // origin_vec2 : Array1<f64>
    //   Cartesian vec of the second Gaussian function.
    //
    // # Returns
    // --------
    // gaussian_prod_center : Array1<f64>
    //   Cartesian vec of the center of the Gaussian product of the two Gaussian functions.
    //

    let p_recip: f64 = (alpha1 + alpha2).recip();
    let gaussian_prod_center: Array1<f64> =
        (gauss1_center_pos * alpha1 + alpha2 * gauss2_center_pos) * p_recip;

    gaussian_prod_center
}

pub fn calc_nuc_attr_int_prim(
    alpha1: &f64,
    alpha2: &f64,
    ang_mom_vec1: &Array1<i32>,
    ang_mom_vec2: &Array1<i32>,
    gauss1_center_pos: &Array1<f64>,
    gauss2_center_pos: &Array1<f64>,
    nuc_center: &Array1<f64>,
) -> f64 {
    let gaussian_prod_center: Array1<f64> = calc_gaussian_prod_center(
        alpha1.to_owned(),
        alpha2.to_owned(),
        gauss1_center_pos,
        gauss2_center_pos,
    );
    let p_recip = (alpha1 + alpha2).recip();
    let dist_P_C: f64 = calc_r_ij_general(&gaussian_prod_center, nuc_center);

    let mut result: f64 = 0.0;

    for t in 0..(ang_mom_vec1[0] + ang_mom_vec2[0] + 1) {
        for u in 0..(ang_mom_vec1[1] + ang_mom_vec2[1] + 1) {
            for v in 0..(ang_mom_vec1[2] + ang_mom_vec2[2] + 1) {
                let mut result_tmp: f64 = 1.0; //* Intialize the result_tmp variable
                for cart_coord in 0..3 {
                    result_tmp *= calc_expansion_coeff_overlap_int(
                        ang_mom_vec1[cart_coord],
                        ang_mom_vec2[cart_coord],
                        t,
                        gauss1_center_pos[cart_coord] - gauss2_center_pos[cart_coord],
                        alpha1,
                        alpha2,
                    )
                }
                result += result_tmp
                    * calc_expansion_coeff_attr_int(
                        t,
                        u,
                        v,
                        0,
                        alpha1,
                        alpha2,
                        &gaussian_prod_center,
                        dist_P_C,
                    );
            }
        }
    }

    result *= 2.0 * PI * p_recip;
    result
}

pub fn calc_nuc_attr_int_cgto(
    ContrGaus1: &CGTO,
    ContrGaus2: &CGTO,
    nuc_center: &Array1<f64>,
) -> f64 {
    let mut nuc_attr_int_val: f64 = 0.0;
    for prim1 in ContrGaus1.pgto_vec.iter() {
        for prim2 in ContrGaus2.pgto_vec.iter() {
            nuc_attr_int_val += prim1.norm_const
                * prim2.norm_const
                * prim1.cgto_coeff
                * prim2.cgto_coeff
                * calc_nuc_attr_int_prim(
                    &prim1.alpha,
                    &prim2.alpha,
                    &prim1.ang_mom_vec,
                    &prim2.ang_mom_vec,
                    &prim1.gauss_center_pos,
                    &prim2.gauss_center_pos,
                    nuc_center,
                );
        }
    }

    nuc_attr_int_val
}

pub fn calc_elec_elec_repul_prim(
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
    let P: Array1<f64> = calc_gaussian_prod_center(
        alpha1.to_owned(),
        alpha2.to_owned(),
        gauss1_center_pos,
        gauss2_center_pos,
    );
    let Q: Array1<f64> = calc_gaussian_prod_center(
        alpha3.to_owned(),
        alpha4.to_owned(),
        gauss3_center_pos,
        gauss4_center_pos,
    );
    let dist_P_Q: f64 = calc_r_ij_general(&P, &Q);
    let P_Q_vec: Array1<f64> = &P - &Q;

    let mut ERI_result: f64 = 0.0;

    for t in 0..(ang_mom_vec1[0] + ang_mom_vec2[0] + 1) {
        for u in 0..(ang_mom_vec1[1] + ang_mom_vec2[1] + 1) {
            for v in 0..(ang_mom_vec1[2] + ang_mom_vec2[2] + 1) {
                for tau in 0..(ang_mom_vec3[0] + ang_mom_vec4[0] + 1) {
                    for nu in 0..(ang_mom_vec3[1] + ang_mom_vec4[1] + 1) {
                        for phi in 0..(ang_mom_vec3[2] + ang_mom_vec4[2] + 1) {
                            ERI_result += (-1.0).powi(tau + nu + phi)
                                * calc_expansion_coeff_overlap_int(
                                    ang_mom_vec1[0],
                                    ang_mom_vec2[0],
                                    t,
                                    gauss1_center_pos[0] - gauss2_center_pos[0],
                                    alpha1,
                                    alpha2,
                                )
                                * calc_expansion_coeff_overlap_int(
                                    ang_mom_vec1[1],
                                    ang_mom_vec2[1],
                                    u,
                                    gauss1_center_pos[1] - gauss2_center_pos[1],
                                    alpha1,
                                    alpha2,
                                )
                                * calc_expansion_coeff_overlap_int(
                                    ang_mom_vec1[2],
                                    ang_mom_vec2[2],
                                    v,
                                    gauss1_center_pos[2] - gauss2_center_pos[2],
                                    alpha1,
                                    alpha2,
                                )
                                * calc_expansion_coeff_overlap_int(
                                    ang_mom_vec3[0],
                                    ang_mom_vec4[0],
                                    tau,
                                    gauss3_center_pos[0] - gauss4_center_pos[0],
                                    alpha3,
                                    alpha4,
                                )
                                * calc_expansion_coeff_overlap_int(
                                    ang_mom_vec3[1],
                                    ang_mom_vec4[1],
                                    nu,
                                    gauss3_center_pos[1] - gauss4_center_pos[1],
                                    alpha3,
                                    alpha4,
                                )
                                * calc_expansion_coeff_overlap_int(
                                    ang_mom_vec3[2],
                                    ang_mom_vec4[2],
                                    phi,
                                    gauss3_center_pos[2] - gauss4_center_pos[2],
                                    alpha3,
                                    alpha4,
                                )
                                * calc_expansion_coeff_attr_int(
                                    t + tau,
                                    u + nu,
                                    v + phi,
                                    0,
                                    &p,
                                    &q,
                                    &P_Q_vec,
                                    dist_P_Q,
                                );
                        }
                    }
                }
            }
        }
    }

    ERI_result *= 2.0 * PI.powf(2.5) * (p * q * (p + q).sqrt()).recip();
    ERI_result
}

pub fn calc_elec_elec_repul_cgto(
    ContrGaus1: &CGTO,
    ContrGaus2: &CGTO,
    ContrGaus3: &CGTO,
    ContrGaus4: &CGTO,
) -> f64 {
    let mut ERI_val: f64 = 0.0;

    for prim1 in ContrGaus1.pgto_vec.iter() {
        for prim2 in ContrGaus2.pgto_vec.iter() {
            for prim3 in ContrGaus3.pgto_vec.iter() {
                for prim4 in ContrGaus4.pgto_vec.iter() {
                    ERI_val += prim1.norm_const
                        * prim2.norm_const
                        * prim3.norm_const
                        * prim4.norm_const
                        * prim1.cgto_coeff
                        * prim2.cgto_coeff
                        * prim3.cgto_coeff
                        * prim4.cgto_coeff
                        * calc_elec_elec_repul_prim(
                            &prim1.alpha,
                            &prim2.alpha,
                            &prim3.alpha,
                            &prim4.alpha,
                            &prim1.ang_mom_vec,
                            &prim2.ang_mom_vec,
                            &prim3.ang_mom_vec,
                            &prim4.ang_mom_vec,
                            &prim1.gauss_center_pos,
                            &prim2.gauss_center_pos,
                            &prim3.gauss_center_pos,
                            &prim4.gauss_center_pos,
                        );
                }
            }
        }
    }

    ERI_val
}

pub fn calc_E_nn_val(geom_matr: &Array2<f64>) -> f64 {
    let mut E_nn_val: f64 = 0.0;
    for (i, atom1_pos) in geom_matr.axis_iter(ndarray::Axis(0)).enumerate() {
        for (j, atom2_pos) in geom_matr.axis_iter(ndarray::Axis(0)).enumerate() {
            if i == j {
                continue;
            }
            E_nn_val += calc_r_ij_general(&atom1_pos.to_owned(), &atom2_pos.to_owned());
        }
    }

    E_nn_val
}
