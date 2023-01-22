use crate::molecule::wfn::ContractedGaussian;
use boys::micb25::boys;
use ndarray::Array1;
use std::{f64::consts::PI, ops::Range};

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
        (x, _, _) if x < 0 || x > (l1 + l2) => return 0.0,
        (0, 0, 0) => return (-q * gauss_dist.powi(2)).exp(),
        (_, _, 0) => {
            //* decrement index l1
            return 0.5
                * p_recip
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
                    );
        }
        _ => {
            //* decrement index l2
            return 0.5
                * p_recip
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
                    );
        }
    }
}

pub fn calc_overlap_int_prim(
    alpha1: &f64,
    alpha2: &f64,
    angular_momentum_vec1: &Array1<i32>,
    angular_momentum_vec2: &Array1<i32>,
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

    let S_x: f64 = calc_expansion_coeff_overlap_int(
        angular_momentum_vec1[0],
        angular_momentum_vec2[0],
        0,
        &gauss1_center_pos[0] - &gauss2_center_pos[0],
        &alpha1,
        &alpha2,
    );
    let S_y: f64 = calc_expansion_coeff_overlap_int(
        angular_momentum_vec1[1],
        angular_momentum_vec2[1],
        0,
        &gauss1_center_pos[1] - &gauss2_center_pos[1],
        &alpha1,
        &alpha2,
    );
    let S_z: f64 = calc_expansion_coeff_overlap_int(
        angular_momentum_vec1[2],
        angular_momentum_vec2[2],
        0,
        &gauss1_center_pos[2] - &gauss2_center_pos[2],
        &alpha1,
        &alpha2,
    );

    S_x * S_y * S_z * PI.powf(1.5) * (alpha1 + alpha2).recip().powf(1.5)
}

pub fn calc_overlap_int_cgto(
    ContrGaus1: &ContractedGaussian,
    ContrGaus2: &ContractedGaussian,
) -> f64 {
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
    for (idx1, prim1) in ContrGaus1.PrimGauss_vec.iter().enumerate() {
        for (idx2, prim2) in ContrGaus2.PrimGauss_vec.iter().enumerate() {
            overlap_int_val += ContrGaus1.PrimGauss_vec[idx1].norm_const
                * ContrGaus2.PrimGauss_vec[idx2].norm_const
                * ContrGaus1.PrimGauss_vec[idx1].cgto_coeff
                * ContrGaus2.PrimGauss_vec[idx2].cgto_coeff
                * calc_overlap_int_prim(
                    &prim1.alpha,
                    &prim2.alpha,
                    &prim1.angular_momentum_vec,
                    &prim2.angular_momentum_vec,
                    &prim1.gauss_center_pos,
                    &prim2.gauss_center_pos,
                );
        }
    }
    overlap_int_val
}

pub fn calc_kinetic_energy_int_prim(
    alpha1: &f64,
    alpha2: &f64,
    angular_momentum_vec1: &Array1<i32>,
    angular_momentum_vec2: &Array1<i32>,
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
    let mut ang_mom_vec2_tmp = angular_momentum_vec2.clone();
    let part1: f64 = alpha2
        * (2.0 * angular_momentum_vec2.sum() as f64 + 3.0)
        * calc_overlap_int_prim(
            &alpha1,
            &alpha2,
            &angular_momentum_vec1,
            &ang_mom_vec2_tmp,
            &gauss1_center_pos,
            &gauss2_center_pos,
        );
    let mut part2_1: f64 = 0.0;
    for i in 0..3 {
        ang_mom_vec2_tmp[i] += 2;
        part2_1 += calc_overlap_int_prim(
            &alpha1,
            &alpha2,
            &angular_momentum_vec1,
            &ang_mom_vec2_tmp,
            &gauss1_center_pos,
            &gauss2_center_pos,
        );
        ang_mom_vec2_tmp[i] -= 2;
    }
    let part2_2: f64 = -2.0 * alpha2.powi(2) * part2_1;
    let mut part3: f64 = 0.0;
    for i in 0..3 {
        ang_mom_vec2_tmp[i] += 2;
        part3 += ((ang_mom_vec2_tmp[i] - 2) as f64)
            * ((ang_mom_vec2_tmp[i]) as f64)
            * calc_overlap_int_prim(
                &alpha1,
                &alpha2,
                &angular_momentum_vec1,
                &ang_mom_vec2_tmp,
                &gauss1_center_pos,
                &gauss2_center_pos,
            );
        ang_mom_vec2_tmp[i] -= 2;
    }
    part3 *= -0.5;

    part1 + part2_2 + part3
}

pub fn calc_kinetic_energy_int_cgto(
    ContrGaus1: &ContractedGaussian,
    ContrGaus2: &ContractedGaussian,
) -> f64 {
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
    for (idx1, prim1) in ContrGaus1.PrimGauss_vec.iter().enumerate() {
        for (idx2, prim2) in ContrGaus2.PrimGauss_vec.iter().enumerate() {
            kinetic_energy_int_val += ContrGaus1.PrimGauss_vec[idx1].norm_const
                * ContrGaus2.PrimGauss_vec[idx2].norm_const
                * ContrGaus1.PrimGauss_vec[idx1].cgto_coeff
                * ContrGaus2.PrimGauss_vec[idx2].cgto_coeff
                * calc_kinetic_energy_int_prim(
                    &prim1.alpha,
                    &prim2.alpha,
                    &prim1.angular_momentum_vec,
                    &prim2.angular_momentum_vec,
                    &prim1.gauss_center_pos,
                    &prim2.gauss_center_pos,
                );
        }
    }

    kinetic_energy_int_val
}

pub fn calc_expansion_coeff_attraction_int(
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
                    * calc_expansion_coeff_attraction_int(
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
                * calc_expansion_coeff_attraction_int(
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
                    * calc_expansion_coeff_attraction_int(
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
                * calc_expansion_coeff_attraction_int(
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
                    * calc_expansion_coeff_attraction_int(
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
                * calc_expansion_coeff_attraction_int(
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
    origin_vec1: &Array1<f64>,
    origin_vec2: &Array1<f64>,
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
    let gaussian_prod_center: Array1<f64> = (origin_vec1 * alpha1 + alpha2 * origin_vec2) * p_recip;

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
        alpha1.clone(),
        alpha2.clone(),
        gauss1_center_pos,
        gauss2_center_pos,
    );
    let p_recip = (alpha1 + alpha2).recip();
    let dist_P_C: f64 = calc_r_ij_general(&gaussian_prod_center, &nuc_center);

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
                    * calc_expansion_coeff_attraction_int(
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
    ContrGaus1: &ContractedGaussian,
    ContrGaus2: &ContractedGaussian,
    nuc_center: &Array1<f64>,
) -> f64 {
    let mut nuc_attr_int_val: f64 = 0.0;
    for (idx1, prim1) in ContrGaus1.PrimGauss_vec.iter().enumerate() {
        for (idx2, prim2) in ContrGaus2.PrimGauss_vec.iter().enumerate() {
            nuc_attr_int_val += ContrGaus1.PrimGauss_vec[idx1].norm_const
                * ContrGaus2.PrimGauss_vec[idx2].norm_const
                * ContrGaus1.PrimGauss_vec[idx1].cgto_coeff
                * ContrGaus2.PrimGauss_vec[idx2].cgto_coeff
                * calc_kinetic_energy_int_prim(
                    &prim1.alpha,
                    &prim2.alpha,
                    &prim1.angular_momentum_vec,
                    &prim2.angular_momentum_vec,
                    &prim1.gauss_center_pos,
                    &prim2.gauss_center_pos,
                );
        }
    }

    nuc_attr_int_val
}
