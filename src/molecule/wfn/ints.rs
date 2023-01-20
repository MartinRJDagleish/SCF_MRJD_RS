use crate::molecule::wfn::ContractedGaussian;
use ndarray::Array1;
use std::f64::consts::PI;

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
    position1: &Array1<f64>,
    position2: &Array1<f64>,
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
    // position1 : Array1<f64>
    //   Position of the first Gaussian function. (Center of the Gaussian)
    // position2 : Array1<f64>
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
        &position1[0] - &position2[0],
        &alpha1,
        &alpha2,
    );
    let S_y: f64 = calc_expansion_coeff_overlap_int(
        angular_momentum_vec1[1],
        angular_momentum_vec2[1],
        0,
        &position1[1] - &position2[1],
        &alpha1,
        &alpha2,
    );
    let S_z: f64 = calc_expansion_coeff_overlap_int(
        angular_momentum_vec1[2],
        angular_momentum_vec2[2],
        0,
        &position1[2] - &position2[2],
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
                    &prim1.position,
                    &prim2.position,
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
    position1: &Array1<f64>,
    position2: &Array1<f64>,
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
    // position1 : Array1<f64>
    //   Position of the first Gaussian function. (Center of the Gaussian)
    // position2 : Array1<f64>
    //   Position of the second Gaussian function. (Center of the Gaussian)
    //
    // # Returns
    // --------
    // overlap_int : f64
    //   The overlap integral between the two Gaussian functions.
    //

    let mut ang_mom_vec2_tmp = angular_momentum_vec2.clone();
    let part1: f64 = alpha2
        * (2.0 * angular_momentum_vec2.sum() as f64 + 3.0)
        * calc_overlap_int_prim(
            &alpha1,
            &alpha2,
            &angular_momentum_vec1,
            &ang_mom_vec2_tmp,
            &position1,
            &position2,
        );
    let mut part2_1: f64 = 0.0;
    for i in 0..3 {
        ang_mom_vec2_tmp[i] += 2;
        part2_1 += calc_overlap_int_prim(
            &alpha1,
            &alpha2,
            &angular_momentum_vec1,
            &ang_mom_vec2_tmp,
            &position1,
            &position2,
        );
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
                &position1,
                &position2,
            );
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
                    &prim1.position,
                    &prim2.position,
                );
        }
    }

    kinetic_energy_int_val
}
