use std::f64::consts::PI;

use ndarray::Array1;

#[doc(hidden)]
pub fn calc_expansion_coeff_overlap_int(
    l1: i32,
    l2: i32,
    no_nodes: i32,
    gauss_dist: f64,
    alpha1: f64,
    alpha2: f64,
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
    //TODO: can I use match here?
    if no_nodes < 0 || no_nodes > (l1 + l2) {
        return 0.0;
    } else if l1 == l2 && l2 == no_nodes && no_nodes == 0 {
        return (-q * gauss_dist.powi(2)).exp();
    } else if l2 == 0 {
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
    } else {
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

pub fn calc_overlap_int_prim(
    alpha1: f64,
    alpha2: f64,
    angular_momentum_vec1: Array1<i32>,
    angular_momentum_vec2: Array1<i32>,
    position1: Array1<f64>,
    position2: Array1<f64>,
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
        position1[0] - position2[0],
        alpha1,
        alpha2,
    );
    let S_y: f64 = calc_expansion_coeff_overlap_int(
        angular_momentum_vec1[1],
        angular_momentum_vec2[1],
        0,
        position1[1] - position2[1],
        alpha1,
        alpha2,
    );
    let S_z: f64 = calc_expansion_coeff_overlap_int(
        angular_momentum_vec1[2],
        angular_momentum_vec2[2],
        0,
        position1[2] - position2[2],
        alpha1,
        alpha2,
    );

    S_x * S_y * S_z * PI.powf(1.5) * (alpha1 + alpha2).recip().powf(1.5)

}

//TODO: this function is not complete yet -> use integrals.pdf to finish it
pub fn calc_overlap_int_cgto(ContrGaus1: &ContrGaus, ContrGaus2: &ContrGaus) -> f64 {
    // Calculate the overlap integral between two contracted Gaussian functions.
    //
    // # Arguments
    // ----------
    // ContrGaus1 : ContrGaus
    //   The first contracted Gaussian function.
    // ContrGaus2 : ContrGaus
    //   The second contracted Gaussian function.
    //
    // # Returns
    // --------
    // overlap_int : f64
    //   The overlap integral between the two contracted Gaussian functions.
    //

    let mut overlap_int: f64 = 0.0;
    for i in 0..ContrGaus1.no_contr {
        for j in 0..ContrGaus2.no_contr {
            overlap_int += ContrGaus1.coeff[i]
                * ContrGaus2.coeff[j]
                * calc_overlap_int_prim(
                    ContrGaus1.alpha[i],
                    ContrGaus2.alpha[j],
                    ContrGaus1.angular_momentum_vec,
                    ContrGaus2.angular_momentum_vec,
                    ContrGaus1.position,
                    ContrGaus2.position,
                );
        }
    }
    overlap_int *= ContrGaus1.norm_const * ContrGaus2.norm_const;
    overlap_int
}