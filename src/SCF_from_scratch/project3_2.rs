use ndarray::prelude::*;

use crate::molecule;

pub fn run_project3_2() {
    println!("\nRunning project 3.2 (SCF from 'scratch')");
    // let test_pg = mol.
    // let alpha_test: f64 = 1.0;
    // let cgto_coeff_test: f64 = 0.4;
    // let position_test = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    // let angular_momentum_test = Array1::<i32>::from_vec(vec![0, 0, 0]);
    // let norm_const_test: f64 = 0.3;

    // let mut prim_test = molecule::wfn::molecule::wfn::PrimitiveGaussian::new(
    //     alpha_test,
    //     cgto_coeff_test,
    //     position_test,
    //     angular_momentum_test,
    // );

    //* DEBUG
    // println!("prim_test: {:?}", prim_test);

    // let mol = molecule::Molecule::new("H2O", "sto-3g", "h2o.xyz");
    //* Define the primitive gaussians
    //* STO-3G here
    println!("Defining the primitive gaussians");
    println!("Test molecule: H2 (STO-3G)");
    let H1_prim_gaus_1s_1 = molecule::wfn::PrimitiveGaussian::new(
        0.3425250914E1,
        0.1543289673E0,
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0, 0, 0]),
    );
    let H1_prim_gaus_1s_2 = molecule::wfn::PrimitiveGaussian::new(
        0.6239137298E0,
        0.5353281423E0,
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0, 0, 0]),
    );
    let H1_prim_gaus_1s_3 = molecule::wfn::PrimitiveGaussian::new(
        0.1688554040E0,
        0.4446345422E0,
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0, 0, 0]),
    );

    let H2_prim_gaus_1s_1 = molecule::wfn::PrimitiveGaussian::new(
        0.3425250914E1,
        0.1543289673E0,
        Array1::from_vec(vec![0.0, 0.0, 1.4]),
        Array1::from_vec(vec![0, 0, 0]),
    );
    let H2_prim_gaus_1s_2 = molecule::wfn::PrimitiveGaussian::new(
        0.6239137298E0,
        0.5353281423E0,
        Array1::from_vec(vec![0.0, 0.0, 1.4]),
        Array1::from_vec(vec![0, 0, 0]),
    );
    let H2_prim_gaus_1s_3 = molecule::wfn::PrimitiveGaussian::new(
        0.1688554040E0,
        0.4446345422E0,
        Array1::from_vec(vec![0.0, 0.0, 1.4]),
        Array1::from_vec(vec![0, 0, 0]),
    );

    let H1_contr_gaus = molecule::wfn::ContractedGaussian::new(vec![
        H1_prim_gaus_1s_1,
        H1_prim_gaus_1s_2,
        H1_prim_gaus_1s_3,
    ]);
    let H2_contr_gaus = molecule::wfn::ContractedGaussian::new(vec![
        H2_prim_gaus_1s_1,
        H2_prim_gaus_1s_2,
        H2_prim_gaus_1s_3,
    ]);

    let mol_basis_set_STO_3G = molecule::wfn::BasisSet::new(vec![H1_contr_gaus, H2_contr_gaus]);
    //* Test:
    // println!("{:?}", mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[0].alpha);
    let S_matr = mol_basis_set_STO_3G.calc_S_matr_l_eq_0();
    println!("S_matr:\n{:^5.6}\n", &S_matr);
    let T_matr = mol_basis_set_STO_3G.calc_T_matr_l_eq_0();
    println!("T_matr:\n{:^5.6}\n", &T_matr);
    let V_ne_matr = mol_basis_set_STO_3G.calc_V_ne_matr_l_eq_0();
    println!("V_ne_matr:\n{:^5.6}\n", &V_ne_matr);
    let V_ee_matr = mol_basis_set_STO_3G.calc_V_ee_matr_l_eq_0();
    println!("V_ee_matr:\n{:^5.6}\n", &V_ee_matr);

    //* Define the primitive gaussians
    //* 6-311G here
    println!("Defining the primitive gaussians");
    println!("Test molecule: H2 (6-311G)");

    let H1_prim_gaus_1s_1 = molecule::wfn::PrimitiveGaussian::new(
        33.86500,
        0.0254938,
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0, 0, 0]),
    );
    let H1_prim_gaus_1s_2 = molecule::wfn::PrimitiveGaussian::new(
        5.094790,
        0.190373,
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0, 0, 0]),
    );
    let H1_prim_gaus_1s_3 = molecule::wfn::PrimitiveGaussian::new(
        1.158790,
        0.852161,
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0, 0, 0]),
    );

    let H1_prim_gaus_2s_1 = molecule::wfn::PrimitiveGaussian::new(
        0.325840,
        1.000000,
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0, 0, 0]),
    );

    let H1_prim_gaus_3s_1 = molecule::wfn::PrimitiveGaussian::new(
        0.102741,
        1.000000,
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0, 0, 0]),
    );

    let H2_prim_gaus_1s_1 = molecule::wfn::PrimitiveGaussian::new(
        33.86500,
        0.0254938,
        Array1::from_vec(vec![0.0, 0.0, 1.4]),
        Array1::from_vec(vec![0, 0, 0]),
    );
    let H2_prim_gaus_1s_2 = molecule::wfn::PrimitiveGaussian::new(
        5.094790,
        0.190373,
        Array1::from_vec(vec![0.0, 0.0, 1.4]),
        Array1::from_vec(vec![0, 0, 0]),
    );
    let H2_prim_gaus_1s_3 = molecule::wfn::PrimitiveGaussian::new(
        1.158790,
        0.852161,
        Array1::from_vec(vec![0.0, 0.0, 1.4]),
        Array1::from_vec(vec![0, 0, 0]),
    );

    let H2_prim_gaus_2s_1 = molecule::wfn::PrimitiveGaussian::new(
        0.325840,
        1.000000,
        Array1::from_vec(vec![0.0, 0.0, 1.4]),
        Array1::from_vec(vec![0, 0, 0]),
    );

    let H2_prim_gaus_3s_1 = molecule::wfn::PrimitiveGaussian::new(
        0.102741,
        1.000000,
        Array1::from_vec(vec![0.0, 0.0, 1.4]),
        Array1::from_vec(vec![0, 0, 0]),
    );

    let H1_contr_gaus_1s = molecule::wfn::ContractedGaussian::new(vec![
        H1_prim_gaus_1s_1,
        H1_prim_gaus_1s_2,
        H1_prim_gaus_1s_3,
    ]);
    let H1_contr_gaus_2s = molecule::wfn::ContractedGaussian::new(vec![H1_prim_gaus_2s_1]);
    let H1_contr_gaus_3s = molecule::wfn::ContractedGaussian::new(vec![H1_prim_gaus_3s_1]);

    let H2_contr_gaus_1s = molecule::wfn::ContractedGaussian::new(vec![
        H2_prim_gaus_1s_1,
        H2_prim_gaus_1s_2,
        H2_prim_gaus_1s_3,
    ]);
    let H2_contr_gaus_2s = molecule::wfn::ContractedGaussian::new(vec![H2_prim_gaus_2s_1]);
    let H2_contr_gaus_3s = molecule::wfn::ContractedGaussian::new(vec![H2_prim_gaus_3s_1]);

    let mol_basis_set_6311G = molecule::wfn::BasisSet::new(vec![
        H1_contr_gaus_1s,
        H1_contr_gaus_2s,
        H1_contr_gaus_3s,
        H2_contr_gaus_1s,
        H2_contr_gaus_2s,
        H2_contr_gaus_3s,
    ]);
    //* Test:
    // println!("{:?}", mol_basis_set_6311G.ContrGauss_vec[0].PrimGauss_vec[0].alpha);
    let S_matr = mol_basis_set_6311G.calc_S_matr_l_eq_0();
    println!("S_matr:\n{:^5.6}\n", &S_matr);
    let T_matr = mol_basis_set_6311G.calc_T_matr_l_eq_0();
    println!("T_matr:\n{:^5.6}\n", &T_matr);
    let V_ne_matr = mol_basis_set_6311G.calc_V_ne_matr_l_eq_0();
    println!("V_ne_matr:\n{:^5.6}\n", &V_ne_matr);
    let V_ee_matr = mol_basis_set_6311G.calc_V_ee_matr_l_eq_0();
    println!("V_ee_matr:\n{:^5.6}\n", &V_ee_matr);

    //* Test new code for overlap_int:
    use crate::molecule::wfn::ints::*; //* for testing
    let S_matr_new_val_test: f64 = calc_overlap_int_prim(
        mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[0].alpha,
        mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[1].alpha,
        mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[0]
            .angular_momentum_vec
            .clone(),
        mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[1]
            .angular_momentum_vec
            .clone(),
        mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[0]
            .position
            .clone(),
        mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[1]
            .position
            .clone(),
    );
    println!("S_matr_new_val_test: {}", S_matr_new_val_test);
}

// ! Template for tests: (run with: 'cargo test')
// #[cfg(test)]
// mod tests{
//     use super::*;    

//     #[test]
//     fn test_calc_overlap_int_prim() {
//     }
// }
