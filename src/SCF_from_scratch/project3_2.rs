use ndarray::prelude::*;

use crate::molecule;
//* For testing:
use crate::molecule::wfn::ints::{
    calc_kinetic_energy_int_cgto, calc_nuc_attr_int_cgto, calc_overlap_int_cgto,
};
use crate::molecule::wfn::*;

pub fn run_project3_2() {
    println!("\nRunning project 3.2 (SCF from 'scratch')");

    let mut mol_sto_3g = molecule::Molecule::new("inp/Project3_2/geom/h2.xyz", 0);
    let mut mol_6_311g = molecule::Molecule::new("inp/Project3_2/geom/h2.xyz", 0);

    // let mol = molecule::Molecule::new("H2O", "sto-3g", "h2o.xyz");
    //* Define the primitive gaussians
    //* STO-3G here
    println!("Defining the primitive gaussians");
    println!("Test molecule: H2 (STO-3G)");

    // * The first H atom -> H1
    mol_sto_3g.wfn_total.ContrGauss_vec =
        vec![ContractedGaussian::new(vec![PrimitiveGaussian::new(
            0.3425250914E1,
            0.1543289673E0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        )])];
    mol_sto_3g.wfn_total.ContrGauss_vec[0]
        .PrimGauss_vec
        .push(PrimitiveGaussian::new(
            0.6239137298E0,
            0.5353281423E0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_sto_3g.wfn_total.ContrGauss_vec[0]
        .PrimGauss_vec
        .push(PrimitiveGaussian::new(
            0.1688554040E0,
            0.4446345422E0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    // * The second H atom -> H2
    mol_sto_3g
        .wfn_total
        .ContrGauss_vec
        .push(ContractedGaussian::new(vec![PrimitiveGaussian::new(
            0.3425250914E1,
            0.1543289673E0,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));
    mol_sto_3g.wfn_total.ContrGauss_vec[1]
        .PrimGauss_vec
        .push(PrimitiveGaussian::new(
            0.6239137298E0,
            0.5353281423E0,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_sto_3g.wfn_total.ContrGauss_vec[1]
        .PrimGauss_vec
        .push(PrimitiveGaussian::new(
            0.1688554040E0,
            0.4446345422E0,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    mol_sto_3g.wfn_total.update_no_of_contr_gauss();

    // let mol_basis_set_STO_3G = molecule::wfn::WfnTotal::new(vec![H1_contr_gaus, H2_contr_gaus]);
    //* Test:
    // println!("{:?}", mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[0].alpha);
    let S_matr = mol_sto_3g.wfn_total.calc_S_matr_l_eq_0();
    println!("S_matr:\n{:^5.6}\n", &S_matr);
    let T_matr = mol_sto_3g.wfn_total.calc_T_matr_l_eq_0();
    println!("T_matr:\n{:^5.6}\n", &T_matr);
    let V_ne_matr = mol_sto_3g.wfn_total.calc_V_ne_matr_l_eq_0();
    println!("V_ne_matr:\n{:^5.6}\n", &V_ne_matr);
    let V_ee_matr = mol_sto_3g.wfn_total.calc_V_ee_matr_l_eq_0();
    println!("V_ee_matr:\n{:^5.6}\n", &V_ee_matr);

    //* Define the primitive gaussians
    //* 6-311G here
    println!("Defining the primitive gaussians");
    println!("Test molecule: H2 (6-311G)");

    //* The first H atom -> H1
    // H1_contr_gauss_1s
    mol_6_311g.wfn_total.ContrGauss_vec =
        vec![ContractedGaussian::new(vec![PrimitiveGaussian::new(
            33.86500,
            0.0254938,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        )])];
    mol_6_311g.wfn_total.ContrGauss_vec[0]
        .PrimGauss_vec
        .push(PrimitiveGaussian::new(
            5.094790,
            0.190373,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_6_311g.wfn_total.ContrGauss_vec[0]
        .PrimGauss_vec
        .push(PrimitiveGaussian::new(
            1.158790,
            0.852161,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    // H1_contr_gauss_2s
    mol_6_311g
        .wfn_total
        .ContrGauss_vec
        .push(ContractedGaussian::new(vec![PrimitiveGaussian::new(
            0.325840,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    // H1_contr_gauss_3s
    mol_6_311g
        .wfn_total
        .ContrGauss_vec
        .push(ContractedGaussian::new(vec![PrimitiveGaussian::new(
            0.102741,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    // H2_contr_gauss_1s
    mol_6_311g.wfn_total.ContrGauss_vec =
        vec![ContractedGaussian::new(vec![PrimitiveGaussian::new(
            33.86500,
            0.0254938,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )])];
    mol_6_311g.wfn_total.ContrGauss_vec[0]
        .PrimGauss_vec
        .push(PrimitiveGaussian::new(
            5.094790,
            0.190373,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_6_311g.wfn_total.ContrGauss_vec[0]
        .PrimGauss_vec
        .push(PrimitiveGaussian::new(
            1.158790,
            0.852161,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    // H2_contr_gauss_2s
    mol_6_311g
        .wfn_total
        .ContrGauss_vec
        .push(ContractedGaussian::new(vec![PrimitiveGaussian::new(
            0.325840,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    // H2_contr_gauss_3s
    mol_6_311g
        .wfn_total
        .ContrGauss_vec
        .push(ContractedGaussian::new(vec![PrimitiveGaussian::new(
            0.102741,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    mol_6_311g.wfn_total.update_no_of_contr_gauss();

    //* Tests:
    let S_matr = mol_6_311g.wfn_total.calc_S_matr_l_eq_0();
    println!("S_matr:\n{:^5.6}\n", &S_matr);
    let T_matr = mol_6_311g.wfn_total.calc_T_matr_l_eq_0();
    println!("T_matr:\n{:^5.6}\n", &T_matr);
    // This implementation of V_ne_matr only works for STO-3G basis set:
    // let V_ne_matr = mol_6_311g.wfn_total.calc_V_ne_matr_l_eq_0();
    // println!("V_ne_matr:\n{:^5.6}\n", &V_ne_matr);
    let V_ee_matr = mol_6_311g.wfn_total.calc_V_ee_matr_l_eq_0();
    println!("V_ee_matr:\n{:^5.6}\n", &V_ee_matr);

    //* Test new code for overlap_int:
    // use crate::molecule::wfn::ints::*; //* for testing

    println!("Overlap integrals:");
    let mut S_matr_test = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.no_of_contr_gauss,
        mol_6_311g.wfn_total.no_of_contr_gauss,
    ));
    for i in 0..mol_6_311g.wfn_total.no_of_contr_gauss {
        for j in 0..mol_6_311g.wfn_total.no_of_contr_gauss {
            S_matr_test[(i, j)] = calc_overlap_int_cgto(
                &mol_6_311g.wfn_total.ContrGauss_vec[i],
                &mol_6_311g.wfn_total.ContrGauss_vec[j],
            );
        }
    }
    println!("{:^5.6}\n", &S_matr_test);

    println!("Kinetic energy integrals:");
    let mut T_matr_test = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.no_of_contr_gauss,
        mol_6_311g.wfn_total.no_of_contr_gauss,
    ));
    for i in 0..mol_6_311g.wfn_total.no_of_contr_gauss {
        for j in 0..mol_6_311g.wfn_total.no_of_contr_gauss {
            T_matr_test[(i, j)] = calc_kinetic_energy_int_cgto(
                &mol_6_311g.wfn_total.ContrGauss_vec[i],
                &mol_6_311g.wfn_total.ContrGauss_vec[j],
            );
        }
    }
    println!("{:^5.6}\n", &T_matr_test);

    println!("Nuclear attraction integrals:");
    let mut V_ne_matr_test = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.no_of_contr_gauss,
        mol_6_311g.wfn_total.no_of_contr_gauss,
    ));
    for i in 0..mol_6_311g.wfn_total.no_of_contr_gauss {
        for j in 0..mol_6_311g.wfn_total.no_of_contr_gauss {
            for atom_pos in mol_6_311g.geom_obj.geom_matr.axis_iter(ndarray::Axis(0)) {
                V_ne_matr_test[(i, j)] += calc_nuc_attr_int_cgto(
                    &mol_6_311g.wfn_total.ContrGauss_vec[i],
                    &mol_6_311g.wfn_total.ContrGauss_vec[j],
                    &atom_pos.to_owned(),
                );

                // }
            }
        }
    }
    println!("{:^5.6}\n", &V_ne_matr_test);

    //* Test new code for parse_basis_set_file
    {
        // use crate::molecule::wfn::parse_BSSE_basis_set::BasisSet::parse_basis_set_file;
        // parse_basis_set_file("STO-3G");
    }
}

// ! Template for tests: (run with: 'cargo test')
// #[cfg(test)]
// mod tests{
//     use super::*;

//     #[test]
//     fn test_calc_overlap_int_prim() {
//     }
// }
