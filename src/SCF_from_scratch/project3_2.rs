use ndarray::prelude::*;

use crate::molecule;
//* For testing:
use crate::molecule::wfn::*;
use crate::molecule::wfn::basisset::*;

pub fn run_project3_2() {
    println!("\nRunning project 3.2 (SCF from 'scratch')");

    let mut mol_sto_3g = molecule::Molecule::new("inp/Project3_2/geom/h2.xyz", Some(0));
    let mut mol_6_311g = molecule::Molecule::new("inp/Project3_2/geom/h2.xyz", Some(0));

    // * Parse the basis set file
    let basis_set_name = "sto-3g";
    let basis_set_tot_def: BasisSetTotalDef = parse_basis_set_file_gaussian(basis_set_name);


    // let mol = molecule::Molecule::new("H2O", "sto-3g", "h2o.xyz");
    //* Define the primitive gaussians
    //* STO-3G here
    println!("Defining the primitive gaussians");
    println!("Test molecule: H2 (STO-3G)");

    // * The first H atom -> H1
    mol_sto_3g.wfn_total.basis_set_total.basis_set_cgtos =
        vec![CGTO::new(vec![PGTO::new(
            0.3425250914E1,
            0.1543289673E0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        )])];
    mol_sto_3g.wfn_total.basis_set_total.basis_set_cgtos[0]
        .pgto_vec
        .push(PGTO::new(
            0.6239137298E0,
            0.5353281423E0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_sto_3g.wfn_total.basis_set_total.basis_set_cgtos[0]
        .pgto_vec
        .push(PGTO::new(
            0.1688554040E0,
            0.4446345422E0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_sto_3g.wfn_total.basis_set_total.basis_set_cgtos[0].update_no_pgtos();

    // * The second H atom -> H2
    mol_sto_3g
        .wfn_total
        .basis_set_total.basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.3425250914E1,
            0.1543289673E0,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));
    mol_sto_3g.wfn_total.basis_set_total.basis_set_cgtos[1]
        .pgto_vec
        .push(PGTO::new(
            0.6239137298E0,
            0.5353281423E0,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_sto_3g.wfn_total.basis_set_total.basis_set_cgtos[1]
        .pgto_vec
        .push(PGTO::new(
            0.1688554040E0,
            0.4446345422E0,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    for cgto in &mut mol_sto_3g.wfn_total.basis_set_total.basis_set_cgtos {
        cgto.update_no_pgtos();
    }

    // mol_sto_3g.wfn_total.update_no_of_contr_gauss();
    mol_sto_3g.wfn_total.basis_set_total.update_no_cgtos();

    // let mol_basis_set_STO_3G = molecule::wfn::WfnTotal::new(vec![H1_contr_gaus, H2_contr_gaus]);
    //* Test:
    // println!("{:?}", mol_basis_set_STO_3G.ContrGauss_vec[0].pgto_vec[0].alpha);
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
    mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos =
        vec![CGTO::new(vec![PGTO::new(
            33.86500,
            0.0254938,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        )])];
    mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[0]
        .pgto_vec
        .push(PGTO::new(
            5.094790,
            0.190373,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[0]
        .pgto_vec
        .push(PGTO::new(
            1.158790,
            0.852161,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    // H1_contr_gauss_2s
    mol_6_311g
        .wfn_total
        .basis_set_total.basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.325840,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    // H1_contr_gauss_3s
    mol_6_311g
        .wfn_total
        .basis_set_total.basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.102741,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    //* The second H atom -> H2
    // H2_contr_gauss_1s
    mol_6_311g
        .wfn_total
        .basis_set_total.basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            33.86500,
            0.0254938,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));
    mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[3]
        .pgto_vec
        .push(PGTO::new(
            5.094790,
            0.190373,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        ));
    mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[3]
        .pgto_vec
        .push(PGTO::new(
            1.158790,
            0.852161,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        ));

    // H2_contr_gauss_2s
    mol_6_311g
        .wfn_total
        .basis_set_total.basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.325840,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));

    // H2_contr_gauss_3s
    mol_6_311g
        .wfn_total
        .basis_set_total.basis_set_cgtos
        .push(CGTO::new(vec![PGTO::new(
            0.102741,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        )]));


    for cgto in &mut mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos {
        cgto.update_no_pgtos();
    }
    // mol_6_311g.wfn_total.update_no_of_contr_gauss();
    mol_6_311g.wfn_total.basis_set_total.update_no_cgtos();
    
    
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
    use crate::molecule::wfn::integrals::*; //* for testing

    println!("\n****************************************");
    println!("           TESTS FOR INTEGRALS         ");
    println!("****************************************");
    // println!("mol_6_311g : {:?}", &mol_6_311g.wfn_total);
    println!("Basis: 6-311G");
    println!("\nOverlap integrals (S matrix):");
    let mut S_matr_test = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
    ));
    for i in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
        for j in 0..=i {
            if i == j {
                S_matr_test[(i, j)] = 1.0;
                continue;
            } else {
                S_matr_test[(i, j)] = calc_overlap_int_cgto(
                    &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[i],
                    &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[j],
                );

                S_matr_test[(j, i)] = S_matr_test[(i, j)];
            }
        }
    }
    println!("{:^5.6}\n", &S_matr_test);

    println!("Kinetic energy integrals (T matrix):");
    let mut T_matr_test = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
    ));
    for i in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
        for j in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            T_matr_test[(i, j)] = calc_kin_energy_int_cgto(
                &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[i],
                &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[j],
            );
        }
    }
    println!("{:^5.6}\n", &T_matr_test);

    println!("Nuclear attraction integrals (V_ne matrix):");
    let mut V_ne_matr_test = Array2::<f64>::zeros((
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
    ));
    for i in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
        for j in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            for (idx, atom_pos) in mol_6_311g
                .geom_obj
                .geom_matr
                .axis_iter(ndarray::Axis(0))
                .enumerate()
            {
                V_ne_matr_test[(i, j)] -= (mol_6_311g.geom_obj.Z_vals[idx] as f64)
                    * calc_nuc_attr_int_cgto(
                        &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[i],
                        &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[j],
                        &atom_pos.to_owned(),
                    );
            }
        }
    }
    println!("{:^5.6}\n", &V_ne_matr_test);

    println!("Electron-electron repulsion integrals (V_ee / ERI matrix):");
    let mut V_ee_matr_test = Array4::<f64>::zeros((
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
        mol_6_311g.wfn_total.basis_set_total.no_cgtos,
    ));
    for i in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
        for j in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
            for k in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
                for l in 0..mol_6_311g.wfn_total.basis_set_total.no_cgtos {
                    V_ee_matr_test[(i, j, k, l)] = calc_elec_elec_repul_cgto(
                        &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[i],
                        &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[j],
                        &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[k],
                        &mol_6_311g.wfn_total.basis_set_total.basis_set_cgtos[l],
                    );
                }
            }
        }
    }

    println!("{:^5.6}\n", &V_ee_matr_test);

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
