use humantime::format_duration;
use ndarray::prelude::*;
use ndarray_linalg::{EigValsh, Eigh, Inverse, SymmetricSqrt};
use std::f64::consts::PI;
use std::fs;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use crate::molecule::wfn::{BasisSet, ContractedGaussian, PrimitiveGaussian};
use crate::molecule::wfn::ints::*;
use crate::molecule::*;
mod molecule;
mod Crawford_projects;

#[allow(non_snake_case)] // * -> I need this due to QM naming conventions
fn main() {
    let start_exec_time = Instant::now();
    let ASCII_art_logo: String = String::from(r#"
    _____/\\\\\\\\\\\___________/\\\\\\\\\___/\\\\\\\\\\\\\\\_                                
     ___/\\\/////////\\\______/\\\////////___\/\\\///////////__                               
      __\//\\\______\///_____/\\\/____________\/\\\_____________                              
       ___\////\\\___________/\\\______________\/\\\\\\\\\\\_____                             
        ______\////\\\_______\/\\\______________\/\\\///////______                            
         _________\////\\\____\//\\\_____________\/\\\_____________                           
          __/\\\______\//\\\____\///\\\___________\/\\\_____________                          
           _\///\\\\\\\\\\\/_______\////\\\\\\\\\__\/\\\_____________                         
            ___\///////////____________\/////////___\///______________                        
    __/\\\\____________/\\\\_____/\\\\\\\\\___________/\\\\\\\\\\\___/\\\\\\\\\\\\____        
     _\/\\\\\\________/\\\\\\___/\\\///////\\\________\/////\\\///___\/\\\////////\\\__       
      _\/\\\//\\\____/\\\//\\\__\/\\\_____\/\\\____________\/\\\______\/\\\______\//\\\_      
       _\/\\\\///\\\/\\\/_\/\\\__\/\\\\\\\\\\\/_____________\/\\\______\/\\\_______\/\\\_     
        _\/\\\__\///\\\/___\/\\\__\/\\\//////\\\_____________\/\\\______\/\\\_______\/\\\_    
         _\/\\\____\///_____\/\\\__\/\\\____\//\\\____________\/\\\______\/\\\_______\/\\\_   
          _\/\\\_____________\/\\\__\/\\\_____\//\\\____/\\\___\/\\\______\/\\\_______/\\\__  
           _\/\\\_____________\/\\\__\/\\\______\//\\\__\//\\\\\\\\\_______\/\\\\\\\\\\\\/___ 
            _\///______________\///___\///________\///____\/////////________\////////////_____
     _______________/\\\\\\\\\__________/\\\\\\\\\\\___                                       
      _____________/\\\///////\\\______/\\\/////////\\\_                                      
       ____________\/\\\_____\/\\\_____\//\\\______\///__                                     
        ____________\/\\\\\\\\\\\/_______\////\\\_________                                    
         ____________\/\\\//////\\\__________\////\\\______                                   
          ____________\/\\\____\//\\\____________\////\\\___                                  
           ____________\/\\\_____\//\\\____/\\\______\//\\\__                                 
            ____________\/\\\______\//\\\__\///\\\\\\\\\\\/___                                
             ____________\///________\///_____\///////////_____  
        "#,
    );
    println!("{}", ASCII_art_logo); // test

    let mut mol: Molecule = Molecule::new("inp/Project3_1/STO-3G/h2o_v2.xyz", 0);

    let is_run_project1: bool = true; //* General molecule geometry stuff
    let is_run_project2: bool = true; //* Hessian -> eigenfreqs from file
    let is_run_project3_1: bool = true; //* SCF from precomputed integrals
    let is_run_project3_2: bool = true; //* SCF from "scratch"
    let is_run_project4: bool = false; //* MP2 from precomputed integrals
    
    
    if is_run_project1 {
        Crawford_projects::project1::run_project1(mol.clone());
    }

    if is_run_project2 {
        Crawford_projects::project2::run_project2(mol.clone());
    }

    if is_run_project3_1 {
        Crawford_projects::project3_1::run_project3_1(mol.clone(), is_run_project4);
    }

    if is_run_project3_2 {
        println!("\nRunning project 3.2 (SCF from 'scratch')");
        // let test_pg = mol.
        // let alpha_test: f64 = 1.0;
        // let cgto_coeff_test: f64 = 0.4;
        // let position_test = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        // let angular_momentum_test = Array1::<i32>::from_vec(vec![0, 0, 0]);
        // let norm_const_test: f64 = 0.3;

        // let mut prim_test = molecule::wfn::PrimitiveGaussian::new(
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
        let H1_prim_gaus_1s_1 = PrimitiveGaussian::new(
            0.3425250914E1,
            0.1543289673E0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        );
        let H1_prim_gaus_1s_2 = PrimitiveGaussian::new(
            0.6239137298E0,
            0.5353281423E0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        );
        let H1_prim_gaus_1s_3 = PrimitiveGaussian::new(
            0.1688554040E0,
            0.4446345422E0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        );

        let H2_prim_gaus_1s_1 = PrimitiveGaussian::new(
            0.3425250914E1,
            0.1543289673E0,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        );
        let H2_prim_gaus_1s_2 = PrimitiveGaussian::new(
            0.6239137298E0,
            0.5353281423E0,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        );
        let H2_prim_gaus_1s_3 = PrimitiveGaussian::new(
            0.1688554040E0,
            0.4446345422E0,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        );

        let H1_contr_gaus = ContractedGaussian::new(vec![
            H1_prim_gaus_1s_1,
            H1_prim_gaus_1s_2,
            H1_prim_gaus_1s_3,
        ]);
        let H2_contr_gaus = ContractedGaussian::new(vec![
            H2_prim_gaus_1s_1,
            H2_prim_gaus_1s_2,
            H2_prim_gaus_1s_3,
        ]);

        let mol_basis_set_STO_3G = BasisSet::new(vec![H1_contr_gaus, H2_contr_gaus]);
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

        let H1_prim_gaus_1s_1 = PrimitiveGaussian::new(
            33.86500,
            0.0254938,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        );
        let H1_prim_gaus_1s_2 = PrimitiveGaussian::new(
            5.094790,
            0.190373,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        );
        let H1_prim_gaus_1s_3 = PrimitiveGaussian::new(
            1.158790,
            0.852161,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        );

        let H1_prim_gaus_2s_1 = PrimitiveGaussian::new(
            0.325840,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        );

        let H1_prim_gaus_3s_1 = PrimitiveGaussian::new(
            0.102741,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![0, 0, 0]),
        );

        let H2_prim_gaus_1s_1 = PrimitiveGaussian::new(
            33.86500,
            0.0254938,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        );
        let H2_prim_gaus_1s_2 = PrimitiveGaussian::new(
            5.094790,
            0.190373,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        );
        let H2_prim_gaus_1s_3 = PrimitiveGaussian::new(
            1.158790,
            0.852161,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        );

        let H2_prim_gaus_2s_1 = PrimitiveGaussian::new(
            0.325840,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        );

        let H2_prim_gaus_3s_1 = PrimitiveGaussian::new(
            0.102741,
            1.000000,
            Array1::from_vec(vec![0.0, 0.0, 1.4]),
            Array1::from_vec(vec![0, 0, 0]),
        );

        let H1_contr_gaus_1s = ContractedGaussian::new(vec![
            H1_prim_gaus_1s_1,
            H1_prim_gaus_1s_2,
            H1_prim_gaus_1s_3,
        ]);
        let H1_contr_gaus_2s = ContractedGaussian::new(vec![H1_prim_gaus_2s_1]);
        let H1_contr_gaus_3s = ContractedGaussian::new(vec![H1_prim_gaus_3s_1]);

        let H2_contr_gaus_1s = ContractedGaussian::new(vec![
            H2_prim_gaus_1s_1,
            H2_prim_gaus_1s_2,
            H2_prim_gaus_1s_3,
        ]);
        let H2_contr_gaus_2s = ContractedGaussian::new(vec![H2_prim_gaus_2s_1]);
        let H2_contr_gaus_3s = ContractedGaussian::new(vec![H2_prim_gaus_3s_1]);

        let mol_basis_set_6311G = BasisSet::new(vec![
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
        let S_matr_new_val_test: f64 = calc_overlap_int_prim(
            mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[0].alpha,
            mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[1].alpha,
            mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[0].angular_momentum_vec.clone(),
            mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[1].angular_momentum_vec.clone(),
            mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[0].position.clone(),
            mol_basis_set_STO_3G.ContrGauss_vec[0].PrimGauss_vec[1].position.clone(),
        );
        println!("S_matr_new_val_test: {}", S_matr_new_val_test);
    }
    //*****************************************************************
    //*****************************************************************
    //*****************************************************************
    let end_of_calc_string: String = format!("{:^29}", "RUN ENDED SUCCESSFULLY!");
    println!("\n{}", "*".repeat(31));
    println!("*{}*", end_of_calc_string);
    println!("{}", "*".repeat(31));
    let end_exec_time = Instant::now();
    let duration_exec_time = end_exec_time.duration_since(start_exec_time);
    let formatted_duration_exec_time = format_duration(duration_exec_time).to_string();
    println!(
        "\nTime elapsed in execution is: {}",
        formatted_duration_exec_time
    );
}
