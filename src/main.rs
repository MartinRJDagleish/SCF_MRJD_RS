#![allow(non_snake_case)]

use std::time::Instant;

use crate::molecule::Molecule;
// use clap::Parser;

/// Simple program to greet a person
// #[command(author, version, about, long_about = None)]
// #[derive(Parser, Default, Debug)]
// #[clap(
//     author = "Martin Dagleish",
//     version,
//     about = "A RHF SCF programme written in Rust;
//     implemented from scratch using the McMurchie-Davidson algorithm"
// )]
// struct Args {
//     test_str: String,
// }

// pub mod Crawford_projects;
// pub mod dev_SCF_from_scratch;
pub mod molecule;
pub mod print_utils;

// #[allow(non_snake_case)] // * -> I need this due to QM naming conventions
fn main() {
    let start_time = Instant::now();
    print_utils::print_header();
    print_utils::print_general_information();

    // *****************************************************************
    // * Development of the code
    // *****************************************************************
    let mol: Molecule = Molecule::new("inp/Project3_1/STO-3G/h2o_v2.xyz", 0);

    // let _is_run_project1: bool = false; //* General molecule geometry stuff
    // let _is_run_project2: bool = false; //* Hessian -> eigenfreqs from file
    // let _is_run_project3_1: bool = false; //* SCF from precomputed integrals
    // let is_run_project3_2: bool = false; //* Ints for SCF from "scratch"
    // let is_run_project3_3: bool = false; //* actual SCF from "scratch"
    // let _is_run_project4: bool = false; //* MP2 from precomputed integrals

    // if is_run_project1 {
    //     Crawford_projects::project1::run_project1(mol.clone());
    // }

    // if is_run_project2 {
    //     Crawford_projects::project2::run_project2(mol.clone());
    // }

    // if is_run_project3_1 {
    //     Crawford_projects::project3_1::run_project3_1(mol.clone(), is_run_project4);
    // }

    // if is_run_project3_2 {
    //     use crate::dev_SCF_from_scratch::project3_2::*;
    //     run_project3_2();
    // }

    // if is_run_project3_3 {
    //     // use crate::SCF_from_scratch::project3_3::*;
    //     // run_project3_3();
    //     use crate::dev_SCF_from_scratch::project3_3_h2o::*;
    //     run_project3_3_h2o();
    // }
    // *****************************************************************

    let is_run_dev = true; //* Development of the code
    if is_run_dev {
        use crate::molecule::wfn::scf::*;
        let mut scf: SCF = SCF::new(mol);
        scf.RHF_par(false, "sto-3g");
        // scf.MP2(false, "def2-SVP");
    }

    // let args = Args::parse();
    // println!("Hello {}!", args.test_str);

    //*****************************************************************
    //*****************************************************************
    //*****************************************************************
    print_utils::print_footer(start_time);
}
