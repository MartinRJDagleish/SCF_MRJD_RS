use ndarray::prelude::*;
use ndarray_linalg::{EigValsh, SymmetricSqrt};
use std::f64::consts::PI;
use std::fs;
// use physical_constants;
// use std::collections::HashSet;
// use std::io::{BufRead, BufReader};
// use std::{collections::HashMap, fs};

use crate::molecule::Molecule;
mod molecule;

fn main() {
    
    let output_beginning_string: String = String::from(r#"
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
        "#);
    println!("{}", output_beginning_string);

    //* Natural constants
    let h: f64 = physical_constants::PLANCK_CONSTANT;
    let c: f64 = physical_constants::SPEED_OF_LIGHT_IN_VACUUM;
    //*******************************************************************
    //*                         OOP WAY
    //*******************************************************************
    //* OOP way:

    let mut mol: Molecule = Molecule::new("inp/Project3/h2o.xyz", "inp/Project2/h2o.hess", 0);

    let run_project1: bool = true;
    let run_project2: bool = false;
    let run_project3: bool = false;


    if run_project1 {
        println!("Project 1 implementation:\n");
        // println!("Z values of atoms:\n{:?}\n", &mol.Z_vals);
        // println!("Geometry of molecule:\n{:?}\n", &mol.geom);
        //* Fancy print of geometry of file (check for valid input)
        println!("Input (bohr):");
        mol.print_geom_input();

        //* Step 2: Bond lengths
        println!("\nInteratomic distances (in bohr):");
        for i in 0..mol.no_atoms {
            for j in 0..i {
                if i != j {
                    let bond_length: f64 = mol.calc_r_ij(i, j);

                    println!("Distance between {}-{} is: {:3.5}", i, j, bond_length);
                } else {
                    continue;
                }
            }
        }

        //* Step 3: Bond angles
        println!("\nBond angles (in degrees):");
        for i in 0..mol.no_atoms {
            for j in 0..i {
                for k in 0..j {
                    if mol.calc_r_ij(i, j) < 4.0 && mol.calc_r_ij(j, k) < 4.0 {
                        let bond_angle: f64 = mol.calc_bond_angle(i, j, k);
                        println!("Angle for {}-{}-{} is: {:.5}", i, j, k, bond_angle);
                    }
                }
            }
        }

        //* Step 4: OOP angles
        println!("\nOut-of-plane angles (in degrees):\n");
        for i in 0..mol.no_atoms {
            for j in 0..mol.no_atoms {
                for k in 0..mol.no_atoms {
                    for l in 0..mol.no_atoms {
                        let bond_dist_jk: f64 = mol.calc_r_ij(j, k);
                        let bond_dist_kl: f64 = mol.calc_r_ij(k, l);
                        let bond_dist_ik: f64 = mol.calc_r_ij(i, k);
                        if i != j
                            && i != k
                            && i != l
                            && j != k
                            && k != l
                            && j != l
                            && bond_dist_jk < 4.0
                            && bond_dist_kl < 4.0
                            && bond_dist_ik < 4.0
                        {
                            let oop_angle: f64 = mol.calc_oop_angle(i, j, k, l);

                            println!("OOP angle for {}-{}-{}-{} is: {:.5}", i, j, k, l, oop_angle);
                        }
                    }
                }
            }
        }

        // * Step 5: Torsion / dihedral angles
        println!("\nTorsion angles (in degrees):\n");
        for i in 0..mol.no_atoms {
            for j in 0..i {
                for k in 0..j {
                    for l in 0..k {
                        let bond_dist_ij: f64 = mol.calc_r_ij(i, j);
                        let bond_dist_jk: f64 = mol.calc_r_ij(j, k);
                        let bond_dist_kl: f64 = mol.calc_r_ij(k, l);
                        if bond_dist_ij < 4.0 && bond_dist_jk < 4.0 && bond_dist_kl < 4.0 {
                            let dihedral_angle: f64 = mol.calc_dihedral_angle(i, j, k, l);
                            println!(
                                "Dihedral angle for {}-{}-{}-{} is: {:.5}",
                                i, j, k, l, dihedral_angle
                            );
                        }
                    }
                }
            }
        }

        //* Step 6: Center of mass
        println!("\nCenter of mass: {:?}", &mol.calc_center_mass());

        //* Step 6.5: Translate molecule such that center of mass is in middle of coordinate system
        println!("\nTranslate molecule such that center of mass is in middle of coordinate system");
        println!("Before translation:");
        mol.print_geom_input();

        println!("After translation:");
        mol.translate_mol_to_center_mass();
        mol.print_geom_input();

        //* Step 7: Inertia tensor
        println!("\nPrinting the moment of inertia tensor:");
        let mut inertia_tensor: Array2<f64> = mol.calc_inertia_tensor();
        println!("Inertia tensor: \n{:?}", inertia_tensor);

        //* Step 7.1 : Get eigenvalues and eigenvectors of inertia tensor
        let eigenvals: Array1<f64> = inertia_tensor
            .eigvalsh(ndarray_linalg::UPLO::Upper)
            .unwrap();

        println!(
            "Principal moments of inertia (amu * bohr^2): \n{:?}\n",
            &eigenvals
        );
        println!(
            "Principal moments of inertia (amu * Angstrom^2): \n{:?}\n",
            &eigenvals * (1.0e10 * physical_constants::BOHR_RADIUS).powi(2) //* prefactor but not exponent for conversion
        );
        println!(
            "Principal moments of inertia (g * cm^2): \n{:?}\n",
            &eigenvals
                * physical_constants::ATOMIC_MASS_CONSTANT
                * (100. * physical_constants::BOHR_RADIUS).powi(2)
        );

        //* Step 8: Rotational constants
        let conv_factor_recip_cm: f64 = 1.0
            / (100.0
                * physical_constants::ATOMIC_MASS_CONSTANT
                * physical_constants::BOHR_RADIUS.powi(2));
        // println!("\nConversion factor: {}\n", conv_factor_recip_cm);
        let rot_const_A_per_cm: f64 =
            conv_factor_recip_cm * (h / (8.0 * PI.powi(2) * c * &eigenvals[0]));
        let rot_const_B_per_cm: f64 =
            conv_factor_recip_cm * (h / (8.0 * PI.powi(2) * c * &eigenvals[1]));
        let rot_const_C_per_cm: f64 =
            conv_factor_recip_cm * (h / (8.0 * PI.powi(2) * c * &eigenvals[2]));
        println!(
            "Rotational constants (cm^-1): \nA: {:.4}\nB: {:.4}\nC: {:.4}\n",
            &rot_const_A_per_cm, &rot_const_B_per_cm, &rot_const_C_per_cm
        );

        //* Step 8.1: Classify the type of rotor for molecule
        println!("Classifying the type of rotor for molecule...");
        if mol.no_atoms == 2 {
            println!("Molecule is linear and diatomic!");
        } else if &eigenvals[0] < &1.0e-4 {
            println!("Molecule is linear!");
        } else if (&eigenvals[0] - &eigenvals[1]).abs() < 1.0e-4
            && (&eigenvals[1] - &eigenvals[2]).abs() < 1.0e-4
        {
            println!("Molecule is symmetric top!");
        } else if (&eigenvals[0] - &eigenvals[1]).abs() < 1.0e-4
            && (&eigenvals[1] - &eigenvals[2]).abs() > 1.0e-4
        {
            println!("Molecule is oblate symmetric top!")
        } else if (&eigenvals[0] - &eigenvals[1]).abs() > 1.0e-4
            && (&eigenvals[1] - &eigenvals[2]).abs() < 1.0e-4
        {
            println!("Molecule is a prolate symmetric top!")
        } else {
            println!("Molecule is an asymmetric top!");
        }
    }

    if run_project2 {
        // * Project 2: read coordinate data DONE -> read hessian
        // * Step 1: Read coordinates (see above in Molecule struct) */
        // * Step 2: Read the cartessian hessian data
        println!("\n\nReading the hessian data...");
        println!("{:1.5}\n", mol.hessian);

        //* Step 3: Mass-weight the hessian matrix
        mol.mass_weight_hessian();
        println!("Mass-weighted hessian matrix: \n{:1.5}\n", mol.hessian);

        //* Step 4: Calculate eigenvalues of the hessian matrix
        println!("\n\nCalculating eigenvalues of the hessian matrix...\n");
        println!("Eigenvalues: \n{:1.5}\n", mol.calc_hess_eigenvals());
        let conv_hess_to_waveno: f64 = (physical_constants::HARTREE_ENERGY
            / (physical_constants::BOHR_RADIUS.powi(2) * physical_constants::ATOMIC_MASS_CONSTANT))
            .sqrt()
            * (2.0 * PI * 100.0 * c).recip(); //* recip is ^-1, but "faster"
        let harm_vib_freqs: Array1<f64> = mol
            .calc_hess_eigenvals()
            .mapv(|x| conv_hess_to_waveno * x.sqrt());
        println!(
            "Harmonic vibrational frequencies:\n{:1.2}\n",
            harm_vib_freqs
        );
    }

    if run_project3 {
        //* Project 3: SCF (with data provided)
        // ! THIS IS A QUICK FIX AND NOT A GOOD SOLUTION
        let no_basis_funcs: usize = 7;
        //* Step 1: Read Nuclear Repulsion Energy (enuc) from file
        let e_nuc_val: f64 = fs::read_to_string("inp/Project3/STO-3G/enuc.dat")
            .expect("Failed to open enuc data!")
            .parse()
            .expect("Failed to parse enuc data file!");
        println!("Nuclear Repulsion Energy: {}\n", e_nuc_val);

        //* Step 2.1: Read the overlap matrix
        let mut S_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));
        let S_matr_file_contents =
            fs::read_to_string("inp/Project3/STO-3G/s.dat").expect("Failed to open S matrix data!");

        for line in S_matr_file_contents.lines() {
            let mut line_split: Vec<&str> = line.trim().split_whitespace().collect();
            let row: usize = line_split[0].parse::<usize>().unwrap() - 1;
            let col: usize = line_split[1].parse::<usize>().unwrap() - 1;
            let val: f64 = line_split[2].parse().unwrap();
            S_matr[(row, col)] = val;
            S_matr[(col, row)] = val;
        }
        println!("Overlap matrix S:\n{:1.5}\n", S_matr);

        //* Step 2.2: Read the kinetic energy matrix
        let mut T_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));
        let T_matr_file_contents =
            fs::read_to_string("inp/Project3/STO-3G/t.dat").expect("Failed to open T matrix data!");

        for line in T_matr_file_contents.lines() {
            let mut line_split: Vec<&str> = line.trim().split_whitespace().collect();
            let row: usize = line_split[0].parse::<usize>().unwrap() - 1;
            let col: usize = line_split[1].parse::<usize>().unwrap() - 1;
            let val: f64 = line_split[2].parse().unwrap();
            T_matr[(row, col)] = val;
            T_matr[(col, row)] = val;
        }
        println!("Kinetic energy matrix T:\n{:1.5}\n", T_matr);

        //* Step 2.3: Read the nuclear attraction matrix
        let mut V_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));
        let V_matr_file_contents =
            fs::read_to_string("inp/Project3/STO-3G/v.dat").expect("Failed to open V matrix data!");

        for line in V_matr_file_contents.lines() {
            let mut line_split: Vec<&str> = line.trim().split_whitespace().collect();
            let row: usize = line_split[0].parse::<usize>().unwrap() - 1;
            let col: usize = line_split[1].parse::<usize>().unwrap() - 1;
            let val: f64 = line_split[2].parse().unwrap();
            V_matr[(row, col)] = val;
            V_matr[(col, row)] = val;
        }
        println!("Nuclear attraction:\n{:1.5}\n", V_matr);

        //* Step 2.4: Form core Hamiltonian H_core = T + V
        let mut H_matr: Array2<f64> = &T_matr + &V_matr;
        println!("Core Hamiltonian:\n{:1.5}\n", H_matr);

        //* Step 3: Read the 2-electron integrals -> ERI tensor
        let (mut i, mut j, mut k, mut l) = (0, 0, 0, 0);
        let (mut ij, mut kl, mut ijkl) = (0, 0, 0);

        let ERI_file_contents = fs::read_to_string("inp/Project3/STO-3G/eri.dat")
            .expect("Failed to open ERI matrix data!");

        // let mut sums = Vec::new();
        // for i in 0..=100 {
        //     sums.push(sum_up_to(i));
        // }

        //* 2nd try but with using a vec!
        let mut ERI_vec: Vec<f64> = Vec::new();

        for line in ERI_file_contents.lines() {
            let mut line_split: Vec<&str> = line.trim().split_whitespace().collect();
            let i: usize = line_split[0].parse::<usize>().unwrap() - 1;
            let j: usize = line_split[1].parse::<usize>().unwrap() - 1;
            let k: usize = line_split[2].parse::<usize>().unwrap() - 1;
            let l: usize = line_split[3].parse::<usize>().unwrap() - 1;
            let val: f64 = line_split[4].parse().unwrap();
            let idx: usize = calc_ijkl_idx(i, j, k, l);

            if ERI_vec.len() < idx {
                ERI_vec.resize(idx, 0.0);
            }

            ERI_vec.insert(calc_ijkl_idx(i, j, k, l), val);
        }
        let ERI_array: Array1<f64> = Array1::from_vec(ERI_vec);
        println!("Electron-electron repulsion array:\n{:1.5}\n", &ERI_array);

        //* Step 4: Build the orthogonalization matrix
        let mut S_matr_inv_sqrt: Array2<f64> = S_matr.ssqrt(ndarray_linalg::UPLO::Upper).unwrap();

        println!("S^-1/2:\n{:1.5}\n", S_matr_inv_sqrt);
    }

    //*****************************************************************
    //*****************************************************************
    //*****************************************************************
    let end_of_calc_string: String = format!("{:^29}", "RUN ENDED SUCCESSFULLY!");
    println!("\n{}", "*".repeat(31));
    println!("*{}*", end_of_calc_string);
    println!("{}", "*".repeat(31));
}

fn calc_ijkl_idx(i: usize, j: usize, k: usize, l: usize) -> usize {
    let ij: usize = if i > j {
        calc_cmp_idx(i, j)
    } else {
        calc_cmp_idx(j, i)
    };
    let kl: usize = if k > l {
        calc_cmp_idx(k, l)
    } else {
        calc_cmp_idx(l, k)
    };
    let ijkl: usize = if ij > kl {
        calc_cmp_idx(ij, kl)
    } else {
        calc_cmp_idx(kl, ij)
    };
    ijkl
}

fn calc_cmp_idx(idx1: usize, idx2: usize) -> usize {
    let idx1idx2: usize = (idx1 * (idx1 + 1)) / 2 + idx2;
    idx1idx2
}

fn sum_up_to(i: i32) -> i32 {
    i * (i + 1) / 2
}
