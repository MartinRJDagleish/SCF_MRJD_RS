use std::f64::consts::PI;
use ndarray::{Array1, Array2};
use ndarray_linalg::EigValsh;

use crate::molecule::Molecule;

#[allow(non_snake_case)] 
pub fn run_project1(mut mol: Molecule) {
    println!("\nProject 1 implementation:\n");
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
    println!("\nCenter of mass: {:^.6}", &mol.calc_center_mass());

    //* Step 6.5: Translate molecule such that center of mass is in middle of coordinate system
    println!("\nTranslate molecule such that center of mass is in middle of coordinate system");
    println!("\nBefore translation:");
    mol.print_geom_input();

    println!("\nAfter translation:");
    mol.translate_mol_to_center_mass();
    mol.print_geom_input();

    //* Step 7: Inertia tensor
    println!("\nPrinting the moment of inertia tensor:");
    let inertia_tensor: Array2<f64> = mol.calc_inertia_tensor();
    println!("Inertia tensor: \n{:^.5}\n", inertia_tensor);

    //* Step 7.1 : Get eigenvalues and eigenvectors of inertia tensor
    let eigenvals: Array1<f64> = inertia_tensor
        .eigvalsh(ndarray_linalg::UPLO::Upper)
        .unwrap();

    println!(
        "Principal moments of inertia (amu * bohr^2): \n{:^.5}\n",
        &eigenvals
    );
    println!(
        "Principal moments of inertia (amu * Angstrom^2): \n{:^.5}\n",
        &eigenvals * (1.0e10 * physical_constants::BOHR_RADIUS).powi(2) //* prefactor but not exponent for conversion
    );
    println!(
        "Principal moments of inertia (g * cm^2): \n{:^.5e}\n",
        &eigenvals
            * physical_constants::ATOMIC_MASS_CONSTANT
            * (100. * physical_constants::BOHR_RADIUS).powi(2)
    );

    //* Step 8: Rotational constants
    let conv_factor_recip_cm: f64 = (100.0
        * physical_constants::ATOMIC_MASS_CONSTANT
        * physical_constants::BOHR_RADIUS.powi(2))
    .recip();
    // println!("\nConversion factor: {}\n", conv_factor_recip_cm);
    let rot_const_A_per_cm: f64 = conv_factor_recip_cm
        * physical_constants::PLANCK_CONSTANT
        * (8.0 * PI.powi(2) * physical_constants::SPEED_OF_LIGHT_IN_VACUUM * &eigenvals[0]).recip();
    let rot_const_B_per_cm: f64 = conv_factor_recip_cm
        * physical_constants::PLANCK_CONSTANT
        * (8.0 * PI.powi(2) * physical_constants::SPEED_OF_LIGHT_IN_VACUUM * &eigenvals[1]).recip();
    let rot_const_C_per_cm: f64 = conv_factor_recip_cm
        * physical_constants::PLANCK_CONSTANT
        * (8.0 * PI.powi(2) * physical_constants::SPEED_OF_LIGHT_IN_VACUUM * &eigenvals[2]).recip();
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

    // mol
}
