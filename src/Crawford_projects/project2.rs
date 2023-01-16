use crate::molecule::Molecule;
use std::{fs, io::{BufReader, BufRead}, f64::consts::PI};
use ndarray::{Array1, Array2};
use physical_constants;


pub fn run_project2(mut mol: Molecule) {
    println!("\nProject 2 implementation:\n");
    // * Project 2: read coordinate data DONE -> read hessian
    // * Step 1: Read coordinates (see above in Molecule struct) */
    // * Step 2: Read the cartessian hessian data

    //* READING THE HESSIAN
    let hess_file_path: &str = "inp/Project2/h2o.hess";
    let hess_file = fs::File::open(hess_file_path).unwrap();
    let hess_reader = BufReader::new(hess_file);

    let mut hessian = Array2::<f64>::zeros((3 * mol.no_atoms * mol.no_atoms, 3));

    for (i, line) in hess_reader.lines().enumerate() {
        if i == 0 {
            let hess_no_atoms: usize = line.unwrap().trim().parse::<usize>().unwrap();
            println!("No of atoms in hessian: {}", hess_no_atoms);
            if mol.no_atoms != hess_no_atoms {
                panic!("Number of atoms in geom file and hessian file are not the same!");
            }
            continue;
        }
        let i = i - 1;
        let line = line.unwrap();
        let values = line
            .split_whitespace()
            .map(|s| s.parse::<f64>().unwrap())
            .collect::<Vec<f64>>();
        for (j, value) in values.iter().enumerate() {
            hessian[[i, j]] = *value;
        }
    }

    mol.hessian = hessian
        .into_shape((3 * mol.no_atoms, 3 * mol.no_atoms))
        .unwrap();
    // DEBUG
    // println!("Hessian before reshape: \n{:1.5}\n", &hessian);
    // println!("Hessian after reshape: \n{:1.5}\n", &mol.hessian);

    //* Step 3: Mass-weight the hessian matrix
    mol.mass_weight_hessian();
    println!("Mass-weighted hessian matrix: \n{:1.5}\n", mol.hessian);

    //* Step 4: Calculate eigenvalues of the hessian matrix
    println!("\n\nCalculating eigenvalues of the hessian matrix...\n");
    println!("Eigenvalues: \n{:1.5}\n", mol.calc_hess_eigenvals());
    let conv_hess_to_waveno: f64 = (physical_constants::HARTREE_ENERGY
        / (physical_constants::BOHR_RADIUS.powi(2) * physical_constants::ATOMIC_MASS_CONSTANT))
        .sqrt()
        * (2.0 * PI * 100.0 * physical_constants::SPEED_OF_LIGHT_IN_VACUUM).recip(); //* recip is ^-1, but "faster"
    let harm_vib_freqs: Array1<f64> = mol
        .calc_hess_eigenvals()
        .mapv(|x| conv_hess_to_waveno * x.sqrt());
    println!(
        "Harmonic vibrational frequencies:\n{:1.2}\n",
        harm_vib_freqs
    );
}
