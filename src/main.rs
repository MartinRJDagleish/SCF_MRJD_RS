use ndarray::prelude::*;
use ndarray::{Array1, Array2};
use ndarray_linalg::{EigValsh, Scalar};
use physical_constants;
use std::f64::consts::PI;
use std::{collections::HashMap, fs};
// use array2d::{Array2D}; # does not work as intended
// use nalgebra::{Vector3, Matrix3};
// use std::io;

// * Version 2 of trying to implement an OOP approach
struct Molecule {
    // charge
    charge: i32,
    // number of atoms
    no_atoms: usize,
    // cartesian coordinates (geometry of molecule)
    geom: Array2<f64>,
    // atomic numbers
    Z_vals: Vec<usize>,
    // HashMap of atomic masses
    mass_map: HashMap<usize, f64>,
    // point_group
    // point_group: String,
}

impl Molecule {
    fn new(geomfile: &str, charge: i32) -> Molecule {
        //* Step 1: Read the coord data from input
        println!("In file {}", geomfile);

        //* Show contents of file
        let geomfile_contents: String =
            fs::read_to_string(geomfile).expect("Failed to read geomfile!");

        //* Read no of atoms first for array size
        let no_atoms: usize = geomfile_contents.lines().nth(0).unwrap().parse().unwrap();
        println!("No of atoms: {}", no_atoms);

        let mut Z_vals: Vec<usize> = vec![0; no_atoms];
        let mut geom: Array2<f64> = Array2::zeros((no_atoms, 3));

        for (line_idx, line) in geomfile_contents.lines().skip(1).enumerate() {
            let line_split: Vec<&str> = line.split_whitespace().collect();

            Z_vals[line_idx] = line_split[0].parse().unwrap();

            for cart_coord in 0..3 {
                geom[(line_idx, cart_coord)] = line_split[cart_coord + 1].parse().unwrap();
            }
        }

        // Main hashmap for masses
        let mut mass_map: HashMap<usize, f64> = HashMap::new();

        let masses_path: &str = "inp/masses.csv";
        let massses_content: String =
            fs::read_to_string(masses_path).expect("Failed to read geomfile!");

        for (Z_val, line) in massses_content.lines().enumerate() {
            if Z_val == 0 {
                continue;
            }
            let line_vec: Vec<&str> = line.split(',').collect();
            let mass: f64 = line_vec[0].parse().unwrap();
            // println!("Mass of atom with Z value {}: {}", Z_val, mass);
            mass_map.insert(Z_val, mass);
        }

        Molecule {
            charge,
            no_atoms,
            geom,
            Z_vals,
            mass_map,
        }
    }

    /// Returns the geom of this [`Molecule`].
    fn print_geom(&self) {
        // println!("Printing geometry of molecule:\n");
        for i in 0..self.no_atoms {
            println!(
                "{}\t{:.4}\t{:.4}\t{:.4}",
                self.Z_vals[i],
                self.geom[(i, 0)],
                self.geom[(i, 1)],
                self.geom[(i, 2)]
            );
        }
    }

    // Calculate the distance between 2 3-d cart coords points
    fn calc_r_ij(&self, i: usize, j: usize) -> f64 {
        let mut bond_length: f64 = 0.0;
        for cart_coord in 0..3 {
            bond_length += (&self.geom[(j, cart_coord)] - &self.geom[(i, cart_coord)]).powi(2);
        }
        return bond_length.sqrt();
    }

    fn calc_e_ij(&self, i: usize, j: usize) -> Vec<f64> {
        let mut unit_vec: Vec<f64> = vec![0.0; 3];
        let r_ij: f64 = self.calc_r_ij(i, j);
        for cart_coord in 0..3 {
            unit_vec[cart_coord] =
                (&self.geom[(j, cart_coord)] - &self.geom[(i, cart_coord)]) / r_ij;
        }
        return unit_vec;
    }

    fn calc_vec_norm(vec1: &Vec<f64>) -> f64 {
        let mut vec_norm: f64 = 0.0;
        for cart_coord in 0..3 {
            vec_norm += vec1[cart_coord].powi(2);
        }
        return vec_norm.sqrt();
    }

    fn calc_vec_cross_prod(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let mut vec_cros_prod: Vec<f64> = vec![0.0; 3];

        vec_cros_prod[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        vec_cros_prod[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        vec_cros_prod[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

        return vec_cros_prod;
    }

    fn calc_unit_vec_cross_prod(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let mut vec_cros_prod: Vec<f64> = vec![0.0; 3];
        let cos_v1_v2: f64 = Self::calc_scalar_prod(&vec1, &vec2);
        let sin_v1_v2: f64 = (1.0f64 - cos_v1_v2.powi(2)).sqrt();

        vec_cros_prod[0] = (vec1[1] * vec2[2] - vec1[2] * vec2[1]) / sin_v1_v2;
        vec_cros_prod[1] = (vec1[2] * vec2[0] - vec1[0] * vec2[2]) / sin_v1_v2;
        vec_cros_prod[2] = (vec1[0] * vec2[1] - vec1[1] * vec2[0]) / sin_v1_v2;

        return vec_cros_prod;
    }

    fn calc_scalar_prod(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
        let mut scalar_prod: f64 = 0.0;
        for cart_coord in 0..3 {
            scalar_prod += vec1[cart_coord] * vec2[cart_coord];
        }
        return scalar_prod;
    }

    fn calc_bond_angle(&self, idx1: usize, idx2: usize, idx3: usize) -> f64 {
        let unit_21: Vec<f64> = self.calc_e_ij(idx2, idx1);
        let unit_23: Vec<f64> = self.calc_e_ij(idx2, idx3);
        let bond_angle: f64 = Self::calc_scalar_prod(&unit_21, &unit_23);
        return bond_angle.acos().to_degrees();
    }

    fn calc_oop_angle(&self, idx1: usize, idx2: usize, idx3: usize, idx4: usize) -> f64 {
        // // Version 1 -> wrong?
        // let bond_dist_jk: f64 = self.calc_r_ij(idx2, idx3);
        // let bond_dist_kl: f64 = self.calc_r_ij(idx3, idx2);
        // let bond_dist_ik: f64 = self.calc_r_ij(idx1, idx3);

        // let mut unit_kj: Vec<f64> = vec![0.0; 3];
        // let mut unit_kl: Vec<f64> = vec![0.0; 3];
        // let mut unit_ki: Vec<f64> = vec![0.0; 3];

        // for cart_coord in 0..3 {
        //     unit_kj[cart_coord] =
        //         (&self.geom[(idx2, cart_coord)] - &self.geom[(idx3, cart_coord)]) / bond_dist_jk;
        //     unit_kl[cart_coord] =
        //         (&self.geom[(idx4, cart_coord)] - &self.geom[(idx3, cart_coord)]) / bond_dist_kl;
        //     unit_ki[cart_coord] =
        //         (&self.geom[(idx1, cart_coord)] - &self.geom[(idx3, cart_coord)]) / bond_dist_ik;
        // }

        // // println!("unit_kj: {:?}", unit_kj);
        // // println!("unit_kl: {:?}", unit_kl);
        // // println!("unit_ki: {:?}", unit_ki);
        // let cross_prod: Vec<f64> = Self::calc_unit_vec_cross_prod(&unit_kj, &unit_kl);
        // // println!("unit_kj: {:?}", unit_kj);
        // // println!("unit_kl: {:?}", unit_kl);
        // // println!("unit_ki: {:?}", unit_ki);
        // // println!("cross_prod: {:?}", cross_prod);

        // let mut oop_angle: f64 = (Self::calc_scalar_prod(&cross_prod, &unit_ki))
        //     / self.calc_bond_angle(idx2, idx3, idx4).sin();

        // if oop_angle < -1.0 {
        //     oop_angle = -1.0f64.asin();
        // } else if oop_angle > 1.0 {
        //     oop_angle = 1.0f64.asin();
        // } else {
        //     oop_angle = oop_angle.asin();
        // }

        // return oop_angle.to_degrees();

        //! Version 2 -> correct
        let unit_kj: Vec<f64> = self.calc_e_ij(idx3, idx2);
        let unit_kl: Vec<f64> = self.calc_e_ij(idx3, idx4);
        let unit_ki: Vec<f64> = self.calc_e_ij(idx3, idx1);

        //* Working, but nicer below
        // let mut x_prod_kj_kl: Vec<f64> = Self::calc_vec_cross_prod(&unit_kj, &unit_kl);
        // let sin_phi_jkl = self.calc_bond_angle(idx2, idx3, idx4).to_radians().sin();
        // x_prod_kj_kl = x_prod_kj_kl
        //     .iter()
        //     .map(|x| x / sin_phi_jkl)
        //     .collect();

        let x_prod_kj_kl_norm = Self::calc_unit_vec_cross_prod(&unit_kj, &unit_kl);

        let oop_angle: f64 = Self::calc_scalar_prod(&x_prod_kj_kl_norm, &unit_ki);

        if oop_angle < -1.0 {
            return -1.0f64.asin().to_degrees();
        } else if oop_angle > 1.0 {
            return 1.0f64.asin().to_degrees();
        } else {
            return oop_angle.asin().to_degrees();
        }
    }

    fn calc_dihedral_angle(&self, idx1: usize, idx2: usize, idx3: usize, idx4: usize) -> f64 {
        //! Version1 just from math formula (Crawford)
        // let bond_dist_ij: f64 = self.calc_r_ij(i, j);
        // let bond_dist_jk: f64 = self.calc_r_ij(j, k);
        // let bond_dist_kl: f64 = self.calc_r_ij(k, l);

        // let mut unit_ij: Vec<f64> = vec![0.0; 3];
        // let mut unit_jk: Vec<f64> = vec![0.0; 3];
        // let mut unit_kl: Vec<f64> = vec![0.0; 3];

        // for cart_coord in 0..3 {
        //     unit_ij[cart_coord] =
        //         (&self.geom[(i, cart_coord)] - &self.geom[(j, cart_coord)]) / bond_dist_ij;
        //     unit_jk[cart_coord] =
        //         (&self.geom[(j, cart_coord)] - &self.geom[(k, cart_coord)]) / bond_dist_jk;
        //     unit_kl[cart_coord] =
        //         (&self.geom[(k, cart_coord)] - &self.geom[(l, cart_coord)]) / bond_dist_kl;
        // }

        // let mut cross_prod_1: Vec<f64> = Self::calc_vec_cross_prod(&unit_ij, &unit_jk);

        // let mut cross_prod_2: Vec<f64> = Self::calc_vec_cross_prod(&unit_jk, &unit_kl);

        // let numerator: f64 = Self::calc_scalar_prod(&cross_prod_1, &cross_prod_2);
        // let denom: f64 = self.calc_bond_angle(i, j, k).sin()
        //     * self.calc_bond_angle(j, k, l).sin();

        // println!("Numerator: {}", numerator);
        // println!("Denominator: {}", denom);
        // let mut dihedral_angle: f64 = numerator / denom;

        // if dihedral_angle < -1.0 {
        //     dihedral_angle = -1.0f64.acos();
        // } else if dihedral_angle > 1.0 {
        //     dihedral_angle = 1.0f64.acos();
        // } else {
        //     dihedral_angle = dihedral_angle.acos();
        // }

        // //* Compute the sign of the torsion angle
        // let cross_prod_1_norm_factor: f64 = Self::calc_vec_norm(&cross_prod_1);
        // cross_prod_1 = cross_prod_1
        //     .iter()
        //     .map(|x| x / cross_prod_1_norm_factor)
        //     .collect();
        // let cross_prod_2_norm_factor: f64 = Self::calc_vec_norm(&cross_prod_2);
        // cross_prod_2 = cross_prod_2
        //     .iter()
        //     .map(|x| x / cross_prod_2_norm_factor)
        //     .collect();
        // let numerator: f64 = Self::calc_scalar_prod(&cross_prod_1, &cross_prod_2);
        // let sign = if numerator > 1.0 { 1.0 } else { -1.0 };

        // return (sign * dihedral_angle).to_degrees();

        //! Version 2 of dihedral angle calculation -> using the TMP python as refence
        // let unit_21: Vec<f64> = self.calc_e_ij(idx2, idx1);
        // let unit_23: Vec<f64> = self.calc_e_ij(idx2, idx3);
        // let unit_32: Vec<f64> = self.calc_e_ij(idx3, idx2);
        // let unit_34: Vec<f64> = self.calc_e_ij(idx3, idx4);

        // let x_prod_u_21_u_23: Vec<f64> = Self::calc_unit_vec_cross_prod(&unit_21, &unit_23);
        // let x_prod_u_32_u_34: Vec<f64> = Self::calc_unit_vec_cross_prod(&unit_32, &unit_34);

        // let torsion_angle: f64 = Self::calc_scalar_prod(&x_prod_u_21_u_23, &x_prod_u_32_u_34);
        // let mut sign_test_scalar_prod: f64 =
        //     Self::calc_scalar_prod(&x_prod_u_21_u_23, &unit_34);

        // if sign_test_scalar_prod < 0.0 {
        //     sign_test_scalar_prod = 1.0;
        // } else if sign_test_scalar_prod > 0.0 {
        //     sign_test_scalar_prod = -1.0;
        // }

        // return (sign_test_scalar_prod * torsion_angle.acos()).to_degrees();
        //! Version 4 working now!!! -> EITHER calc_unit_vec_cross_prod OR calc_vec_cross_prod and div by sines
        let e_ij: Vec<f64> = self.calc_e_ij(idx1, idx2);
        let e_jk: Vec<f64> = self.calc_e_ij(idx2, idx3);
        let e_kl: Vec<f64> = self.calc_e_ij(idx3, idx4);

        let x_prod_e_ij_e_jk: Vec<f64> = Self::calc_unit_vec_cross_prod(&e_ij, &e_jk);
        let x_prod_e_jk_e_kl: Vec<f64> = Self::calc_unit_vec_cross_prod(&e_jk, &e_kl);

        let cos_tors_angle: f64 = Self::calc_scalar_prod(&x_prod_e_ij_e_jk, &x_prod_e_jk_e_kl);

        if cos_tors_angle > 1.0 {
            return 0.0;
        } else if cos_tors_angle < -1.0 {
            return 180.0;
        } else {
            return cos_tors_angle.acos().to_degrees();
        }

        // let numerator: f64 = dot_prod;
        // let denom: f64 = self.calc_bond_angle(idx1, idx2, idx3).sin()
        //     * self.calc_bond_angle(idx2, idx3, idx4).sin();

        // let mut tau: f64 = 0.0;
        // let mut cos_tau: f64 = numerator / denom;
        // println!("Cos tau: {}", cos_tau);

        // if cos_tau > 1.0 {
        //     tau = 1.0f64.acos();
        // } else if cos_tau < -1.0 {
        //     tau = std::f64::consts::PI;
        // } else {
        //     tau = cos_tau.acos();
        // }
        // return tau.to_degrees();
    }

    fn get_mass_Z_val(&self, Z_val: &usize) -> f64 {
        return self.mass_map.get(Z_val).unwrap().clone();
    }

    fn calc_center_mass(&self) -> Vec<f64> {
        let mut total_mass: f64 = 0.0;
        let mut center_mass_vec: Vec<f64> = vec![0.0; 3];

        for (idx, Z_val) in self.Z_vals.iter().enumerate() {
            let mass_Z_val: f64 = self.get_mass_Z_val(Z_val);
            total_mass += mass_Z_val;

            for cart_coord in 0..3 {
                center_mass_vec[cart_coord] += mass_Z_val * self.geom[(idx, cart_coord)];
            }
        }

        center_mass_vec = center_mass_vec.iter().map(|x| x / total_mass).collect();
        return center_mass_vec;
    }

    fn translate_mol_to_center_mass(&mut self) {
        let center_mass_vec: Vec<f64> = self.calc_center_mass();

        for (idx, Z_val) in self.Z_vals.iter().enumerate() {
            for cart_coord in 0..3 {
                self.geom[(idx, cart_coord)] -= center_mass_vec[cart_coord];
            }
        }
    }

    fn other_two(n: usize) -> Vec<usize> {
        let arr: [usize; 3] = [0, 1, 2];
        arr.iter()
            .filter(|&x| *x != n)
            .map(|x| *x)
            .collect::<Vec<usize>>()
    }

    fn calc_inertia_tensor(&self) -> Array2<f64> {
        let mut inertia_tensor: Array2<f64> = Array2::<f64>::zeros((3, 3));

        for i in 0..3 {
            for j in 0..3 {
                for (idx, Z_val) in self.Z_vals.iter().enumerate() {
                    let mass_Z_val: f64 = self.get_mass_Z_val(Z_val);
                    if i == j {
                        let (i1, i2) = (Self::other_two(i)[0], Self::other_two(i)[1]);
                        inertia_tensor[(i, j)] += mass_Z_val
                            * (self.geom[(idx, i1)].powi(2) + self.geom[(idx, i2)].powi(2));
                    } else {
                        inertia_tensor[(i, j)] -=
                            mass_Z_val * self.geom[(idx, i)] * self.geom[(idx, j)];
                    }
                }
            }
        }

        return inertia_tensor;
    }
}

fn main() {
    //* Natural constants
    let h: f64 = physical_constants::PLANCK_CONSTANT;
    // let h_bar: f64 = physical_constants::REDUCED_PLANCK_CONSTANT;
    let c: f64 = physical_constants::SPEED_OF_LIGHT_IN_VACUUM;
    //*******************************************************************
    //*                         OOP WAY
    //*******************************************************************
    //* OOP way:

    //* Step 1: Read file into buffer
    println!("Starting the OOP way:");
    let mut mol: Molecule = Molecule::new("inp/allene_bohr.xyz", 0);
    // let mol: Molecule = Molecule::new("QM_Programm/inp/geom.xyz", 0);

    // println!("Z values of atoms:\n{:?}\n", mol.Z_vals);
    // println!("Geometry of molecule:\n{:?}\n", mol.geom);
    //* Fancy print of geometry of file (check for valid input)
    mol.print_geom();

    //* Step 2: Bond lengths
    println!("\nInteratomic distances (in bohr):\n");
    for i in 0..mol.no_atoms {
        for j in 0..i {
            if i != j {
                let bond_length: f64 = mol.calc_r_ij(i, j);

                println!("Atom {}-{} is: {:.5}", i, j, bond_length);
            } else {
                continue;
            }
        }
    }

    //* Step 3: Bond angles
    println!("\nBond angles (in degrees):\n");
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
    // TODO: Implement the center of mass calculation

    //* Step 6: Center of mass
    println!("\nCenter of mass: {:?}", mol.calc_center_mass());

    //* Step 6.5: Translate molecule such that center of mass is in middle of coordinate system
    println!("\nTranslate molecule such that center of mass is in middle of coordinate system");
    println!("Before translation:");
    mol.print_geom();

    println!("After translation:");
    mol.translate_mol_to_center_mass();
    mol.print_geom();

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


    println!("***********************");
    println!("RUN ENDED SUCCESSFULLY!");
    println!("***********************");
}
