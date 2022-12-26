use ndarray::{Array2, Array3, Array4};
use std::fs;
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
    Z_vals: Vec<i32>,
    // point_group
    // point_group: String,
}

impl Molecule {
    fn new(geomfile: &str, charge: i32) -> Molecule {
        //* Step 1: Read the coord data from input
        println!("In file {}", geomfile);

        //* Show contents of file
        let contents: String = fs::read_to_string(geomfile).expect("Failed to read geomfile!");

        //* Read no of atoms first for array size
        let no_atoms: usize = contents.lines().nth(0).unwrap().parse().unwrap();
        println!("No of atoms: {}", no_atoms);

        let no_cart_coords: usize = 3;
        let mut Z_vals: Vec<i32> = vec![0; no_atoms];
        let mut geom: Array2<f64> = Array2::zeros((no_atoms, no_cart_coords));

        for (line_idx, line) in contents.lines().skip(1).enumerate() {
            let line_split: Vec<&str> = line.split_whitespace().collect();

            Z_vals[line_idx] = line_split[0].parse().unwrap();

            for coord_idx in 0..3 {
                geom[(line_idx, coord_idx)] = line_split[coord_idx + 1].parse().unwrap();
            }
        }

        Molecule {
            charge,
            no_atoms,
            geom,
            Z_vals,
        }
    }

    fn calc_vec_cros_prod(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let mut vec_cros_prod: Vec<f64> = vec![0.0; 3];

        vec_cros_prod[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        vec_cros_prod[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        vec_cros_prod[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

        return vec_cros_prod;
    }

    fn calc_bond_length(geom: &Array2<f64>, idx1: usize, idx2: usize) -> f64 {
        let mut bond_length: f64 = 0.0;
        for cart_coord in 0..3 {
            bond_length += (geom[(idx1, cart_coord)] - geom[(idx2, cart_coord)]).powi(2);
        }
        return bond_length.sqrt();
    }

    fn calc_bond_angle(geom: &Array2<f64>, idx1: usize, idx2: usize, idx3: usize) -> f64 {
        let mut bond_angle: f64 = 0.0;
        let bond_length_buff_ij: f64 = Molecule::calc_bond_length(&geom, idx1, idx2);
        let bond_length_buff_jk: f64 = Molecule::calc_bond_length(&geom, idx2, idx3);

        let mut unit_ij: Vec<f64> = vec![0.0; 3];
        let mut unit_jk: Vec<f64> = vec![0.0; 3];

        for cart_coord in 0..3 {
            unit_ij[cart_coord] =
                (geom[(idx1, cart_coord)] - geom[(idx2, cart_coord)]) / bond_length_buff_ij;
            unit_jk[cart_coord] =
                (geom[(idx2, cart_coord)] - geom[(idx3, cart_coord)]) / bond_length_buff_jk;

            bond_angle += -unit_ij[cart_coord] * unit_jk[cart_coord];
        }

        return bond_angle.acos() * (180.0 / std::f64::consts::PI);
    }

    fn calc_scalar_prod(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
        let mut scalar_prod: f64 = 0.0;
        for cart_coord in 0..3 {
            scalar_prod += vec1[cart_coord] * vec2[cart_coord];
        }
        return scalar_prod;
    }
    // fn calc_oop_angle(geom: &Array2<f64>, idx1: usize, idx2: usize, idx3: usize, idx4: usize) -> f64 {
    //     let mut oop_angle: f64 = 0.0;
    //     let bond_length_buff_ij: f64 = Molecule::calc_bond_length(&geom, idx1, idx2);
    //     let bond_length_buff_jk: f64 = Molecule::calc_bond_length(&geom, idx2, idx3);
    //     let bond_length_buff_kl: f64 = Molecule::calc_bond_length(&geom, idx3, idx4);

    //     let mut unit_ij: Vec<f64> = vec![0.0; 3];
    //     let mut unit_jk: Vec<f64> = vec![0.0; 3];
    //     let mut unit_kl: Vec<f64> = vec![0.0; 3];

    //     for cart_coord in 0..3 {
    //         unit_ij[cart_coord] =
    //             (geom[(idx1, cart_coord)] - geom[(idx2, cart_coord)]) / bond_length_buff_ij;
    //         unit_jk[cart_coord] =
    //             (geom[(idx2, cart_coord)] - geom[(idx3, cart_coord)]) / bond_length_buff_jk;
    //         unit_kl[cart_coord] =
    //             (geom[(idx3, cart_coord)] - geom[(idx4, cart_coord)]) / bond_length_buff_kl;
    //     }

    //     let mut vec_cross_prod: Vec<f64> = Molecule::calc_vec_cros_prod(&unit_ij, &unit_jk);

    //     return oop_angle
    // }

    fn print_geom(&self) {
        todo!()
    }

    // fn print_geom(&self) -> Result<&str, _> {
    //     println!("Geometry of molecule:");
    //     for i in 0..self.geom.shape()[0] {
    //         for j in 0..self.geom.shape()[1] {
    //             print!("{:10.6} ", self.geom[(i, j)]);
    //         }
    //         println!("");
    //     }
    //     Ok("Done")
    // }

    // fn print_Z_vals(&self) -> Result<&str, _> {
    //     println!("Z values of atoms:");
    //     for i in 0..self.Z_vals.len() {
    //         print!("{:10} ", self.Z_vals[i]);
    //     }
    //     println!("");
    //     Ok("Done")
    // }
}

fn main() {
    //*******************************************************************
    //*                         OOP WAY
    //*******************************************************************
    //* OOP way:

    //* Step 1: Read file into buffer
    println!("Starting the OOP way:");
    let mol: Molecule = Molecule::new("inp/geom.xyz", 0);

    println!("Z values of atoms:\n{:?}\n", mol.Z_vals);
    println!("Geometry of molecule:\n{:?}\n", mol.geom);

    //* Step 2: Bond lengths
    println!("Interatomic distances (in bohr):\n");
    for i in 0..mol.no_atoms {
        for j in 0..i {
            if i != j {
                let bond_length: f64 = Molecule::calc_bond_length(&mol.geom, i, j);

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
                if Molecule::calc_bond_length(&mol.geom, i, j) < 4.0
                    && Molecule::calc_bond_length(&mol.geom, j, k) < 4.0
                {
                    let bond_angle: f64 = Molecule::calc_bond_angle(&mol.geom, i, j, k);

                    println!("Angle for {}-{}-{} is: {:.5}", i, j, k, bond_angle);
                }
            }
        }
    }

    //* Step 4: OOP angles
    println!("\n OOP angles (in degrees):\n");
    for i in 0..mol.no_atoms {
        for j in 0..mol.no_atoms {
            for k in 0..mol.no_atoms {
                for l in 0..mol.no_atoms {
                    let bond_dist_jk: f64 = Molecule::calc_bond_length(&mol.geom, j, k);
                    let bond_dist_kl: f64 = Molecule::calc_bond_length(&mol.geom, k, l);
                    let bond_dist_ik: f64 = Molecule::calc_bond_length(&mol.geom, i, k);
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
                        let mut unit_kj: Vec<f64> = vec![0.0; 3];
                        let mut unit_kl: Vec<f64> = vec![0.0; 3];
                        let mut unit_ki: Vec<f64> = vec![0.0; 3];

                        for cart_coord in 0..3 {
                            unit_kj[cart_coord] = (mol.geom[(k, cart_coord)]
                                - mol.geom[(j, cart_coord)])
                                / bond_dist_jk;
                            unit_kl[cart_coord] = (mol.geom[(k, cart_coord)]
                                - mol.geom[(l, cart_coord)])
                                / bond_dist_kl;
                            unit_ki[cart_coord] = (mol.geom[(k, cart_coord)]
                                - mol.geom[(i, cart_coord)])
                                / bond_dist_ik;
                        }

                        let cross_prod: Vec<f64> = Molecule::calc_vec_cros_prod(&unit_kj, &unit_kl);

                        let mut oop_angle: f64 = Molecule::calc_scalar_prod(&cross_prod, &unit_ki);
                        oop_angle = (oop_angle
                            / Molecule::calc_bond_angle(&mol.geom, j, k, l).sin())
                        .asin() * (180.0 / std::f64::consts::PI);

                        println!("OOP angle for {}-{}-{}-{} is: {:.5}", i, j, k, l, oop_angle);
                    }
                }
            }
        }
    }
}
