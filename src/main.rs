// use array2d::{Array2D}; # does not work as intended
use ndarray::{Array2, Array3, Array4};
// use nalgebra::{Vector3, Matrix3};
use std::fs;

fn main() {
    //* Step 1: Read the coord data from input
    let geomfile: &str = "inp/geom.xyz";
    println!("In file {}", geomfile);

    //* Show contents of file
    let contents: String = fs::read_to_string(geomfile).expect("Failed to read geomfile!");

    //* Read no of atoms first for array size
    let no_atoms: usize = contents.lines().nth(0).unwrap().parse().unwrap();
    println!("No of atoms: {}", no_atoms);

    let num_cart_coords: usize = 3;
    let mut Z_vals: Vec<i32> = vec![0; no_atoms];
    let mut geom: Array2<f64> = Array2::zeros((no_atoms, num_cart_coords));

    for (line_idx, line) in contents.lines().skip(1).enumerate() {
        // println!("Line {}: {}", line_idx, line);

        // println!("{}", line.split_whitespace().next().unwrap());
        Z_vals[line_idx] = line.split_whitespace().next().unwrap().parse().unwrap();

        for coord_idx in 0..3 {
            geom[(line_idx, coord_idx)] = line
                .split_whitespace()
                .nth(coord_idx + 1)
                .unwrap()
                .parse()
                .unwrap();
        }
    }

    //* Debugging
    // println!("{:?}", Z_vals);
    // println!("{:?}", geom);

    //* Step 2: Bond lengths

    let mut bond_lengths: Array2<f64> = Array2::zeros((no_atoms, no_atoms));

    for i in 0..no_atoms {
        for j in 0..i {
            if i == j {
                bond_lengths[(i, j)] = 0.0;
            } else {
                let mut bond_length: f64 = 0.0;
                for k in 0..3 {
                    bond_length += (geom[(i, k)] - geom[(j, k)]).powi(2);
                }
                bond_lengths[(i, j)] = bond_length.sqrt();
            }
        }
    }

    //* Debugging
    // println!("{:?}", bond_lengths);

    //* Step 3: Bond angles
    // let mut bond_angles: Array2<f64> = Array2::zeros((no_atoms, no_atoms));

    let mut bond_angles: Array3<f64> = Array3::zeros((no_atoms, no_atoms, no_atoms));
    let mut unit_vec_ij: Vec<f64> = vec![0.0; 3];
    let mut unit_vec_jk: Vec<f64> = vec![0.0; 3];

    for i in 0..no_atoms {
        for j in 0..i {
            for k in 0..j {
                let mut bond_angle: f64 = 0.0;
                let bond_length_buff_ij: f64 = calc_bond_length(&geom, i, j);
                let bond_length_buff_jk: f64 = calc_bond_length(&geom, j, k);

                for cart_coord in 0..3 {
                    unit_vec_ij[cart_coord] =
                        (geom[(i, cart_coord)] - geom[(j, cart_coord)]) / bond_length_buff_ij;
                    unit_vec_jk[cart_coord] =
                        (geom[(j, cart_coord)] - geom[(k, cart_coord)]) / bond_length_buff_jk;

                    bond_angle += -unit_vec_ij[cart_coord] * unit_vec_jk[cart_coord];
                    bond_angles[(i, j, k)] = bond_angle.acos() * (180.0 / std::f64::consts::PI);
                }
            }
        }
    }

    //* Debugging
    // println!("{:?}", bond_angles);

    //* Step 4: OOP angles

    let mut oop_angles: Array4<f64> = Array4::zeros((no_atoms, no_atoms, no_atoms, no_atoms));
    let mut unit_vec_kj: Vec<f64> = vec![0.0; 3];
    let mut unit_vec_kl: Vec<f64> = vec![0.0; 3];
    let mut unit_vec_ki: Vec<f64> = vec![0.0; 3];

    for i in 0..no_atoms {
        for j in 0..i {
            for k in 0..j {
                for l in 0..k {
                    if i != j || i != k || i != l || j != k || j != l || k != l {
                        let mut oop_angle: f64 = 0.0;
                        let bond_length_buff_kj: f64 = calc_bond_length(&geom, i, j);
                        let bond_length_buff_kl: f64 = calc_bond_length(&geom, j, k);
                        let bond_length_buff_ki: f64 = calc_bond_length(&geom, k, i);

                        //* Unit vec calcs */
                        for cart_coord in 0..3 {
                            unit_vec_kj[cart_coord] = (geom[(k, cart_coord)]
                                - geom[(j, cart_coord)])
                                / bond_length_buff_kj;
                            unit_vec_kl[cart_coord] = (geom[(k, cart_coord)]
                                - geom[(l, cart_coord)])
                                / bond_length_buff_kl;
                            unit_vec_ki[cart_coord] = (geom[(k, cart_coord)]
                                - geom[(i, cart_coord)])
                                / bond_length_buff_ki;
                        }

                        //* Cross product
                        let cross_prod: Vec<f64> = calc_vec_cros_prod(&unit_vec_kj, &unit_vec_kl);
                        println!("{:?}", cross_prod);

                        //* angle for oop
                        let phi_jkl: f64 = calc_bond_angle(&geom, j, k, l);

                        for cart_coord in 0..3 {
                            oop_angle +=
                                cross_prod[cart_coord] * unit_vec_ki[cart_coord] / phi_jkl.sin();
                        }

                        if oop_angle < -1.0 {
                            oop_angle = -1.0_f64.asin();
                        }
                        else if oop_angle > 1.0 {
                            oop_angle = 1.0_f64.asin();
                        }
                        else {
                            oop_angle = oop_angle.asin() * (180.0 / std::f64::consts::PI);
                        }
                        // if oop_angle 
                        oop_angles[(i, j, k, l)] = oop_angle;
                    }
                }
            }
        }
    }
    println!("{:?}", oop_angles);
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
    let bond_length_buff: f64 = calc_bond_length(&geom, idx1, idx2);

    let mut unit_vec_ij: Vec<f64> = vec![0.0; 3];
    let mut unit_vec_jk: Vec<f64> = vec![0.0; 3];

    for cart_coord in 0..3 {
        unit_vec_ij[cart_coord] =
            (geom[(idx1, cart_coord)] - geom[(idx2, cart_coord)]) / bond_length_buff;
        unit_vec_jk[cart_coord] =
            (geom[(idx2, cart_coord)] - geom[(idx3, cart_coord)]) / bond_length_buff;

        bond_angle += -unit_vec_ij[cart_coord] * unit_vec_jk[cart_coord];
        bond_angle = bond_angle.acos() * (180.0 / std::f64::consts::PI);
    }
    return bond_angle;
}

fn calc_vec_cros_prod(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
    let mut vec_cros_prod: Vec<f64> = vec![0.0; 3];

    vec_cros_prod[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    vec_cros_prod[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
    vec_cros_prod[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

    return vec_cros_prod;
}
