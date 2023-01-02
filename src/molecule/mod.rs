use ndarray::prelude::*;
use ndarray_linalg::{EigValsh, Scalar, SymmetricSqrt};
use physical_constants;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::io::{BufRead, BufReader};
use std::{collections::HashMap, fs};

pub struct Molecule {
    // charge
    charge: i32,
    // number of atoms
    no_atoms: usize,
    // cartesian coordinates (geometry of molecule)
    geom: Array2<f64>,
    // atomic numbers
    Z_vals: Vec<usize>,
    // HashMap of atomic masses
    // mass_map: HashMap<usize, f64>,
    // Array of atmoic masses
    mass_array: [f64; 119],
    hessian: Array2<f64>,
    // point_group
    // point_group: String,
}

impl Molecule {
    pub fn new(geom_file: &str, hessian_file: &str, charge: i32) -> Molecule {
        //* Step 1: Read the coord data from input
        println!("In file {}", geom_file);

        //* Show contents of file
        let geom_file_contents: String =
            fs::read_to_string(geom_file).expect("Failed to read geom file!");

        //* Read no of atoms first for array size
        let no_atoms: usize = geom_file_contents.lines().nth(0).unwrap().parse().unwrap();
        println!("No of atoms: {}", no_atoms);

        let mut Z_vals: Vec<usize> = vec![0; no_atoms];
        let mut geom: Array2<f64> = Array2::zeros((no_atoms, 3));

        for (line_idx, line) in geom_file_contents.lines().skip(1).enumerate() {
            let line_split: Vec<&str> = line.split_whitespace().collect();

            Z_vals[line_idx] = line_split[0].parse().unwrap();

            for cart_coord in 0..3 {
                geom[(line_idx, cart_coord)] = line_split[cart_coord + 1].parse().unwrap();
            }
        }

        //* Main hashmap for masses
        // let mut mass_map: HashMap<usize, f64> = HashMap::new();

        // let masses_path: &str = "inp/masses.csv";
        // let massses_content: String =
        //     fs::read_to_string(masses_path).expect("Failed to read geomfile!");

        // for (Z_val, line) in massses_content.lines().enumerate() {
        //     if Z_val == 0 {
        //         continue;
        //     }
        //     let line_vec: Vec<&str> = line.split(',').collect();
        //     let mass: f64 = line_vec[0].parse().unwrap();
        //     // println!("Mass of atom with Z value {}: {}", Z_val, mass);
        //     mass_map.insert(Z_val, mass);
        // }
        //* Array instead of hashamp for masses -> no file reading necessary
        // ! SOURCE: https://iupac.qmul.ac.uk/AtWt/ -> cleaned with OpenRefine 
        let mass_array = [
            0.0,
            1.008,
            4.002602,
            6.94,
            9.0121831,
            10.81,
            12.011,
            14.007,
            15.999,
            18.998403163,
            20.1797,
            22.98976928,
            24.305,
            26.9815384,
            28.085,
            30.973761998,
            32.06,
            35.45,
            39.95,
            39.0983,
            40.078,
            44.955907,
            47.867,
            50.9415,
            51.9961,
            54.938043,
            55.845,
            58.933194,
            58.6934,
            63.546,
            65.38,
            69.723,
            72.630,
            74.921595,
            78.971,
            79.904,
            83.798,
            85.4678,
            87.62,
            88.905838,
            91.224,
            92.90637,
            95.95,
            97.0,
            101.07,
            102.90549,
            106.42,
            107.8682,
            112.414,
            114.818,
            118.710,
            121.760,
            127.60,
            126.90447,
            131.293,
            132.90545196,
            137.327,
            138.90547,
            140.116,
            140.90766,
            144.242,
            145.0,
            150.36,
            151.964,
            157.25,
            158.925354,
            162.500,
            164.930329,
            167.259,
            168.934219,
            173.045,
            174.9668,
            178.486,
            180.94788,
            183.84,
            186.207,
            190.23,
            192.217,
            195.084,
            196.966570,
            200.592,
            204.38,
            207.2,
            208.98040,
            209.0,
            210.0,
            222.0,
            223.0,
            226.0,
            227.0,
            232.0377,
            231.03588,
            238.02891,
            237.0,
            244.0,
            243.0,
            247.0,
            247.0,
            251.0,
            252.0,
            257.0,
            258.0,
            259.0,
            262.0,
            267.0,
            270.0,
            269.0,
            270.0,
            270.0,
            278.0,
            281.0,
            281.0,
            285.0,
            286.0,
            289.0,
            289.0,
            293.0,
            293.0,
            294.0,
        ];

        //* READING THE HESSIAN
        let hess_file = fs::File::open(hessian_file).unwrap();
        let hess_reader = BufReader::new(hess_file);

        let mut hessian: Vec<Vec<f64>> = Vec::new();
        let mut line_iter = hess_reader.lines();

        let hess_no_atoms: usize = line_iter.next().unwrap().unwrap().trim().parse().unwrap();
        println!("No of atoms in hessian: {}", hess_no_atoms);
        if no_atoms != hess_no_atoms {
            panic!("Number of atoms in geom file and hessian file are not the same!");
        }

        for line in line_iter {
            let line = line.unwrap();
            let values: Vec<f64> = line
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            hessian.push(values);
        }

        let mut hessian: Array2<f64> = Array2::from_shape_vec(
            (3 * hess_no_atoms, 3 * hess_no_atoms),
            hessian.into_iter().flatten().collect(),
        )
        .unwrap();

        Molecule {
            charge,
            no_atoms,
            geom,
            Z_vals,
            mass_array,
            hessian,
        }
    }

    /// Returns the geom of this [`Molecule`].
    pub fn print_geom(&self) {
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
    pub fn calc_r_ij(&self, i: usize, j: usize) -> f64 {
        let mut bond_length: f64 = 0.0;
        for cart_coord in 0..3 {
            bond_length += (&self.geom[(j, cart_coord)] - &self.geom[(i, cart_coord)]).powi(2);
        }
        return bond_length.sqrt();
    }

    pub fn calc_e_ij(&self, i: usize, j: usize) -> Vec<f64> {
        let mut unit_vec: Vec<f64> = vec![0.0; 3];
        let r_ij: f64 = self.calc_r_ij(i, j);
        for cart_coord in 0..3 {
            unit_vec[cart_coord] =
                (&self.geom[(j, cart_coord)] - &self.geom[(i, cart_coord)]) / r_ij;
        }
        return unit_vec;
    }

    pub fn calc_vec_norm(vec1: &Vec<f64>) -> f64 {
        let mut vec_norm: f64 = 0.0;
        for cart_coord in 0..3 {
            vec_norm += vec1[cart_coord].powi(2);
        }
        return vec_norm.sqrt();
    }

    pub fn calc_vec_cross_prod(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let mut vec_cros_prod: Vec<f64> = vec![0.0; 3];

        vec_cros_prod[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        vec_cros_prod[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        vec_cros_prod[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

        return vec_cros_prod;
    }

    pub fn calc_unit_vec_cross_prod(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
        let mut vec_cros_prod: Vec<f64> = vec![0.0; 3];
        let cos_v1_v2: f64 = Self::calc_scalar_prod(&vec1, &vec2);
        let sin_v1_v2: f64 = (1.0f64 - cos_v1_v2.powi(2)).sqrt();

        vec_cros_prod[0] = (vec1[1] * vec2[2] - vec1[2] * vec2[1]) / sin_v1_v2;
        vec_cros_prod[1] = (vec1[2] * vec2[0] - vec1[0] * vec2[2]) / sin_v1_v2;
        vec_cros_prod[2] = (vec1[0] * vec2[1] - vec1[1] * vec2[0]) / sin_v1_v2;

        return vec_cros_prod;
    }

    pub fn calc_scalar_prod(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
        let mut scalar_prod: f64 = 0.0;
        for cart_coord in 0..3 {
            scalar_prod += vec1[cart_coord] * vec2[cart_coord];
        }
        return scalar_prod;
    }

    pub fn calc_bond_angle(&self, idx1: usize, idx2: usize, idx3: usize) -> f64 {
        let unit_21: Vec<f64> = self.calc_e_ij(idx2, idx1);
        let unit_23: Vec<f64> = self.calc_e_ij(idx2, idx3);
        let bond_angle: f64 = Self::calc_scalar_prod(&unit_21, &unit_23);
        return bond_angle.acos().to_degrees();
    }

    pub fn calc_oop_angle(&self, idx1: usize, idx2: usize, idx3: usize, idx4: usize) -> f64 {
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

    pub fn calc_dihedral_angle(&self, idx1: usize, idx2: usize, idx3: usize, idx4: usize) -> f64 {
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

    pub fn get_mass_Z_val(&self, Z_val: &usize) -> f64 {
        // return self.mass_map.get(Z_val).unwrap().clone();
        //* new impl with mass_array
        return self
            .mass_array
            .get(*Z_val)
            .unwrap()
            .clone();
    }

    pub fn calc_center_mass(&self) -> Vec<f64> {
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

    pub fn translate_mol_to_center_mass(&mut self) {
        let center_mass_vec: Vec<f64> = self.calc_center_mass();

        for (idx, Z_val) in self.Z_vals.iter().enumerate() {
            for cart_coord in 0..3 {
                self.geom[(idx, cart_coord)] -= center_mass_vec[cart_coord];
            }
        }
    }

    pub fn other_two(n: usize) -> Vec<usize> {
        let arr: [usize; 3] = [0, 1, 2];
        arr.iter()
            .filter(|&x| *x != n)
            .map(|x| *x)
            .collect::<Vec<usize>>()
    }

    pub fn calc_inertia_tensor(&self) -> Array2<f64> {
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

    pub fn mass_weight_hessian(&mut self) {
        for i in 0..self.no_atoms * 3 {
            for j in 0..self.no_atoms * 3 {
                self.hessian[(i, j)] = self.hessian[(i, j)]
                    / (self.get_mass_Z_val(&self.Z_vals[i / 3]) //* this uses integer div by default
                    * self.get_mass_Z_val(&self.Z_vals[j / 3]))
                    .sqrt();
            }
        }
    }

    pub fn calc_hess_eigenvals(&self) -> Array1<f64> {
        let mut hess_eigenvals: Array1<f64> =
            self.hessian.eigvalsh(ndarray_linalg::UPLO::Upper).unwrap(); //* these values are in atomic units

        //* Conversion to cm^-1
        // let conv: f64 = 0.0;
        // hess_eigenvals = conv * hess_eigenvals;
        return hess_eigenvals;
    }
}
