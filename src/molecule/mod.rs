use ndarray::prelude::*;
use ndarray_linalg::{EigValsh, Scalar, SymmetricSqrt};
use physical_constants;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::io::{BufRead, BufReader};
use std::{collections::HashMap, fs};

pub struct Molecule {
    // charge
    pub charge: i32,
    // number of atoms
    pub no_atoms: usize,
    // cartesian coordinates (geometry of molecule)
    pub geom: Array2<f64>,
    // atomic numbers
    pub Z_vals: Vec<usize>,
    // HashMap of atomic masses
    // mass_map: HashMap<usize, f64>,
    // Array of atmoic masses
    pub mass_array: [f64; 119],
    pub hessian: Array2<f64>,
    // point_group
    // point_group: String,
}

mod geom;

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

    #[allow(non_snake_case)]
    pub fn get_mass_Z_val(&self, Z_val: &usize) -> f64 {
        // return self.mass_map.get(Z_val).unwrap().clone();
        //* new impl with mass_array
        return self.mass_array.get(*Z_val).unwrap().clone();
    }

    pub fn other_two(n: usize) -> Vec<usize> {
        let arr: [usize; 3] = [0, 1, 2];
        arr
            .iter()
            .filter(|&x| *x != n)
            .map(|x| *x)
            .collect()
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
