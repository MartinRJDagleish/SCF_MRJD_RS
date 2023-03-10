use ndarray::prelude::*;
use ndarray_linalg::EigValsh;
use std::{
    fs,
    io::{BufRead, BufReader},
};

// #[derive(Clone)] ONLY FOR CRAWFORD PROJECTS -> breaks
// with wfn_total
#[derive(Debug)]
pub struct Molecule {
    pub charge: i32,
    pub geom_obj: geometry::Geometry,
    pub Z_vals: Vec<i32>,
    pub hessian: Array2<f64>,
    pub no_atoms: usize,
    pub wfn_total: wfn::WfnTotal,
}

pub mod geometry;
pub mod wfn;

#[allow(non_snake_case)] // * -> I need this due to QM naming conventions
impl Molecule {
    pub fn new(geom_file: &str, charge: i32) -> Molecule {
        let (Z_vals, geom_matr, no_atoms): (Vec<i32>, Array2<f64>, usize) =
            Self::read_crawford_inputfile(geom_file);

        let geom_obj: geometry::Geometry =
            geometry::Geometry::new(no_atoms, geom_matr, Z_vals.clone());

        //* Define a 0-matrix which can be edited later on ?
        let hessian: Array2<f64> = Array::default((3 *no_atoms, 3 * no_atoms));

        let wfn_total: wfn::WfnTotal = wfn::WfnTotal::new();

        Molecule {
            charge,
            no_atoms,
            geom_obj,
            Z_vals,
            hessian,
            wfn_total,
        }
    }

    pub fn update_no_occ_orb_rhf(&mut self) {
        let mut no_occ_orb: usize = 0;
        self.geom_obj.Z_vals.iter().for_each(|Z_val| {
            no_occ_orb += *Z_val as usize;
        });
        if no_occ_orb > 0 {
            no_occ_orb -= self.charge as usize;
            self.wfn_total.basis_set_total.no_occ_orb = no_occ_orb / 2;
        } else {
            // no_occ_orb = 0;
            println!("Warning: Charge is too high for the number of electrons! Charge set to 0!");
        }

    }

    fn read_crawford_inputfile(geom_filename: &str) -> (Vec<i32>, Array2<f64>, usize) {
        //* Step 1: Read the coord data from input
        println!("Inputfile: {geom_filename}");
        println!("Reading geometry from input file...\n");

        let geom_file = fs::File::open(geom_filename).expect("Geometry file not found!");
        let geom_file_reader = BufReader::new(geom_file);
        let mut geom_file_lines = geom_file_reader
            .lines()
            .map(|l| l.expect("Failed to read line!"));

        //* Read no of atoms first for array size
        let no_atoms = geom_file_lines.next().unwrap().parse().unwrap();

        let mut Z_vals: Vec<i32> = Vec::new();
        let mut geom_matr: Array2<f64> = Array2::zeros((no_atoms, 3));

        for (atom_idx, line) in geom_file_lines.enumerate() {
            //* Print out the read geometry to stdout */
            println!("{line}");
            let mut line_split = line.split_whitespace();

            Z_vals.push(line_split.next().unwrap().parse().unwrap());

            for cart_coord in 0..3 {
                geom_matr[(atom_idx, cart_coord)] = line_split.next().unwrap().parse().unwrap();
            }
        }

        println!("\n...End of geometry input.\n");

        (Z_vals, geom_matr, no_atoms)
    }

    fn read_xyz_xmol_inputfile(geom_filename: &str) -> (Vec<i32>, Array2<f64>, usize) {
        //* Step 1: Read the coord data from input
        println!("Inputfile: {geom_filename}");
        println!("Reading geometry from input file...\n");

        let geom_file = fs::File::open(geom_filename).expect("Geometry file not found!");
        let geom_file_reader = BufReader::new(geom_file);
        let mut geom_file_lines = geom_file_reader
            .lines()
            .map(|l| l.expect("Failed to read line!"));

        //* Read no of atoms first for array size
        let no_atoms = geom_file_lines.next().unwrap().parse().unwrap();

        let mut Z_vals: Vec<i32> = Vec::new();
        let mut geom_matr: Array2<f64> = Array2::zeros((no_atoms, 3));

        geom_file_lines.next(); //* Skip the comment line of xyz file

        for (atom_idx, line) in geom_file_lines.enumerate() {
            //* Print out the read geometry to stdout */
            println!("{line}");
            let mut line_split = line.split_whitespace();

            Z_vals.push(line_split.next().unwrap().parse().unwrap());

            (0..3).for_each(|cart_coord| {
                geom_matr[(atom_idx, cart_coord)] = line_split.next().unwrap().parse().unwrap();
            });
        }
        let bohr_to_AA: f64 = 0.529177210903_f64.recip();
        geom_matr.mapv_inplace(|x| x * bohr_to_AA);

        println!("\n...End of geometry input.\n");

        (Z_vals, geom_matr, no_atoms)
    }

    pub fn mass_weight_hessian(&mut self) {
        for i in 0..self.no_atoms * 3 {
            for j in 0..self.no_atoms * 3 {
                self.hessian[(i, j)] /= (self.geom_obj.get_mass_Z_val(&self.Z_vals[i / 3]) //* this uses integer div by default 
                    * self.geom_obj.get_mass_Z_val(&self.Z_vals[j / 3])) //* -> neat trick to get right index
                .sqrt();
            }
        }
    }

    pub fn calc_hess_eigenvals(&self) -> Array1<f64> {
        let hess_eigenvals: Array1<f64> =
            self.hessian.eigvalsh(ndarray_linalg::UPLO::Upper).unwrap(); //* these values are in atomic units

        //* Conversion to cm^-1
        // let conv: f64 = 0.0;
        // hess_eigenvals = conv * hess_eigenvals;
        hess_eigenvals
    }
}
