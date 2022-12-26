use ndarray::Array2;

#[derive()]
pub struct Molecule {
    // number of atoms
    pub no_atoms: usize,
    // atomic numbers
    pub Z_vals: Vec<i32>,
    // cartesian coordinates (geometry of molecule)
    pub geom: Array2<f64>,
}

impl Molecule {
    pub fn new(no_atoms: usize, Z_vals: Vec<i32>, geom: Array2<f64>) -> Molecule {
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
        Molecule {
            no_atoms,
            Z_vals,
            geom,
            bond_lengths,
        }
    }
}
