use std::f64::consts::PI;

use ndarray::{Array1, Array2};
// use factorial::{DoubleFactorial};

#[derive(Debug)]
pub struct Wavefunction_total {
    pub S_matr: Array2<f64>,
    pub T_matr: Array2<f64>,
    pub V_ne_matr: Array2<f64>,
    pub H_core_matr: Array2<f64>,
    pub ERI_arr1: Array1<f64>,
}

#[derive(Debug)]
pub struct PrimitiveGaussian {
    pub alpha: f64,
    pub cgto_coeff: f64,
    pub position: Array1<f64>,
    pub angular_momentum_vec: Array1<i32>,
    pub norm_const: f64,
}

// mod parse_BSSE_data;

#[allow(non_snake_case)]
impl Wavefunction_total {
    pub fn new() -> Wavefunction_total {
        let S_matr: Array2<f64> = Array2::zeros((1, 1));
        let T_matr: Array2<f64> = Array2::zeros((1, 1));
        let V_ne_matr: Array2<f64> = Array2::zeros((1, 1));
        let H_core_matr: Array2<f64> = Array2::zeros((1, 1));
        let ERI_arr1: Array1<f64> = Array1::zeros(1);

        Wavefunction_total {
            S_matr,
            T_matr,
            V_ne_matr,
            H_core_matr,
            ERI_arr1,
        }
    }

    pub fn calc_S_matr(&mut self, basis_set: &Vec<PrimitiveGaussian>) {
        let no_basis_funcs: usize = basis_set.len();
        let mut S_matr: Array2<f64> = Array2::zeros((no_basis_funcs, no_basis_funcs));
        //TODO: calculate S_matr -> need to give mol (coords) and PrimGaus (alpha, norm_const, position, angular_momentum_vec)
        //TODO: more Rust knowledge how to do this: Wavefunction_total should be part of mol

        self.S_matr = S_matr;
    }
}

impl PrimitiveGaussian {
    pub fn new(
        alpha: f64,
        cgto_coeff: f64,
        position: Array1<f64>,
        angular_momentum_vec: Array1<i32>,
    ) -> Self {
        // let alpha: f64 = 0.0;
        // let cgto_coeff: f64 = 0.0;
        // let position: Array1<f64> = Array1::zeros(3);
        // let angular_momentum_vec: Array1<u8> = Array1::zeros(3);

        //TODO: calculate norm_const in new()
        let mut norm_const: f64 = 0.0;

        PrimitiveGaussian {
            alpha,
            cgto_coeff,
            position,
            angular_momentum_vec,
            norm_const,
        }
    }

    pub fn calc_cart_norm_const(&mut self) {
        let numerator: f64 = (2.0 * &self.alpha / PI).powf(1.5)
            * (4.0 * &self.alpha).powi(self.angular_momentum_vec.sum() as i32);
        let denom: i32 = self
            .angular_momentum_vec
            .mapv(|x| PrimitiveGaussian::double_factorial(&self, 2 * x - 1))
            .product();

        self.norm_const = numerator / denom as f64
    }

    pub fn double_factorial(&self, n: i32) -> i32 {
        if n == -1 {
            return 1
        }
        if n == 1 {
            return 1
        } else {
            n * self.double_factorial(n - 2)
        }
    }
}
