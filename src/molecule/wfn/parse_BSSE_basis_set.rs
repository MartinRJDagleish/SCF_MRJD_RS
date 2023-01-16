use std::{
    fs,
    io::{BufRead, BufReader},
};

struct BasisSetTotal {
    ListofBasisSets: Vec<BasisSet>,
    // chem_elem: String,
    // angular_mom: String,
    // no_primitives: u32,
    // scaling_factor: f64,
    // values: Vec<(f64, f64)>,
}

//TODO: maybe combine PrimitiveGaussian and BasisSet into one struct?
pub struct BasisSet {
    pub element_sym: String,
    pub angular_mom: String,
    pub no_primitives: u32,
    pub scaling_factor: f64,
    pub alphas: Vec<f64>,
    pub coeffs: Vec<f64>,
}

impl BasisSet {
    pub fn new() -> BasisSet {
        BasisSet {
            element_sym: String::new(),
            angular_mom: String::new(),
            no_primitives: 0,
            scaling_factor: 0.0,
            alphas: Vec::new(),
            coeffs: Vec::new(),
        }
    }

    pub fn parse_basis_set_file(basis_set_name: &str) {
        let basis_set_file_path: &str;
        match basis_set_name.to_ascii_lowercase().as_str() {
            "sto-3g" => {
                basis_set_file_path = "inp/Project3_2/basis_sets/sto-3g.gbs";
            }
            "6-311g" => {
                basis_set_file_path = "inp/Project3_2/basis_sets/6-311g.gbs";
            }
            "def2-svp" => {
                basis_set_file_path = "inp/Project3_2/basis_sets/def2-svp.gbs";
            }
            "def2-tzvp" => {
                basis_set_file_path = "inp/Project3_2/basis_sets/def2-tzvp.gbs";
            }
            _ => {
                panic!("Basis set not implemented yet!");
            }
        }

        let basis_set_file =
            fs::File::open(basis_set_file_path).expect("Basis set file not found!");
        let basis_set_reader = BufReader::new(basis_set_file);

        let mut block_count: u32 = 0;
        for line in basis_set_reader.lines() {
            let line = line.unwrap();
            // * Skip comments and empty lines
            if line.starts_with("!") || line.is_empty() {
                continue;
            }

            if line.starts_with("****") {
                block_count += 1;
                continue;
            } else if line.starts_with(" ") {
                let mut values: Vec<f64> = Vec::new();
                line.trim()
                    .split_whitespace()
                    .for_each(|val| values.push(val.parse::<f64>().unwrap()));
                println!("{:?}", values);
            }
        }
        println!("{}", &block_count);
    }
}

// ChatGPT
// struct ChemElem {
//     chem_elem: String,
//     angular_mom: String,
//     no_primitives: u32,
//     scaling_factor: f64,
//     values: Vec<(f64, f64)>,
// }

// fn read_text_block(text: &str) -> Vec<ChemElem> {
//     let mut elements = vec![];
//     let lines = text.split("\n");
//     let mut current_elem = ChemElem {
//         chem_elem: String::new(),
//         angular_mom: String::new(),
//         no_primitives: 0,
//         scaling_factor: 0.0,
//         values: vec![],
//     };
//     for line in lines {
//         if line.starts_with("****") {
//             continue;
//         } else if line.starts_with(" ") {
//             let parts: Vec<&str> = line.split(" ").collect();
//             let value1 = parts[0].parse::<f64>().unwrap();
//             let value2 = parts[1].parse::<f64>().unwrap();
//             current_elem.values.push((value1, value2));
//         } else {
//             let parts: Vec<&str> = line.split(" ").collect();
//             if parts.len() == 2 {
//                 current_elem.chem_elem = parts[0].to_string();
//             } else if parts.len() == 3 {
//                 current_elem.angular_mom = parts[0].to_string();
//                 current_elem.no_primitives = parts[1].parse::<u32>().unwrap();
//                 current_elem.scaling_factor = 0.0;
//             } else {
//                 if !current_elem.chem_elem.is_empty() {
//                     elements.push(current_elem);
//                 }
//                 current_elem = ChemElem {
//                     chem_elem: String::new(),
//                     angular_mom: String::new(),
//                     no_primitives: 0,
//                     scaling_factor: 0.0,
//                     values: vec![],
//                 };
//             }
//         }
//     }
//     elements.push(current_elem);
//     elements
// }


