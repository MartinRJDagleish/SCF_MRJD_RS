use core::panic;
use std::{
    fs,
    io::{BufRead, BufReader},
};

pub struct BasisSetTotal {
    pub name: String,
    pub ListofBasisSets: Vec<BasisSet>,
    // chem_elem: String,
    // angular_mom: String,
    // no_primitives: u32,
    // scaling_factor: f64,
    // values: Vec<(f64, f64)>,
}

//TODO: maybe combine PrimitiveGaussian and BasisSet into one struct?
pub enum L_letter {
    S,
    P,
    D,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    SP,
}

pub struct BasisSet {
    pub element_sym: String,
    pub alphas: Vec<f64>,
    pub cgto_coeffs: Vec<f64>,
    pub L_and_no_prim_tup: Vec<(L_letter, usize)>,
}

impl BasisSetTotal {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            ListofBasisSets: Vec::new(),
        }
    }
}

impl BasisSet {
    pub fn new() -> Self {
        Self {
            element_sym: String::new(),
            alphas: Vec::new(),
            cgto_coeffs: Vec::new(),
            L_and_no_prim_tup: Vec::new(),
        }
    }
}

pub fn parse_basis_set_file_gaussian(basis_set_name: &str) -> BasisSetTotal {
    let mut basis_set_total: BasisSetTotal = BasisSetTotal {
        name: basis_set_name.to_string(),
        ListofBasisSets: Vec::new(),
    };

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

    let basis_set_file = fs::File::open(basis_set_file_path).expect("Basis set file not found!");
    let basis_set_reader = BufReader::new(basis_set_file);

    // let SPDF_str: &str = "SPDFGHIKLMN"; //* With L or without?
    // let mut SPDF_HashMap: HashMap<char, usize> = HashMap::new();

    // for (i, c) in SPDF_str.chars().enumerate() {
    //     SPDF_HashMap.insert(c, i);
    // }

    let block_delimiter: &str = "****";

    // let mut block_count: u32 = 0;

    let mut basis_set: BasisSet = BasisSet::new();

    for line in basis_set_reader.lines() {
        let line = line.unwrap();
        let data = line.trim();
        let mut line_start: char = 0 as char;
        if !data.is_empty() {
            line_start = data.chars().nth(0).unwrap();
        }
        if data.starts_with("!") || data.is_empty() {
            continue;
        } else if data.starts_with(block_delimiter) {
            if !basis_set.alphas.is_empty() {
                //* Check if BasisSet is not empty
                basis_set_total.ListofBasisSets.push(basis_set);
            }
            basis_set = BasisSet::new();
            continue;
        } else if line_start.is_alphabetic() {
            let line_split: Vec<&str> = data.split_whitespace().collect();
            if line_split.len() == 2 {
                basis_set.element_sym = line_split[0].to_string();
                continue;
            } else if line_split[0] == "SP" {
                let no_prim1: usize = line_split[1].parse::<usize>().unwrap();
                basis_set.L_and_no_prim_tup.push((L_letter::SP, no_prim1));
                // basis_set.L_and_no_prim_tup.push((L_letter::SP, no_prim1.clone()));
            } else if line_split[0].len() > 2
                && (line_split[0].starts_with("l=") || line_split[0].starts_with("L="))
            {
                todo!("Add the values for L basis sets");
            } else {
                let L_letter_val = match line_split[0] {
                    "S" => L_letter::S,
                    "P" => L_letter::P,
                    "D" => L_letter::D,
                    "F" => L_letter::F,
                    "G" => L_letter::G,
                    "H" => L_letter::H,
                    "I" => L_letter::I,
                    "J" => L_letter::J,
                    "K" => L_letter::K,
                    "L" => L_letter::L,
                    "M" => L_letter::M,
                    "N" => L_letter::N,
                    "O" => L_letter::O,
                    _ => panic!("This letter is not supported!"),
                };
                // let L_val: usize = SPDF_HashMap.get(&L_val_char).unwrap().clone();
                let no_prim: usize = line_split[1].parse::<usize>().unwrap();
                basis_set.L_and_no_prim_tup.push((L_letter_val, no_prim));
            }
        } else {
            let parameters_vec = data
                .replace("D", "e")
                .split_whitespace()
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<f64>>();
            if parameters_vec.len() > 2 {
                //* This is the SP basis case
                basis_set.alphas.push(parameters_vec[0]);
                basis_set.cgto_coeffs.push(parameters_vec[1]);
                basis_set.cgto_coeffs.push(parameters_vec[2]);
            } else {
                basis_set.alphas.push(parameters_vec[0]);
                basis_set.cgto_coeffs.push(parameters_vec[1]);
            }
        }
    }

    basis_set_total
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
