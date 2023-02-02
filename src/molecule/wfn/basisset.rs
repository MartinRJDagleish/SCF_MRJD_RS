use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader},
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use ndarray::{Array1, Array2};

use crate::molecule::wfn::{BasisSetTotal, CGTO, PGTO};



#[derive(Eq, Hash, PartialEq, Clone, Copy, EnumIter)]
pub enum PSE_element_sym {
    DUMMY,
    H,
    He,
    Li,
    Be,
    B,
    C,
    N,
    O,
    F,
    Ne,
    Na,
    Mg,
    Al,
    Si,
    P,
    S,
    Cl,
    Ar,
    K,
    Ca,
    Sc,
    Ti,
    V,
    Cr,
    Mn,
    Fe,
    Co,
    Ni,
    Cu,
    Zn,
    Ga,
    Ge,
    As,
    Se,
    Br,
    Kr,
    Rb,
    Sr,
    Y,
    Zr,
    Nb,
    Mo,
    Tc,
    Ru,
    Rh,
    Pd,
    Ag,
    Cd,
    In,
    Sn,
    Sb,
    Te,
    I,
    Xe,
    Cs,
    Ba,
    La,
    Ce,
    Pr,
    Nd,
    Pm,
    Sm,
    Eu,
    Gd,
    Tb,
    Dy,
    Ho,
    Er,
    Tm,
    Yb,
    Lu,
    Hf,
    Ta,
    W,
    Re,
    Os,
    Ir,
    Pt,
    Au,
    Hg,
    Tl,
    Pb,
    Bi,
    Po,
    At,
    Rn,
    Fr,
    Ra,
    Ac,
    Th,
    Pa,
    U,
    Np,
    Pu,
    Am,
    Cm,
    Bk,
    Cf,
    Es,
    Fm,
    Md,
    No,
    Lr,
    Rf,
    Db,
    Sg,
    Bh,
    Hs,
    Mt,
    Ds,
    Rg,
    Cn,
    Nh,
    Fl,
    Mc,
    Lv,
    Ts,
    Og,
}


pub enum L_char {
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

pub struct BasisSetAtom {
    pub element_sym: PSE_element_sym,
    pub cgto_list: Vec<CGTO>,
}

pub struct BasisSetDef {
    pub element_sym: PSE_element_sym,
    pub alphas: Vec<f64>,
    pub cgto_coeffs: Vec<f64>,
    pub L_and_no_prim_tup: Vec<(L_char, usize)>,
}

pub struct BasisSetTotalDef {
    pub name: String,
    pub basis_set_defs_dict: HashMap<PSE_element_sym, BasisSetDef>,
}

impl BasisSetTotalDef {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            basis_set_defs_dict: HashMap::new(),
        }
    }
}

impl BasisSetDef {
    pub fn new(element_sym: PSE_element_sym) -> Self {
        Self {
            element_sym,
            alphas: Vec::new(),
            cgto_coeffs: Vec::new(),
            L_and_no_prim_tup: Vec::new(),
        }
    }
}

pub fn match_pse_symb(match_string: &str) -> PSE_element_sym {
    // let mut PSE_element_sym_HashMap = HashMap::<&str,PSE_element_sym>::new();

    let PSE_element_sym_HashMap = [
        ("H", PSE_element_sym::H),
        ("He", PSE_element_sym::He),
        ("Li", PSE_element_sym::Li),
        ("Be", PSE_element_sym::Be),
        ("B", PSE_element_sym::B),
        ("C", PSE_element_sym::C),
        ("N", PSE_element_sym::N),
        ("O", PSE_element_sym::O),
        ("F", PSE_element_sym::F),
        ("Ne", PSE_element_sym::Ne),
        ("Na", PSE_element_sym::Na),
        ("Mg", PSE_element_sym::Mg),
        ("Al", PSE_element_sym::Al),
        ("Si", PSE_element_sym::Si),
        ("P", PSE_element_sym::P),
        ("S", PSE_element_sym::S),
        ("Cl", PSE_element_sym::Cl),
        ("Ar", PSE_element_sym::Ar),
        ("K", PSE_element_sym::K),
        ("Ca", PSE_element_sym::Ca),
        ("Sc", PSE_element_sym::Sc),
        ("Ti", PSE_element_sym::Ti),
        ("V", PSE_element_sym::V),
        ("Cr", PSE_element_sym::Cr),
        ("Mn", PSE_element_sym::Mn),
        ("Fe", PSE_element_sym::Fe),
        ("Co", PSE_element_sym::Co),
        ("Ni", PSE_element_sym::Ni),
        ("Cu", PSE_element_sym::Cu),
        ("Zn", PSE_element_sym::Zn),
        ("Ga", PSE_element_sym::Ga),
        ("Ge", PSE_element_sym::Ge),
        ("As", PSE_element_sym::As),
        ("Se", PSE_element_sym::Se),
        ("Br", PSE_element_sym::Br),
        ("Kr", PSE_element_sym::Kr),
        ("Rb", PSE_element_sym::Rb),
        ("Sr", PSE_element_sym::Sr),
        ("Y", PSE_element_sym::Y),
        ("Zr", PSE_element_sym::Zr),
        ("Nb", PSE_element_sym::Nb),
        ("Mo", PSE_element_sym::Mo),
        ("Tc", PSE_element_sym::Tc),
        ("Ru", PSE_element_sym::Ru),
        ("Rh", PSE_element_sym::Rh),
        ("Pd", PSE_element_sym::Pd),
        ("Ag", PSE_element_sym::Ag),
        ("Cd", PSE_element_sym::Cd),
        ("In", PSE_element_sym::In),
        ("Sn", PSE_element_sym::Sn),
        ("Sb", PSE_element_sym::Sb),
        ("Te", PSE_element_sym::Te),
        ("I", PSE_element_sym::I),
        ("Xe", PSE_element_sym::Xe),
        ("Cs", PSE_element_sym::Cs),
        ("Ba", PSE_element_sym::Ba),
        ("La", PSE_element_sym::La),
        ("Ce", PSE_element_sym::Ce),
        ("Pr", PSE_element_sym::Pr),
        ("Nd", PSE_element_sym::Nd),
        ("Pm", PSE_element_sym::Pm),
        ("Sm", PSE_element_sym::Sm),
        ("Eu", PSE_element_sym::Eu),
        ("Gd", PSE_element_sym::Gd),
        ("Tb", PSE_element_sym::Tb),
        ("Dy", PSE_element_sym::Dy),
        ("Ho", PSE_element_sym::Ho),
        ("Er", PSE_element_sym::Er),
        ("Tm", PSE_element_sym::Tm),
        ("Yb", PSE_element_sym::Yb),
        ("Lu", PSE_element_sym::Lu),
        ("Hf", PSE_element_sym::Hf),
        ("Ta", PSE_element_sym::Ta),
        ("W", PSE_element_sym::W),
        ("Re", PSE_element_sym::Re),
        ("Os", PSE_element_sym::Os),
        ("Ir", PSE_element_sym::Ir),
        ("Pt", PSE_element_sym::Pt),
        ("Au", PSE_element_sym::Au),
        ("Hg", PSE_element_sym::Hg),
        ("Tl", PSE_element_sym::Tl),
        ("Pb", PSE_element_sym::Pb),
        ("Bi", PSE_element_sym::Bi),
        ("Po", PSE_element_sym::Po),
        ("At", PSE_element_sym::At),
        ("Rn", PSE_element_sym::Rn),
        ("Fr", PSE_element_sym::Fr),
        ("Ra", PSE_element_sym::Ra),
        ("Ac", PSE_element_sym::Ac),
        ("Th", PSE_element_sym::Th),
        ("Pa", PSE_element_sym::Pa),
        ("U", PSE_element_sym::U),
        ("Np", PSE_element_sym::Np),
        ("Pu", PSE_element_sym::Pu),
        ("Am", PSE_element_sym::Am),
        ("Cm", PSE_element_sym::Cm),
        ("Bk", PSE_element_sym::Bk),
        ("Cf", PSE_element_sym::Cf),
        ("Es", PSE_element_sym::Es),
        ("Fm", PSE_element_sym::Fm),
        ("Md", PSE_element_sym::Md),
        ("No", PSE_element_sym::No),
        ("Lr", PSE_element_sym::Lr),
        ("Rf", PSE_element_sym::Rf),
        ("Db", PSE_element_sym::Db),
        ("Sg", PSE_element_sym::Sg),
        ("Bh", PSE_element_sym::Bh),
        ("Hs", PSE_element_sym::Hs),
        ("Mt", PSE_element_sym::Mt),
        ("Ds", PSE_element_sym::Ds),
        ("Rg", PSE_element_sym::Rg),
        ("Cn", PSE_element_sym::Cn),
        ("Nh", PSE_element_sym::Nh),
        ("Fl", PSE_element_sym::Fl),
        ("Mc", PSE_element_sym::Mc),
        ("Lv", PSE_element_sym::Lv),
        ("Ts", PSE_element_sym::Ts),
        ("Og", PSE_element_sym::Og),
    ]
    .into_iter()
    .collect::<HashMap<_, _>>();

    let pse_symb = match PSE_element_sym_HashMap.get(match_string) {
        Some(value) => *value,
        None => panic!("Element symbol not found!"),
    };

    pse_symb
}

pub fn parse_basis_set_file_gaussian(basis_set_name: &str) -> BasisSetTotalDef {
    let mut basis_set_total_def: BasisSetTotalDef = BasisSetTotalDef {
        name: basis_set_name.to_string(),
        basis_set_defs_dict: HashMap::new(),
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

    let mut basis_set: BasisSetDef = BasisSetDef::new(PSE_element_sym::DUMMY); //* using dummy element symbol

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
                //* Add the basis using the element symbol as key
                basis_set_total_def
                    .basis_set_defs_dict
                    .insert(basis_set.element_sym.clone(), basis_set);
            }
            basis_set = BasisSetDef::new(PSE_element_sym::DUMMY);
            continue;
        } else if line_start.is_alphabetic() {
            let line_split: Vec<&str> = data.split_whitespace().collect();
            if line_split.len() == 2 {
                //* Old version with string -> new version with enum
                // basis_set.element_sym = line_split[0].to_string();
                //* New version with enum
                basis_set.element_sym = match_pse_symb(line_split[0]);
                continue;
            } else if line_split[0] == "SP" {
                let no_prim1: usize = line_split[1].parse::<usize>().unwrap();
                basis_set.L_and_no_prim_tup.push((L_char::SP, no_prim1));
                // basis_set.L_and_no_prim_tup.push((L_letter::SP, no_prim1.clone()));
            } else if line_split[0].len() > 2
                && (line_split[0].starts_with("l=") || line_split[0].starts_with("L="))
            {
                todo!("Add the values for L basis sets");
            } else {
                let L_letter_val = match line_split[0] {
                    "S" => L_char::S,
                    "P" => L_char::P,
                    "D" => L_char::D,
                    "F" => L_char::F,
                    "G" => L_char::G,
                    "H" => L_char::H,
                    "I" => L_char::I,
                    "J" => L_char::J,
                    "K" => L_char::K,
                    "L" => L_char::L,
                    "M" => L_char::M,
                    "N" => L_char::N,
                    "O" => L_char::O,
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

    basis_set_total_def
}

pub fn create_basis_set_total(
    basis_set_total_def: BasisSetTotalDef,
    geom_matr: Array2<f64>,
    Z_vals: Vec<i32>,
) -> BasisSetTotal {
    let mut basis_set_total = BasisSetTotal::new();

    // fn build_pgto(Z_val: i32) -> PGTO {
    //     let mut pgto = PGTO::new();
    //     let elem_sym: PSE_element_sym = translate_Z_val_to_sym(Z_vals[0]);

    //     pgto
    // }

    // fn build_cgto() -> CGTO {
    //     let mut cgto = CGTO::new();
    //     for 

    //     cgto
    // }


    // fn build_basis_for_atom() {
    //     let mut basis_set_atom = BasisSetAtom::new();
    //     basis_for_atom
    // }

    basis_set_total
}

pub fn translate_Z_val_to_sym(Z_val: i32) -> PSE_element_sym {
    let mut Z_to_sym: HashMap<i32, PSE_element_sym> = HashMap::new();

    for (idx, sym) in PSE_element_sym::iter().enumerate() {
        let idx = (idx + 1) as i32;
        Z_to_sym.insert(idx, sym);
    }

    let pse_symb = match Z_to_sym.get(&Z_val) {
        Some(value) => *value,
        None => panic!("Element symbol not found!"),
    };

    pse_symb
}

pub fn translate_sym_to_Z_val(sym: PSE_element_sym) -> i32 {
    let mut sym_to_Z: HashMap<PSE_element_sym, i32> = HashMap::new();

    for (idx, sym) in PSE_element_sym::iter().enumerate() {
        let idx = (idx + 1) as i32;
        sym_to_Z.insert(sym, idx);
    }

    let Z_val = match sym_to_Z.get(&sym) {
        Some(value) => *value,
        None => panic!("Element symbol not found!"),
    };

    Z_val
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
