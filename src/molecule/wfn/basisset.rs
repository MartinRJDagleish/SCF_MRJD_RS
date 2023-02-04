use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader},
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use ndarray::Array2;

use crate::molecule::wfn::{BasisSetTotal, CGTO, PGTO};

#[derive(Eq, Hash, PartialEq, Clone, Copy, EnumIter)]
pub enum PseElementSym {
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

#[derive(Default)]
pub struct BasisSetAtom {
    pub element_sym: PseElementSym,
    pub cgto_list: Vec<CGTO>,
}

#[derive(Default)]
pub struct BasisSetDef {
    pub element_sym: PseElementSym,
    pub alphas: Vec<f64>,
    pub cgto_coeffs: Vec<f64>,
    pub L_and_no_prim_tup: Vec<(L_char, usize)>,
}

#[derive(Default)]
pub struct BasisSetTotalDef {
    pub name: String,
    pub basis_set_defs_dict: HashMap<PseElementSym, BasisSetDef>,
}

impl Default for PseElementSym {
    fn default() -> Self {
        PseElementSym::DUMMY
    }
}

// impl BasisSetAtom {
//     pub fn new(element_sym: PseElementSym) -> Self {
//         Self {
//             element_sym,
//             cgto_list: Vec::new(),
//         }
//     }
// }

// impl BasisSetTotalDef {
//     pub fn new() -> Self {
//         Self {
//             name: String::new(),
//             basis_set_defs_dict: HashMap::new(),
//         }
//     }
// }

// impl BasisSetDef {
//     pub fn new(element_sym: PseElementSym) -> Self {
//         Self {
//             element_sym,
//             alphas: Vec::new(),
//             cgto_coeffs: Vec::new(),
//             L_and_no_prim_tup: Vec::new(),
//         }
//     }
// }


pub fn match_pse_symb(match_string: &str) -> PseElementSym {
    let PSE_element_sym_HashMap = [
        ("H", PseElementSym::H),
        ("He", PseElementSym::He),
        ("Li", PseElementSym::Li),
        ("Be", PseElementSym::Be),
        ("B", PseElementSym::B),
        ("C", PseElementSym::C),
        ("N", PseElementSym::N),
        ("O", PseElementSym::O),
        ("F", PseElementSym::F),
        ("Ne", PseElementSym::Ne),
        ("Na", PseElementSym::Na),
        ("Mg", PseElementSym::Mg),
        ("Al", PseElementSym::Al),
        ("Si", PseElementSym::Si),
        ("P", PseElementSym::P),
        ("S", PseElementSym::S),
        ("Cl", PseElementSym::Cl),
        ("Ar", PseElementSym::Ar),
        ("K", PseElementSym::K),
        ("Ca", PseElementSym::Ca),
        ("Sc", PseElementSym::Sc),
        ("Ti", PseElementSym::Ti),
        ("V", PseElementSym::V),
        ("Cr", PseElementSym::Cr),
        ("Mn", PseElementSym::Mn),
        ("Fe", PseElementSym::Fe),
        ("Co", PseElementSym::Co),
        ("Ni", PseElementSym::Ni),
        ("Cu", PseElementSym::Cu),
        ("Zn", PseElementSym::Zn),
        ("Ga", PseElementSym::Ga),
        ("Ge", PseElementSym::Ge),
        ("As", PseElementSym::As),
        ("Se", PseElementSym::Se),
        ("Br", PseElementSym::Br),
        ("Kr", PseElementSym::Kr),
        ("Rb", PseElementSym::Rb),
        ("Sr", PseElementSym::Sr),
        ("Y", PseElementSym::Y),
        ("Zr", PseElementSym::Zr),
        ("Nb", PseElementSym::Nb),
        ("Mo", PseElementSym::Mo),
        ("Tc", PseElementSym::Tc),
        ("Ru", PseElementSym::Ru),
        ("Rh", PseElementSym::Rh),
        ("Pd", PseElementSym::Pd),
        ("Ag", PseElementSym::Ag),
        ("Cd", PseElementSym::Cd),
        ("In", PseElementSym::In),
        ("Sn", PseElementSym::Sn),
        ("Sb", PseElementSym::Sb),
        ("Te", PseElementSym::Te),
        ("I", PseElementSym::I),
        ("Xe", PseElementSym::Xe),
        ("Cs", PseElementSym::Cs),
        ("Ba", PseElementSym::Ba),
        ("La", PseElementSym::La),
        ("Ce", PseElementSym::Ce),
        ("Pr", PseElementSym::Pr),
        ("Nd", PseElementSym::Nd),
        ("Pm", PseElementSym::Pm),
        ("Sm", PseElementSym::Sm),
        ("Eu", PseElementSym::Eu),
        ("Gd", PseElementSym::Gd),
        ("Tb", PseElementSym::Tb),
        ("Dy", PseElementSym::Dy),
        ("Ho", PseElementSym::Ho),
        ("Er", PseElementSym::Er),
        ("Tm", PseElementSym::Tm),
        ("Yb", PseElementSym::Yb),
        ("Lu", PseElementSym::Lu),
        ("Hf", PseElementSym::Hf),
        ("Ta", PseElementSym::Ta),
        ("W", PseElementSym::W),
        ("Re", PseElementSym::Re),
        ("Os", PseElementSym::Os),
        ("Ir", PseElementSym::Ir),
        ("Pt", PseElementSym::Pt),
        ("Au", PseElementSym::Au),
        ("Hg", PseElementSym::Hg),
        ("Tl", PseElementSym::Tl),
        ("Pb", PseElementSym::Pb),
        ("Bi", PseElementSym::Bi),
        ("Po", PseElementSym::Po),
        ("At", PseElementSym::At),
        ("Rn", PseElementSym::Rn),
        ("Fr", PseElementSym::Fr),
        ("Ra", PseElementSym::Ra),
        ("Ac", PseElementSym::Ac),
        ("Th", PseElementSym::Th),
        ("Pa", PseElementSym::Pa),
        ("U", PseElementSym::U),
        ("Np", PseElementSym::Np),
        ("Pu", PseElementSym::Pu),
        ("Am", PseElementSym::Am),
        ("Cm", PseElementSym::Cm),
        ("Bk", PseElementSym::Bk),
        ("Cf", PseElementSym::Cf),
        ("Es", PseElementSym::Es),
        ("Fm", PseElementSym::Fm),
        ("Md", PseElementSym::Md),
        ("No", PseElementSym::No),
        ("Lr", PseElementSym::Lr),
        ("Rf", PseElementSym::Rf),
        ("Db", PseElementSym::Db),
        ("Sg", PseElementSym::Sg),
        ("Bh", PseElementSym::Bh),
        ("Hs", PseElementSym::Hs),
        ("Mt", PseElementSym::Mt),
        ("Ds", PseElementSym::Ds),
        ("Rg", PseElementSym::Rg),
        ("Cn", PseElementSym::Cn),
        ("Nh", PseElementSym::Nh),
        ("Fl", PseElementSym::Fl),
        ("Mc", PseElementSym::Mc),
        ("Lv", PseElementSym::Lv),
        ("Ts", PseElementSym::Ts),
        ("Og", PseElementSym::Og),
    ]
    .into_iter()
    .collect::<HashMap<_, _>>();
    // let mut PSE_element_sym_HashMap = HashMap::<&str,PSE_element_sym>::new();


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

    let basis_set_file_path: &str = match basis_set_name.to_ascii_lowercase().as_str() {
        "sto-3g" => {
            "inp/Project3_2/basis_sets/sto-3g.gbs"
        }
        "6-311g" => {
            "inp/Project3_2/basis_sets/6-311g.gbs"
        }
        "def2-svp" => {
            "inp/Project3_2/basis_sets/def2-svp.gbs"
        }
        "def2-tzvp" => {
            "inp/Project3_2/basis_sets/def2-tzvp.gbs"
        }
        _ => {
            panic!("Basis set not yet implemented!");
        }
    };

    let basis_set_file = fs::File::open(basis_set_file_path).expect("Basis set file not found!");
    let basis_set_reader = BufReader::new(basis_set_file);

    // let SPDF_str: &str = "SPDFGHIKLMN"; //* With L or without?
    // let mut SPDF_HashMap: HashMap<char, usize> = HashMap::new();

    // for (i, c) in SPDF_str.chars().enumerate() {
    //     SPDF_HashMap.insert(c, i);
    // }

    let block_delimiter: &str = "****";

    // let mut block_count: u32 = 0;

    let mut basis_set: BasisSetDef = BasisSetDef::default(); //* using dummy element symbol

    for line in basis_set_reader.lines() {
        let line = line.unwrap();
        let data = line.trim();
        let mut line_start: char = 0 as char;
        if !data.is_empty() {
            line_start = data.chars().next().unwrap();
        }
        if data.starts_with('!') || data.is_empty() {
            continue;
        } else if data.starts_with(block_delimiter) {
            if !basis_set.alphas.is_empty() {
                //* Check if BasisSet is not empty
                //* Add the basis using the element symbol as key
                basis_set_total_def
                    .basis_set_defs_dict
                    .insert(basis_set.element_sym, basis_set);
            }
            basis_set = BasisSetDef::default();
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

//TODO: THIS WHOLE FUNCTION DOES NOT WORK YET -> complicated process
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

    //     build_pgto();

    //     cgto
    // }

    // for (idx, Z_val) in Z_vals.iter().enumerate() {
    //     let mut basis_set_atom = BasisSetAtom::new(PSE_element_sym::DUMMY);
    //     let elem_sym: PSE_element_sym = translate_Z_val_to_sym(*Z_val);
    //     basis_set_atom.element_sym = elem_sym;

    //     let basis_set_def = basis_set_total_def
    //         .basis_set_defs_dict
    //         .get(&elem_sym)
    //         .unwrap();

    //     let cgto: CGTO = build_cgto();

    //     for (L_val, no_prim) in basis_set_def.L_and_no_prim_tup.iter() {

    //     }

    // }

    basis_set_total
}

pub fn translate_Z_val_to_sym(Z_val: i32) -> PseElementSym {
    let mut Z_to_sym: HashMap<i32, PseElementSym> = HashMap::new();

    for (idx, sym) in PseElementSym::iter().enumerate() {
        let idx = (idx + 1) as i32;
        Z_to_sym.insert(idx, sym);
    }

    let pse_symb = match Z_to_sym.get(&Z_val) {
        Some(value) => *value,
        None => panic!("Element symbol not found!"),
    };

    pse_symb
}

pub fn translate_sym_to_Z_val(sym: PseElementSym) -> i32 {
    let mut sym_to_Z: HashMap<PseElementSym, i32> = HashMap::new();

    for (idx, sym) in PseElementSym::iter().enumerate() {
        let idx = (idx + 1) as i32;
        sym_to_Z.insert(sym, idx);
    }

    let Z_val = match sym_to_Z.get(&sym) {
        Some(value) => *value,
        None => panic!("Element symbol not found!"),
    };

    Z_val
}
