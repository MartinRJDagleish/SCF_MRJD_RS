use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader},
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use ndarray::prelude::*;

use crate::molecule::wfn::{BasisSetTotal, CGTO, PGTO};

#[derive(Eq, Hash, PartialEq, Clone, Copy, EnumIter, Default)]
pub enum PseElementSym {
    #[default]
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

#[derive(PartialEq)]
pub enum L_CHAR {
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
    pub L_and_no_prim_tup: Vec<(L_CHAR, usize)>,
}

#[derive(Default)]
pub struct BasisSetTotalDef {
    pub name: String,
    pub basis_set_defs_dict: HashMap<PseElementSym, BasisSetDef>,
}

pub fn match_pse_symb(
    pse_hash_map: &HashMap<&str, PseElementSym>,
    match_string: &str,
) -> PseElementSym {
    let pse_symb = match pse_hash_map.get(match_string) {
        Some(value) => *value,
        None => panic!("Element symbol not found! Error in basis set file!"),
    };

    pse_symb
}

pub fn parse_basis_set_file_gaussian(basis_set_name: &str) -> BasisSetTotalDef {
    let mut basis_set_total_def: BasisSetTotalDef = BasisSetTotalDef {
        name: basis_set_name.to_string(),
        basis_set_defs_dict: HashMap::new(),
    };

    let basis_set_file_path: &str = match basis_set_name.to_ascii_lowercase().as_str() {
        // * LEGACY
        // "sto-3g" => "inp/Project3_2/basis_sets/sto-3g.gbs",
        // "6-311g" => "inp/Project3_2/basis_sets/6-311g.gbs",
        // "def2-svp" => "inp/Project3_2/basis_sets/def2-svp.gbs",
        // "def2-tzvp" => "inp/Project3_2/basis_sets/def2-tzvp.gbs",
        "sto-3g" => "src/basis_sets/sto-3g.gbs",
        "sto-6g" => "src/basis_sets/sto-6g.gbs",
        "6-311g" => "src/basis_sets/6-311g.gbs",
        "6-311g*" => "src/basis_sets/6-311g_st.gbs",
        "6-311g**" => "src/basis_sets/6-311g_st_st.gbs",
        "def2-svp" => "src/basis_sets/def2-svp.gbs",
        "def2-tzvp" => "src/basis_sets/def2-tzvp.gbs",
        "def2-qzvp" => "src/basis_sets/def2-qzvp.gbs",
        _ => {
            panic!("Basis set not yet implemented!");
        }
    };

    let basis_set_file = fs::File::open(basis_set_file_path).expect("Basis set file not found!");
    let basis_set_reader = BufReader::new(basis_set_file);

    let block_delimiter: &str = "****";

    let mut basis_set_def: BasisSetDef = BasisSetDef::default(); //* using dummy element symbol
    let PSE_elem_sym_HashMap = [
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

    for line in basis_set_reader.lines() {
        let line = line.unwrap();
        let data = line.trim();
        let mut line_start = 0 as char;
        if !data.is_empty() {
            line_start = data.chars().next().unwrap();
        }
        if data.starts_with('!') || data.is_empty() {
            continue;
        } else if data.starts_with(block_delimiter) {
            if !basis_set_def.alphas.is_empty() {
                //* Check if BasisSet is not empty
                //* Add the basis using the element symbol as key
                basis_set_total_def
                    .basis_set_defs_dict
                    .insert(basis_set_def.element_sym, basis_set_def);
            }
            basis_set_def = BasisSetDef::default();
            continue;
        } else if line_start.is_alphabetic() {
            let line_split: Vec<&str> = data.split_whitespace().collect();
            if line_split.len() == 2 {
                //* Old version with string -> new version with enum
                // basis_set.element_sym = line_split[0].to_string();
                //* New version with enum
                basis_set_def.element_sym = match_pse_symb(&PSE_elem_sym_HashMap, line_split[0]);
                continue;
            } else if line_split[0] == "SP" {
                let no_prim1: usize = line_split[1].parse::<usize>().unwrap();
                basis_set_def.L_and_no_prim_tup.push((L_CHAR::SP, no_prim1));
            } else if line_split[0].len() > 2
                && (line_split[0].starts_with("l=") || line_split[0].starts_with("L="))
            {
                todo!("Add the values for L basis sets");
            } else {
                let L_letter_val = match line_split[0] {
                    "S" => L_CHAR::S,
                    "P" => L_CHAR::P,
                    "D" => L_CHAR::D,
                    "F" => L_CHAR::F,
                    "G" => L_CHAR::G,
                    "H" => L_CHAR::H,
                    "I" => L_CHAR::I,
                    "J" => L_CHAR::J,
                    "K" => L_CHAR::K,
                    "L" => L_CHAR::L,
                    "M" => L_CHAR::M,
                    "N" => L_CHAR::N,
                    "O" => L_CHAR::O,
                    _ => panic!("This letter is not supported!"),
                };
                // let L_val: usize = SPDF_HashMap.get(&L_val_char).unwrap().clone();
                let no_prim: usize = line_split[1].parse::<usize>().unwrap();
                basis_set_def
                    .L_and_no_prim_tup
                    .push((L_letter_val, no_prim));
            }
        } else {
            let parameters_vec = data
                .replace("D", "e")
                .split_whitespace()
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<f64>>();
            if parameters_vec.len() > 2 {
                //* This is the SP basis case
                basis_set_def.alphas.push(parameters_vec[0]);
                basis_set_def.cgto_coeffs.push(parameters_vec[1]); //* Values at even positions (0,2,…) are coeffs for S, odd values are for P (1,3,…) */
                basis_set_def.cgto_coeffs.push(parameters_vec[2]);
            } else {
                basis_set_def.alphas.push(parameters_vec[0]);
                basis_set_def.cgto_coeffs.push(parameters_vec[1]);
            }
        }
    }

    basis_set_total_def
}

pub fn create_basis_set_total(
    basis_set_total_def: BasisSetTotalDef,
    geom_matr: Array2<f64>,
    Z_vals: &[i32],
) -> BasisSetTotal {
    let mut basis_set_total = BasisSetTotal::new();

    let mut Z_to_sym: HashMap<i32, PseElementSym> = HashMap::new();
    let mut sym_to_Z: HashMap<PseElementSym, i32> = HashMap::new();
    for (idx, sym) in PseElementSym::iter().enumerate() {
        let idx = idx as i32;
        Z_to_sym.insert(idx, sym);
        sym_to_Z.insert(sym, idx);
    }

    for (atom_idx, atom_pos) in geom_matr.axis_iter(ndarray::Axis(0)).enumerate() {
        let Z_val = Z_vals[atom_idx];
        let elem_sym: PseElementSym = translate_Z_val_to_sym(&Z_to_sym, Z_val);

        let atom_basis_set: &BasisSetDef = basis_set_total_def
            .basis_set_defs_dict
            .get(&elem_sym)
            .unwrap();

        // * Generate PGTOs and then CGTOs
        let mut alphas_offset = 0_usize;
        for (L_val, no_prim) in atom_basis_set.L_and_no_prim_tup.iter() {
            if *L_val != L_CHAR::SP {
                let list_ang_mom_vec: Vec<Array1<i32>> = match L_val {
                    L_CHAR::S => vec![array![0, 0, 0]],
                    L_CHAR::P => vec![array![1, 0, 0], array![0, 1, 0], array![0, 0, 1]],
                    L_CHAR::D => vec![
                        array![2, 0, 0],
                        array![0, 2, 0],
                        array![0, 0, 2],
                        array![1, 1, 0],
                        array![1, 0, 1],
                        array![0, 1, 1],
                    ],
                    L_CHAR::F => vec![
                        array![3, 0, 0],
                        array![0, 3, 0],
                        array![0, 0, 3],
                        array![2, 1, 0],
                        array![2, 0, 1],
                        array![0, 2, 1],
                        array![1, 2, 0],
                        array![1, 0, 2],
                        array![0, 1, 2],
                    ],
                    L_CHAR::G => vec![
                        array![4, 0, 0],
                        array![0, 4, 0],
                        array![0, 0, 4],
                        array![3, 1, 0],
                        array![3, 0, 1],
                        array![0, 3, 1],
                        array![1, 3, 0],
                        array![1, 0, 3],
                        array![0, 1, 3],
                        array![2, 2, 0],
                        array![2, 0, 2],
                        array![0, 2, 2],
                    ],
                    _ => vec![array![0, 0, 0]],
                };
                for ang_mom_poss in list_ang_mom_vec.iter() {
                    let mut pgto_vec: Vec<PGTO> = Vec::new();
                    for prim_idx in 0..*no_prim {
                        let alpha = atom_basis_set.alphas[prim_idx + alphas_offset];
                        let cgto_coeff = atom_basis_set.cgto_coeffs[prim_idx + alphas_offset];
                        let pgto: PGTO = PGTO::new(
                            alpha,
                            cgto_coeff,
                            atom_pos.to_owned(),
                            ang_mom_poss.to_owned(),
                        );
                        pgto_vec.push(pgto);
                    }
                    let cgto: CGTO = CGTO::new(pgto_vec);
                    basis_set_total.basis_set_cgtos.push(cgto);
                }
                alphas_offset += no_prim;
            } else {
                (0..2).for_each(|coeff_type: usize| {
                    //* 0 is S, 1 is P
                    if coeff_type == 0 {
                        let mut pgto_vec: Vec<PGTO> = Vec::new();
                        for prim_idx in 0..*no_prim {
                            let alpha = atom_basis_set.alphas[prim_idx + alphas_offset];
                            let cgto_coeff = atom_basis_set.cgto_coeffs
                                [(2 * prim_idx) + alphas_offset + coeff_type];
                            //* S
                            let ang_mom_vec: Array1<i32> = array![0, 0, 0];
                            let pgto: PGTO =
                                PGTO::new(alpha, cgto_coeff, atom_pos.to_owned(), ang_mom_vec);
                            pgto_vec.push(pgto);
                        }
                        let cgtos: CGTO = CGTO::new(pgto_vec);
                        basis_set_total.basis_set_cgtos.push(cgtos);
                    } else {
                        (0..3).for_each(|cart_coord: usize| {
                            //* P
                            let mut pgto_vec: Vec<PGTO> = Vec::new();
                            for prim_idx in 0..*no_prim {
                                let alpha = atom_basis_set.alphas[prim_idx + alphas_offset];
                                let cgto_coeff = atom_basis_set.cgto_coeffs
                                    [(2 * prim_idx) + alphas_offset + coeff_type];
                                let mut ang_mom_vec: Array1<i32> = array![0, 0, 0];
                                ang_mom_vec[cart_coord] = 1;
                                let pgto: PGTO =
                                    PGTO::new(alpha, cgto_coeff, atom_pos.to_owned(), ang_mom_vec);
                                pgto_vec.push(pgto);
                            }
                            let cgto: CGTO = CGTO::new(pgto_vec);
                            basis_set_total.basis_set_cgtos.push(cgto);
                        });
                    }
                });
                alphas_offset += no_prim;
            }
        }
    }

    basis_set_total.no_cgtos = basis_set_total.basis_set_cgtos.len();
    basis_set_total
}

pub fn translate_Z_val_to_sym(Z_to_sym: &HashMap<i32, PseElementSym>, Z_val: i32) -> PseElementSym {
    let pse_symb = match Z_to_sym.get(&Z_val) {
        Some(value) => *value,
        None => panic!("Element symbol not found!"),
    };

    pse_symb
}

pub fn translate_sym_to_Z_val(sym_to_Z: &HashMap<PseElementSym, i32>, sym: PseElementSym) -> i32 {
    let Z_val = match sym_to_Z.get(&sym) {
        Some(value) => *value,
        None => panic!("Element symbol not found!"),
    };

    Z_val
}

// pub fn translate_L_char_to_val(L_char: L_CHAR) -> i32 {
//     match L_char {
//         L_CHAR::S => 0,
//         L_CHAR::SP => 1, //* This needs special care */
//         L_CHAR::P => 1,
//         L_CHAR::D => 2,
//         L_CHAR::F => 3,
//         L_CHAR::G => 4,
//         L_CHAR::H => 5,
//         L_CHAR::I => 6,
//         L_CHAR::J => 7,
//         L_CHAR::K => 8,
//         L_CHAR::L => 9,
//         L_CHAR::M => 10,
//         L_CHAR::N => 11,
//         L_CHAR::O => todo!(),
//     }
// }

pub fn calc_center_charge(Z_vals: &[i32], geom_matr: &Array2<f64>) -> Array1<f64> {
    let mut center_charge: Array1<f64> = Array1::zeros(3);
    let sum_Z_vals = Z_vals.iter().sum::<i32>() as f64;

    for (atom_idx, atom_pos) in geom_matr.axis_iter(ndarray::Axis(0)).enumerate() {
        center_charge = center_charge + (Z_vals[atom_idx] as f64) * atom_pos.to_owned();
    }

    center_charge / sum_Z_vals
}
