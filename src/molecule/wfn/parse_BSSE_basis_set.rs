use std::{fs, io::{BufReader, BufRead}, f64::consts::PI};
pub fn parse_basis_set_file(basis_set_name: &str)  {
    match basis_set_name.to_ascii_lowercase().as_str() {
        "sto-3g" => { 
            let basis_set_file_path: &str = "inp/Project3_2/basis_sets/sto-3g.gbs";
        }
        "6-311g" => {
            let basis_set_file_path: &str = "inp/Project3_2/basis_sets/6-311g.gbs";
        }
        _ => {
            panic!("Basis set not implemented yet!");
        }
    }
    let basis_set_file = fs::File::open(basis_set_file_path).expect("Basis set file not found!");
    let basis_set_reader = BufReader::new(basis_set_file);
    for line in basis_set_reader.lines() {
        println!("{}", line.unwrap());
    }
    let mut basis_set_file = File::open("basis_sets/6-311G.txt").unwrap();
    let mut basis_set_file_contents = String::new();
    basis_set_file.read_to_string(&mut basis_set_file_contents).unwrap();
}