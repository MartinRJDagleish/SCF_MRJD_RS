use humantime::format_duration;
use std::time::Instant;

use crate::{molecule::Molecule};

// pub mod Crawford_projects;
pub mod molecule;
pub mod SCF_from_scratch;

#[allow(non_snake_case)] // * -> I need this due to QM naming conventions
fn main() {
    let start_exec_time = Instant::now();
    let ASCII_art_logo: String = String::from(
        r#"
    _____/\\\\\\\\\\\___________/\\\\\\\\\___/\\\\\\\\\\\\\\\_                                
     ___/\\\/////////\\\______/\\\////////___\/\\\///////////__                               
      __\//\\\______\///_____/\\\/____________\/\\\_____________                              
       ___\////\\\___________/\\\______________\/\\\\\\\\\\\_____                             
        ______\////\\\_______\/\\\______________\/\\\///////______                            
         _________\////\\\____\//\\\_____________\/\\\_____________                           
          __/\\\______\//\\\____\///\\\___________\/\\\_____________                          
           _\///\\\\\\\\\\\/_______\////\\\\\\\\\__\/\\\_____________                         
            ___\///////////____________\/////////___\///______________                        
    __/\\\\____________/\\\\_____/\\\\\\\\\___________/\\\\\\\\\\\___/\\\\\\\\\\\\____        
     _\/\\\\\\________/\\\\\\___/\\\///////\\\________\/////\\\///___\/\\\////////\\\__       
      _\/\\\//\\\____/\\\//\\\__\/\\\_____\/\\\____________\/\\\______\/\\\______\//\\\_      
       _\/\\\\///\\\/\\\/_\/\\\__\/\\\\\\\\\\\/_____________\/\\\______\/\\\_______\/\\\_     
        _\/\\\__\///\\\/___\/\\\__\/\\\//////\\\_____________\/\\\______\/\\\_______\/\\\_    
         _\/\\\____\///_____\/\\\__\/\\\____\//\\\____________\/\\\______\/\\\_______\/\\\_   
          _\/\\\_____________\/\\\__\/\\\_____\//\\\____/\\\___\/\\\______\/\\\_______/\\\__  
           _\/\\\_____________\/\\\__\/\\\______\//\\\__\//\\\\\\\\\_______\/\\\\\\\\\\\\/___ 
            _\///______________\///___\///________\///____\/////////________\////////////_____
     _______________/\\\\\\\\\__________/\\\\\\\\\\\___                                       
      _____________/\\\///////\\\______/\\\/////////\\\_                                      
       ____________\/\\\_____\/\\\_____\//\\\______\///__                                     
        ____________\/\\\\\\\\\\\/_______\////\\\_________                                    
         ____________\/\\\//////\\\__________\////\\\______                                   
          ____________\/\\\____\//\\\____________\////\\\___                                  
           ____________\/\\\_____\//\\\____/\\\______\//\\\__                                 
            ____________\/\\\______\//\\\__\///\\\\\\\\\\\/___                                
             ____________\///________\///_____\///////////_____  
        "#,
    );
    println!("{}", ASCII_art_logo);

    let mut mol: Molecule = Molecule::new("inp/Project3_1/STO-3G/h2o_v2.xyz", 0);

    let _is_run_project1: bool = false; //* General molecule geometry stuff
    let _is_run_project2: bool = false; //* Hessian -> eigenfreqs from file
    let _is_run_project3_1: bool = false; //* SCF from precomputed integrals
    let is_run_project3_2: bool = true; //* Ints for SCF from "scratch"
    let is_run_project3_3: bool = false; //* actual SCF from "scratch" 
    let _is_run_project4: bool = false; //* MP2 from precomputed integrals

    // if is_run_project1 {
    //     Crawford_projects::project1::run_project1(mol.clone());
    // }

    // if is_run_project2 {
    //     Crawford_projects::project2::run_project2(mol.clone());
    // }

    // if is_run_project3_1 {
    //     Crawford_projects::project3_1::run_project3_1(mol.clone(), is_run_project4);
    // }

    if is_run_project3_2 {
        use crate::SCF_from_scratch::project3_2::*;
        run_project3_2();
    }

    if is_run_project3_3 {
        use crate::SCF_from_scratch::project3_3::*;
        run_project3_3();
    }
    //*****************************************************************
    //*****************************************************************
    //*****************************************************************
    let end_of_calc_string: String = format!("{:^29}", "RUN ENDED SUCCESSFULLY!");
    println!("\n{}", "*".repeat(31));
    println!("*{}*", end_of_calc_string);
    println!("{}", "*".repeat(31));
    let end_exec_time = Instant::now();
    let duration_exec_time = end_exec_time.duration_since(start_exec_time);
    let formatted_duration_exec_time = format_duration(duration_exec_time).to_string();
    println!(
        "\nTime elapsed in execution is: {}",
        formatted_duration_exec_time
    );
}
