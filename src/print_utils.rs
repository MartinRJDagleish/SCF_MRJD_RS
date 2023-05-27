use std::time::Instant;

use humantime::format_duration;

#[inline]
pub(crate) fn print_header() {
    let programme_name = "M   R   J   D      S   C   F";
    println!("\n{}{}", " ".repeat(25), "*".repeat(32));
    println!(
        "{}* {programme_name} *",
        " ".repeat(25),
        programme_name = programme_name
    );
    println!("{}{}", " ".repeat(25), "*".repeat(32));
    let ASCII_art_logo = String::from(
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
    println!("{ASCII_art_logo}\n");
}

#[inline]
pub(crate) fn print_general_information() {
    let msg = r#"
================================================================================
                             GENERAL INFORMATION
================================================================================
This programme was written by MRJD after finishing his Bachelor's in Chemistry 
at University of Regensburg and before starting his Master's at the LMU Munich. 
(Dec 2022 - Mar 2023) It was only written in his spare time and is not intended 
to be used for any serious calculations.
It was merely written to learn the Rust programming language and to get a better
understanding of the SCF / RHF / quantum chemical methodes.

The programme is written in Rust and is a "simple" (yet highly parallelized) 
implementation of the SCF method for the calculation of the ground state energy 
of a molecule for a given geometry. As of now, the programme can only calculate 
the energy of a molecule for "paired" electrons, which is called "RHF" for 
Restricted Hartree-Fock. No DFT-methods are implemented yet. 

A number of common basis sets are added to code. Because the author has written 
a parser for the Gaussian basis set format, it is possible to add any basis set 
to the code. As of now the following basis sets are implemented:

STO-3G, STO-6G, 6-311G**, 6-311G*, 
def2-SVP, def2-TZVP, def2-TZVPP and def2-QZVP.

(Although this programme uses the McMurchie-Davidson algorithm to calculate the 
integrals, which utilizies the evaluation of Cartesian Gaussians, which 
may not conform to the definition of the def2-group basis-sets defined by Ahlrichs.)
********************************************************************************
"#;

    println!("{msg}");
}

pub fn print_header_with_long_barrier(header_title: &str) {
    let barrier = format!("{}", "=".repeat(79));
    let header_title = format!("{:^79}", header_title);

    println!("{}\n{}\n{}", barrier, header_title, barrier);
}

pub fn print_input_file_line(line_no: usize, line: &String) {
    println!("{}: {}", line_no + 1, line);
}

pub(crate) fn print_footer(start_exec_time: Instant) {
    let end_of_calc_string: String = format!("{:^79}", "RUN ENDED SUCCESSFULLY!");
    println!("\n{}", "*".repeat(81));
    println!("*{end_of_calc_string}*");
    println!("{}", "*".repeat(81));
    let end_exec_time = Instant::now();
    let duration_exec_time = end_exec_time.duration_since(start_exec_time);
    let formatted_duration_exec_time = format_duration(duration_exec_time).to_string();
    println!("\nTime elapsed in execution is: {formatted_duration_exec_time}");
}
