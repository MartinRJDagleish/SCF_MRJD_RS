use serde::Deserialize;
use serde_json::Value;



//TODO: THIS IS NOT WORKING -> whole code can be deleted
//TODO: starting with manual impl first and adding json parser later


#[derive(Deserialize, Debug)]
struct Root {
    elements: Elements,
}

#[derive(Deserialize, Debug)]
struct Elements {
    #[serde(rename = "1")]
    element_1: Element,
    #[serde(rename = "2")]
    element_2: Element,
}

#[derive(Deserialize, Debug)]
struct Element {
    electron_shells: Vec<ElectronShell>,
}

#[derive(Deserialize, Debug)]
struct ElectronShell {
    exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

pub fn parse_bsse_json() {
    let json_str = r#"{
        "elements": {
            "1": {
                "electron_shells": [
                    {
                        "exponents": [
                            "0.3425250914E+01",
                            "0.6239137298E+00",
                            "0.1688554040E+00"
                        ],
                        "coefficients": [
                            [
                                "0.1543289673E+00",
                                "0.5353281423E+00",
                                "0.4446345422E+00"
                            ]
                        ]
                    }
                ]
            },
            "2": {
                "electron_shells": [
                    {
                        "exponents": [
                            "0.6362421394E+01",
                            "0.1158922999E+01",
                            "0.3136497915E+00"
                        ],
                        "coefficients": [
                            [
                                "0.1543289673E+00",
                                "0.5353281423E+00",
                                "0.4446345422E+00"
                            ]
                        ]
                    }
                ]
            }
        }
    }"#;
    let root: Root = serde_json::from_str(json_str).unwrap();
    for element in vec![root.elements.element_1, root.elements.element_2] {
        for electron_shell in element.electron_shells {
            println!("Exponents: {:?}", electron_shell.exponents);
            println!("Coefficients: {:?}", electron_shell.coefficients);
        }
    }
}
