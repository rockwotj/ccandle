use anyhow::Result;
use ccandle::*;

fn main() -> Result<()> {
    println!("hello");
    let model_name = "mistral";
    let mut model = unsafe {
        ccandle_load_model(UnownedString {
            data: model_name.as_ptr(),
            length: model_name.len(),
        })
        .unwrap()
    };
    println!("loaded");
    let prompt = "write a haiku about a redpanda";
    let prompt = UnownedString {
        data: prompt.as_ptr(),
        length: prompt.len(),
    };
    let resp: String = unsafe {
        ccandle_run_model(&mut model, prompt, 100)
            .unwrap()
            .into_string()
    };
    println!("final resp: {}", resp);
    Ok(())
}
