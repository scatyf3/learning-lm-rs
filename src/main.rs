mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod template;

use std::path::PathBuf;
use template::Template;
use tensor::Tensor;
use tokenizers::{ Tokenizer};
// a most common arg parser in rust
use clap::Parser;


#[derive(Clone, clap::ValueEnum, Debug)]
enum Service{
    Generate,
    Chat,
}
//clap 默认使用小写的匹配？

//clap::Parser requires this
impl ToString for Service {
    fn to_string(&self) -> String {
        match self {
            Service::Generate => "generate".to_string(),
            Service::Chat => "chat".to_string(),
        }
    }
}

#[command(version, about, long_about = None)]
#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value_t = String::from("Once upon a time"))]
    prompts: String,
    
    #[arg(short, long, default_value_t = Service::Generate)]
    service: Service
}



fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let args = Args::parse();
    match args.service {
        Service::Generate => {
            let model_dir = PathBuf::from(project_dir).join("models").join("story");
            let llama = model::Llama::<f32>::from_safetensors(&model_dir);
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            let input = args.prompts;
            print!("\n{}", input);
            let binding = tokenizer.encode(input, true).unwrap();
            let input_ids = binding.get_ids();
            let output_ids = llama.generate(
                input_ids,
                500,
                0.9,
                4,
                1.,
            );
            //greedy
            //let output_ids = llama.generate(input_ids,500,1.0,1,1.);
            println!("{}", tokenizer.decode(&output_ids, true).unwrap());
        }
        Service::Chat => {
            println!("进入chat模式");
            let model_dir = PathBuf::from(project_dir).join("models").join("chat");
            let llama = model::Llama::<f32>::from_safetensors(&model_dir);
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            let mut kv_cache = llama.new_cache();
            let mut t = Template::new(String::from("You are an imaginative story generator. Create engaging and creative stories based on user prompts."));
            println!("{}", t.get_system_message());
            let binding = tokenizer.encode(t.get_system_message(), true).unwrap();
            let input_system_prompt = binding.get_ids();
            //先forward system prompt
            llama.forward(&Tensor::<u32>::new(input_system_prompt.to_vec(), &vec![t.get_system_message().len()]), &mut kv_cache);

            //然后交互地获取用户信息
            loop {
                t.ask_user_input();
                // t.add(String::from("Once upon a time"));
                println!("{}", t.get_user_message());
                let binding = tokenizer.encode(t.get_user_message(), true).unwrap();
                let input_user_prompt = binding.get_ids();
                let res = llama.chat(input_user_prompt, 500, 0.9, 4, 1., &mut kv_cache);
                println!("{}", tokenizer.decode(&res, true).unwrap());
                break;
            }
            
        }
    }

}
