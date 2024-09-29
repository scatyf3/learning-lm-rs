use std::{fmt, io::{self, Write}};

pub struct Template{
    system_message: String,
    user_message: String,
}

impl fmt::Display for Template {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", self.system_message, self.user_message);
        Ok(())
    }
}

impl Template {

    pub fn new(system_message: String) -> Self {
        Self {
            system_message,
            user_message: String::new(), 
        }
    }

    // 手动修改user_message
    pub fn add(&mut self, user_message: String) {
        self.user_message = user_message;
    }

    // 通过用户输入修改 user_message
    pub fn ask_user_input(&mut self) {
        io::stdout().flush().unwrap(); 

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        // 去掉输入中的换行符
        self.user_message = input.trim().to_string();
    }

    pub fn get_system_message(&self)-> String{
        format!("<|im_start|>system\n{}<|im_end|>\n",self.system_message)
    }

    pub fn get_user_message(&self)-> String{
        format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",self.user_message)
    }

}
