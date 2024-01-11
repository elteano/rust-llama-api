use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use curl::easy::Easy;
use clap::Parser;

#[derive(Parser)]
struct InputOptions
{
    #[arg(short, long)]
    file: Option<String>,
    #[arg(short, long, conflicts_with = "file")]
    prompt: bool,
    #[arg(short, long, default_value = "llama2:7b-chat", help = "Name of the model to query", value_name = "MODEL:TAG")]
    model: Option<String>
}

#[derive(Serialize, Deserialize, Default)]
struct ModelOptions
{
    num_keep: Option<u32>,
    seed: Option<u32>,
    num_predict: Option<u32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    tfs_z: Option<f32>,
    typical_p: Option<f32>,
    repeat_last_n: Option<u32>,
    temperature: Option<f32>,
    repeat_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    mirostat: Option<i32>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    penalize_newline: Option<bool>,
    stop: Option<Vec<String>>,
    numa: Option<bool>,
    num_ctx: Option<u32>,
    num_batch: Option<u32>,
    num_gqa: Option<u32>,
    num_gpu: Option<u32>,
    main_gpu: Option<u32>,
    low_vram: Option<bool>,
    f16_kv: Option<bool>,
    vocab_only: Option<bool>,
    use_mmap: Option<bool>,
    use_mlock: Option<bool>,
    embedding_only: Option<bool>,
    rope_frequency_base: Option<f32>,
    rope_frequency_scale: Option<f32>,
    num_thread: Option<u8>
}

#[derive(Serialize, Deserialize, Default)]
struct Message
{
    role: String,
    content: String,
    done: Option<bool>,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u32>,
    eval_duration: Option<u64>
}

#[derive(Serialize, Deserialize)]
struct LlamaResponse
{
    model: String,
    created_at: String,
    message: Message
}

#[derive(Serialize, Deserialize, Default)]
struct LlamaRequest
{
    model: String,
    stream: bool,
    messages: Vec<Message>,
    options: Option<ModelOptions>
}

#[derive(Serialize, Deserialize, Default)]
struct ErrorResponse
{
    error: String
}

fn individual_request(request_object: &LlamaRequest) -> Result<String, String>
{
    let data = serde_json::to_string(&request_object).unwrap();

    // Buffer to hold curl response data
    let mut buf = Vec::new();

    let mut curl_easy = Easy::new();
    curl_easy.url("http://localhost:11434/api/chat").unwrap();

    curl_easy.read_function(move |into| {
        Ok(data.as_bytes().read(into).unwrap())
    }).unwrap();
    curl_easy.post(true).unwrap();

    {
        let mut transfer = curl_easy.transfer();
        transfer.write_function(|data| {
            buf.extend_from_slice(data);
            Ok(buf.len())
        }).unwrap();

        println!("Making request...");
        match transfer.perform()
        {
            Ok(_) => { () }
            Err(msg) => { eprintln!("{}", msg); return Err(msg.to_string()) }
        };
    }

    let decoded = std::str::from_utf8(&buf).unwrap();

    match serde_json::from_str::<ErrorResponse>(&decoded)
    {
        Ok(val) => {
            eprintln!("Error received: {}", val.error);
            return Err(val.error)
        }
        Err(_) => { () }
    };

    let r: LlamaResponse = match serde_json::from_str(&decoded)
    {
        Ok(val) => { val }
        Err(err) => {
            eprintln!("Unable to decode response:\n{0}", err.to_string());
            return Err(err.to_string());
        }
    };

    // Everything should be serializable so no error expected
    return Ok(r.message.content);
}

fn make_request(model_name: String, prompt: String) -> Result<String, String>
{
    let mut req = LlamaRequest {
        model: model_name,
        stream: false,
        messages: Vec::new(),
        options: Some(ModelOptions {
            temperature: Some(0.8),
            ..Default::default()
        })
    };

    req.messages.push(Message {
        role: "user".to_string(),
        content: prompt,
        ..Default::default()
    });

    return individual_request(&req);
}

fn main()
{
    let args = InputOptions::parse();

    /*
    opts.optopt("f", "file", "Read input from a specified file.", "FILE");
    opts.optflag("p", "prompt", "Prompt for user input.");
    opts.optopt("m", "model",
,
                "");
    */

    let model_tag = match args.model
    {
        Some(s) => { s }
        None => { "llama2:7b-chat".to_string() }
    };

    let mut prompt = String::new();

    if args.file.is_some()
    {
        // we already verified that the 'f' option exists, so no need to check again
        let fname = args.file.unwrap();
        if fname == "-"
        {
            match std::io::stdin().read_to_string(&mut prompt)
            {
                Ok(_) => { () }
                Err(err) => { eprintln!("Error reading stdin: {err}"); return }
            };
        }
        else
        {
            // Currently assume the file exists
            let mut fhandle = match File::open(&fname)
            {
                Ok(f) => { f }
                Err(msg) => { 
                    match msg.kind()
                    {
                        std::io::ErrorKind::NotFound => { eprintln!("File {fname} not found.") }
                        _ => { eprintln!("{msg}") }
                    };
                    return }
            };
            fhandle.read_to_string(&mut prompt).unwrap();
        }
    }
    else if args.prompt
    {
        print!("Enter your prompt on a single line:\n>");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut prompt).expect("Expected user input but could not use STDIN.");
    }
    else
    {
        eprintln!("--file or --prompt are required parameters.");
        return;
    }

    match make_request(model_tag, prompt)
    {
        Ok(res) => { println!("{res}") }
        Err(err) => { eprintln!("Error received: {err}"); return }
    }
}
