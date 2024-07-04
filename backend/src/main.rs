use candle_core::Device;
use hf_hub::api::tokio::Api;

#[tokio::main]
async fn main() {
    let device = Device::Cpu;
    let api = Api::new().unwrap();
    let repo = api.model("BAAI/bge-small-en".to_string());

    let weights_filename = repo.get("model.safetensors").await.unwrap();

    let weights = candle_core::safetensors::load(weights_filename, &device).unwrap();

    println!("{:#?}", weights);


}
