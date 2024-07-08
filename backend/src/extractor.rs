use super::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};

pub async fn process_data(sentences: &[&str]) -> Result<()> {
    let device = Device::Cpu;

    let (model_path, tokenizer_path, config_path) = download_model_files().await?;
    let (model, mut tokenizer) =
        load_model(&model_path, &config_path, &tokenizer_path, &device).await?;

    println!("INFO: {:#?}", model.device);
    // println!("INFO: {:#?}", tokenizer);

    let n_sentences = sentences.len();

    if let Some(pp) = tokenizer.get_padding_mut() {
        println!("INFO: Tokenizer Padding Params: {pp:?}");
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        println!("INFO: Longest Batch Padding Params: {pp:?}");
        tokenizer.with_padding(Some(pp));
    }

    let tokens = tokenizer.encode_batch(sentences.to_vec(), true).unwrap();

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let token_type_ids = token_ids.zeros_like()?;
    println!("INFO: Running inference on batch {:?}", token_ids.shape());
    let embeddings = model.forward(&token_ids, &token_type_ids)?;
    println!("INFO: Generated embeddings {:?}", embeddings.shape());
    // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    let embeddings = normalize_l2(&embeddings)?;
    println!("INFO: Pooled embeddings {:?}", embeddings.shape());

    let mut similarities = find_cosine_similarities(&embeddings, n_sentences)?;

    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities[..5].iter() {
        println!(
            "INFO: Score: {score:.2} '{}' '{}'",
            sentences[i], sentences[j]
        )
    }

    Ok(())
}

fn find_cosine_similarities(
    embeddings: &Tensor,
    n_sentences: usize,
) -> Result<Vec<(f32, usize, usize)>> {
    let mut similarities = vec![];
    for i in 0..n_sentences {
        let e_i = embeddings.get(i)?;
        for j in (i + 1)..n_sentences {
            let e_j = embeddings.get(j)?;
            let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
            let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            similarities.push((cosine_similarity, i, j))
        }
    }
    Ok(similarities)
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

#[allow(unused)]
async fn download_model_files() -> Result<(PathBuf, PathBuf, PathBuf)> {
    let api = Api::new().unwrap();
    let repo = api.repo(Repo::with_revision(
        "BAAI/bge-small-en".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let model_path = repo.get("model.safetensors").await.unwrap();
    let config_path = repo.get("config.json").await.unwrap();
    let tokenizer_path = repo.get("tokenizer.json").await.unwrap();

    // let vocab_path = repo.get("vocab.txt").await.unwrap();
    // let config_sentence_transformers_path = repo.get("config_sentence_transformers.json").await.unwrap();
    // let modules_path = repo.get("modules.json").await.unwrap();
    // let pytorch_model_bin_path= repo.get("pytorch_model.bin").await.unwrap();
    // let sentence_bert_config_path= repo.get("sentence_bert_config.json").await.unwrap();
    // let special_tokens_map_path= repo.get("special_tokens_map.json").await.unwrap();
    // let tokenizer_config_path = repo.get("tokenizer_config.json").await.unwrap();

    println!(
        "INFO: model path:{:?}, tokenizer path:{:?}, config path: {:?}",
        model_path, tokenizer_path, config_path
    );
    Ok((model_path, tokenizer_path, config_path))
}

async fn load_model(
    model_path: &PathBuf,
    config_path: &PathBuf,
    tokenizer_path: &PathBuf,
    device: &Device,
) -> Result<(BertModel, Tokenizer)> {
    let config = std::fs::read_to_string(config_path).unwrap();
    let config: Config = serde_json::from_str(&config).unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DTYPE, &device)? };
    // let vb = VarBuilder::from_pth(&model_path, DTYPE, &device).unwrap();
    let model = BertModel::load(vb, &config).unwrap();
    Ok((model, tokenizer))
}
