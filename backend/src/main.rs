use backend::extractor::process_data;
use backend::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ];

    // read data from html or pdf

    // process data using BAAI/bge-small-en model
    process_data(&sentences).await?;

    Ok(())
}
