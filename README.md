# Template Extractor in Rust

## Idea: 
> This is just a rough idea or a starting point.
* Extract the template from given dataset of invoices or bills.
* The format can be any but lets start with pdf or html only.
* Their are a lot of open source resources that can be used. 
* The user should be able to provide a lot of files. 
* Backend will consume the files and use models to create embeddings. 
* Then it should normalize those embeddings and use k-means clustering to identify different types of templates. 
* It should use cosine similarity to group similar items.

## Why Rust? 
* I like rust-lang. 
* Their are a lot of recources availble for it. 
* It is fast, might not be the fastest with Gen AI and ML as it is still new.

## What is implemented? 
* It is able to download the model and create embeddings. 
* It is able to normalize vectors and also use cosine similarity.


## References: 
* https://huggingface.co/BAAI/bge-small-en/tree/main
* https://huggingface.github.io/candle/guide/hello_world.html
* https://huggingface.github.io/candle/inference/hub.html
* https://github.com/huggingface/candle/tree/main/candle-examples/examples/bert
* https://github.com/huggingface/candle/blob/main/candle-examples/examples/bert/main.rs
* https://github.com/invoice-x/invoice2data

## Datasets: 
* https://huggingface.co/datasets/aharley/rvl_cdip
* https://github.com/femstac/Sample-Pdf-invoices
* https://zenodo.org/records/5113009#.YRK6IIj7RPY
* https://data.mendeley.com/datasets/tnj49gpmtz/2
