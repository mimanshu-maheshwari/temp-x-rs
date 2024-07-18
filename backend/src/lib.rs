pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub mod extractor;
pub mod document;

