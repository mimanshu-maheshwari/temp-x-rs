use std::path::Path;

use pdf::file::File as PdfFile;
// use std::fs::File as PdfFile;

use walkdir::WalkDir;
use super::Result;

#[derive(Debug, Clone, Copy)]
enum DocumentType {
    PDF,
    HTML,
}

pub fn read_dir(dir_name: &str) {
    let texts = extract_texts_from_pdfs(dir_name);
}

fn extract_texts_from_pdfs(dir_path: &str) -> Vec<String> {
    let mut texts = Vec::new();

    for entry in WalkDir::new(dir_path).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().and_then(|s| s.to_str()) == Some("pdf") {
            if let Ok(text) = extract_text_from_pdf(entry.path()) {
                texts.push(text);
            }
        }
    }

    texts
}

fn extract_text_from_pdf(file_path: &Path) -> Result<String> {
    // let file = PdfFile::open(file_path)?;
    let mut text = String::new();

//    for page in file.pages() {
//        let page = page?;
//        if let Some(content) = page.contents.as_ref() {
//            for stream in content.iter() {
//                if let pdf::primitive::Primitive::String(content) = stream {
//                    text.push_str(&content.to_string()?);
//                }
//            }
//        }
//    }

    Ok(text)
}

