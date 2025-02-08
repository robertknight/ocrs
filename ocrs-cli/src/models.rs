use std::fmt;
use std::io::Read;
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
use std::{fs, path::Path};

use anyhow::anyhow;
use rten::Model;

#[cfg(not(target_arch = "wasm32"))]
use url::Url;

/// Return the path to the directory in which cached models etc. should be
/// saved.
#[cfg(not(target_arch = "wasm32"))]
fn cache_dir() -> Result<PathBuf, anyhow::Error> {
    let mut cache_dir: PathBuf =
        home::home_dir().ok_or(anyhow!("Failed to determine home directory"))?;
    cache_dir.push(".cache");
    cache_dir.push("ocrs");

    fs::create_dir_all(&cache_dir)?;

    Ok(cache_dir)
}

/// Extract the last path segment from a URL.
///
/// eg. "https://models.com/text-detection.rten" => "text-detection.rten".
#[cfg(not(target_arch = "wasm32"))]
#[allow(rustdoc::bare_urls)]
fn filename_from_url(url: &str) -> Option<String> {
    let parsed = Url::parse(url).ok()?;
    let path = Path::new(parsed.path());
    path.file_name()
        .and_then(|f| f.to_str())
        .map(|s| s.to_string())
}

/// Download a file from `url` to a local cache, if not already fetched, and
/// return the path to the local file.
#[cfg(not(target_arch = "wasm32"))]
fn download_file(url: &str, filename: Option<&str>) -> Result<PathBuf, anyhow::Error> {
    let cache_dir = cache_dir()?;
    let filename = match filename {
        Some(fname) => fname.to_string(),
        None => filename_from_url(url).ok_or(anyhow!("Could not get destination filename"))?,
    };
    let file_path = cache_dir.join(filename);
    if file_path.exists() {
        return Ok(file_path);
    }

    eprintln!("Downloading {}...", url);

    let mut body = ureq::get(url).call()?.into_body();
    let mut buf = Vec::new();
    body.as_reader().read_to_end(&mut buf)?;

    fs::write(&file_path, &buf)?;

    Ok(file_path)
}

#[cfg(target_arch = "wasm32")]
fn download_file(_url: &str, _filename: Option<&str>) -> Result<PathBuf, anyhow::Error> {
    Err(anyhow!(
        "Downloading models from a URL is not supported on the current platform"
    ))
}

/// Location that a model can be loaded from.
#[derive(Clone, Copy)]
pub enum ModelSource<'a> {
    /// Load model from an HTTP(S) URL.
    Url(&'a str),

    /// Load model from a local file path.
    Path(&'a str),
}

impl fmt::Display for ModelSource<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ModelSource::Url(url) => url,
                ModelSource::Path(path) => path,
            }
        )
    }
}

/// Load a model from a given source.
///
/// If the source is a URL, the model will be downloaded and cached locally if
/// needed.
pub fn load_model(source: ModelSource) -> Result<Model, anyhow::Error> {
    let model_path = match source {
        ModelSource::Url(url) => download_file(url, None)?,
        ModelSource::Path(path) => path.into(),
    };
    let model = Model::load_file(model_path)?;
    Ok(model)
}
