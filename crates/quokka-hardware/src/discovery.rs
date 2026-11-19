//! Découverte GGUF locale — politique « zéro doublon » de §4.1.
//!
//! Avant de frapper Hugging Face, scanne les magasins de modèles pouvant
//! déjà exister sur l'hôte (Ollama, LM Studio, cache HF) et rapporte les
//! blobs GGUF réutilisables. L'identification se fait par **magic GGUF**,
//! pas par nom de fichier : les blobs Ollama et le cache HF sont des hashes
//! adressés par contenu, sans extension `.gguf`.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

/// Octets magiques à l'offset 0 de tout conteneur GGUF.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// Les fichiers plus petits ne sont jamais des modèles chargeables — élimine
/// manifests, JSON de config et blobs tokenizer *avant* toute lecture disque.
pub const MIN_GGUF_BYTES: u64 = 16 * 1024 * 1024;

/// Plafond de profondeur du walk : le layout le plus profond connu est HF hub
/// `models--org--name/blobs/<sha256>`, soit 3 niveaux sous la racine.
const MAX_WALK_DEPTH: usize = 6;

/// Application possédant le magasin scanné.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    Ollama,
    LmStudio,
    HuggingFaceCache,
    /// Magasin propre de Quokka (`~/.quokka/models`).
    Quokka,
}

/// Un GGUF réutilisable trouvé sur disque.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalModel {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub source: ModelSource,
}

/// Racine de scan étiquetée avec son application propriétaire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootDir {
    pub path: PathBuf,
    pub source: ModelSource,
}

/// Racines par défaut, en honorant les variables de réinstallation
/// `OLLAMA_MODELS` et `HF_HOME`.
#[must_use]
pub fn default_roots() -> Vec<RootDir> {
    let home = home_dir();
    let mut roots = Vec::new();

    let ollama = std::env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .or_else(|| home.as_deref().map(|h| h.join(".ollama").join("models")));
    if let Some(path) = ollama {
        roots.push(RootDir {
            path,
            source: ModelSource::Ollama,
        });
    }

    if let Some(home) = home.as_deref() {
        // Magasin des modèles téléchargés par `quokka pull`.
        roots.push(RootDir {
            path: home.join(".quokka").join("models"),
            source: ModelSource::Quokka,
        });

        // LM Studio ≥ 0.3 : ~/.lm-studio/models ; legacy : ~/.cache/lm-studio.
        roots.push(RootDir {
            path: home.join(".lm-studio").join("models"),
            source: ModelSource::LmStudio,
        });
        roots.push(RootDir {
            path: home.join(".cache").join("lm-studio"),
            source: ModelSource::LmStudio,
        });
    }

    let hf_hub = std::env::var_os("HF_HOME")
        .map(|root| PathBuf::from(root).join("hub"))
        .or_else(|| {
            home.as_deref()
                .map(|h| h.join(".cache").join("huggingface").join("hub"))
        });
    if let Some(path) = hf_hub {
        roots.push(RootDir {
            path,
            source: ModelSource::HuggingFaceCache,
        });
    }

    roots
}

/// Scanne les racines par défaut. Ne peut pas échouer : les répertoires
/// illisibles sont sautés, jamais propagés.
#[must_use]
pub fn discover() -> Vec<LocalModel> {
    discover_in(&default_roots())
}

/// Scanne des racines explicites ; plus gros modèles d'abord (les experts
/// pèsent le plus dans la décision « peut-on éviter le téléchargement ? »).
#[must_use]
pub fn discover_in(roots: &[RootDir]) -> Vec<LocalModel> {
    let mut found: Vec<LocalModel> = Vec::new();

    for root in roots {
        if !root.path.is_dir() {
            tracing::debug!(root = %root.path.display(), "model root absent, skipping");
            continue;
        }

        for entry in WalkDir::new(&root.path)
            .max_depth(MAX_WALK_DEPTH)
            // Les snapshots HF sont des symlinks vers ../blobs, qu'on scanne
            // directement : pas de suivi de liens (immunité aux boucles).
            .follow_links(false)
            .into_iter()
            .filter_map(|entry| entry.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.len() < MIN_GGUF_BYTES {
                continue;
            }
            let path = entry.path();
            if is_gguf(path) {
                found.push(LocalModel {
                    path: path.to_owned(),
                    size_bytes: metadata.len(),
                    source: root.source,
                });
            }
        }
    }

    found.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then_with(|| a.path.cmp(&b.path))
    });
    found
}

/// Vérifie la magic GGUF sur 4 octets. Moins cher que parser l'en-tête et
/// suffisamment exact : aucun format mainstream ne commence par "GGUF".
#[must_use]
pub fn is_gguf(path: &Path) -> bool {
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic).is_ok() && magic == GGUF_MAGIC
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fake_gguf(path: &Path, size: u64) {
        let mut file = File::create(path).expect("create fake gguf");
        file.write_all(&GGUF_MAGIC).expect("write magic");
        // Fichier creux : instantané, aucune allocation réelle de `size` octets.
        file.set_len(size).expect("extend (sparse) to final size");
    }

    #[test]
    fn discovers_only_gguf_blobs_above_min_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        let blobs = dir.path().join("blobs");
        std::fs::create_dir_all(&blobs).expect("mkdir blobs");

        write_fake_gguf(&blobs.join("sha256-expert"), MIN_GGUF_BYTES + 5 * 1024 * 1024);
        write_fake_gguf(&blobs.join("sha256-flash"), MIN_GGUF_BYTES + 1);

        // Un gros fichier non-GGUF doit être ignoré…
        let mut junk = File::create(blobs.join("sha256-not-a-model")).expect("create junk");
        junk.write_all(b"JUNK").expect("write junk");
        junk.set_len(MIN_GGUF_BYTES * 2).expect("extend junk");
        // …et un petit fichier est éliminé avant toute lecture de magic.
        write_fake_gguf(&blobs.join("tiny.gguf"), 1024);

        let found = discover_in(&[RootDir {
            path: dir.path().to_path_buf(),
            source: ModelSource::Ollama,
        }]);
        assert_eq!(found.len(), 2, "exactly the two valid GGUF blobs");
        // Les plus gros d'abord.
        assert!(found[0].size_bytes > found[1].size_bytes);
        assert_eq!(found[0].source, ModelSource::Ollama);
        assert!(is_gguf(&found[0].path));
        assert!(!is_gguf(&blobs.join("sha256-not-a-model")));
    }

    #[test]
    fn absent_root_is_skipped_silently() {
        let found = discover_in(&[RootDir {
            path: PathBuf::from("/nonexistent/quokka-test-root"),
            source: ModelSource::LmStudio,
        }]);
        assert!(found.is_empty());
    }
}
