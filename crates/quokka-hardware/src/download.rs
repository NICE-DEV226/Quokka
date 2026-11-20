//! Téléchargement de modèles GGUF depuis Hugging Face.
//!
//! Pipeline : `HEAD` sur l'URL `resolve` (taille + ETag sha256) → GET
//! segmenté. Les gros fichiers (> [`SEGMENT_MIN_BYTES`]) sont découpés en
//! jusqu'à [`MAX_PARALLEL_WORKERS`] plages `Range` écrites en parallèle à
//! offsets absolus d'un `.part` pré-alloué ; les petits fichiers — ou les
//! serveurs sans support `Range` — passent par un flux séquentiel reprenant
//! sur la taille déjà présente dans `.part`.
//!
//! L'ETag `x-linked-etag` de HF est le SHA-256 du blob LFS : il est
//! vérifié avant le rename atomique `.part` → nom final. Un fichier final
//! déjà complet est renvoyé tel quel (téléchargement idempotent).
//!
//! `shortcut: le mode parallèle repart de zéro si `.part` existe déjà —
//! la reprise fine par segment demanderait un fichier de contrôle ;
//! à reprendre si les coupures sur gros modèles deviennent fréquentes.`

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use reqwest::StatusCode;
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_LENGTH, ETAG, RANGE};
use sha2::{Digest, Sha256};

use crate::error::{HardwareError, Result};
use crate::registry::ModelRef;

/// Plancher par segment parallèle : en dessous, l'overhead HTTP domine.
const SEGMENT_MIN_BYTES: u64 = 64 * 1024 * 1024;
/// Le mode parallèle ne se justifie qu'à partir de deux segments pleins.
const PARALLEL_MIN_BYTES: u64 = 2 * SEGMENT_MIN_BYTES;
const MAX_PARALLEL_WORKERS: usize = 8;
/// Tentatives par segment : une coupure CDN ne doit pas corrompre le
/// fichier — les écritures à offset absolu sont idempotentes.
const MAX_SEGMENT_ATTEMPTS: u32 = 3;
/// Tampon de hash et de lecture : 1 Mio, cache-friendly.
const IO_BUF_BYTES: usize = 1024 * 1024;
const USER_AGENT: &str = concat!("quokka/", env!("CARGO_PKG_VERSION"));

/// Télécharge `model` dans `dest_dir` (créé si absent) et renvoie le chemin
/// final. `on_progress(downloaded, total)` est appelé à chaque chunk ;
/// `total` vaut 0 tant que la taille n'est pas connue. Le callback doit être
/// `'static` : il est partagé entre les workers de segments.
pub fn pull(
    model: &ModelRef,
    dest_dir: &Path,
    on_progress: impl Fn(u64, u64) + Send + Sync + 'static,
) -> Result<PathBuf> {
    let url = model.hf_url();
    let client = build_client()?;
    let meta = head_metadata(&client, &url)?;
    let total = meta
        .size_bytes
        .ok_or_else(|| HardwareError::DownloadSizeUnknown { url: url.clone() })?;

    let final_path = dest_dir.join(&model.filename);
    if file_len(&final_path) == Some(total) {
        on_progress(total, total);
        tracing::info!(path = %final_path.display(), "model already downloaded, skipping");
        return Ok(final_path);
    }

    fs::create_dir_all(dest_dir).map_err(|source| HardwareError::DownloadIo {
        path: dest_dir.display().to_string(),
        source,
    })?;
    let part_path = dest_dir.join(format!("{}.part", model.filename));

    let progress = Arc::new(Progress {
        downloaded: AtomicU64::new(0),
        total,
        report: Mutex::new(Box::new(on_progress)),
    });

    let supports_range = probe_range_support(&client, &url)?;
    if supports_range && total >= PARALLEL_MIN_BYTES {
        download_parallel(&client, &url, &part_path, total, progress.clone())?;
    } else {
        download_sequential(&client, &url, &part_path, total, progress.clone())?;
    }

    // Source d'intégrité : `lfs.oid` de l'API HF en priorité — l'ETag du
    // HEAD expose le *xetHash* sur les dépôts migrés Xet, pas le sha256
    // (constaté sur bartowski/*-GGUF : hash attendu 4cb1…, fichier réel
    // 6c1a… — le téléchargement était sain, c'est l'ETag qui mentait).
    let expected_sha = hf_lfs_sha256(&client, model).or_else(|| {
        if meta.sha256.is_some() {
            tracing::warn!(
                url = %url,
                "HF API unreachable — falling back to ETag which may be a Xet hash"
            );
        }
        meta.sha256.clone()
    });
    if let Some(expected) = &expected_sha {
        let actual = sha256_file(&part_path)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(HardwareError::Integrity {
                path: part_path.display().to_string(),
                expected: expected.clone(),
                actual,
            });
        }
    } else {
        tracing::debug!(url = %url, "no sha256 ETag exposed, skipping integrity check");
    }

    fs::rename(&part_path, &final_path).map_err(|source| HardwareError::DownloadIo {
        path: final_path.display().to_string(),
        source,
    })?;
    Ok(final_path)
}

/// `~/.quokka/models` — destination par défaut des modèles tirés par
/// `quokka pull`. `None` si aucune racine home n'est résolvable.
#[must_use]
pub fn models_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| PathBuf::from(home).join(".quokka").join("models"))
}

fn build_client() -> Result<Client> {
    Client::builder()
        .user_agent(USER_AGENT)
        // Les redirects HF → CDN (Xet/LFS) doivent être suivis ; la limite
        // par défaut de reqwest (10) suffit.
        .build()
        .map_err(|source| http_err("<client builder>", source))
}

/// Métadonnées utiles extraites des en-têtes de réponse.
struct RemoteMeta {
    size_bytes: Option<u64>,
    sha256: Option<String>,
}

fn head_metadata(client: &Client, url: &str) -> Result<RemoteMeta> {
    let response = client
        .head(url)
        .send()
        .map_err(|source| http_err(url, source))?;
    let response = response
        .error_for_status()
        .map_err(|source| http_err(url, source))?;
    Ok(meta_from_headers(response.headers()))
}

/// Requête `Range: bytes=0-0` : prouve le support Range et récupère au
/// passage les ETag que le HEAD aurait pu omettre.
fn probe_range_support(client: &Client, url: &str) -> Result<bool> {
    let response = client
        .get(url)
        .header(RANGE, "bytes=0-0")
        .send()
        .map_err(|source| http_err(url, source))?;
    if !response.status().is_success() {
        return Ok(false);
    }
    Ok(response.status() == StatusCode::PARTIAL_CONTENT)
}

/// SHA-256 LFS faisant foi, via `POST /api/models/{repo}/paths-info`.
/// `None` si l'API est injoignable ou le fichier n'est pas en LFS.
fn hf_lfs_sha256(client: &Client, model: &ModelRef) -> Option<String> {
    let url = format!(
        "https://huggingface.co/api/models/{}/paths-info/main",
        model.repo_id
    );
    let entries: Vec<serde_json::Value> = client
        .post(&url)
        .json(&serde_json::json!({ "paths": [model.filename] }))
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .ok()?
        .json()
        .ok()?;
    entries
        .iter()
        .find(|entry| {
            entry.get("path").and_then(|p| p.as_str()) == Some(model.filename.as_str())
        })
        .and_then(|entry| entry.pointer("/lfs/oid"))
        .and_then(|oid| oid.as_str())
        .filter(|oid| {
            oid.len() == 64 && oid.bytes().all(|b| b.is_ascii_hexdigit())
        })
        .map(str::to_owned)
}

fn meta_from_headers(headers: &reqwest::header::HeaderMap) -> RemoteMeta {
    let size_bytes = headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|text| text.parse::<u64>().ok());
    // `x-linked-etag` porte le sha256 LFS chez HF ; `etag` est le repli.
    let sha256 = headers
        .get("x-linked-etag")
        .or_else(|| headers.get(ETAG))
        .and_then(|value| value.to_str().ok())
        .and_then(parse_sha256_etag);
    RemoteMeta { size_bytes, sha256 }
}

/// Extrait un SHA-256 hexadécimal d'un ETag HTTP (`"abc…"`, `W/"abc…"`).
/// Renvoie `None` pour les ETag non-sha256 (MD5, étiquettes faibles…).
fn parse_sha256_etag(value: &str) -> Option<String> {
    let candidate = value
        .trim()
        .trim_start_matches("W/")
        .trim_matches('"')
        .to_ascii_lowercase();
    (candidate.len() == 64 && candidate.bytes().all(|b| b.is_ascii_hexdigit())).then_some(candidate)
}

/// Flux séquentiel avec reprise : `Range: bytes=<déjà présent>-` quand le
/// serveur accepte, sinon réécriture complète.
fn download_sequential(
    client: &Client,
    url: &str,
    part_path: &Path,
    total: u64,
    progress: Arc<Progress>,
) -> Result<()> {
    // Un `.part` plus long que `total` est corrompu (autre fichier, autre
    // révision) : on le supprime plutôt que de produire un blob trop gros.
    let raw_len = file_len(part_path).unwrap_or(0);
    if raw_len > total {
        fs::remove_file(part_path).map_err(|source| io_err(part_path, source))?;
    }
    let have = raw_len.min(total);
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(part_path)
        .map_err(|source| io_err(part_path, source))?;

    let mut response = if have == total {
        return Ok(());
    } else if have > 0 {
        let response = client
            .get(url)
            .header(RANGE, format!("bytes={have}-"))
            .send()
            .map_err(|source| http_err(url, source))?;
        if response.status() == StatusCode::PARTIAL_CONTENT {
            file.seek(SeekFrom::End(0))
                .map_err(|source| io_err(part_path, source))?;
            progress.add(have);
            response
        } else {
            // Serveur sans Range : repartir de zéro.
            file.set_len(0)
                .map_err(|source| io_err(part_path, source))?;
            file.seek(SeekFrom::Start(0))
                .map_err(|source| io_err(part_path, source))?;
            client
                .get(url)
                .send()
                .map_err(|source| http_err(url, source))?
        }
    } else {
        client
            .get(url)
            .send()
            .map_err(|source| http_err(url, source))?
    };

    let mut counted = CountingReader {
        inner: io::BufReader::with_capacity(IO_BUF_BYTES, &mut response),
        progress: progress.clone(),
    };
    io::copy(&mut counted, &mut file).map_err(|source| io_err(part_path, source))?;
    file.flush().map_err(|source| io_err(part_path, source))
}

/// Découpage en plages contiguës `[start, end)` couvrant `[0, total)`.
fn plan_segments(total: u64, max_workers: usize) -> Vec<(u64, u64)> {
    let workers = ((total / SEGMENT_MIN_BYTES) as usize).clamp(1, max_workers);
    let base = total / workers as u64;
    let remainder = total % workers as u64;
    let mut ranges = Vec::with_capacity(workers);
    let mut start = 0_u64;
    for index in 0..workers {
        let len = base + u64::from(index < remainder as usize);
        ranges.push((start, start + len));
        start += len;
    }
    ranges
}

/// Télécharge `total` octets en `plan_segments` plages parallèles. Chaque
/// worker écrit à son offset absolu dans le `.part` pré-alloué : aucune
/// coordination d'écriture n'est nécessaire.
fn download_parallel(
    client: &Client,
    url: &str,
    part_path: &Path,
    total: u64,
    progress: Arc<Progress>,
) -> Result<()> {
    // Un `.part` antérieur ne garantit aucune plage complète : on repart
    // propre (voir le `shortcut:` du module).
    if part_path.exists() {
        fs::remove_file(part_path).map_err(|source| io_err(part_path, source))?;
    }
    File::create(part_path)
        .and_then(|file| file.set_len(total))
        .map_err(|source| io_err(part_path, source))?;

    let ranges = plan_segments(total, MAX_PARALLEL_WORKERS);
    let mut handles = Vec::with_capacity(ranges.len());
    for (segment, &(start, end)) in ranges.iter().enumerate() {
        let client = client.clone();
        let url = url.to_owned();
        let part_path = part_path.to_owned();
        let progress = progress.clone();
        handles.push(thread::spawn(move || {
            fetch_segment(&client, &url, &part_path, start, end, progress).map_err(|error| {
                HardwareError::DownloadSegment {
                    segment,
                    reason: error.to_string(),
                }
            })
        }));
    }

    for handle in handles {
        match handle.join() {
            Ok(result) => result?,
            Err(_) => {
                return Err(HardwareError::DownloadSegment {
                    segment: usize::MAX,
                    reason: "worker thread panicked".into(),
                });
            }
        }
    }
    Ok(())
}

/// Un segment `[start, end)` : GET `Range`, écriture à offset absolu,
/// avec retry — une réponse tronquée ne doit jamais laisser de trou.
fn fetch_segment(
    client: &Client,
    url: &str,
    part_path: &Path,
    start: u64,
    end: u64,
    progress: Arc<Progress>,
) -> Result<()> {
    let mut attempt = 0_u32;
    loop {
        match fetch_segment_once(client, url, part_path, start, end, progress.clone()) {
            Ok(()) => return Ok(()),
            Err(err) if attempt + 1 < MAX_SEGMENT_ATTEMPTS => {
                attempt += 1;
                tracing::warn!(
                    %url, start, end, attempt, error = %err,
                    "segment download failed, retrying"
                );
                thread::sleep(std::time::Duration::from_millis(200 * u64::from(attempt)));
            }
            Err(err) => return Err(err),
        }
    }
}

/// Une tentative de segment. Toute anomalie (statut, longueur annoncée,
/// octets écrits ≠ plage demandée) est une erreur — jamais de trou muet.
fn fetch_segment_once(
    client: &Client,
    url: &str,
    part_path: &Path,
    start: u64,
    end: u64,
    progress: Arc<Progress>,
) -> Result<()> {
    let expected = end - start;
    let response = client
        .get(url)
        .header(RANGE, format!("bytes={start}-{}", end - 1))
        .send()
        .map_err(|source| http_err(url, source))?;
    let status = response.status();
    if status != StatusCode::PARTIAL_CONTENT {
        return Err(HardwareError::DownloadStatus {
            url: url.to_owned(),
            status: status.as_u16(),
            start,
            end,
        });
    }
    if let Some(declared) = response.content_length()
        && declared != expected
    {
        return Err(io_err(
            part_path,
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("segment {start}-{end}: server sent {declared} bytes, expected {expected}"),
            ),
        ));
    }

    let mut file = OpenOptions::new()
        .write(true)
        .open(part_path)
        .map_err(|source| io_err(part_path, source))?;
    file.seek(SeekFrom::Start(start))
        .map_err(|source| io_err(part_path, source))?;

    let mut counted = CountingReader {
        inner: io::BufReader::with_capacity(IO_BUF_BYTES, response),
        progress,
    };
    let written = io::copy(&mut counted, &mut file).map_err(|source| io_err(part_path, source))?;
    file.flush().map_err(|source| io_err(part_path, source))?;

    // `io::copy` s'arrête sur EOF propre : un body plus court que la plage
    // demandée passe inaperçu sans ce contrôle — c'est ce qui a corrompu
    // le premier téléchargement (trou de ~120 Mio → sha256 mismatch).
    if written != expected {
        return Err(io_err(
            part_path,
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("segment {start}-{end}: wrote {written} bytes, expected {expected}"),
            ),
        ));
    }
    Ok(())
}

/// Compteur de progression thread-safe : `report` est sérialisé par mutex
/// (un appel par chunk de ~1 Mio par worker — contention négligeable).
struct Progress {
    downloaded: AtomicU64,
    total: u64,
    report: Mutex<Box<dyn Fn(u64, u64) + Send + Sync>>,
}

impl Progress {
    fn add(&self, bytes: u64) {
        let downloaded = self.downloaded.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if let Ok(report) = self.report.lock() {
            // Les retries de segment recompteraient leurs octets : on borne
            // l'affichage à `total` (cosmétique, le fichier reste exact).
            report(downloaded.min(self.total), self.total);
        }
    }
}

/// `Read` qui rapporte chaque octet lu au [`Progress`] partagé.
struct CountingReader<R> {
    inner: R,
    progress: Arc<Progress>,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buf)?;
        if read > 0 {
            self.progress.add(read as u64);
        }
        Ok(read)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|source| io_err(path, source))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; IO_BUF_BYTES];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| io_err(path, source))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn file_len(path: &Path) -> Option<u64> {
    fs::metadata(path).ok().map(|meta| meta.len())
}

fn http_err(url: &str, source: reqwest::Error) -> HardwareError {
    HardwareError::DownloadHttp {
        url: url.to_owned(),
        source,
    }
}

fn io_err(path: &Path, source: io::Error) -> HardwareError {
    HardwareError::DownloadIo {
        path: path.display().to_string(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn etag_parsing_accepts_only_sha256_hex() {
        let sha = "a".repeat(64);
        assert_eq!(parse_sha256_etag(&format!("\"{sha}\"")), Some(sha.clone()));
        assert_eq!(
            parse_sha256_etag(&format!("W/\"{sha}\"")),
            Some(sha.clone())
        );
        assert_eq!(
            parse_sha256_etag(&sha.to_uppercase()),
            Some(sha.to_lowercase())
        );
        // MD5 (32 hex) et étiquettes arbitraires : rejetés.
        assert_eq!(parse_sha256_etag(&"b".repeat(32)), None);
        assert_eq!(parse_sha256_etag("\"not-a-hash\""), None);
    }

    #[test]
    fn segments_cover_file_exactly() {
        for total in [
            SEGMENT_MIN_BYTES,
            PARALLEL_MIN_BYTES,
            1 << 30,
            (1 << 30) + 1,
        ] {
            let ranges = plan_segments(total, MAX_PARALLEL_WORKERS);
            assert_eq!(ranges.first().map(|r| r.0), Some(0));
            assert_eq!(ranges.last().map(|r| r.1), Some(total));
            let mut covered = 0_u64;
            for &(start, end) in &ranges {
                assert!(start < end && start >= covered, "overlap or gap at {start}");
                covered = end;
            }
            assert_eq!(covered, total);
        }
    }

    #[test]
    fn small_file_gets_single_segment() {
        assert_eq!(plan_segments(1000, MAX_PARALLEL_WORKERS), vec![(0, 1000)]);
    }
}
