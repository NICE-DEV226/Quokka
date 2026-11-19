//! Surface d'erreur de `quokka-hardware` (`thiserror`, DoD §8.1 :
//! zéro panic, tout remonte en `Result`).

use thiserror::Error;

/// Erreurs du probe matériel, du chargement de registre et de la découverte.
#[derive(Debug, Error)]
pub enum HardwareError {
    /// Un probe a retourné des données structurellement invalides
    /// (0 octet de RAM, 0 cœur CPU — quasi impossible, mais fatal au contrat).
    #[error("hardware probe failed: {reason}")]
    Probe {
        /// Cause lisible par un humain.
        reason: String,
    },

    /// Le fichier de registre existe mais est illisible.
    #[error("cannot read model registry `{path}`: {source}")]
    RegistryIo {
        /// Localisation (chemin filesystem ou `<embedded>`).
        path: String,
        /// Erreur I/O sous-jacente.
        source: std::io::Error,
    },

    /// Le registre n'a pas pu être désérialisé.
    #[error("invalid model registry `{path}`: {source}")]
    RegistryParse {
        /// Localisation (chemin filesystem ou `<embedded>`).
        path: String,
        /// Erreur de désérialisation JSON (avec offset).
        source: serde_json::Error,
    },

    /// Une requête HTTP du downloader a échoué (HEAD de métadonnées ou GET
    /// de segment).
    #[error("download request failed for `{url}`: {source}")]
    DownloadHttp {
        /// URL demandée (avant redirections).
        url: String,
        /// Erreur transport/statut HTTP.
        source: reqwest::Error,
    },

    /// Écriture du fichier `.part` impossible.
    #[error("cannot write download file `{path}`: {source}")]
    DownloadIo {
        /// Chemin du fichier partiel.
        path: String,
        /// Erreur I/O sous-jacente.
        source: std::io::Error,
    },

    /// Le serveur n'a pas fourni de `Content-Length` exploitable.
    #[error("download of `{url}` has no usable content length")]
    DownloadSizeUnknown {
        /// URL demandée.
        url: String,
    },

    /// Le SHA-256 calculé ne correspond pas à l'ETag annonce par le CDN.
    #[error("integrity check failed for `{path}`: expected sha256 {expected}, got {actual}")]
    Integrity {
        /// Fichier vérifié.
        path: String,
        /// Empreinte attendue (hex).
        expected: String,
        /// Empreinte calculée (hex).
        actual: String,
    },

    /// Le serveur a répondu autre chose que `206 Partial Content` à une
    /// requête `Range` de segment.
    #[error("download of `{url}` returned status {status} for range {start}-{end}")]
    DownloadStatus {
        /// URL demandée.
        url: String,
        /// Statut HTTP observé.
        status: u16,
        /// Début de plage demandé.
        start: u64,
        /// Fin de plage demandée (exclue).
        end: u64,
    },

    /// Un thread de segment de téléchargement est mort sur une erreur.
    #[error("download segment {segment} failed: {reason}")]
    DownloadSegment {
        /// Indice du segment.
        segment: usize,
        /// Cause lisible.
        reason: String,
    },
}

/// Alias de commodité utilisé dans tout le crate.
pub type Result<T> = std::result::Result<T, HardwareError>;
