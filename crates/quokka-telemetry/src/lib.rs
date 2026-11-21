//! Télémétrie Quokka : compteurs d'usage agrégés en mémoire.
//!
//! Enregistré une fois par complétion terminée (événement `Done`), avec
//! le modèle réellement utilisé et la raison de routage. Persisté
//! nulle part — un redémarrage remet les compteurs à zéro (volontaire :
//! la télémétrie décrit la session courante, pas un historique).
//!
//! Les métriques « RAM économisée » ne sont pas mesurées directement :
//! ce qui est réellement observé, c'est `cached_prompt_tokens` — les
//! tokens de préfill **évités** par la réutilisation du KV-cache. C'est
//! le proxy honnête du calcul économisé.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Compteurs agrégés sur une fenêtre (process ou par modèle).
#[derive(Debug, Default, Clone, Serialize)]
pub struct UsageStats {
    /// Complétions terminées.
    pub completions: u64,
    /// Tokens de prompt traités (cached inclus).
    pub prompt_tokens: u64,
    /// Tokens de prompt servis par le KV-cache (préfill évité).
    pub cached_prompt_tokens: u64,
    /// Tokens générés.
    pub generated_tokens: u64,
    /// Temps d'inférence cumulé en millisecondes.
    pub elapsed_ms: f64,
}

impl UsageStats {
    fn add(&mut self, prompt: u64, cached: u64, generated: u64, elapsed_ms: f64) {
        self.completions += 1;
        self.prompt_tokens += prompt;
        self.cached_prompt_tokens += cached;
        self.generated_tokens += generated;
        self.elapsed_ms += elapsed_ms;
    }
}

/// Instantané sérialisable exposé par `GET /telemetry`.
#[derive(Debug, Serialize)]
pub struct TelemetrySnapshot {
    /// Secondes depuis le démarrage du processus.
    pub uptime_secs: u64,
    /// Totaux toutes modèles confondus.
    pub totals: UsageStats,
    /// `cached_prompt_tokens / prompt_tokens` — part du préfill évitée.
    pub cache_hit_ratio: f64,
    /// Détail par modèle servi (`flash`, `expert`, …).
    pub by_model: BTreeMap<String, UsageStats>,
    /// Histogramme des raisons de routage (`ShortPrompt`, `Semantic`, …).
    pub routes: BTreeMap<String, u64>,
    /// Histogramme des cibles de routage (`flash`, `expert`).
    pub route_targets: BTreeMap<String, u64>,
}

#[derive(Debug, Default)]
struct Inner {
    totals: UsageStats,
    by_model: BTreeMap<String, UsageStats>,
    routes: BTreeMap<String, u64>,
    route_targets: BTreeMap<String, u64>,
}

/// Registre de télémétrie — `Clone` à faible coût (partage l'état).
#[derive(Clone, Debug)]
pub struct Telemetry {
    inner: std::sync::Arc<Mutex<Inner>>,
    started: Instant,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl Telemetry {
    /// Crée un registre vide.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(Mutex::new(Inner::default())),
            started: Instant::now(),
        }
    }

    /// Enregistre une complétion terminée.
    pub fn record_completion(
        &self,
        model: &str,
        reason: &str,
        prompt_tokens: u64,
        cached_prompt_tokens: u64,
        generated_tokens: u64,
        elapsed_ms: f64,
    ) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        inner.totals.add(
            prompt_tokens,
            cached_prompt_tokens,
            generated_tokens,
            elapsed_ms,
        );
        inner.by_model.entry(model.to_owned()).or_default().add(
            prompt_tokens,
            cached_prompt_tokens,
            generated_tokens,
            elapsed_ms,
        );
        *inner.routes.entry(reason.to_owned()).or_insert(0) += 1;
        *inner.route_targets.entry(model.to_owned()).or_insert(0) += 1;
    }

    /// Instantané courant, sérialisable en JSON.
    #[must_use]
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let inner = self.inner.lock().map(|i| TelemetrySnapshot {
            uptime_secs: self.started.elapsed().as_secs(),
            totals: i.totals.clone(),
            cache_hit_ratio: if i.totals.prompt_tokens > 0 {
                i.totals.cached_prompt_tokens as f64 / i.totals.prompt_tokens as f64
            } else {
                0.0
            },
            by_model: i.by_model.clone(),
            routes: i.routes.clone(),
            route_targets: i.route_targets.clone(),
        });
        inner.unwrap_or_else(|_| TelemetrySnapshot {
            uptime_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            totals: UsageStats::default(),
            cache_hit_ratio: 0.0,
            by_model: BTreeMap::new(),
            routes: BTreeMap::new(),
            route_targets: BTreeMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_aggregates() {
        let t = Telemetry::new();
        t.record_completion("flash", "ShortPrompt", 20, 0, 10, 100.0);
        t.record_completion("flash", "Semantic", 60, 40, 12, 200.0);
        t.record_completion("expert", "DomainDetected", 80, 0, 30, 900.0);

        let snap = t.snapshot();
        assert_eq!(snap.totals.completions, 3);
        assert_eq!(snap.totals.prompt_tokens, 160);
        assert_eq!(snap.totals.cached_prompt_tokens, 40);
        assert_eq!(snap.totals.generated_tokens, 52);
        assert!((snap.cache_hit_ratio - 0.25).abs() < 1e-9);
        assert_eq!(snap.by_model["flash"].completions, 2);
        assert_eq!(snap.by_model["expert"].generated_tokens, 30);
        assert_eq!(snap.routes["ShortPrompt"], 1);
        assert_eq!(snap.route_targets["expert"], 1);
        let json = serde_json::to_string(&snap).expect("serialize");
        assert!(json.contains("\"flash\""));
    }

    #[test]
    fn empty_snapshot_is_zeroed() {
        let snap = Telemetry::new().snapshot();
        assert_eq!(snap.totals.completions, 0);
        assert_eq!(snap.cache_hit_ratio, 0.0);
    }
}
