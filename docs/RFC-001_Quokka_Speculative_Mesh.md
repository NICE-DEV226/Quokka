# RFC-001 : Architecture Hybride *Speculative Pipeline Parallelism* pour Quokka (`quokka-core` + `quokka-mesh`)

**Statut :** Spécification Technique Complète (Prête pour implémentation)  
**Cible :** Écosystème Quokka (100 % Rust — `quokka-core`, `quokka-router`, `quokka-hardware`, `quokka-mesh`)  
**Objectif :** Exécution distribuée de grands modèles de langage (32B–70B+) sur réseau local standard (Wi-Fi / Ethernet Gigabit) en combinant un modèle *Draft* local avec un modèle *Target* découpé en tranches (*shards*) sur le LAN.

---

## 1. Contexte, Diagnostic et Principes Directeurs

### 1.1 Pourquoi cette architecture et pas une autre ?
Une analyse lucide de l'état de l'art (`llama.cpp` / `ggml`) impose de ne **pas** réinventer ce que `llama.cpp` fait déjà de manière optimale sur une seule machine :
- **Pas de surcouche Go/Python/IPC local :** Quokka dispose déjà de `tokio` et `axum` en un seul binaire Rust. Ajouter une frontière de processus locale et un *ring buffer* partagé résoudrait une latence IPC de `~1 µs` alors qu'un token prend `~50–100 ms` à décoder.
- **Pas de réécriture de noyaux SIMD/WGSL :** Les kernels quantifiés de `ggml` (`Q4_K`, `Q8_0`, Vulkan, AVX2, NEON) sont à l'état de l'art.
- **Pas de streaming SSD custom :** Le mapping mémoire `mmap` de `llama.cpp` exploite déjà le *page cache* du noyau OS.

### 1.2 Le vrai verrou physique à faire sauter
Sur un réseau local ordinaire (Wi-Fi ou Ethernet 1 Gbps sans NVLink/Thunderbolt), le **Pipeline Parallelism** classique échoue sur les requêtes uniques car chaque token généré doit traverser tous les nœuds séquentiellement (latence réseau payée à chaque mot).

**La solution architecturale :** **La Spéculation Asymétrique Distribuée (*Distributed Speculative Decoding*)**.
1. Le **Nœud Maître** (machine cliente, même modeste avec 8 Go de RAM) héberge un petit modèle *Draft* local (ex: `Qwen2.5-0.5B`) qui génère un micro-lot de `K` tokens candidats en local en quelques millisecondes.
2. Ce micro-lot de `K` tokens est envoyé **en un seul paquet** à travers le pipeline réseau (`quokka-mesh`) qui héberge le gros modèle *Target* (32B–70B) découpé par couches sur plusieurs machines du LAN.
3. Les Workers exécutent la passe avant sur les `K` tokens en parallèle, divisant le coût de latence réseau par le taux d'acceptation (typiquement **3x à 5x moins d'allers-retours réseau**).

### 1.3 Invariants non-négociables
1. **Un seul binaire Rust :** Aucune dépendance externe lourde (pas de Python, Docker ou broker externe).
2. **Cohérence mathématique stricte du KV-Cache :** Tout token spéculé puis rejeté doit être purgé de manière déterministe sur l'ensemble des nœuds avant le tour suivant.
3. **Zero-Config LAN :** Découverte automatique des nœuds via mDNS et répartition dynamique des couches selon les sondes `quokka-hardware`.

---

## 2. Vue d'Ensemble de l'Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│ NŒUD MAÎTRE (Initiateur / Client — ex: PC Portable 8 Go RAM)               │
│                                                                            │
│  [quokka-router] ──► Gestion session, RTT réseau, adaptation dynamique K   │
│         │                                                                  │
│         ├──► [Draft Model Local (0.5B - 1.5B)]                             │
│         │     Génère K tokens candidats [t_1..t_K] + probs P_draft         │
│         │                                                                  │
│         └──► [Target Shard 0 : token_embd + Couches 0 .. L_1]              │
│               Calcule les états cachés initiaux H_0 de forme [K+1, dim]    │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │ QUIC Stream (quinn)
                                      │ ActivationFrame (Quantifié Q8_0)
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ NŒUD WORKER 1 (Relais — ex: PC Bureau 16 Go RAM/VRAM)                      │
│                                                                            │
│  [quokka-mesh Worker] ──► Applique kv_truncate_to -> Déquantifie Q8_0      │
│         │                                                                  │
│         └──► [Target Shard 1 : Couches L_1+1 .. L_2]                       │
│               Passe avant parallèle sur K+1 positions -> H_1 [K+1, dim]    │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │ QUIC Stream (quinn)
                                      │ ActivationFrame (Quantifié Q8_0)
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ NŒUD WORKER FINAL (Head — ex: Workstation / GPU)                           │
│                                                                            │
│  [quokka-mesh Worker] ──► Applique kv_truncate_to -> Déquantifie Q8_0      │
│         │                                                                  │
│         └──► [Target Shard Final : Couches L_2+1 .. L_max + norm + output] │
│               Calcule les logits pour les K+1 positions                    │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │ QUIC Stream Retour Direct vers Maître
                                      │ LogitsFrame (Top-P / Distributions)
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ VALIDATION STOCHASTIQUE SUR LE NŒUD MAÎTRE                                 │
│  • Compare P_target(t_i) vs P_draft(t_i) pour i = 1..K                    │
│  • Accepte M tokens valides + échantillonne 1 token de correction t'_r     │
│  • Met à jour n_past_new = n_past + M + 1                                  │
│  • Programme kv_truncate_to = n_past_new pour le prochain tour             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chantier A : Spéculation Locale `Draft → Target` (`quokka-core`)

### 3.1 Contrainte de compatibilité des modèles
Le modèle *Draft* et le modèle *Target* **doivent** appartenir à la même famille et partager exactement le même vocabulaire et le même tokenizer :
- `Qwen2.5-0.5B-Instruct` (Draft) → `Qwen2.5-7B / 14B / 32B / 72B` (Target)
- `Llama-3.2-1B-Instruct` (Draft) → `Llama-3.1-8B / Llama-3.3-70B` (Target)
- `DeepSeek-R1-Distill-Qwen-1.5B` (Draft) → `DeepSeek-R1-Distill-Qwen-32B` (Target)

### 3.2 Structures de données Rust (`quokka-core/src/speculative.rs`)

```rust
use std::collections::HashMap;

/// Représente un token candidat proposé par le modèle Draft
#[derive(Debug, Clone)]
pub struct DraftToken {
    pub token_id: i32,
    /// Probabilité attribuée par le Draft à ce token : P_draft(token_id)
    pub prob: f32,
    /// Distribution complète (ou top-k) utile pour le calcul résiduel en cas de rejet
    pub distribution: Vec<(i32, f32)>,
}

/// Résultat d'un cycle complet de spéculation + vérification
#[derive(Debug, Clone)]
pub struct SpeculationStepResult {
    /// Tokens définitivement validés lors de ce tour (entre 1 et K+1 tokens)
    pub accepted_tokens: Vec<i32>,
    /// Nombre de tokens spéculés acceptés (M) avant le premier rejet
    pub draft_accepted_count: usize,
    /// Nouvelle position exacte dans le KV-Cache après validation
    pub new_n_past: i32,
}

/// Contrôleur adaptatif de la fenêtre de spéculation K
pub struct AdaptiveDraftController {
    /// Moyenne mobile exponentielle du taux d'acceptation (0.0 à 1.0)
    pub ema_acceptance: f32,
    /// Facteur de lissage EMA (ex: 0.2)
    pub alpha: f32,
    /// Longueur courante de spéculation K
    pub current_k: usize,
    /// Bornes min et max selon le mode (Local vs Cluster LAN)
    pub min_k: usize,
    pub max_k: usize,
}

impl AdaptiveDraftController {
    pub fn new(is_distributed: bool) -> Self {
        if is_distributed {
            Self {
                ema_acceptance: 0.7,
                alpha: 0.2,
                current_k: 6,
                min_k: 3,
                max_k: 12,
            }
        } else {
            Self {
                ema_acceptance: 0.7,
                alpha: 0.2,
                current_k: 5,
                min_k: 1,
                max_k: 8,
            }
        }
    }

    /// Met à jour K après chaque tour de vérification
    pub fn update(&mut self, proposed_k: usize, accepted_m: usize, network_rtt_ms: f32) {
        if proposed_k == 0 {
            return;
        }
        let step_rate = accepted_m as f32 / proposed_k as f32;
        self.ema_acceptance = self.alpha * step_rate + (1.0 - self.alpha) * self.ema_acceptance;

        // Si le réseau a de la latence (> 2ms), on favorise un K plus grand tant que l'acceptation reste correcte
        let latency_bonus = if network_rtt_ms > 2.0 && self.ema_acceptance > 0.45 {
            2.0
        } else {
            0.0
        };

        let target_k = (1.0 + (self.max_k as f32 - 1.0) * self.ema_acceptance + latency_bonus).round() as usize;
        self.current_k = target_k.clamp(self.min_k, self.max_k);
    }
}
```

### 3.3 Algorithme de Validation Stochastique (Leviathan et al.)
Cet algorithme garantit mathématiquement que la distribution de sortie du couple `Draft + Target` est **strictement identique** à celle du modèle `Target` seul (aucune dégradation de qualité ni d'hallucination supplémentaire).

```rust
use rand::Rng;

/// Valide les K tokens spéculés contre les K+1 distributions du modèle Target
pub fn verify_speculative_tokens(
    draft_tokens: &[DraftToken],
    target_probs: &[Vec<f32>], // Taille K + 1, chaque vecteur a la taille du vocabulaire
    rng: &mut impl Rng,
) -> (Vec<i32>, usize) {
    let k = draft_tokens.len();
    assert_eq!(target_probs.len(), k + 1);

    let mut final_tokens = Vec::with_capacity(k + 1);
    let mut accepted_m = 0;

    for i in 0..k {
        let candidate = &draft_tokens[i];
        let p_target = target_probs[i][candidate.token_id as usize];
        let p_draft = candidate.prob.max(1e-9);

        let ratio = (p_target / p_draft).min(1.0);
        let r: f32 = rng.gen::<f32>();

        if r <= ratio {
            // Token accepté
            final_tokens.push(candidate.token_id);
            accepted_m += 1;
        } else {
            // Rejet à l'indice i : échantillonnage dans la distribution résiduelle norm(max(0, P_target - P_draft))
            let correction_token = sample_residual_distribution(
                &target_probs[i],
                &candidate.distribution,
                rng,
            );
            final_tokens.push(correction_token);
            return (final_tokens, accepted_m);
        }
    }

    // Si tous les K tokens ont été acceptés, on échantillonne le (K+1)-ième token directement depuis target_probs[k]
    let bonus_token = sample_from_distribution(&target_probs[k], rng);
    final_tokens.push(bonus_token);

    (final_tokens, accepted_m)
}

fn sample_residual_distribution(
    p_target: &[f32],
    p_draft_sparse: &[(i32, f32)],
    rng: &mut impl Rng,
) -> i32 {
    let mut residual = p_target.to_vec();
    for &(tok_id, prob) in p_draft_sparse {
        if let Some(val) = residual.get_mut(tok_id as usize) {
            *val = (*val - prob).max(0.0);
        }
    }

    let sum: f32 = residual.iter().sum();
    if sum <= 1e-9 {
        // Fallback sécurisé sur la distribution Target pure
        return sample_from_distribution(p_target, rng);
    }

    let mut threshold = rng.gen::<f32>() * sum;
    for (idx, &val) in residual.iter().enumerate() {
        threshold -= val;
        if threshold <= 0.0 {
            return idx as i32;
        }
    }
    (residual.len() - 1) as i32
}

fn sample_from_distribution(probs: &[f32], rng: &mut impl Rng) -> i32 {
    let mut threshold = rng.gen::<f32>();
    for (idx, &val) in probs.iter().enumerate() {
        threshold -= val;
        if threshold <= 0.0 {
            return idx as i32;
        }
    }
    (probs.len() - 1) as i32
}
```

### 3.4 Gestion stricte du KV-Cache (`llama.cpp` FFI)
À la fin de chaque cycle :
1. Soit `n_past` la position avant spéculation, et `M` le nombre de tokens acceptés (`final_tokens.len() == M + 1`).
2. `new_n_past = n_past + (M as i32) + 1`.
3. **Purge du Draft :** Le Draft avait avancé jusqu'à `n_past + K`. On tronque tout ce qui dépasse `new_n_past - 1` (le dernier token validé `final_tokens.last()` sera réinjecté comme amorce au tour suivant) :
   ```c
   llama_kv_cache_seq_rm(ctx_draft, 0, new_n_past - 1, -1);
   ```
4. **Purge du Target :** Le Target avait évalué `K + 1` tokens (`n_past` jusqu'à `n_past + K`). On purge les positions rejetées à partir de `new_n_past` :
   ```c
   llama_kv_cache_seq_rm(ctx_target, 0, new_n_past, -1);
   ```

---

## 4. Chantier B : Exécution par Tranches (`Sharding` GGUF)

### 4.1 Découpage d'un modèle Transformer dans `quokka-core`
Un modèle GGUF standard est composé séquentiellement de :
1. `token_embd.weight` (Embedding d'entrée : `Token ID -> Vecteur [hidden_dim]`)
2. `blk.0` à `blk.(N-1)` (Les `N` couches Transformer : Attention + KV-Cache + FFN)
3. `output_norm.weight` + `output.weight` (Normalisation finale + Projection vocabulaire : `Vecteur [hidden_dim] -> Logits [vocab_size]`)

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelShardSpec {
    pub model_id: String,
    pub layer_start: u32,     // Incluse (ex: 0)
    pub layer_end: u32,       // Exclue (ex: 16 -> couches 0..15)
    pub has_embedding: bool,  // Vrai uniquement pour le Nœud Maître
    pub has_lm_head: bool,    // Vrai uniquement pour le Worker Final
    pub hidden_dim: u32,      // Ex: 4096 (8B), 5120 (14B), 8192 (70B)
    pub vocab_size: u32,
}
```

### 4.2 Mécanisme d'injection/extraction dans `llama.cpp` / `ggml`
Pour exécuter uniquement la plage `[layer_start, layer_end)` sans charger les autres couches en RAM :
- **Filtrage au chargement :** Lors de l'ouverture du fichier GGUF (`gguf_init_from_file`), seuls les tenseurs dont le nom correspond à `blk.{i}.*` avec `layer_start <= i < layer_end` (plus `token_embd` si `has_embedding` et `output*` si `has_lm_head`) sont alloués et mappés sur le backend (CPU/Vulkan/CUDA).
- **Entrée du graphe de calcul (`ggml_cgraph`) :**
  - Si `has_embedding == true` : l'entrée est un tenseur d'entiers `tokens [K+1]`.
  - Si `has_embedding == false` : l'entrée est un tenseur flottant `hidden_in [hidden_dim, K+1]` reçu du réseau, copié directement dans le buffer d'entrée du graphe `ggml`.
- **Sortie du graphe de calcul :**
  - Si `has_lm_head == false` : le graphe s'arrête après la couche `layer_end - 1` et retourne le tenseur `hidden_out [hidden_dim, K+1]`.
  - Si `has_lm_head == true` : le graphe exécute `output_norm` et `output` et retourne les `logits [vocab_size, K+1]`.

---

## 5. Chantier C : Protocole Réseau `quokka-mesh` (QUIC + Compression `Q8_0`)

### 5.1 Découverte automatique LAN (`mDNS`)
Chaque instance Quokka exposant sa capacité de calcul publie un enregistrement mDNS `_quokka-mesh._udp.local` enrichi par `quokka-hardware` :

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeCapability {
    pub node_id: String,
    pub addr: std::net::SocketAddr,
    pub available_vram_mb: u64,
    pub available_ram_mb: u64,
    pub compute_score_gflops: f32,
    pub backend: ComputeBackend, // Vulkan, Cuda, Metal, CpuAvx2, CpuNeon
    pub active_sessions: u32,
}
```

### 5.2 Algorithme de partitionnement (`ClusterPlanner`)
Lorsqu'un modèle de taille totale `total_model_mb` et de `total_layers` couches doit être chargé :
1. **Score de capacité mémoire pondéré :** Pour chaque nœud `i`, on calcule la mémoire utilisable avec une marge de sécurité de 20 % :
   $$M_i = 0.8 	imes (	ext{VRAM}_i 	imes 1.0 + 	ext{RAM}_i 	imes 0.85)$$
2. **Poids de vitesse (priorité GPU) :** Si le nœud dispose d'un GPU dédié (`Vulkan`/`CUDA`/`Metal`) avec assez de VRAM, son score d'allocation est multiplié par `2.5` pour concentrer un maximum de couches sur le silicium le plus rapide.
3. **Attribution des couches :**
   $$	ext{Couches}_i = \max\left(1, \left\lfloor 	ext{total\_layers} 	imes rac{	ext{Score}_i}{\sum_j 	ext{Score}_j} ightflooright)$$
   *(Ajustement du reste sur le nœud ayant le plus grand score pour que $\sum 	ext{Couches}_i = 	ext{total\_layers}$).*

### 5.3 Format binaire des trames (`Wire Protocol`)
Communication sur connexions persistantes **QUIC (`quinn`)** avec contrôle de congestion BBR et chiffrement TLS 1.3 éphémère (certificats auto-signés épinglés lors du handshake mDNS).

```rust
pub const FRAME_MAGIC: [u8; 4] = *b"QKSH";

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorQuantType {
    F16 = 0,
    Q8_0 = 1, // Bloc de 32 f32 -> 1 scale f16 (2 octets) + 32 i8 (32 octets) = 34 octets
}

pub mod frame_flags {
    pub const RESET_SESSION: u8 = 1 << 0; // Réinitialise complètement le KV-Cache de la session
    pub const FINAL_NODE: u8    = 1 << 1; // Indique au worker qu'il doit calculer les logits
    pub const PING_PROBE: u8    = 1 << 2; // Mesure de latence RTT pure sans inférence
}

/// En-tête fixe de 24 octets (Little Endian)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct ActivationFrameHeader {
    pub magic: [u8; 4],        // "QKSH"
    pub session_id: u64,       // ID unique de la session d'inférence
    pub seq_step: u32,         // Compteur monotone de tour (rejette les paquets obsolètes)
    pub kv_truncate_to: u32,   // Position exacte où tronquer le KV-Cache avant forward()
    pub token_count: u16,      // Nombre de tokens dans ce micro-lot (K + 1)
    pub hidden_dim: u16,       // Dimension cachée (ex: 4096, 8192)
    pub quant_type: u8,        // TensorQuantType
    pub flags: u8,             // Bitmask frame_flags
    pub reserved: [u8; 2],     // Alignement 24 octets
}
```

### 5.4 Quantification réseau `Q8_0` à la volée
Pour éviter de saturer le Wi-Fi lors du transfert des tenseurs d'activation `[token_count, hidden_dim]` :
- Chaque vecteur de `hidden_dim` éléments est découpé en blocs de `32` flottants (`f32`).
- Pour chaque bloc `B[0..32]` :
  1. `amax = max(|B[j]|)` pour `j in 0..32`
  2. `scale = amax / 127.0` (stocké en `half::f16` sur 2 octets)
  3. `inv_scale = if amax != 0.0 { 127.0 / amax } else { 0.0 }`
  4. `q[j] = round(B[j] * inv_scale) as i8` (stocké sur 1 octet)
- **Gain réseau :** Un bloc de 32 `f32` (128 octets) est compressé en **34 octets** (facteur **3.76x**). Pour un modèle 70B (`hidden_dim = 8192`) avec `K + 1 = 7` tokens, la trame complète pèse **60.9 Ko** au lieu de 229.4 Ko.

### 5.5 Trame de retour (`LogitsFrame`) du Worker Final vers le Maître
Le Worker Final n'a pas besoin de renvoyer tout le vocabulaire (`151 936` floats pour Qwen2.5 = 600 Ko par token).
- **Optimisation bande passante :** Le Worker Final applique le `softmax` (avec température/top-p) localement, et renvoie uniquement :
  1. Les probabilités exactes `P_target(t_i)` pour les `K` tokens candidats envoyés par le Maître (`K * 4` octets).
  2. Le **Top-64** des tokens les plus probables (ID `i32` + prob `f32`) pour chacune des `K + 1` positions afin de permettre au Maître d'échantillonner le token résiduel ou le token bonus (`(K + 1) * 64 * 8` octets ≈ **3.5 Ko**).

---

## 6. Tolérance aux Pannes et Gestion des Anomalies Réseau

| Scénario d'incident | Détection | Réaction automatique de `quokka-mesh` |
| :--- | :--- | :--- |
| **Perte / Retard d'un paquet QUIC** | Timeout dynamique (`3 * EMA_RTT`) ou `seq_step` incohérent | Le Maître réémet le lot avec `kv_truncate_to = n_past`. Comme `kv_truncate_to` est idempotent, le Worker nettoie son cache et recalcule proprement. |
| **Déconnexion brutale d'un Worker** | Fermeture de stream QUIC ou perte heartbeat mDNS (`> 1.5s`) | Le `ClusterPlanner` recalcule la partition sur les nœuds survivants, recharge les couches manquantes depuis le GGUF local/partagé et reprend la session à `n_past` sans perdre la conversation. |
| **Chute brutale du débit Wi-Fi** | Augmentation du `RTT` mesuré par `AdaptiveDraftController` | Augmentation automatique de `K` (jusqu'à 12) si l'acceptation est bonne, ou bascule automatique sur un modèle local plus petit si le RTT dépasse le seuil critique (ex: `> 150 ms`). |

---

## 7. Feuille de Route d'Implémentation (4 Jalons)

### Jalon 1 : Spéculation Locale `Draft → Target` (`quokka-core`)
- **Livrables :** Implémentation de `AdaptiveDraftController`, `verify_speculative_tokens` et `SpeculativeSession` dans `quokka-core`.
- **Critère de succès :** Faire tourner `Qwen2.5-0.5B` + `Qwen2.5-7B` sur une seule machine avec cohérence bit-à-bit du KV-Cache (`llama_kv_cache_seq_rm`) et mesurer un gain de vitesse **>= 2.0x** sur des tâches de génération de code/JSON.

### Jalon 2 : Sharding GGUF & Graphe Partiel (`quokka-core`)
- **Livrables :** Support de `ModelShardSpec` permettant d'instancier un worker ne chargeant qu'une plage de couches `[layer_start, layer_end)` et exposant `forward_activations(hidden_in, kv_truncate_to) -> hidden_out`.
- **Critère de succès :** Découper un modèle 7B en 3 instances locales en mémoire et vérifier que les logits finaux sont strictement identiques au modèle monolithique.

### Jalon 3 : Transport QUIC & Quantification `Q8_0` (`quokka-mesh`)
- **Livrables :** Nouveau crate `quokka-mesh` intégrant `quinn`, la sérialisation `ActivationFrameHeader`, la compression SIMD `Q8_0` des activations et le retour compact `LogitsFrame`.
- **Critère de succès :** Exécuter un modèle 32B réparti sur 2 machines via Ethernet/Wi-Fi avec une latence de sérialisation + quantification `< 0.5 ms`.

### Jalon 4 : Découverte mDNS & Auto-Partitionnement (`quokka-router` + `quokka-hardware`)
- **Livrables :** Découverte zéro-config via `mdns-sd`, intégration des métriques `quokka-hardware` dans `ClusterPlanner` et bascule résiliente en cas de perte d'un nœud.
- **Critère de succès :** Lancer `quokka` sur 3 machines du même réseau local sans aucun fichier de configuration IP et exécuter un modèle 70B de manière fluide.
