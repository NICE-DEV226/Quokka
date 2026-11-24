# Recherche — Inférence LLM locale : qualité maximale, matériel minimal

> Synthèse d'une revue approfondie (articles, benchmarks, code llama.cpp).
> Objectif : cartographier tous les leviers connus pour servir des LLM
> localement avec la meilleure qualité par watt/Go de RAM, et évaluer
> lesquels s'appliquent à Quokka.

## 0. Le cadre mental qui ordonne tout

Le décodage autoregressif est **bound par la bande passante mémoire**,
pas par le compute :

```
tok/s_max ≈ bandwidth_mémoire (Go/s) / taille_modèle (Go)
```

Conséquences pratiques :

- **Quantifier les poids accélère le décodage quasi-proportionnellement**
  (Q4 = 2× plus rapide que Q8 à matériel égal).
- **Plus de cœurs n'aide pas** une fois la bande passante saturée.
- **Le GPU gagne sur la bande passante** (GDDR6X ~1 To/s vs DDR5 ~60 Go/s),
  pas sur les FLOPS — d'où l'offload CUDA/Vulkan.
- Le **prefill** (prompt processing) est *compute-bound* : c'est là que
  les tensor cores / AVX-512 / flash attention comptent.

Toute optimisation se range donc dans 4 cases : réduire les octets lus
par token (quant, KV cache, early-exit), lire moins de tokens (spec
decoding, cache), ne pas relire le prompt (prefix/KV reuse), ou router
vers un modèle plus petit quand c'est suffisant.

---

## 1. Leviers classés par impact/effort

### 1.1 Quantification des poids — déjà en place, à affiner

| Format | Taille vs BF16 | Qualité | Verdict |
|---|---|---|---|
| Q8_0 | ~53 % | quasi-lossless (KLD 0.0017, GSM8K = baseline) | trop lourd pour CPU-only |
| Q6_K | ~41 % | excellente | bon compromis si la RAM le permet |
| **Q4_K_M** | ~31 % | très bonne (KLD 0.034) | **défaut industriel** |
| **IQ4_XS** | ~29 % | *meilleure* que Q4_K_M sur KLD/GSM8K dans un bench 2025 | à évaluer |
| Q3_K_M | ~26 % | correcte (KLD 0.08) | limite basse acceptable |
| Q2_K | ~21 % | dégradée (KLD 0.23) | à éviter |

Faits saillants :

- **IQ4_XS bat Q4_K_M** en qualité *et* en taille dans le benchmark
  ComputingForGeeks (GSM8K 86.4 % vs 84.8 %) — les formats "importance
  matrix" de llama.cpp sont sous-estimés.
- Une étude unifiée (arXiv 2601.14277) montre que **Q5_0/Q5_1 peuvent
  égaler ou dépasser le baseline F16** sur certaines métriques — le
  quant n'est pas toujours une perte nette (variance d'évaluation).
- Le nom "N-bit" compte moins que l'implémentation : à 3 bits, Q3_K_M
  vs Q3_K_S = écart énorme. Toujours lire le suffixe _M/_S/_L.

**Application Quokka** : nos paires Flash/Expert en Q4_K_M sont le bon
défaut. Action : évaluer IQ4_XS sur notre corpus de test — si la qualité
tient, on gagne ~8 % de bande passante = ~8 % de tok/s gratuits.

### 1.2 KV cache quantifié — on a Q8, le mixte K/V est la prochaine étape

| Type | Bits/valeur | vs f16 | Qualité mesurée |
|---|---|---|---|
| f16 | 16 | 100 % | référence |
| q8_0 | 8.5 | 53 % | +0.03 % perplexité — **quasi-gratuit** |
| q4_0 | 4.5 | 28 % | +3.31 % perplexité — cher en qualité |
| **q8_0 K + q4_0 V** | ~6.5 | ~40 % | le compromissweet spot mesuré |

Nuance CPU importante (étude Graviton4, 6 modèles, 4 familles) : la
quantification KV **double le débit de prefill** mais **ralentit le
décodage** sur CPU dans la plupart des cas — le surcoût de déquant
dépasse le gain bandwidth à contexte court. À contexte long (≥8-16k) le
cache ne tient plus en cache L3 et le gain mémoire reprend.

**Application Quokka** : on est déjà en Q8 KV. Le mixte `q8_0` K +
`q4_0` V diviserait par ~1.6 l'empreinte KV — utile pour les longs
contextes. Mais mesurer l'impact decode sur notre i7-8565U avant
d'adopter : à contexte court ça peut être une régression.

### 1.3 Speculative decoding — LE levier n°1 non exploité (1.5×–3.5×)

Vérification par batch : un petit modèle *draft* propose N tokens, le
gros *vérifie* en un seul forward. Perte nulle par construction
(rejet sampling), gain = tokens acceptés / coût du draft.

| Méthode | Gain mesuré | Condition |
|---|---|---|
| Draft model (même famille) | 2–3.5× (TensorRT-LLM H200 : 3.16× sur 70B) | draft + target **même tokenizer** |
| **MTP heads** (Qwen3.6+) | 2–2.5×, acceptance ~75 % à γ=3 | heads dans le même GGUF, zéro draft externe |
| **EAGLE-3** | ~2× | draft léger lisant les hidden states |
| **N-gram / prompt-lookup** | 2–4× sur tâches input-grounded | aucun modèle requis |
| Self-speculative (`--spec-self`) | jusqu'à ~1.75× | surtout code/édition avec répétitions |

Règles mesurées :

- **Acceptance < ~50 % → perte nette**. Le draft doit être *petit et
  assez bon*, pas intelligent.
- Gain maximal sur : code, JSON, reformulation, tâches qui copient
  l'input. Minimal sur : écriture créative, haute température.
- MoE déjà rapide → gain faible. Batch élevé → gain faible (le batch
  utilise déjà le compute idle).

**Insight stratégique pour Quokka** : notre architecture Flash/Expert
ressemble à du speculative decoding — **mais nos deux modèles sont de
familles différentes (Llama 3.2 vs Qwen 2.5 Coder) donc tokenizers
incompatibles : impossible de faire draft→verify entre eux.**

Deux voies :

1. **Paire même-famille** : ex. Qwen2.5-Coder-1.5B (draft) →
   Qwen2.5-Coder-7B (expert). Le Flash servirait toujours de route
   rapide, et le draft accélérerait l'Expert 2-3×. C'est LE changement
   de registre possible : Expert passe de ~2.7 à ~6-8 tok/s sur CPU.
2. **MTP** : les Qwen3.6 intègrent des têtes MTP dans le GGUF —
   llama.cpp les exploite automatiquement (PR #22673, ~2.5× mesuré).
   Si on passe l'Expert sur un modèle MTP-compatible, gain sans
   aucun draft à gérer.

llama.cpp supporte déjà `--spec-type draft-simple|draft-eagle3|
draft-mtp|ngram-*` côté server ; nos bindings `llama-cpp-2` exposent
`LlamaContext` brut — il faudra vérifier si le chemin spéculatif est
accessible ou s'il faut passer par une API plus haute.

### 1.4 Routage Flash/Expert — on est dans la bonne lignée (RouteLLM)

RouteLLM (LMSYS, ICLR 2025) formalise exactement notre feature : router
entre weak/strong models avec un classifieur, et montre **2-3× d'économies
à qualité égale** sur MT-Bench/MMLU/Arena. Leur approche :

- **Matrix Factorization / Similarity-weighted ranking** : apprend
  P(win_strong | query) depuis des données de préférence Arena —
  nécessite un dataset de préférences qu'on n'a pas.
- **Notre approche** (lexical + centroïdes IDF + routing.toml) est la
  version zero-data : moins fine sur les cas ambigus mais <2 ms,
  déterministe, auditable. C'est le bon choix pour du local single-user.

Amélioration possible documentée : **coût de routage asymétrique** —
RouteLLM formalise un seuil α ; notre équivalent est le seuil de score
du classifieur + la garde mémoire. À outiller : logger les décisions +
feedback implicite (l'utilisateur régénère = mauvais route ?) pour
calibrer α dans le futur.

### 1.5 KV-cache et réutilisation de prompt — déjà leaders

Ce qu'on a et qui est state-of-the-art côté llama.cpp :

- **Prefix reuse** (nous) ≈ `--cache-reuse` + prompt cache de
  llama-server : matcher le plus long préfixe et ne ré-évaluer que le
  suffixe. On l'a mesuré : ~88 % de tokens économisés au 2ᵉ tour.
- **Unified KV pool + slots** (`-kvu`, `-np`) : llama-server partage un
  pool KV entre N slots avec éviction des slots idle — notre limite
  connue (1 session/worker) pourrait passer à N slots par rôle sans
  changer l'API. C'est le prochain chantier concurrency.
- **Cache idle slots** (`--cache-idle-slots`) : sauvegarder un slot au
  repos — exactement notre offload LZ4, sauf que eux le font en RAM.
- **Prompt lookup / n-gram KV** : `ngram-cache` réutilise des chunks
  n-grammes du KV existant comme draft — spec decoding *sans draft
  model*, complémentaire au §1.3.

### 1.6 Cache de réponses (sémantique) — feature serveur absente

Benchmark AWS/ElastiCache sur 64k requêtes réelles : cache sémantique →
**-73 à -86 % de coût LLM**, hit rate 67-90 %, latence ÷10 sur hit,
précision ~91 % au seuil 0.75-0.8. Exact-match ne capture que ~18 %
des doublons ; la similarité sémantique en capture ~47 % de plus.

**Application Quokka** : un cache exact (hash normalisé du prompt →
réponse) coûte ~50 lignes et zéro modèle — à ajouter d'abord. Le cache
sémantique demanderait un embedder (~90 Mo MiniLM, ou réutiliser les
features du routeur) — Phase ultérieure, attention aux faux positifs
("cancel subscription" vs "cancel order" à 0.87 similarité = réponse
fausse). Mitigation : seuil strict + clés de contexte (modèle, system
prompt) + jamais cacher les requêtes avec outils.

### 1.7 Compression de prompts — mitigée, à utiliser avec parcimonie

- **Extractive compression** (reranker/sélection de chunks) : jusqu'à
  10× compression *sans perte* sur QA multi-doc — c'est du retrieve,
  pas du prune.
- **Token pruning (LLMLingua/2)** : les études de comparaison montrent
  qu'il **sous-performe l'extractive** et détruit les needles — le
  gain prefill ne vaut pas le risque qualité en local où le prefill
  est déjà amorti par notre prefix reuse.
- **Verdict** : ne pas intégrer LLMLingua. Si un jour on fait du RAG,
  la compression extractive est la bonne technique — et c'est
  exactement ce que `/v1/embeddings` + retrieval servirait.

### 1.8 Early exit / CALM — recherche, pas pour nous

CALM (Google) et suivre : sortie anticipée par couche avec garantie
statistique de cohérence — jusqu'à 3× en théorie. **Non disponible
dans llama.cpp/GGUF**, nécessite des têtes de confiance par couche.
Maturité : papier + Google interne. À surveiller, pas à implémenter.

### 1.9 MoE — la contre-intuition importante

Sur hardware bandwidth-bound (edge/CPU), **le coût d'inférence suit les
paramètres TOTAUX, pas les actifs** (étude arXiv 2606.21428 : OLMoE-1B-7B
~10-31 % *plus lent* que Llama-3.2-1B dense à qualité égale ; le "qs
inequality" montre jusqu'à 4.5× de désavantage vs dense qualité-équivalent
à long contexte). Le routing MoE fragmente le batch et réduit le reuse
des poids en HBM.

**Verdict** : notre paire dense Flash/Expert est le bon pattern pour
le local. Les gros MoE (Qwen3-30B-A3B, gpt-oss) sont intéressants
uniquement si la RAM totale permet de les charger — et là ils restent
handicapés par le total de paramètres. Ne pas router *vers* du MoE en
croyant que "actif = rapide" : sur CPU, actif ≠ rapide.

### 1.10 Choix des modèles — le levier le plus simple

Benchmarks SLM 2025 (MMLU/HumanEval/GSM8K, Q4_K_M) :

| Modèle | Taille | Points forts |
|---|---|---|
| **Phi-3 mini (3.8B)** | 2.2 Go | Meilleur rapport qualité/taille mesuré (MMLU 67.5) |
| Qwen2.5-1.5B | 0.9 Go | surprisingly bon (MMLU 55.8) — excellent draft |
| Llama-3.2-3B | 1.9 Go | correct, pas spectacular |
| Qwen2.5-Coder-7B | 4.4 Go | specialist code — notre Expert actuel, justifié |

Considérations :

- **Même famille pour speculation** : Qwen2.5-Coder-1.5B → 7B ou
  Llama-3.2-1B → 3B débloquent le §1.3.
- **MTP-compatible** : Qwen3.6 emporte des têtes MTP dans le GGUF —
  le prochain Expert "gratuit" en vitesse.
- Les modèles "assistant" (petit modèle qui reformule le prompt pour
  le gros) sont un pattern RouteLLM adjacent — notre routeur joue déjà
  ce rôle en choisissant le modèle, pas besoin de reformulation.

### 1.11 Efficacité énergétique

- CPU vs GPU sur edge : le GPU gagne en latence mais le CPU gagne en
  tok/J sur petits modèles — notre position CPU-only est défendable
  écologiquement à petite échelle.
- **Sub-1B = meilleur tok/J** ; 7-8B = ~4× plus cher en énergie/token.
- Conséquence design : la frontière Flash/Expert devrait être
  *énergétique* autant que qualitative — router vers Flash économise
  ~2-4× l'énergie par token. Notre `MemoryPressureFallback` va dans ce
  sens ; un futur `EcoMode` (déjà esquissé dans routing.toml) pourrait
  baisser la qualité *volontairement* pour l'énergie.

### 1.12 NUMA / multi-socket — niche, pas pour notre cible

Sur dual-socketXeon, le split NUMA peut doubler la bande passante
utile mais la config est experte (`--numa distribute`, interleave,
drop_caches+mmap). **Notre cible = consumer single-socket** : hors
scope. À documenter comme non-objectif.

---

## 2. Ce que Quokka fait déjà (et fait bien)

| Technique | État | benchmark interne |
|---|---|---|
| mmap poids (zéro copie) | ✅ | dirty RAM 4.4 Gio → 55 Mio |
| GGML_CPU_REPACK=OFF | ✅ | poids en pages récupérables |
| Flash attention | ✅ | activé |
| KV Q8 | ✅ | +0.03 % ppl mesuré ailleurs |
| Prefix reuse intra-session | ✅ | ~88 % tokens économisés tour 2 |
| KV offload LZ4 disque | ✅ | survit aux redémarrages, LRU 8 |
| Routeur <2 ms | ✅ | ~µs mesurés, corpus 137 @100 % |
| Garde mémoire → Flash | ✅ | MemoryPressureFallback |

## 3. Recommandations priorisées

| # | Action | Gain estimé | Effort | Statut |
|---|---|---|---|---|
| P0 | **Vérifier l'accès au speculative decoding** dans llama-cpp-2 (draft-simple / MTP / ngram) | 1.5-3× decode | M | à investiguer |
| P0 | **Paire même-famille** Flash/Expert (ex. Qwen2.5-Coder-1.5B→7B) pour débloquer draft→verify | 2-3× sur Expert | S (config) | décision produit |
| P1 | **Multi-slots KV par rôle** (équivalent -np/-kvu) | concurrency réelle | M | roadmap |
| P1 | **Cache réponses exact-match** (hash prompt normalisé) | -coût sur requêtes répétées | S | facile |
| P1 | Évaluer **IQ4_XS vs Q4_K_M** sur notre corpus | ~8 % tok/s si égal | S | bench |
| P2 | KV mixte **q8_0 K + q4_0 V** si contexte long | -40 % KV RAM | S | mesurer decode d'abord |
| P2 | Expert **MTP-compatible** (Qwen3.6) à la prochaine itération modèles | ~2× gratuit | S | attendre modèle |
| P2 | Cache sémantique (embedder léger) | -50-80 % sur requêtes paraphrasées | M-L | phase ultérieure |
| P3 | `EcoMode` : routing plus agressif vers Flash (énergie) | -2-4× énergie/requête | S | routing.toml existe |
| P3 | N-gram/self-speculative pour code & édition | 1.5-2× sur workloads input-grounded | M | suit P0 |

## 4. Explicitement rejeté

- **Moteur pur Rust from scratch** (kora-engine) : llama.cpp est
  5-10× plus optimisé sur decode ; voir discussion ADR à venir si besoin.
- **LLMLingua / token pruning** : sous-performe l'extractive, détruit
  les needles, et notre prefix reuse amortit déjà le prefill.
- **MoE "actif = rapide"** : sur CPU/edge le coût suit les paramètres
  totaux — nos denses sont le bon choix.
- **Early exit (CALM)** : pas dans llama.cpp, recherche-only.
- **NUMA multi-socket** : hors cible consumer.

## 5. Sources principales

- llama.cpp docs/speculative.md, PR #22673 (MTP), PR #18471 (self-spec),
  discussion #13606 (KV reuse), server README (slots/unified KV)
- RouteLLM — arXiv:2406.18665, LMSYS blog (ICLR 2025)
- Speculative decoding — arXiv:2211.17192 ; aliteq.com ; gpuwerk.com
- KV quant — modelvram.com ; StephenSook/gravitonkv ; anbeeld.com
- Quant formats — computingforgeeks.com GGUF vs AWQ/GPTQ/FP8 ;
  arXiv:2601.14277 (unified eval Q4_K_M vs Q5)
- MoE edge — arXiv:2606.21428, arXiv:2603.08960 (qs inequality)
- Prompt compression — arXiv:2310.06839 (LongLLMLingua),
  arXiv:2407.08892 (comparative study)
- Semantic cache — AWS ElastiCache benchmarks, VentureBeat, arXiv:2411.05276
- SLM quality — neelmishra.github.io benchmarks, ACL 2025 SLM-Bench,
  arXiv edge-deployment studies
- Early exit — CALM (Schuster et al. 2022), arXiv:2407.20272
