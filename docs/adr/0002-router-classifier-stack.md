# 0002 — Classifieur du routeur : heuristique + ONNX INT8, jamais de LLM

- Statut : Accepted
- Date : 2026-11-19

## Contexte

Chaque prompt doit être routé Flash/Expert en **< 2,0 ms** de surcharge
(§4.2). Le routeur tourne avant l'inférence : il ne peut ni charger un
modèle lourd, ni allouer, ni faire d'I/O. La classification doit rester
déterministe et testable sur CPU seul.

## Décision

Pipeline en deux étages, les deux sans GPU ni réseau :

1. **Stage 1 — porte lexicale/structurelle (< 0,1 ms)** : fences de code,
   mots-clés de programmation, overrides `@flash`/`@expert`, seuil de
   tokens. Court-circuite la majorité des prompts triviaux.
2. **Stage 2 — similarité cosinus sur embeddings** : `all-MiniLM-L6-v2`
   quantisé INT8 via `ort` (ONNX Runtime), 384 dims, 2 threads CPU,
   comparé à des centroïdes d'ancres pré-calculés (`coding`,
   `math_logic`, `multi_step_analysis` vs `translation`, `summarization`,
   `chit_chat`).

Le tout gardé par le **Memory Pressure Guard** : si
`available_ram_mb < expert_size + 1500 MiB`, repli Flash + événement
`MemoryPressureFallback`.

## Conséquences

### Positives
- Latence bornée et mesurable (fixture criterion `router_bench`, budget
  2 ms vérifié en CI).
- Zéro dépendance au modèle expert pour router — pas de bootstrap circulaire.
- `ort` est déjà prévu au stack (§3) et couvre d'autres usages (embeddings
  `/v1/embeddings`).

### Négatives / risques
- Le premier appel ONNX paie le coût de création de session ; la session
  est initialisée au démarrage, jamais à la requête.
- Les centroïdes doivent être calibrés sur un corpus représentatif —
  risque de biais de routage documenté, à itérer avec `llm-evaluation`.
- `ort` ajoute un binaire natif à la matrice de release (ONNX Runtime par
  plateforme).

### Alternatives écartées
- Router par LLM (même Flash) : ~100 ms de latence, hors budget.
- Embeddings servis par llama.cpp/GGUF : viable en Phase 3+ si le gain le
  justifie ; l'INT8 ONNX est plus simple à embarquer au MVP.
- Pure heuristique sans Stage 2 : taux de faux routage trop élevé sur
  requêtes ambiguës (ex. « explique ce code » sans mot-clé).
