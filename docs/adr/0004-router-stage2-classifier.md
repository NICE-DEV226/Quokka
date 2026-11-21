# 0004 — Classifieur Stage 2 : centroïdes lexicaux IDF, ONNX différé

- Statut : Accepted (réversible)
- Date : 2026-11-20

## Contexte

ROADMAP §3.2 exige un gate de décision pour le backend d'embedding du
routeur : `ort` rc.15 + all-MiniLM-L6-v2 INT8, `rten`, `candle`, ou un
repli heuristique — avec pour critère une latence < 1,8 ms sur CPU et
une précision mesurée sur corpus labellisé. ADR-0002 avait choisi `ort`
par défaut mais sans l'implémenter.

## Décision

Livrer d'abord un **classifieur par centroïdes sur features lexicales** :
sac de mots + bigrammes hashés (FNV-1a → 512 buckets), poids IDF calculés
sur les ancres, similarité cosinus contre deux centroïdes (Expert/Flash).
Les ancres couvrent ~45 prompts représentatifs par classe, calibrées sur
le corpus de test.

Le pipeline complet (Stage 1, garde mémoire, `routing.toml`, API de
décision) est inchangé : `ort`/`rten`/`candle` pourra remplacer le
scoring Stage 2 sans toucher au reste.

## Mesures

| Métrique | Valeur |
| --- | --- |
| Latence routeur (bench criterion, CPU de référence) | 0,3–5,5 µs/appel, corpus 100 prompts = 203 µs |
| Latence live observée (avec sonde RAM sysinfo) | 0,03–0,15 ms |
| Précision corpus 137 prompts labellisés | 100 % (après itération) |
| Budget spec | < 2 ms — respecté avec ~400× de marge |

## Conséquences

### Positives
- Zéro dépendance native, binaire unique préservé, cross-compilation
  triviale — les trois options ONNX ajoutaient un runtime par plateforme.
- Déterministe et testable : 8 tests unitaires + corpus d'intégration.
- Le budget < 2 ms est tenu par construction, pas par espoir.

### Négatives / risques — honnêtes
- **Sur-apprentissage du corpus** : le corpus a été itéré contre le
  classifieur ; 100 % ne prédit pas la précision réelle. Un corpus
  externe plus large est nécessaire avant de revendiquer une précision.
- **Limite sémantique** : les centroïdes ne comprennent pas le sens —
  « explain what a black hole is » (Flash) et « explain how TLS works »
  (Expert) se distinguent par marqueurs de domaine, pas par compréhension.
  Les cas ambigus (« résumé de ce papier de recherche ») peuvent rater.
- **Dérive des ancres** : la qualité dépend des phrases choisies ; un
  utilisateur avec un vocabulaire technique hors ancres peut voir des
  faux Flash.

### Conditions de révision

Reconsidérer `ort`/`rten`/`candle` si :
1. Un corpus externe de ≥ 200 prompts montre une précision < 85 %.
2. Des domaines manquent durablement (droit, médical, multilingue).
3. Le besoin `/v1/embeddings` (spec §4.4) justifie de toute façon un
   runtime d'embedding — alors MiniLM devient quasi gratuit.

## Alternatives écartées

- **`ort` + MiniLM INT8 (choix initial ADR-0002)** : ~100 Mo de runtime
  natif par plateforme + modèle à distribuer ; sur une machine sans GPU
  le gain de précision n'est pas démontré — on paie le coût avant la
  preuve.
- **`rten`/`candle`** : même ordre de grandeur de travail, écosystème
  plus jeune ; `candle` alourdit l'arbre de deps.
- **Embeddings via llama.cpp** : ~50–200 ms par prompt, hors budget 2 ms.
- **Pure heuristique sans Stage 2** : rejetée par ADR-0002, confirmé par
  les 15 faux routages Expert→Flash mesurés avant les marqueurs de
  domaine.
