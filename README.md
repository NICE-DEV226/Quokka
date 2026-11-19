# QUOKKA — Smart Local AI Router & Engine

<div align="center">
  <pre>
   (\__/)  
   (•ᴗ•)  QUOKKA
  / >⚡<\  Smart Local AI Router & Engine
  </pre>

  **Faites tourner les meilleurs modèles d'IA sur votre PC. Zéro config. Zéro surchauffe.**
</div>

---

## Pourquoi Quokka ?
Les outils locaux actuels vous obligent à choisir manuellement vos modèles et saturent votre RAM. **Quokka** scanne votre PC, télécharge le duo de modèles idéal, et aiguille chaque requête en moins de 2 millisecondes :

* **Mode Éclair (1B-3B) :** Répond instantanément aux questions simples sans faire tourner les ventilateurs.
* **Mode Expert (8B-32B) :** S'active automatiquement via `mmap` uniquement pour le code et le raisonnement complexe.
* **Compatible partout :** API 100 % compatible OpenAI (`http://localhost:8080/v1`) et support natif MCP (Cursor, Claude Code, VS Code).

## Démarrage en 10 secondes
```bash
# Détecte votre matériel, configure les modèles et lance le routeur
quokka run auto
```

## Architecture
* **`quokka-core`** : Moteur d'inférence C++ (`llama.cpp`) piloté en Rust avec chargement `mmap` et gestion du cache KV.
* **`quokka-hardware`** : Détection automatique CPU/RAM/GPU/NPU et téléchargement multi-thread depuis Hugging Face.
* **`quokka-router`** : Classifieur sémantique ultra-rapide (`< 2 ms`) pour aiguiller chaque requête vers le bon modèle.
* **`quokka-server`** : Serveur HTTP Axum compatible OpenAI, serveur MCP et interface Web embarquée.

## Licence
Apache-2.0
