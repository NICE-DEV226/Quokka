# Architecture Decision Records

ADR pour les décisions structurantes de Quokka. Une décision acceptée ne se
modifie pas : on la **remplace** par un nouvel ADR qui la supersède.

## Index

| ADR | Titre | Statut | Date |
| --- | --- | --- | --- |
| [0001](0001-inference-backend-bindings.md) | Bindings d'inférence : `llama-cpp-2` | Accepted | 2026-11-19 |
| [0002](0002-router-classifier-stack.md) | Classifieur du routeur : heuristique + ONNX INT8 | Accepted | 2026-11-19 |
| [0003](0003-hardware-probe-stack.md) | Stack de probe matériel : sysinfo + nvml-wrapper + ash | Accepted | 2026-11-19 |

## Créer un ADR

1. Copier le gabarit ci-dessous en `NNNN-titre-avec-tirets.md`.
2. Remplir Contexte / Décision / Conséquences (positives **et** négatives).
3. Lier les ADR liés ; mettre à jour cet index.

## Gabarit

```markdown
# NNNN — Titre de la décision

- Statut : Proposed | Accepted | Deprecated | Superseded by NNNN
- Date : AAAA-MM-JJ

## Contexte
Pourquoi une décision est nécessaire (contraintes, forces en présence).

## Décision
Ce qui a été décidé, formulé à l'impératif.

## Conséquences
### Positives
### Négatives / risques
### Alternatives écartées (une ligne chacune)
```
