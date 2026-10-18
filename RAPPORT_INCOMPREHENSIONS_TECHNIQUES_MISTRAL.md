# 🔍 **RAPPORT TECHNIQUE — Points d'Incompréhension sur la Recette de Quantisation Canonique**

*Document d'accompagnement à [MAIL_POUR_MISTRAL_AI.md](MAIL_POUR_MISTRAL_AI.md), rédigé après exécution réelle du pipeline sur un checkpoint Mixtral-8x7B complet.*

---

## 📌 Contexte de ce rapport

Avant d'envoyer la proposition de collaboration à Mistral AI, j'ai exécuté `scripts/02_quantize_canonical.py` contre un vrai checkpoint **Mixtral-8x7B fp32** (995 tenseurs, 19 shards safetensors, ~87 GB) pour vérifier que les chiffres annoncés dans le mail (18.8×, >94-95% de précision conservée) sont reproductibles. Ce n'est pas le cas, et les raisons sont des incohérences mathématiques dans le code lui-même, pas des problèmes d'infrastructure. Ce document liste ces points précisément, pour qu'ils soient tranchés **avant** tout envoi à Mistral.

---

## 1️⃣ La méthode ne fonctionne que pour des vecteurs de dimension 3 — pas pour des poids de modèle

`coordonnées_sphériques()` (dans `apourvoir/quantization/canonical.py` **et** sa copie dans `scripts/02_quantize_canonical.py`) lit explicitement :

```python
theta = torch.atan2(v[..., 1], v[..., 0])
phi = torch.acos(torch.clamp(v[..., 2] / (r + 1e-10), -1.0, 1.0))
```

Ceci n'a de sens que si la dernière dimension du tenseur vaut **exactement 3**. Le propre test embarqué du module (`if __name__ == "__main__":` dans `canonical.py`) ne teste que `torch.randn(10, 10, 3)` — jamais une matrice de poids réelle.

Or les poids de Mixtral ont des largeurs de **1024 à 14336** (`q_proj`, `k_proj`, `w1`/`w2`/`w3` des experts, `embed_tokens`, etc.). Pour toute ligne de poids de largeur ≠ 3, la fonction prend silencieusement les 3 premières composantes (`v[...,0]`, `v[...,1]`, `v[...,2]`) et **ignore le reste** — il n'existe aucune logique de découpage en triplets (`tiling`) dans le code.

**Preuve expérimentale (exécution réelle, 2026-10-17) :**
```
2026-10-17 22:35:20 - INFO - Quantification terminée: 86.99GB -> 0.02GB (ratio 5318.86x)
2026-10-17 22:35:21 - INFO - Validation: 0.0% de succès
2026-10-17 22:35:21 - INFO - Différence maximale: inf
```
`déquantifier_tensor_canonique()` ne peut même pas reconstruire la forme d'origine (`view(original_shape)` échoue : `shape '[14336, 4096]' is invalid for input of size 43008`), car on ne peut pas reformer une matrice `(14336, 4096)` à partir de `(14336, 3)`. Le ratio de 5318× n'est donc pas de la compression : c'est la perte quasi-totale (>99.9%) de chaque ligne de poids.

**Question pour Mistral / pour la suite du projet :** la méthode `canonical_torus` a-t-elle été conçue pour quantifier des **embeddings de tokens** projetés au préalable sur 3 axes sémantiques (pou/voir/z(f), cf. `RECETTE_QUANTISATION_MOE_CANONIQUE.md`), et non pour quantifier des **matrices de poids** de couches linéaires/MoE ? Si oui, le script `02_quantize_canonical.py` applique la méthode au mauvais objet.

---

## 2️⃣ Écart entre les chiffres du mail et ce que le code produit réellement

`MAIL_POUR_MISTRAL_AI.md` annonce :

| Modèle | Ratio annoncé | Précision annoncée |
|---|---|---|
| Mixtral-8x7B | 18.8× | >95% |
| Mixtral-8x22B | 18.8× | >94% |

Ces chiffres proviennent de `sandbox_quantisation.py`, qui ne teste que **11 tokens sémantiques** (`pouvoir`, `lumière`, `savoir`, ...) construits *par définition* en dimension 3. Rien dans le dépôt ne démontre ces ratios sur un vrai modèle MoE. Mon exécution sur le vrai checkpoint donne un tout autre résultat (5318× / 0% de validation / erreur infinie — point 1). Avant d'envoyer le mail, il faudrait soit :
- reproduire 18.8×/>94% sur un vrai modèle (ce qui suppose de résoudre le point 1), soit
- corriger le mail pour ne pas présenter des résultats "sandbox sur 11 tokens" comme des "résultats concrets sur Mixtral-8x7B/8x22B".

---

## 3️⃣ Le chemin SDK n'est jamais réellement exécuté

En tête de `scripts/02_quantize_canonical.py` :
```python
try:
    from apourvoir.quantization import (
        quantifier_tensor_canonique,
        déquantifier_tensor_canonique,
        ...
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
```
Mais plus loin dans le même fichier, des fonctions **locales** portant exactement les mêmes noms sont redéfinies (lignes ~100-205). En Python, ces définitions ultérieures écrasent silencieusement les imports. Donc `SDK_AVAILABLE` est calculé mais jamais utilisé pour choisir quel code tourne : **le code local tourne toujours**, même quand le SDK est disponible. Les deux implémentations sont actuellement identiques, donc cela n'a pas changé le résultat de ce test — mais si l'une évolue sans l'autre, ce flag `SDK_AVAILABLE` donnera une fausse impression de quelle implémentation est active.

---

## 4️⃣ La "recette" n'a jamais été testée à l'échelle d'un vrai modèle

Deux problèmes d'implémentation, indépendants des choix mathématiques ci-dessus, montrent que le pipeline n'avait jamais tourné sur un modèle de taille réelle avant ce test :

- **Format de poids** : le chargeur attendait un unique fichier `pytorch_model.bin` (pickle). Tout modèle HF moderne, Mixtral inclus, est distribué en **safetensors shardé + index** (`model-00001-of-00019.safetensors`, ..., `model.safetensors.index.json`). *(Corrigé — le chargeur lit maintenant l'index et les shards safetensors, avec repli sur fichier unique ou `.bin`.)*
- **Mémoire** : le pipeline chargeait l'intégralité du `state_dict` (~87 GB) puis construisait une seconde copie quantifiée en parallèle, sur une machine avec 15.4 GB de RAM + 28 GB de swap. Impossible à exécuter tel quel. *(Corrigé — le pipeline traite maintenant les poids shard par shard, en écrivant chaque shard quantifié sur disque avant de libérer la mémoire et de passer au shard suivant.)*

Ces deux correctifs étaient nécessaires juste pour que l'exécution se termine sans crasher — ils n'ont aucun rapport avec la validité mathématique de la méthode elle-même (point 1).

---

## 📝 Résumé — ce qui doit être tranché avant tout envoi à Mistral

1. **Décider de l'objet réel de `canonical_torus`** : embeddings de tokens en 3D (cas pour lequel le code fonctionne), ou poids de modèle en pleine dimension (cas pour lequel il faudrait d'abord inventer un découpage en triplets, avec une vraie étude de la perte d'information induite).
2. **Ne pas présenter les chiffres du sandbox 11-tokens comme des résultats Mixtral-8x7B/8x22B** dans le mail tant qu'ils n'ont pas été reproduits sur un vrai modèle.
3. **Clarifier si le chemin SDK (`apourvoir.quantization`) doit remplacer le code local**, ou si la duplication est volontaire (fallback).
