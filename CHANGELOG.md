# Changelog

Tous les chiffres ci‑dessous sont **[MESURÉ]** ; les commandes de reproduction sont dans le README.

## [2.0.0] — 2026‑10‑18 — Implémentation honnête

### Corrigé
- **Quantisation canonique N‑dimensionnelle** (`apourvoir/quantization/canonical.py`) : l'ancienne version ne lisait que les 3 premières composantes, jetait la norme et ne savait reconstruire que des points 3D d'un tore. Elle est remplacée par une quantisation sphérique par triplets `(r, θ, φ)` qui fonctionne sur toute forme de tenseur.
  - randn(100, 4096) à 8 bits : erreur relative 0.0076, 8.01 bits/valeur, 3.99× vs FP32.
  - randn(100, 14336) à 8 bits : erreur relative 0.0077, 4.00× vs FP32.
  - TinyLlama‑1.1B (poids réels) : 3.99× vs FP32, 2.00× vs BF16, cosinus pondéré 0.999971, perplexité 11.484 (référence FP32 : 11.492).
- `φ = π` était quantifié en 0 (modulo erroné) ; `torch.degrees` (inexistant) dans `ThalesTriangle`.
- **Ratio de compression** : il est maintenant calculé à partir des octets réellement stockés (codes + échelles), et non plus d'une formule (`n_bits // 8` valait 0 pour n_bits < 8).
- **Imports SDK** (`scripts/02_quantize_canonical.py`) : les définitions locales qui masquaient le SDK sont supprimées. Le script lit et écrit du safetensors et mesure la taille du fichier.
- `BabelSpace.quantize` : les angles sont pris sur les coordonnées 3D, et non plus sur le point déjà projeté sur le tore, ce qui rend la déquantification cohérente.
- `src/utils/quantization.py` : débordement int16 de `UniformQuantizer` à 16 bits. Les tests historiques sont réparés : imports `src.*`, et signe du gradient attendu dans le test (le code était correct).

### Remplacé
- Embeddings aléatoires (`torch.randn`, `hash(word)`) supprimés :
  - **M** : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` ; 19/20 triplets sémantiques français bien ordonnés ; roi/reine 0.740 contre roi/banane 0.267 ;
  - **P** : traits articulatoires IPA déterministes ; bateau/gâteau 0.876 contre bateau/chien 0.261 ;
  - **S** : fréquences de lettres + POS one‑hot, déterministes.
- La fusion P+S+M concatène désormais les vues complètes. Avant, elle sommait les « 3 premières composantes » de chaque vue.

### Ajouté
- `apourvoir/quantization/baselines.py` : INT8 par ligne, réimplémentations de Q8_0 et Q4_0. Q8_0 reproduit à l'identique l'erreur du GGUF Q8_0 de llama.cpp (0.00548 sur TinyLlama).
- `apourvoir/quantization/model_bench.py` : mesure tenseur par tenseur à partir des safetensors (Mixtral sans GPU).
- `scripts/benchmark_real.py`, `scripts/compare_baselines.py`, `scripts/validate_quantization.py`.
- Tests : `tests/test_canonical_quantization.py`, `tests/test_embeddings.py`, `tests/test_sdk_imports.py` (110 tests au total, tous passent).

### Supprimé
- `scripts/03_benchmark_inference.py` (temps d'inférence simulés avec bruit aléatoire), `04_benchmark_memory.py` (taille supposée = 1/18.8), `05_validate_semantics.py` (embeddings `hash` + aléatoires), `07_compare_results.py` (tableau codé en dur, dont « 18.8× »).

### Rétracté
- « Compression 18.8× » et « précision > 95 % » sur Mixtral‑8x7B : jamais mesurés. Le 18.8× provenait de 11 vecteurs 3D jouets à 4 bits (24 % d'erreur moyenne). Mesure réelle : 4.0× vs FP32 à 8 bits.
- « Validation par le théorème de Thalès » : ce n'est qu'un test d'angle droit, sans portée sémantique démontrée.
