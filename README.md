# Apourvoir

> **A** pour désigner l'angle et la mesure de l'omnipotence qui survient à l'assemblage du savoir dans son espace qui est son champ de vision

**Apourvoir** est un projet expérimental de **quantisation « canonique »** (sphérique) des poids de modèles de langage, accompagné d'un espace d'embeddings « Babel » (phonétique + syntaxe + sémantique).

**Règle du dépôt : tout chiffre de ce README provient d'une exécution réelle et indique la commande qui permet de le reproduire.** Ce qui n'a pas été mesuré est marqué **[NON MESURÉ]**.

---

## ⚠️ Correctifs d'intégrité (octobre 2026)

Les versions précédentes affirmaient une **compression de 18.8×** et une **précision > 95 %** sur Mixtral‑8x7B. **Ces chiffres n'avaient jamais été mesurés sur un modèle**, et ils sont faux :

- le 18.8× venait de `results/rapport_quantisation_optimisee.json`, c'est‑à‑dire de **11 vecteurs 3D jouets** quantifiés à 4 bits, avec **24 % d'erreur moyenne et 73 % d'erreur maximale** ;
- l'ancienne quantisation ne traitait que les **3 premières composantes** d'un vecteur et jetait sa norme : elle ne pouvait **pas reconstruire** un poids de dimension 4096 ;
- les embeddings Babel étaient **aléatoires** (`torch.randn`, `hash(word)`) ;
- les scripts `03`/`04`/`05`/`07` produisaient des **chiffres simulés ou codés en dur**. Ils ont été supprimés.

**Mesure réelle :** la méthode canonique à 8 bits compresse **4.0× par rapport au FP32**, soit **2.0× par rapport au BF16** dans lequel Mistral distribue ses modèles. C'est le même budget qu'un INT8. Voir les tableaux ci‑dessous.

---

## Méthode canonique (sphérique par triplets)

`apourvoir/quantization/canonical.py` :

1. Le tenseur est aplati en lignes `(N, D)`, et chaque ligne est découpée en `ceil(D/3)` triplets `(x, y, z)`. La dernière dimension peut être quelconque (4096, 14336, …).
2. Chaque triplet est converti en `(r, θ, φ)`. `θ` et `φ` sont quantifiés uniformément sur `n_bits` ; `r` est normalisé par le rayon maximal de la ligne (une échelle float32 par ligne) et encodé en `sqrt(r / r_max)` sur `n_bits`.
3. Coût exact : `3·n_bits` bits par triplet, plus 32 bits par ligne, soit **≈ n_bits bits par paramètre**.

```python
import torch
from apourvoir.quantization import quantifier_tensor_canonique, déquantifier_tensor_canonique, erreur_relative

w = torch.randn(14336, 4096)
codes, meta = quantifier_tensor_canonique(w, n_bits=8)
w_hat = déquantifier_tensor_canonique(codes, meta)
print(meta["compression_ratio"], erreur_relative(w, w_hat))   # ratio calculé sur les octets réellement stockés
```

**Pourquoi pas une PCA en 3D ?** Une reconstruction de rang 3 d'une matrice `100×4096` laisse **97.5 % d'erreur relative** ; sur des poids réels de TinyLlama, l'erreur va de **77 % à 99 %**. Une PCA 3D détruit donc l'information (`python scripts/validate_quantization.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

---

## Résultats mesurés

Machine : Windows 11, AMD Ryzen 5 4600H (6 threads utilisés), 16 Go RAM, torch 2.14.1+cpu, llama.cpp b9873 (a4107133a). Toutes les méthodes sont appliquées **aux mêmes poids**.

### Roundtrip sur tenseurs synthétiques — `python scripts/validate_quantization.py`

| Tenseur (graine 0) | canonique 8 bits | canonique 4 bits | INT8 / ligne | Q8_0 (réf.) | Q4_0 (réf.) |
|---|---|---|---|---|---|
| randn(100, 4096) | 0.0076 | 0.1249 | 0.0087 | 0.0054 | 0.0862 |
| randn(100, 14336) | 0.0077 | 0.1257 | 0.0094 | 0.0054 | 0.0860 |
| Student‑t 3 ddl (100, 4096) | 0.0090 | 0.1463 | 0.0367 | 0.0083 | 0.1289 |

*(erreur relative de Frobenius ‖x − x̂‖ / ‖x‖ ; bits/valeur : 8.01, 4.01, 8.0, 8.5, 4.5)*

### TinyLlama‑1.1B (poids réels BF16) — `python scripts/compare_baselines.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model-level --llama-cpp-dir <llama.cpp>`

1 100 048 384 paramètres ; 2.200 Go en BF16 ; 4.400 Go en FP32.

| Méthode | Taille | bits/param | vs FP32 | vs BF16 | cos. moyen | err. rel. | Perplexité ¹ |
|---|---|---|---|---|---|---|---|
| référence | — | — | — | — | 1 | 0 | 11.492 (HF FP32) / 9.2915 (GGUF BF16) |
| **canonique 8 bits** | 1.102 Go | 8.018 | 3.99× | 2.00× | 0.999971 | 0.00763 | **11.484** |
| **canonique 4 bits** | 0.552 Go | 4.016 | 7.97× | 3.98× | 0.992175 | 0.12529 | **12.870** |
| INT8 par ligne | 1.102 Go | 8.014 | 3.99× | 2.00× | 0.999952 | 0.00969 | 11.509 |
| Q8_0 (réimpl. PyTorch) | 1.169 Go | 8.501 | 3.76× | 1.88× | 0.999985 | 0.00548 | 11.488 |
| Q4_0 (réimpl. PyTorch) | 0.619 Go | 4.501 | 7.11× | 3.55× | 0.996148 | 0.08781 | 12.063 |
| GGUF Q8_0 (llama.cpp) | 1.170 Go ² | 8.507 | 3.76× | 1.88× | 0.999985 | 0.00548 | 9.2948 |
| GGUF Q4_K_M (llama.cpp) | 0.668 Go ² | 4.857 | 6.59× | 3.30× | 0.997805 | 0.06308 | 9.6241 |
| AWQ | **[NON MESURÉ]** : AutoAWQ requiert un GPU CUDA ; torch est installé en version CPU sur cette machine | | | | | | |

¹ wikitext‑2 (test), 20 fenêtres de 512 tokens. Les deux pipelines **ne sont pas comparables entre eux** : llama.cpp ne note que la seconde moitié de chaque fenêtre. Il faut comparer chaque méthode à **sa propre** référence (HF FP32 ou GGUF BF16).
² taille du fichier GGUF (métadonnées et tokenizer inclus). La réimplémentation `q8_0_ref` donne **exactement** la même erreur que le vrai GGUF Q8_0, ce qui valide le banc de mesure.

**Lecture honnête :**
- À 8 bits, la méthode canonique fait **mieux que l'INT8 par ligne** (erreur 0.0076 contre 0.0097, et elle résiste aux queues lourdes), mais **moins bien que Q8_0** (0.0055), qui coûte 0.5 bit de plus par paramètre. L'effet sur la perplexité n'est pas mesurable (11.484 contre 11.492 en FP32).
- À 4 bits, la méthode canonique est **nettement moins bonne** que Q4_0 et que Q4_K_M : +12.0 % de perplexité, contre +5.0 % pour Q4_0 et +3.6 % pour Q4_K_M.

**Vitesse de génération** (32 tokens, 6 threads) :

| Moteur / format | CPU (tokens/s) | GPU Vulkan, GTX 1660 Ti, toutes couches (tokens/s) |
|---|---|---|
| transformers FP32 (et toutes les méthodes PyTorch, qui s'exécutent sur poids déquantifiés) | 4.41 | — (torch CPU) |
| llama.cpp GGUF BF16 | 15.3 | 117.7 |
| llama.cpp GGUF Q8_0 | 28.2 | 148.7 |
| llama.cpp GGUF Q4_K_M | 49.4 | 218.6 |

- Les méthodes PyTorch (canonique, INT8, Q8_0/Q4_0 réf.) **n'ont aucun noyau bas‑bit** : l'inférence tourne à la vitesse FP32. La quantification puis la déquantification du modèle complet prennent 20.9 s en canonique 8 bits, contre 5.6 s en INT8.
- Les colonnes CPU viennent de `llama-bench -ngl 0`. Le backend chargé reste Vulkan : la vitesse de *traitement du prompt* (dans le JSON) peut donc utiliser le GPU même avec `-ngl 0`, et elle est **[À VALIDER]**. La génération, elle, est faite sur CPU.
- Deux exécutions complètes donnent des perplexités identiques à 4 décimales. Les vitesses varient : 3.62 et 4.41 tok/s pour transformers FP32 selon la charge de la machine.

### Mixtral‑8x7B — `python scripts/benchmark_real.py` et `python scripts/compare_baselines.py --model mistralai/Mixtral-8x7B-v0.1 --llama-cpp-dir <llama.cpp>`

Les 19 fichiers safetensors (995 tenseurs, dont 930 tenseurs 2‑D quantifiés) sont lus tenseur par tenseur, sur CPU, en 1814 s.

| Mesure | Valeur [MESURÉ] |
|---|---|
| Paramètres | 46 702 792 704 |
| Taille d'origine (safetensors BF16) | 93.406 Go |
| Taille équivalente FP32 | 186.811 Go |
| **Taille compressée (canonique 8 bits)** | **46.755 Go** |
| Bits par paramètre | 8.0089 |
| **Compression ratio vs FP32** | **4.00×** |
| Compression ratio vs BF16 | 2.00× |
| Erreur relative pondérée / max | 0.007657 / 0.008662 |
| Cosinus pondéré / min | 0.999971 / 0.999963 |

Détails par tenseur : `results/benchmark_mistralai__Mixtral-8x7B-v0.1_canonical8.json`. *(L'ancien README annonçait 2.5 Go et 18.8× : c'est faux d'un facteur ~19.)*

Comparaison avec les baselines sur Mixtral (INT8, Q8_0/Q4_0, GGUF Q8_0/Q4_K_M) : [EN COURS DE MESURE].

---

## Espace Babel (P + S + M)

| Dimension | Encodage actuel | Avant |
|---|---|---|
| **P** phonétique | traits articulatoires IPA (voisement, nasalité, lieu, mode, aperture, antériorité, arrondissement), vecteur `[moyenne | 1er phonème | dernier phonème]` de 64 dimensions, déterministe | `torch.randn` |
| **S** syntaxe | fréquences de lettres du token + POS one‑hot + encodages sinusoïdaux de profondeur et de position, déterministe | `torch.randn` / `hash` |
| **M** sémantique | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 dim, téléchargé depuis HuggingFace puis mis en cache), projeté en 3D sur la base orthonormée (Gram‑Schmidt) des embeddings de « pou », « voir » et « savoir » | `hash(word)` |

Mesures (`python -m pytest tests/test_embeddings.py`) :
- sémantique (mpnet) : similarité « roi »/« reine » = 0.740, contre 0.267 pour « roi »/« banane ». Sur 20 triplets français (ancre, proche, éloigné), **19/20** sont bien ordonnés avec mpnet ; l'échec est « homme/femme/voiture ». Pour comparaison : 17/20 avec paraphrase‑multilingual‑MiniLM‑L12‑v2, 19/20 avec multilingual‑e5‑small, 15/20 avec all‑MiniLM‑L6‑v2 ;
- phonétique : « bateau »/« gâteau » = 0.88, contre 0.26 pour « bateau »/« chien ».

La projection 3D de M **ne conserve qu'une partie de l'information**. Les similarités doivent se calculer sur l'embedding complet (`BabelSpace.semantic_embedding`). Les deux angles produits par `BabelSpace.quantize` codent seulement la direction du point 3D : c'est un code avec perte, **pas une compression réversible** de l'embedding.

---

## Installation

```bash
git clone https://github.com/Albo-co/Apourvoir.git
cd Apourvoir
pip install -r requirements.txt
python -c "from apourvoir.quantization import quantifier_tensor_canonique; from apourvoir.babel import BabelSpace; print('ok')"
python -m pytest -q
```

Pour les baselines GGUF : installer les binaires llama.cpp (`llama-quantize`, `llama-perplexity` et `llama-bench` dans le PATH), et cloner le dépôt llama.cpp **à la même version** (ce clone fournit `convert_hf_to_gguf.py` et `gguf-py`), puis passer `--llama-cpp-dir`.

## Serveur d'inférence

```bash
python scripts/launcher/launch.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant-method canonical --n-bits 8 --api-host 127.0.0.1
curl -s localhost:8080/v1/completions -d '{"prompt": "The capital of France is", "max_tokens": 16}'
curl -s localhost:8080/v1/chat/completions -d '{"messages": [{"role": "user", "content": "Bonjour"}], "max_tokens": 32}'
```

- `--quant-method canonical` : chaque poids 2‑D passe par la quantisation canonique puis est déquantifié. La génération utilise donc exactement les valeurs quantifiées, et `/status` rapporte le ratio et l'erreur mesurés au chargement. `none` laisse les poids intacts ; `bitsandbytes`, `gguf` et `awq` sont refusés explicitement par le moteur.
- Paramètres de requête acceptés : `max_tokens` (1 à 4096), `temperature` (0 = glouton, déterministe), `top_p`, `stop`, `seed`. `stream` est refusé (400).
- Avant le chargement, le moteur vérifie la RAM ou la VRAM disponible (taille des safetensors × dtype) et refuse un modèle trop gros.
- `--no-inference` démarre le serveur sans modèle : les complétions répondent alors 501.

## Scripts de mesure

| Script | Rôle |
|---|---|
| `scripts/validate_quantization.py` | roundtrip quantize → dequantize sur tenseurs synthétiques et réels, et comparaison avec la PCA 3D |
| `scripts/benchmark_real.py` | méthode canonique sur tous les poids d'un modèle HF (Mixtral par défaut), lecture tenseur par tenseur : **pas de GPU requis**, quelques Go de RAM |
| `scripts/compare_baselines.py` | canonique contre INT8, Q8_0/Q4_0 (réimpl.), GGUF Q8_0/Q4_K_M (llama.cpp) et AWQ, plus perplexité et vitesse avec `--model-level` |
| `scripts/02_quantize_canonical.py` | écrit un modèle quantifié en safetensors (codes + échelles) et mesure la taille du fichier |
| `scripts/run_all_benchmarks.sh` | lance tout |

Chaque script affiche la commande exécutée et écrit ses résultats détaillés (par tenseur) dans `results/`.

---

## Limitations et bugs connus

- **Pas de noyau d'inférence bas‑bit** : un modèle quantifié de façon canonique doit être déquantifié avant l'inférence. Il n'y a donc aucun gain de vitesse, ni de mémoire à l'exécution.
- **4 bits** : la méthode canonique est moins précise que Q4_0 et Q4_K_M au même budget (voir tableau).
- **Mixtral, niveau modèle** : perplexité et vitesse **[NON MESURÉ]**. Le modèle (93 Go en BF16) ne tient pas en RAM sur la machine de mesure (16 Go).
- **AWQ** : **[NON MESURÉ]** (pas de GPU CUDA utilisable par torch sur la machine de mesure).
- **Couches concentriques** (`CanonicalQuantizer`, couches 1 à 4 avec 8/6/4/4 bits) : simple politique de bits, sans bénéfice mesuré.
- **Relations « de Thalès »** : `ThalesTriangle` et `validate_thales` mesurent seulement si un angle vaut 90°. **Aucune propriété sémantique n'est démontrée.**
- **SDK `apourvoir/optimizer`** (tiers matériels, presets, batching adaptatif, parallélisme d'experts) : **[NON MESURÉ]**. `adaptive_batching.py` contient encore une démo avec des valeurs `np.random` ; les tableaux de tiers et presets sont des heuristiques non validées.
- **Lanceur et API** (`scripts/launcher/launch.py`) : `/v1/completions` et `/v1/chat/completions` sont servis par un vrai moteur (`apourvoir/inference`, backend transformers, CPU ou CUDA). Voir « Serveur d'inférence » ci‑dessous. Limites :
  - pas de streaming, et une seule génération à la fois (verrou) ;
  - pas de noyau bas‑bit : la RAM et la vitesse sont celles du dtype déquantifié ;
  - Mixtral ne tient pas sur la machine de mesure, et le launcher le refuse avec un message explicite ;
  - la compatibilité avec des agents de codage n'a pas été vérifiée.
- **Documents historiques** (`MAIL_POUR_MISTRAL_AI.md`, `RECETTE_QUANTISATION_MOE_CANONIQUE.md`, `README_APOURVOIR.md`, `results/*.json|png` d'avant octobre 2026) : ils contiennent les anciennes affirmations non mesurées et sont signalés comme tels.
- `src/` est un module historique indépendant (alignement savoir/pouvoir) ; ses tests passent, mais il n'est pas utilisé par la quantisation.

---

## Licence

MIT.
