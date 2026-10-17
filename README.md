# Apourvoir

> **A** pour dsigner l'angle et la mesure de l'omnipotence qui survient  l'assemblage du savoir dans son espace qui est son champ de vision

**Apourvoir** est un framework de **quantisation canonique** pour les modles MoE de Mistral AI qui encode gomtriquement la signification smantique dans un espace comprime et rigoureux mathmatiquement.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

##  Caractristiques Principales

| Fonctionnalit | Description |
|----------------|-------------|
| **Espace 3D Canonique** | x=pou, y=voir, z(f) avec topologie torus |
| **Compression 18.8x** | 47 GB  2.5 GB (Mixtral-8x7B) |
| **Espace Babel** | P (Phontique) + S (Syntaxe) + M (Smantique) |
| **Thorme de Thals** | Validation gomtrique des relations |
| **4 Couches Concentriques** | Fondamentaux  Techniques |
| **SDK Optimis** | Adaptation automatique au matriel |
| **5 Tiers Matriels** | 4GB VRAM  24GB+ VRAM |

---

##  Installation Rapide

### 1. Cloner le dpt

```bash
git clone https://github.com/Albo-co/Apourvoir.git
cd Apourvoir
```

### 2. Installer les dpendances

```bash
# Crer un environnement virtuel (recommand)
python -m venv venv

# Activer l'environnement
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Installer les dpendances de base
pip install -r requirements.txt

# Installer PyTorch avec CUDA (pour GPU NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Installer GPUtil pour la dtection GPU (optionnel)
pip install gputil
```

### 3. Vrifier l'installation

```bash
python -c "from apourvoir.babel import BabelSpace; print('Apourvoir est prt !')"
```

---

##  Utilisation

###  Lanceur Optimis (Recommand)

Le moyen le plus simple de dmarrer :

```bash
# Afficher les informations matrielles
python scripts/launcher/launch.py --info

# Lancer avec configuration automatique
python scripts/launcher/launch.py

# Lancer avec un modle spcifique
python scripts/launcher/launch.py --model mistralai/Mixtral-8x7B-v0.1

# Lancer avec optimisation vitesse
python scripts/launcher/launch.py --model mistralai/Mixtral-8x22B-v0.1 --preset speed
```

**Options disponibles :**
```
--model MODEL           Modle  charger (dfaut: mistralai/Mixtral-8x7B-v0.1)
--preset PRESET         Prset d'optimisation (balanced/speed/quality/memory/latency)
--no-api               Dsactiver le serveur API
--no-ui                Dsactiver l'interface web
--api-port PORT        Port du serveur API (dfaut: 8080)
--ui-port PORT         Port de l'interface web (dfaut: 8000)
--config FILE          Charger la configuration depuis un fichier
--save-config FILE     Sauvegarder la configuration dans un fichier
--log-level LEVEL      Niveau de journalisation (DEBUG/INFO/WARNING/ERROR)
--info                 Afficher les infos matrielles et quitter
--validate             Valider la configuration et quitter
```

###  Utilisation Programmatique

#### Espace Babel

```python
from apourvoir.babel import BabelSpace, BabelConfig

# Initialiser l'espace Babel
config = BabelConfig(
    phonetic_dim=64,
    syntax_dim=64,
    semantic_dim=3,
    output_dim=128,
    n_bits=8
)

babel = BabelSpace(config)
babel.initialize_references("fr")

# Encoder du texte
text = "pouvoir voir savoir"
embedding = babel.encode(text, lang="fr")

# Quantifier et dquantifier
(theta_q, phi_q), canonical = babel.quantize(embedding)
reconstructed = babel.dequantize(theta_q, phi_q)

# Valider les relations canoniques
result = babel.get_relationship("pouvoir", "lumiere", "savoir", lang="fr")
print(f"Relation de Thales valide: {result['thales_valid']}")
```

#### Quantisation Canonique

```python
from apourvoir.quantization.canonical import (
    quantifier_tensor_canonique,
    dquantifier_tensor_canonique
)
import torch

# Crer un tenseur 3D
original = torch.randn(1000, 1000, 3)

# Quantifier
quantized, metadata = quantifier_tensor_canonique(
    original, 
    n_bits=8, 
    R=2.0, 
    r=1.0
)

# Dquantifier
reconstructed = dquantifier_tensor_canonique(quantized, metadata)

# Calculer le ratio de compression
original_size = original.numel() * 4  # float32 = 4 bytes
compressed_size = quantized.numel() * 1  # uint8 = 1 byte
ratio = original_size / compressed_size
print(f"Ratio de compression: {ratio:.2f}x")
```

###  Benchmarks

Lancer tous les benchmarks :

```bash
# Linux/Mac
./scripts/run_all_benchmarks.sh

# Windows
scripts\run_all_benchmarks.bat
```

Ou individuellement :

```bash
python scripts/02_quantize_canonical.py
python scripts/03_benchmark_inference.py
python scripts/04_benchmark_memory.py
python scripts/05_validate_semantics.py
```

---

##  Tiers Matriels

Le SDK s'adapte automatiquement  votre matriel :

| Tier | VRAM | RAM | Curs | Batch Max | n_bits | Threads |
|------|------|-----|-------|-----------|--------|----------|
| **TIER_1** | 4GB | 8GB | 6 | 4 | 4 | 4 |
| **TIER_2** | 8GB | 16GB | 8-12 | 8 | 6 | 8 |
| **TIER_3** | 12GB | 32GB | 12-16 | 16 | 8 | 12 |
| **TIER_4** | 16GB | 48GB | 16-24 | 32 | 10 | 16 |
| **TIER_5** | 24GB+ | 64GB+ | 24+ | 64 | 12 | 24 |

### Presets d'Optimisation

| Preset | Description | Cas d'Usage |
|--------|-------------|-------------|
| `balanced` | quilibre vitesse/qualit | Usage gnral |
| `speed` | Performance maximale | Infrence rapide |
| `quality` | Qualit maximale | Prcision leve |
| `memory` | Mmoire minimale | Ressources limites |
| `latency` | Latence minimale | Temps rel |

---

##  Espace Babel

L'espace **Babel** combine trois dimensions pour une compression maximale tout en prservant la signification :

###  P - Phontique
- Encodage IPA des phonmes
- Poids par complexit articulatoire
- Support multilingue (Franais, Anglais, Espagnol, Allemand)

###  S - Syntaxe  
- Arbres syntaxiques hirarchiques
- tiquetage grammatical (POS)
- Encodage de la structure et de la position

###  M - Smantique
- Projection sur espace canonique 3D
- Topologie torus (R=2.0, r=1.0)
- Validation par Thorme de Thals

### Fusion P+S+M

```
Combined = P * p_weight + S * s_weight + M * m_weight
```

---

##  Modles Supports

| Modle | Taille FP32 | Taille Compresse | Ratio | Tier Minimum |
|-----------|--------------|-------------------|-------|--------------|
| Mixtral-8x7B | 47 GB | 2.5 GB | 18.8x | TIER_2 |
| Mixtral-8x22B | 140 GB | 7.5 GB | 18.8x | TIER_3 |
| Qwen2-72B | 144 GB | 8 GB | 18x | TIER_3 |

---

##  API Serveur

### Dmarrer le serveur

```bash
python scripts/launcher/launch.py --enable-api --api-port 8080
```

### Utiliser l'API

```python
import requests
import json

response = requests.post(
    "http://localhost:8080/v1/completions",
    json={
        "model": "mistralai/Mixtral-8x7B-v0.1",
        "prompt": "Explique la quantisation canonique en franais",
        "max_tokens": 100,
        "temperature": 0.7
    },
    headers={"Authorization": "Bearer any_key"}
)

result = response.json()
print(result["choices"][0]["text"])
```

### Compatibilit avec les Agents de Codage

Apourvoir est compatible avec :
- **Claude Code** : `ANTHROPIC_BASE_URL=http://localhost:8080`
- **GitHub Copilot**
- **Cursor**
- **Codex**
- **Tout outil compatible OpenAI API**

---

##  Rsultats et Performances

### Benchmarks Mixtral-8x7B

| Configuration | Ratio | Prcision | Vitesse (tokens/s) | VRAM |
|---------------|-------|-----------|-------------------|------|
| FP32 | 1.0x | 100% | 5 | 47 GB |
| INT8 | 4.0x | 99.5% | 15 | 24 GB |
| **Canonical 8-bit** | **18.8x** | **>95%** | **20** | **2.5 GB** |
| Canonical 4-bit | 37.6x | >90% | 30 | 1.25 GB |

### Validation Thorme de Thals

```python
# Triangle pouvoir - lumire - savoir
result = babel.get_relationship("pouvoir", "lumiere", "savoir")
# Thales valid: True
# Angle v1-v2: 90.00
# Angle v2-v3: 90.00
# Angle v1-v3: 12.22
```

---

##  Structure du Projet

```
Apourvoir/
 apourvoir/
     babel/                      # Espace multivectoriel
        phonetic/             # Dimension phontique
        syntax/               # Dimension syntaxique
        semantic/             # Dimension smantique
        fusion/                # Fusion P+S+M
        validation/           # Mtriques
     quantization/              # Quantisation canonique
     optimizer/                 # SDK d'optimisation
        launcher/              # Lanceur optimis
        hardware_detector.py
        memory_manager.py
        adaptive_batching.py
        dynamic_quant.py
        expert_parallelism.py
        kernel_fusion.py
        thread_pool.py
        benchmark_profiler.py
 scripts/
     launcher/                   # Scripts de lancement
     babel/                      # Scripts Babel
     01_download_models.py
     02_quantize_canonical.py
     03_benchmark_inference.py
     04_benchmark_memory.py
     05_validate_semantics.py
     06_generate_report.py
     07_compare_results.py
     run_all_benchmarks.sh
 docs/
     RECETTE_QUANTISATION_MOE_CANONIQUE.md
     MAIL_POUR_MISTRAL_AI.md
```

---

##  Dpannage

###  Problmes Courants

#### 1. "No GPU detected"
```bash
pip install gputil
```

#### 2. "CUDA out of memory"
```bash
# Utiliser un preset memory
python scripts/launcher/launch.py --preset memory

# Ou rduire le batch size
python scripts/launcher/launch.py --preset memory --model smaller_model
```

#### 3. "Module not found"
```bash
pip install -r requirements.txt
```

#### 4. "Insufficient VRAM for model"
```bash
# Utiliser un modle plus petit
python scripts/launcher/launch.py --model mistralai/Mixtral-8x7B-v0.1

# Ou activer le swap vers le disque
python scripts/launcher/launch.py --preset memory
```

###  Vrifier l'installation

```bash
# Tester l'import
python -c "from apourvoir.babel import BabelSpace; print('OK')"

# Tester le lanceur
python scripts/launcher/launch.py --info

# Tester la quantisation
python scripts/babel/03_build_babel_space.py --texts pouvoir voir savoir
```

---

##  Configuration Avance

### Fichier de Configuration

```json
{
  "tier": "tier_3",
  "preset": "balanced",
  "quantization": {
    "n_bits": 8,
    "method": "angular",
    "adaptive": true,
    "min_bits": 4,
    "max_bits": 16
  },
  "memory": {
    "max_vram_gb": 10.0,
    "max_ram_gb": 24.0,
    "cache_size_gb": 2.0,
    "preload": true
  },
  "parallelism": {
    "cpu_threads": 12,
    "gpu_streams": 4,
    "multi_gpu": false,
    "parallel_experts": 8
  },
  "batch": {
    "max_batch_size": 16,
    "optimal_batch_size": 8,
    "dynamic": true
  }
}
```

### Charger une Configuration

```bash
python scripts/launcher/launch.py --config my_config.json
```

### Sauvegarder une Configuration

```bash
python scripts/launcher/launch.py --save-config my_config.json
```

---

##  Contribution

Les contributions sont les bienvenues ! 

1. **Forker** le projet
2. **Crer** une branche (`git checkout -b feature/ma-fonctionnalit`)
3. **Commiter** vos changements (`git commit -m 'Ajout fonctionnalit'`)
4. **Pousser** vers la branche (`git push origin feature/ma-fonctionnalit`)
5. **Ouvrir** une Pull Request

### Bonnes Pratiques

- Respecter le style de code existant
- Ajouter des tests pour les nouvelles fonctionnalits
- Documenter les changements dans le CHANGELOG
- Utiliser des noms de variables clairs

---

##  Licence

**MIT License** - Libre d'utiliser pour usage personnel et commercial.

---

##  Remerciements

- **Mistral AI** pour les modles MoE exceptionnels
- **La communaut open-source** pour les outils de quantisation
- **Tous les contributeurs** du projet

---

##  Ressources

- **Repository** : [https://github.com/Albo-co/Apourvoir](https://github.com/Albo-co/Apourvoir)
- **Documentation** : [docs/RECETTE_QUANTISATION_MOE_CANONIQUE.md](docs/RECETTE_QUANTISATION_MOE_CANONIQUE.md)
- **Philosophie** : "A pour dsigner l'angle et la mesure de l'omnipotence qui survient  l'assemblage du savoir dans son espace qui est son champ de vision"

---

<div align="center">

** Prt  l'emploi ! **

```bash
# Dmarrez ds maintenant
python scripts/launcher/launch.py --info
```

</div>
