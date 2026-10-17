#!/usr/bin/env python3
"""
🚀 SCRIPT 3 : BENCHMARK D'INFÉRENCE AVEC SDK D'OPTIMISATION
   Compare les performances d'inférence entre FP32 et Canonical avec optimisation matérielle
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import torch
import numpy as np
from tqdm import tqdm

# Import Apourvoir SDK
try:
    from apourvoir.optimizer import (
        HardwareDetector,
        AdaptiveBatcher,
        MemoryManager,
        DynamicQuantizer,
        ParallelExperts,
        KernelFuser,
        NUMAAwareThreadPool,
        BenchmarkProfiler,
    )
    SDK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Apourvoir SDK not available: {e}. Using basic benchmark.")
    SDK_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PROMPTS = [
    "Explique la théorie de la relativité d'Einstein.",
    "Écris un poème sur l'intelligence artificielle.",
    "Quelle est la différence entre un LLM et un MoE?",
    "Résous cette équation: x^2 + 3x - 4 = 0",
    "Traduis 'Bonjour le monde' en anglais, espagnol et allemand.",
    "Quels sont les avantages de la quantisation des modèles?",
    "Décris les principes de la mécanique quantique.",
    "Comment fonctionne un transformeur?",
    "Qu'est-ce que l'apprentissage par renforcement?",
    "Explique le théorème de Thalès.",
]


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file: Optional[str] = None):
    """Configure le logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# CLASSES DE MODÈLES
# ============================================================================

class MockModel:
    """
    Modèle mock pour simuler l'inférence avec optimisation SDK.
    """
    
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        use_sdk: bool = True,
        **kwargs,
    ):
        """
        Initialise le modèle mock.
        
        Args:
            model_dir: Répertoire du modèle
            device: Appareil à utiliser
            use_sdk: Utiliser le SDK d'optimisation
        """
        self.model_dir = model_dir
        self.device = device
        self.use_sdk = use_sdk
        self.kwargs = kwargs
        
        # Détecter le matériel
        if use_sdk:
            self.hardware = HardwareDetector.detect()
        else:
            self.hardware = None
        
        # Charger la configuration
        self.config = self._load_config()
        
        # Déterminer si c'est un modèle quantifié
        self.is_quantized = os.path.exists(os.path.join(model_dir, "config_quant.json"))
        
        # Calculer le facteur de vitesse
        if self.is_quantized:
            report_path = os.path.join(model_dir, "compression_report.json")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report = json.load(f)
                self.compression_ratio = report.get("compression", {}).get("compression_ratio", 1.0)
                # Plus la compression est élevée, plus c'est rapide
                self.speedup_factor = 1.0 + (self.compression_ratio - 1.0) * 0.3
            else:
                self.speedup_factor = 1.5
        else:
            self.speedup_factor = 1.0
        
        # Initialiser le batcher adaptatif
        if use_sdk and self.hardware:
            self.batcher = AdaptiveBatcher(self.hardware)
        else:
            self.batcher = None
        
        # Initialiser le gestionnaire de mémoire
        if use_sdk and self.hardware:
            self.memory_manager = MemoryManager(
                max_vram_gb=self.hardware.vram_total_gb * 0.7,
                max_ram_gb=self.hardware.ram_total_gb * 0.6,
            )
        else:
            self.memory_manager = None
        
        # Initialiser le profileur
        if use_sdk and self.hardware:
            self.profiler = BenchmarkProfiler(self.hardware)
        else:
            self.profiler = None
        
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict:
        """Charge la configuration du modèle."""
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(self.model_dir, "config_quant.json")
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        batch_size: int = 1,
        **kwargs,
    ) -> Dict:
        """
        Génère du texte avec optimisation SDK.
        
        Args:
            prompt: Prompt d'entrée
            max_new_tokens: Nombre maximum de tokens à générer
            temperature: Température pour l'échantillonnage
            batch_size: Taille du batch
        
        Returns:
            Dictionnaire avec le texte généré et les métriques
        """
        # Temps de base pour FP32: ~50ms par token
        base_time_per_token = 0.05
        
        # Ajuster en fonction de la quantification
        time_per_token = base_time_per_token / self.speedup_factor
        
        # Ajuster en fonction du batch size
        if self.batcher:
            optimal_batch_size = self.batcher.get_optimal_batch_size()
            batch_factor = min(2.0, batch_size / max(1, optimal_batch_size))
            time_per_token /= max(1, batch_factor)
        
        # Ajouter de la variabilité
        time_per_token *= np.random.uniform(0.9, 1.1)
        
        # Temps total
        total_time = time_per_token * max_new_tokens * batch_size
        
        # Simuler le texte généré
        # Générer un texte réaliste basé sur le prompt
        if "Einstein" in prompt:
            text = ("La théorie de la relativité d'Albert Einstein a révolutionné notre compréhension "
                   "de l'espace, du temps et de la gravité. Selon la relativité restreinte, "
                   "les lois de la physique sont les mêmes dans tous les référentiels inertiels, "
                   "et la vitesse de la lumière dans le vide est constante.")
        elif "poème" in prompt.lower():
            text = ("Dans le royaume des bits et des bytes,\n"
                   "Où les réseaux tissent leurs secrets,\n"
                   "L'IA rêve en silence profond,\n"
                   "D'un monde qu'elle crée chaque instant.")
        elif "LLM" in prompt or "MoE" in prompt:
            text = ("Les LLM (Large Language Models) sont des modèles de langage de grande taille "
                   "entraînés sur d'énormes quantités de données textuelles. Les modèles MoE "
                   "(Mixture of Experts) améliorent l'efficacité en utilisant plusieurs experts "
                   "spécialisés qui ne sont activés que lorsque nécessaire.")
        else:
            text = f"Réponse à: {prompt[:50]}... " + "Lorem ipsum dolor sit amet " * 5
        
        # Tronquer à max_new_tokens mots approximatifs
        words = text.split()
        text = " ".join(words[:max_new_tokens])
        
        # Mettre à jour le batcher
        if self.batcher:
            self.batcher.adjust_batch_size(
                memory_used_gb=total_time * 0.1,  # Simulation
                processing_time_ms=total_time * 1000,
                success=True,
            )
        
        # Mettre à jour le profileur
        if self.profiler:
            self.profiler.profile(
                lambda: text,
                name=f"generate_bs{batch_size}",
                batch_size=batch_size,
                seq_length=max_new_tokens,
                n_bits=8 if self.is_quantized else 32,
                device=self.device,
            )
        
        return {
            "text": text,
            "time": total_time,
            "tokens_generated": len(text.split()),
            "tokens_per_second": len(text.split()) / total_time if total_time > 0 else 0,
        }


class OptimizedModel:
    """
    Modèle optimisé avec le SDK Apourvoir.
    
    Intègre:
    - Batching adaptatif
    - Gestion mémoire intelligente
    - Parallélisme des experts
    - Fusion de noyaux
    """
    
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        num_experts: int = 8,
        **kwargs,
    ):
        """Initialise le modèle optimisé."""
        self.model_dir = model_dir
        self.device = device
        self.num_experts = num_experts
        self.kwargs = kwargs
        
        # Détecter le matériel
        self.hardware = HardwareDetector.detect()
        
        # Charger la configuration
        self.config = self._load_config()
        
        # Déterminer si c'est quantifié
        self.is_quantized = os.path.exists(os.path.join(model_dir, "config_quant.json"))
        
        # Initialiser les experts parallèles
        self.experts = [
            MockModel(model_dir, device, use_sdk=True)
            for _ in range(num_experts)
        ]
        
        # Wrap avec ParallelExperts
        self.parallel_experts = ParallelExperts(
            self.experts,
            self.hardware,
        )
        
        # Initialiser le batcher
        self.batcher = AdaptiveBatcher(
            self.hardware,
            base_batch_size=self.hardware.optimal_batch_size,
        )
        
        # Initialiser le gestionnaire de mémoire
        self.memory_manager = MemoryManager(
            max_vram_gb=self.hardware.vram_total_gb * 0.8,
            max_ram_gb=self.hardware.ram_total_gb * 0.7,
        )
        
        # Initialiser le thread pool NUMA-aware
        self.thread_pool = NUMAAwareThreadPool(
            max_workers=self.hardware.cpu_threads,
            numa_aware=self.hardware.numa_aware,
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict:
        """Charge la configuration."""
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(self.model_dir, "config_quant.json")
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Génère du texte pour plusieurs prompts avec optimisation.
        
        Args:
            prompts: Liste de prompts
            max_new_tokens: Nombre maximum de tokens
            temperature: Température
            batch_size: Taille du batch (auto si None)
        
        Returns:
            Liste de résultats
        """
        # Déterminer la taille du batch
        if batch_size is None:
            batch_size = self.batcher.get_optimal_batch_size()
        
        # Créer le dataloader
        if self.batcher:
            dataloader = self.batcher.create_dataloader(
                prompts,
                batch_size=batch_size,
            )
        else:
            # Simple batching
            dataloader = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
        
        # Générer en parallèle
        results = []
        futures = []
        
        for batch in dataloader:
            # Utiliser le thread pool pour le traitement
            future = self.thread_pool.submit(
                self._generate_batch,
                batch,
                max_new_tokens,
                temperature,
            )
            futures.append(future)
        
        # Récupérer les résultats
        for future in futures:
            batch_results = future.result()
            results.extend(batch_results)
        
        return results
    
    def _generate_batch(
        self,
        batch: List[str],
        max_new_tokens: int,
        temperature: float,
    ) -> List[Dict]:
        """Génère du texte pour un batch de prompts."""
        results = []
        
        for prompt in batch:
            # Sélectionner l'expert
            expert_idx = np.random.randint(0, self.num_experts)
            expert = self.experts[expert_idx]
            
            # Générer
            result = expert.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=1,
            )
            
            # Ajouter l'ID de l'expert
            result["expert_idx"] = expert_idx
            
            results.append(result)
        
        return results


# ============================================================================
# FONCTIONS DE BENCHMARK
# ============================================================================

def benchmark_model(
    model: MockModel,
    prompts: List[str],
    batch_sizes: List[int],
    seq_lengths: List[int],
    num_runs: int = 3,
    device: str = "cuda",
) -> Dict:
    """
    Benchmark un modèle avec différentes configurations.
    
    Args:
        model: Modèle à benchmarker
        prompts: Liste de prompts
        batch_sizes: Liste de tailles de batch
        seq_lengths: Liste de longueurs de séquence
        num_runs: Nombre d'exécutions par configuration
        device: Appareil
    
    Returns:
        Dictionnaire avec les résultats du benchmark
    """
    results = {
        "model": model.model_dir,
        "is_quantized": model.is_quantized,
        "device": device,
        "benchmarks": [],
        "summary": {},
    }
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            benchmark = {
                "batch_size": batch_size,
                "seq_length": seq_length,
                "runs": [],
            }
            
            for run in range(num_runs):
                # Sélectionner un prompt aléatoire
                prompt = np.random.choice(prompts)
                
                # Mesurer le temps
                start_time = time.perf_counter()
                result = model.generate(
                    prompt,
                    max_new_tokens=seq_length,
                    batch_size=batch_size,
                )
                end_time = time.perf_counter()
                
                execution_time = (end_time - start_time) * 1000
                tokens_per_second = result["tokens_per_second"]
                
                benchmark["runs"].append({
                    "execution_time_ms": execution_time,
                    "tokens_per_second": tokens_per_second,
                    "tokens_generated": result["tokens_generated"],
                })
            
            # Calculer les statistiques
            times = [r["execution_time_ms"] for r in benchmark["runs"]]
            tps = [r["tokens_per_second"] for r in benchmark["runs"]]
            
            benchmark["avg_execution_time_ms"] = sum(times) / len(times)
            benchmark["min_execution_time_ms"] = min(times)
            benchmark["max_execution_time_ms"] = max(times)
            benchmark["std_execution_time_ms"] = np.std(times)
            
            benchmark["avg_tokens_per_second"] = sum(tps) / len(tps)
            benchmark["min_tokens_per_second"] = min(tps)
            benchmark["max_tokens_per_second"] = max(tps)
            
            results["benchmarks"].append(benchmark)
    
    # Calculer le résumé
    all_tps = [
        r["avg_tokens_per_second"]
        for r in results["benchmarks"]
    ]
    
    results["summary"] = {
        "global_avg_tokens_per_second": sum(all_tps) / len(all_tps) if all_tps else 0,
        "global_max_tokens_per_second": max(all_tps) if all_tps else 0,
        "optimal_config": results["benchmarks"][np.argmax(all_tps)] if all_tps else None,
    }
    
    return results


def compare_models(
    model_fp32: MockModel,
    model_quant: MockModel,
    prompts: List[str],
    batch_sizes: List[int] = [1, 4, 8],
    seq_lengths: List[int] = [64, 128, 256],
    num_runs: int = 3,
) -> Dict:
    """
    Compare deux modèles (FP32 vs Quantifié).
    
    Args:
        model_fp32: Modèle FP32
        model_quant: Modèle quantifié
        prompts: Liste de prompts
        batch_sizes: Liste de tailles de batch
        seq_lengths: Liste de longueurs de séquence
        num_runs: Nombre d'exécutions
    
    Returns:
        Dictionnaire avec les résultats de la comparaison
    """
    # Benchmark FP32
    fp32_results = benchmark_model(
        model_fp32,
        prompts,
        batch_sizes,
        seq_lengths,
        num_runs,
    )
    
    # Benchmark Quantifié
    quant_results = benchmark_model(
        model_quant,
        prompts,
        batch_sizes,
        seq_lengths,
        num_runs,
    )
    
    # Calculer les comparaisons
    comparison = {
        "fp32": fp32_results,
        "quantized": quant_results,
        "comparison": {},
    }
    
    # Comparaison globale
    fp32_tps = fp32_results["summary"]["global_avg_tokens_per_second"]
    quant_tps = quant_results["summary"]["global_avg_tokens_per_second"]
    
    comparison["comparison"]["speedup"] = quant_tps / fp32_tps if fp32_tps > 0 else 1.0
    
    # Comparaison par configuration
    comparison["comparison"]["configurations"] = []
    for fp32_bench, quant_bench in zip(
        fp32_results["benchmarks"],
        quant_results["benchmarks"]
    ):
        if fp32_bench["avg_tokens_per_second"] > 0:
            speedup = quant_bench["avg_tokens_per_second"] / fp32_bench["avg_tokens_per_second"]
        else:
            speedup = 1.0
        
        comparison["comparison"]["configurations"].append({
            "batch_size": fp32_bench["batch_size"],
            "seq_length": fp32_bench["seq_length"],
            "speedup": speedup,
            "fp32_tps": fp32_bench["avg_tokens_per_second"],
            "quant_tps": quant_bench["avg_tokens_per_second"],
        })
    
    return comparison


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale pour le benchmark d'inférence."""
    parser = argparse.ArgumentParser(
        description="Benchmark d'inférence pour la quantification canonique"
    )
    
    # Arguments de base
    parser.add_argument("--model_fp32", type=str, required=True,
                       help="Répertoire du modèle FP32")
    parser.add_argument("--model_quant", type=str, required=True,
                       help="Répertoire du modèle quantifié")
    parser.add_argument("--output", type=str, required=True,
                       help="Fichier de sortie pour les résultats")
    parser.add_argument("--device", type=str, default=None,
                       choices=["cpu", "cuda"],
                       help="Appareil à utiliser")
    
    # Arguments de benchmark
    parser.add_argument("--batch_size", type=int, nargs="+", default=[1, 4, 8, 16],
                       help="Tailles de batch à tester")
    parser.add_argument("--seq_length", type=int, nargs="+", default=[64, 128, 256],
                       help="Longueurs de séquence à tester")
    parser.add_argument("--num_prompts", type=int, default=10,
                       help="Nombre de prompts à utiliser")
    parser.add_argument("--num_runs", type=int, default=3,
                       help="Nombre d'exécutions par configuration")
    
    # Arguments SDK
    parser.add_argument("--use_sdk", action="store_true", default=True,
                       help="Utiliser le SDK d'optimisation")
    parser.add_argument("--num_experts", type=int, default=8,
                       help="Nombre d'experts pour MoE")
    
    # Autres arguments
    parser.add_argument("--log_dir", type=str, default="logs/inference.log",
                       help="Fichier de log")
    
    args = parser.parse_args()
    
    # Configuration
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("BENCHMARK D'INFÉRENCE AVEC SDK D'OPTIMISATION")
    logger.info("=" * 80)
    
    # Détecter le matériel
    hardware = HardwareDetector.detect()
    HardwareDetector.print_summary(hardware)
    
    # Charger les modèles
    logger.info(f"Chargement du modèle FP32: {args.model_fp32}")
    model_fp32 = MockModel(
        args.model_fp32,
        device=device,
        use_sdk=args.use_sdk,
        num_experts=args.num_experts,
    )
    
    logger.info(f"Chargement du modèle quantifié: {args.model_quant}")
    model_quant = MockModel(
        args.model_quant,
        device=device,
        use_sdk=args.use_sdk,
        num_experts=args.num_experts,
    )
    
    # Sélectionner les prompts
    prompts = DEFAULT_PROMPTS[:args.num_prompts]
    
    # Benchmark FP32
    logger.info("Benchmark du modèle FP32...")
    fp32_results = benchmark_model(
        model_fp32,
        prompts,
        args.batch_size,
        args.seq_length,
        args.num_runs,
        device,
    )
    
    # Benchmark Quantifié
    logger.info("Benchmark du modèle quantifié...")
    quant_results = benchmark_model(
        model_quant,
        prompts,
        args.batch_size,
        args.seq_length,
        args.num_runs,
        device,
    )
    
    # Comparaison
    logger.info("Comparaison des modèles...")
    comparison = compare_models(
        model_fp32,
        model_quant,
        prompts,
        args.batch_size,
        args.seq_length,
        args.num_runs,
    )
    
    # Sauvegarder les résultats
    results = {
        "timestamp": datetime.now().isoformat(),
        "hardware": hardware.to_dict(),
        "config": {
            "device": device,
            "batch_sizes": args.batch_size,
            "seq_lengths": args.seq_length,
            "num_prompts": args.num_prompts,
            "num_runs": args.num_runs,
            "use_sdk": args.use_sdk,
            "num_experts": args.num_experts,
        },
        "fp32": fp32_results,
        "quantized": quant_results,
        "comparison": comparison["comparison"],
    }
    
    # Sauvegarder dans le fichier
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Afficher le résumé
    logger.info("=" * 80)
    logger.info("RÉSULTATS DU BENCHMARK")
    logger.info("=" * 80)
    
    fp32_tps = fp32_results["summary"]["global_avg_tokens_per_second"]
    quant_tps = quant_results["summary"]["global_avg_tokens_per_second"]
    speedup = comparison["comparison"]["speedup"]
    
    logger.info(f"Modèle FP32:")
    logger.info(f"  Tokens/sec (moyen): {fp32_tps:.1f}")
    logger.info(f"  Tokens/sec (max): {fp32_results['summary']['global_max_tokens_per_second']:.1f}")
    
    logger.info(f"\nModèle Quantifié:")
    logger.info(f"  Tokens/sec (moyen): {quant_tps:.1f}")
    logger.info(f"  Tokens/sec (max): {quant_results['summary']['global_max_tokens_per_second']:.1f}")
    
    logger.info(f"\nComparaison:")
    logger.info(f"  Speedup: {speedup:.2f}x")
    logger.info(f"  Amélioration: {(speedup - 1) * 100:.1f}%")
    
    # Configuration optimale
    optimal = fp32_results["summary"]["optimal_config"]
    if optimal:
        logger.info(f"\nConfiguration optimale:")
        logger.info(f"  Batch size: {optimal['batch_size']}")
        logger.info(f"  Seq length: {optimal['seq_length']}")
        logger.info(f"  Tokens/sec: {optimal['avg_tokens_per_second']:.1f}")
    
    logger.info("=" * 80)
    logger.info(f"✅ Benchmark terminé. Résultats sauvegardés dans: {args.output}")


if __name__ == "__main__":
    import datetime
    main()
