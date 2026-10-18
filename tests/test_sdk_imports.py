"""Le SDK et les scripts s'importent sans erreur ; le script 02 utilise le SDK."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("statement", [
    "from apourvoir.quantization import quantifier_tensor_canonique",
    "from apourvoir.quantization import déquantifier_tensor_canonique, CanonicalQuantizer",
    "from apourvoir.babel import BabelSpace",
    "from apourvoir.optimizer import HardwareDetector",
    "import apourvoir",
])
def test_import_in_fresh_interpreter(statement):
    proc = subprocess.run([sys.executable, "-c", statement], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("script", [
    "02_quantize_canonical.py", "benchmark_real.py", "compare_baselines.py", "validate_quantization.py",
])
def test_scripts_import(script):
    spec = importlib.util.spec_from_file_location(script[:-3].lstrip("0123456789_"), ROOT / "scripts" / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


def test_script_02_does_not_redefine_sdk_functions():
    tree = ast.parse((ROOT / "scripts" / "02_quantize_canonical.py").read_text(encoding="utf-8"))
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not defined & {"quantifier_tensor_canonique", "déquantifier_tensor_canonique",
                          "coordonnées_sphériques", "projeter_sur_tore", "quantifier_angles"}


def test_script_02_end_to_end(tmp_path):
    import torch
    from safetensors.torch import save_file
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    g = torch.Generator().manual_seed(0)
    weights = {"layer.weight": torch.randn(64, 300, generator=g), "layer.bias": torch.randn(64, generator=g)}
    save_file(weights, str(model_dir / "model.safetensors"))
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "02_quantize_canonical.py"),
                           "--input", str(model_dir), "--output", str(out)],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert proc.returncode == 0, proc.stderr
    assert "[MESURÉ]" in proc.stdout

    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("q02", ROOT / "scripts" / "02_quantize_canonical.py")
    q02 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q02)
    state = q02.load_quantized(str(out))
    assert torch.equal(state["layer.bias"], weights["layer.bias"])
    err = (state["layer.weight"] - weights["layer.weight"]).norm() / weights["layer.weight"].norm()
    assert err < 0.02
