"""Tests de la quantisation canonique N-dimensionnelle et des baselines."""

import math

import pytest
import torch

from apourvoir.quantization import (
    CanonicalQuantizer,
    ThalesTriangle,
    cartésiennes,
    coordonnées_sphériques,
    déquantifier_angles,
    déquantifier_tensor_canonique,
    erreur_relative,
    quantifier_angles,
    quantifier_tensor_canonique,
    taille_octets,
)
from apourvoir.quantization.baselines import BASELINES
from apourvoir.quantization.model_bench import fake_quantize_state_dict, get_method, measure_tensor


def randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("dim", [4096, 14336])
def test_high_dimensional_roundtrip_under_5_percent(dim):
    x = randn(100, dim)
    q, meta = quantifier_tensor_canonique(x, n_bits=8)
    y = déquantifier_tensor_canonique(q, meta)
    assert y.shape == x.shape
    assert erreur_relative(x, y) < 0.05
    # Toutes les dimensions sont reconstruites (pas seulement les 3 premières)
    assert erreur_relative(x[:, 3:], y[:, 3:]) < 0.05


@pytest.mark.parametrize("dim", [4096, 14336])
def test_compression_ratio_matches_stored_bytes(dim):
    x = randn(100, dim)
    q, meta = quantifier_tensor_canonique(x, n_bits=8)
    stored = q.numel() * q.element_size() + meta["scales"].numel() * meta["scales"].element_size()
    assert meta["stored_bytes"] == stored == taille_octets(q, meta)
    assert meta["compression_ratio"] == pytest.approx(x.numel() * 4 / stored)
    # 8 bits par composante : au plus 4x vs FP32 (pas de chiffre magique)
    assert 3.9 < meta["compression_ratio"] <= 4.0


@pytest.mark.parametrize("shape", [(7, 5), (10, 10, 3), (4097,), (2, 3, 4, 11), (1, 1)])
def test_arbitrary_shapes(shape):
    x = randn(*shape)
    q, meta = quantifier_tensor_canonique(x, n_bits=8)
    y = déquantifier_tensor_canonique(q, meta)
    assert y.shape == x.shape
    assert erreur_relative(x, y) < 0.05


@pytest.mark.parametrize("n_bits,max_err", [(16, 1e-3), (8, 0.02), (6, 0.05), (4, 0.2), (2, 0.7)])
def test_error_decreases_with_bits(n_bits, max_err):
    x = randn(64, 4096)
    q, meta = quantifier_tensor_canonique(x, n_bits=n_bits)
    assert erreur_relative(x, déquantifier_tensor_canonique(q, meta)) < max_err
    assert meta["bits_per_value"] == pytest.approx(n_bits, rel=0.02)


@pytest.mark.parametrize("r_mode", ["sqrt", "lin"])
def test_r_modes(r_mode):
    x = randn(32, 999)
    q, meta = quantifier_tensor_canonique(x, n_bits=8, r_mode=r_mode)
    assert erreur_relative(x, déquantifier_tensor_canonique(q, meta)) < 0.02


def test_zero_rows_and_bf16():
    x = randn(8, 300)
    x[2] = 0
    q, meta = quantifier_tensor_canonique(x.to(torch.bfloat16), n_bits=8)
    y = déquantifier_tensor_canonique(q, meta)
    assert torch.all(y[2] == 0)
    assert erreur_relative(x.to(torch.bfloat16).float(), y) < 0.02


def test_deterministic():
    x = randn(16, 4096)
    q1, m1 = quantifier_tensor_canonique(x)
    q2, m2 = quantifier_tensor_canonique(x)
    assert torch.equal(q1, q2) and torch.equal(m1["scales"], m2["scales"])


def test_invalid_arguments():
    with pytest.raises(ValueError):
        quantifier_tensor_canonique(randn(4, 4), n_bits=1)
    with pytest.raises(ValueError):
        quantifier_tensor_canonique(randn(4, 4), r_mode="log")
    with pytest.raises(ValueError):
        déquantifier_tensor_canonique(torch.zeros(1), {"method": "torus"})


def test_spherical_coordinates_roundtrip():
    v = randn(1000, 3)
    r, theta, phi = coordonnées_sphériques(v)
    assert torch.allclose(cartésiennes(r, theta, phi), v, atol=1e-5)
    tq, pq = quantifier_angles(theta, phi, 8)
    assert int(tq.max()) < 256 and int(pq.max()) < 256
    # phi = pi doit rester pi (l'ancienne version le ramenait à 0)
    _, phi_pi = déquantifier_angles(torch.tensor([0]), quantifier_angles(torch.tensor([0.0]), torch.tensor([math.pi]), 8)[1], 8)
    assert phi_pi.item() == pytest.approx(math.pi)


def test_canonical_quantizer_layers():
    quantizer = CanonicalQuantizer(n_bits=8)
    x = randn(16, 4096)
    assert [quantizer.bits_for_layer(i) for i in (1, 2, 3, 4)] == [8, 6, 4, 4]
    q, meta = quantizer.quantize(x, layer=2)
    assert meta["n_bits"] == 6
    assert erreur_relative(x, quantizer.dequantize(q, meta)) < 0.05
    assert 3.9 < quantizer.get_compression_ratio(x) <= 4.0


def test_thales_triangle():
    t = ThalesTriangle()
    A, B, C = torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])
    assert t.calculate_angles(A, B, C) == pytest.approx(90.0, abs=1e-4)
    assert t.is_right_triangle(A, B, C)
    assert not t.is_right_triangle(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.5, 0.8, 0.0]))


@pytest.mark.parametrize("name,max_err,bits", [("int8_per_row", 0.02, 8.0), ("q8_0_ref", 0.02, 8.5), ("q4_0_ref", 0.15, 4.5)])
def test_baselines_roundtrip(name, max_err, bits):
    quant, dequant = BASELINES[name]
    x = randn(64, 4096)
    codes, meta = quant(x)
    y = dequant(codes, meta)
    assert y.shape == x.shape
    assert erreur_relative(x, y) < max_err
    assert meta["stored_bytes"] * 8 / x.numel() == pytest.approx(bits, rel=0.01)


def test_baselines_odd_dims():
    x = randn(5, 45)
    for quant, dequant in BASELINES.values():
        codes, meta = quant(x)
        assert dequant(codes, meta).shape == x.shape


def test_model_bench_helpers():
    quant, dequant = get_method("canonical8")
    r = measure_tensor("w", randn(32, 512).to(torch.bfloat16), quant, dequant)
    assert r.quantized and r.rel_error < 0.02 and 0.999 < r.cosine <= 1.0
    bias = measure_tensor("b", randn(512), quant, dequant)
    assert not bias.quantized and bias.stored_bytes == bias.original_bytes
    state = {"w": randn(16, 64), "b": randn(64)}
    fq = fake_quantize_state_dict(state, "q4_0_ref")
    assert torch.equal(fq["b"], state["b"]) and not torch.equal(fq["w"], state["w"])
    with pytest.raises(ValueError):
        get_method("magic")
