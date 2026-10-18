"""Tests des embeddings Babel : déterministes, sans aléatoire, sémantiquement cohérents."""

import inspect
import re

import pytest
import torch

from apourvoir.babel import PhoneticEncoder, SyntaxEncoder, SyntaxTreeBuilder
from apourvoir.babel.phonetic import phonetic_encoder, features
from apourvoir.babel.semantic import CanonicalProjector, embedder
from apourvoir.babel.syntax import syntax_encoder
from apourvoir.babel.fusion import babel_space


@pytest.mark.parametrize("module", [phonetic_encoder, features, syntax_encoder, embedder, babel_space])
def test_no_random_or_hash_embeddings(module):
    source = inspect.getsource(module)
    assert not re.search(r"torch\.randn|torch\.rand\(|np\.random|\bhash\(", source)


def test_phonetic_deterministic_across_instances():
    a = PhoneticEncoder("fr").get_phonetic_embedding("pouvoir")
    b = PhoneticEncoder("fr").get_phonetic_embedding("pouvoir")
    assert torch.equal(a, b)
    assert a.shape == (64,)
    assert torch.norm(a).item() == pytest.approx(1.0, abs=1e-5)


def test_phonetic_similarity_reflects_sound():
    enc = PhoneticEncoder("fr")
    cos = lambda x, y: float(torch.dot(enc.get_phonetic_embedding(x), enc.get_phonetic_embedding(y)))
    assert cos("bateau", "gâteau") > cos("bateau", "chien")
    assert cos("voir", "savoir") > cos("voir", "intelligence")


def test_phonetic_features_minimal_pair():
    p, b = features.segment_features("p"), features.segment_features("b")
    assert torch.count_nonzero(p != b).item() == 1  # seul le voisement diffère
    assert features.segment_ipa("/tɑ̃/") == ["t", "ɑ~"]


def test_phonetic_unknown_word_is_zero_not_random():
    enc = PhoneticEncoder("fr")
    assert torch.count_nonzero(enc.get_phonetic_embedding("")) == 0


def test_syntax_encoder_deterministic():
    builder = SyntaxTreeBuilder(lang="fr", granularity="word")
    tree = builder.build_tree("pouvoir voir savoir")
    a = SyntaxEncoder().encode_tree(tree)
    b = SyntaxEncoder().encode_tree(builder.build_tree("pouvoir voir savoir"))
    assert torch.allclose(a, b)
    enc = SyntaxEncoder()
    assert torch.dot(enc._get_token_embedding("chien"), enc._get_token_embedding("niche")).item() == pytest.approx(1.0)


def test_canonical_projector_requires_basis_for_high_dim():
    proj = CanonicalProjector()
    with pytest.raises(ValueError):
        proj.project_to_canonical(torch.ones(768))
    x, y, z = torch.eye(5)[:3]
    proj.set_axes(x, y, z)
    v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert torch.allclose(proj.project_to_canonical(v, method="linear"), torch.tensor([1.0, 2.0, 3.0]))
    batch = proj.project_to_canonical(torch.stack([v, v]))
    assert batch.shape == (2, 3)


# ---------------------------------------------------------------------------
# Embeddings sémantiques réels (nécessitent le modèle sentence-transformers,
# téléchargé depuis HuggingFace au premier lancement puis mis en cache)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def semantic():
    pytest.importorskip("sentence_transformers")
    emb = embedder.SemanticEmbedder()
    try:
        emb.encode("test")
    except Exception as e:  # pas de réseau ni de cache
        pytest.skip(f"modèle d'embedding indisponible : {e}")
    return emb


TRIPLETS = [
    ("roi", "reine", "banane"), ("homme", "femme", "voiture"), ("chien", "chat", "montagne"),
    ("voiture", "camion", "pomme"), ("pomme", "poire", "ordinateur"), ("paris", "londres", "fourchette"),
    ("médecin", "hôpital", "guitare"), ("soleil", "lune", "chaussure"), ("livre", "lecture", "tonnerre"),
    ("eau", "rivière", "clavier"), ("voir", "regarder", "manger"), ("savoir", "connaissance", "vélo"),
    ("pouvoir", "puissance", "fromage"), ("lumière", "lampe", "dette"), ("mer", "océan", "crayon"),
    ("piano", "violon", "nuage"), ("père", "mère", "tournevis"), ("hiver", "neige", "téléphone"),
    ("acheter", "vendre", "planète"), ("rouge", "bleu", "parlement"),
]


def test_roi_close_to_reine(semantic):
    assert semantic.similarity("roi", "reine") > semantic.similarity("roi", "banane")


def test_semantic_triplets(semantic):
    ok = sum(semantic.similarity(a, p) > semantic.similarity(a, n) for a, p, n in TRIPLETS)
    # Mesuré : 19/20 avec paraphrase-multilingual-mpnet-base-v2 (échec : "homme/femme/voiture")
    assert ok >= 18


def test_semantic_deterministic(semantic):
    fresh = embedder.SemanticEmbedder(semantic.model_name)
    assert torch.allclose(semantic.encode("pouvoir"), fresh.encode("pouvoir"), atol=1e-6)


def test_babel_space_uses_real_embeddings(semantic):
    space = babel_space.BabelSpace()
    space._semantic_embedder = semantic
    m = space.encode("pouvoir", method="semantic")
    assert m.shape == (3,)
    combined = space.encode("pouvoir")
    assert combined.shape == (64 + 64 + semantic.dim,)
    (theta_q, phi_q), canonical = space.quantize(combined)
    assert 0 <= theta_q < 256 and 0 <= phi_q < 256
