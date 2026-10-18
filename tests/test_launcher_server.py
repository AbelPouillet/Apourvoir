"""Le launcher garde ses serveurs HTTP actifs après l'initialisation, jusqu'à stop()."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from apourvoir.optimizer.launcher import LaunchOptions, OptimizedLauncher


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read().decode("utf-8")


@pytest.fixture
def launcher():
    options = LaunchOptions(api_host="127.0.0.1", api_port=0, ui_port=0, enable_api=True, enable_ui=True,
                            enable_inference=False)
    launcher = OptimizedLauncher(options=options)
    thread = threading.Thread(target=launcher.launch, kwargs={"block": True}, daemon=True)
    thread.start()
    deadline = time.time() + 120
    while len(launcher.servers) < 2 and time.time() < deadline:
        time.sleep(0.1)
    yield launcher
    launcher.stop()
    thread.join(timeout=10)


def test_servers_stay_up_after_launch(launcher):
    api, ui = (s.address for s in launcher.servers)
    time.sleep(1.0)  # launch() a rendu la main au thread d'attente : le serveur doit toujours répondre
    assert _get(api + "/health") == (200, json.dumps({"status": "ok"}))
    assert json.loads(_get(api + "/status")[1])["running"] is True
    assert "<title>Apourvoir</title>" in _get(ui + "/")[1]


def test_completions_without_engine_answer_501(launcher):
    api = launcher.servers[0].address
    req = urllib.request.Request(api + "/v1/completions", data=b'{"prompt": "x"}', method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=5)
    assert err.value.code == 501


def test_stop_shuts_servers_down(launcher):
    api = launcher.servers[0].address
    launcher.stop()
    assert launcher.servers == []
    with pytest.raises(OSError):
        _get(api + "/health")


# ---------------------------------------------------------------------------
# Real inference engine (tiny random Llama from HuggingFace: checks the
# plumbing and OpenAI response format, not text quality)
# ---------------------------------------------------------------------------

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def engine_launcher():
    pytest.importorskip("transformers")
    options = LaunchOptions(model=TINY_MODEL, quant_method="canonical", n_bits=8, api_host="127.0.0.1",
                            api_port=0, enable_ui=False, device="cpu")
    launcher = OptimizedLauncher(options=options)
    try:
        launcher.launch(block=False)
    except OSError as e:  # no network and no cache
        pytest.skip(f"tiny model unavailable: {e}")
    yield launcher
    launcher.stop()


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read())


def test_completion_endpoint_generates(engine_launcher):
    api = engine_launcher.servers[0].address
    code, body = _post(api + "/v1/completions", {"prompt": "Hello", "max_tokens": 5})
    assert code == 200
    assert body["object"] == "text_completion"
    assert isinstance(body["choices"][0]["text"], str)
    assert body["usage"]["completion_tokens"] == 5
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["apourvoir"]["quant_method"] == "canonical"
    # Greedy decoding is deterministic
    assert _post(api + "/v1/completions", {"prompt": "Hello", "max_tokens": 5})[1]["choices"][0]["text"] == \
        body["choices"][0]["text"]


def test_chat_endpoint_generates(engine_launcher):
    api = engine_launcher.servers[0].address
    code, body = _post(api + "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
    assert code == 200 and body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.parametrize("payload", [{"prompt": 3}, {"prompt": "x", "max_tokens": 0}, {"prompt": "x", "stream": True},
                                     {"prompt": "x", "temperature": -1}])
def test_invalid_requests_answer_400(engine_launcher, payload):
    api = engine_launcher.servers[0].address
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(api + "/v1/completions", payload)
    assert err.value.code == 400


def test_status_reports_measured_canonical_stats(engine_launcher):
    inference = json.loads(_get(engine_launcher.servers[0].address + "/status")[1])["inference"]
    assert inference["ready"] and inference["quant_method"] == "canonical"
    assert inference["canonical_weighted_rel_error"] < 0.02


def test_unsupported_quant_method_is_explicit():
    from apourvoir.inference import InferenceEngine
    with pytest.raises(NotImplementedError):
        InferenceEngine(TINY_MODEL, quant_method="awq")


def test_memory_check_refuses_too_large_model(monkeypatch):
    from apourvoir.inference import engine as engine_mod
    monkeypatch.setattr(engine_mod, "_weights_bytes_on_disk", lambda model: 10 ** 15)
    with pytest.raises(engine_mod.InsufficientMemoryError):
        engine_mod.InferenceEngine("huge/model").load()
