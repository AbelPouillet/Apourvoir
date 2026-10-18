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
    options = LaunchOptions(api_host="127.0.0.1", api_port=0, ui_port=0, enable_api=True, enable_ui=True)
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


def test_completions_are_honestly_not_implemented(launcher):
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
