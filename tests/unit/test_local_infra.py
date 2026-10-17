"""local-multimodal-infra as a first-tier service: settings, catalog,
client and the providers it creates."""

import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from astrbot.core.local_infra import catalog as catalog_module
from astrbot.core.local_infra import config as infra_config
from astrbot.core.local_infra.catalog import InfraCatalog, realtime_asr_model
from astrbot.core.local_infra.client import LocalInfraClient, LocalInfraError

MODELS = [
    {
        "id": "sensevoice",
        "name": "SenseVoice",
        "task_kinds": ["asr.transcribe"],
        "enabled": True,
        "downloaded": True,
    },
    {
        "id": "qwen3-tts",
        "task_kinds": ["tts.synthesize"],
        "enabled": True,
        "downloaded": True,
    },
    {
        "id": "indextts",
        "task_kinds": ["tts.synthesize"],
        "enabled": False,
        "downloaded": True,
    },
    {
        "id": "e5",
        "task_kinds": ["text.embed"],
        "enabled": True,
        "downloaded": False,
    },
    {
        "id": "guard",
        "task_kinds": ["text.moderate"],
        "enabled": True,
        "downloaded": True,
    },
    {
        "id": "voice-cascade",
        "task_kinds": ["voice.realtime"],
        "enabled": True,
        "downloaded": True,
    },
]


class FakeInfra:
    """The routes of a local-multimodal-infra server the client uses."""

    def __init__(self) -> None:
        self.models = MODELS
        self.rpc: list[dict] = []
        self.uploads: dict[str, bytes] = {}
        self.auth: list[str | None] = []
        self.embeddings: list[dict] = []
        self.kind = ""

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/models", self.list_models)
        app.router.add_post("/rpc/infer", self.handle_rpc)
        app.router.add_post("/files/upload/{task}/{slot}", self.upload)
        app.router.add_get("/files/out/{name}", self.download)
        app.router.add_post("/v1/embeddings", self.embed)
        app.router.add_post("/v1/rerank", self.rerank)
        return app

    async def list_models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": self.models})

    async def handle_rpc(self, request: web.Request) -> web.Response:
        self.auth.append(request.headers.get("Authorization"))
        body = await request.json()
        self.rpc.append(body)
        method, params = body["method"], body["params"]
        if method == "create_task":
            self.kind = params["task_kind"]
            uploads = [
                {
                    "role": f["role"],
                    "upload_url": f"/files/upload/t1/{f['role']}?sig=x",
                }
                for f in params["files"]
            ]
            result = {"task_id": "t1", "uploads": uploads}
        elif method in ("start_task", "run_task"):
            kind = params.get("task_kind", self.kind)
            if kind == "asr.transcribe":
                output = {"type": "asr_transcription", "text": "你好"}
            elif kind == "text.embed":
                output = {"type": "text_embeddings", "embeddings": [[9.0, 9.0]]}
            elif kind == "text.rerank":
                output = {
                    "type": "text_rerank",
                    "results": [{"index": 0, "relevance_score": 0.5}],
                }
            elif kind == "chat.complete" and "fail" in json.dumps(params):
                return web.json_response(
                    {"jsonrpc": "2.0", "id": body["id"], "error": {"message": "boom"}}
                )
            else:
                output = {
                    "type": "tts_audio",
                    "audio": {"url": "/files/out/a.wav", "mime": "audio/wav"},
                }
            result = {"task_id": "t1", "state": "succeeded", "output": output}
        else:
            raise AssertionError(method)
        return web.json_response({"jsonrpc": "2.0", "id": body["id"], "result": result})

    async def upload(self, request: web.Request) -> web.Response:
        self.uploads[request.match_info["slot"]] = await request.read()
        return web.json_response({"ok": True})

    async def download(self, request: web.Request) -> web.Response:
        return web.Response(body=b"RIFFwav", content_type="audio/wav")

    async def embed(self, request: web.Request) -> web.Response:
        body = await request.json()
        if "model" not in body:  # as the real server: `model` is required
            return web.Response(status=422, text="missing field `model`")
        self.embeddings.append(body)
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        data = [
            {"index": i, "embedding": [float(i), 1.0, 2.0]}
            for i in reversed(range(len(inputs)))
        ]
        return web.json_response({"data": data})

    async def rerank(self, request: web.Request) -> web.Response:
        body = await request.json()
        if "model" not in body:
            return web.Response(status=422, text="missing field `model`")
        return web.json_response(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ][: body.get("top_n") or 2]
            }
        )


@pytest_asyncio.fixture
async def infra():
    fake = FakeInfra()
    server = TestServer(fake.app())
    await server.start_server()
    fake.base_url = str(server.make_url("")).rstrip("/")
    yield fake
    await server.close()


def settings(base_url="http://127.0.0.1:17890", **services):
    raw = {"enable": True, "base_url": base_url, "token": "tok", "services": services}
    return infra_config.normalize_local_infra(raw)


# ------------------------------------------------------------ settings


def test_settings_are_normalized():
    s = infra_config.normalize_local_infra(
        {
            "enable": "yes",
            "base_url": " http://h:1/ ",
            "timeout": "x",
            "services": {"tts": {"model": " m ", "enable": False}, "bogus": {}},
        }
    )
    assert s["enable"] is False  # not a bool: the default
    assert s["base_url"] == "http://h:1"
    assert s["timeout"] == 120
    assert s["services"]["tts"] == {
        "enable": False,
        "model": "m",
        "reference_audio": "",
        "reference_text": "",
    }
    assert "bogus" not in s["services"]
    voice = infra_config.normalize_local_infra(
        {"services": {"realtime_voice": {"asr_model": " fun-asr-nano "}}}
    )["services"]["realtime_voice"]
    assert voice == {"enable": True, "asr_model": "fun-asr-nano"}
    assert infra_config.normalize_local_infra(None) == (
        infra_config.default_local_infra()
    )


def test_urls_follow_the_base():
    assert infra_config.realtime_url({"base_url": "http://h:17890/v1"}) == (
        "ws://h:17890/v1/realtime"
    )
    assert infra_config.realtime_url({"base_url": "https://h/infra"}) == (
        "wss://h/infra/v1/realtime"
    )
    assert infra_config.base_from_url("wss://h/infra/v1/realtime") == "https://h/infra"
    assert infra_config.base_from_url("http://h:1/v1/moderations") == "http://h:1"
    assert infra_config.base_from_url("") == ""


# ------------------------------------------------------------ catalog


@pytest.mark.asyncio
async def test_services_resolve_to_enabled_models(infra):
    catalog = InfraCatalog()
    s = settings(infra.base_url)
    await catalog.refresh(s)
    assert catalog.known and not catalog.error

    stt = catalog.resolve("stt", s)
    assert stt["available"] and stt["model"] == "sensevoice"
    # The disabled IndexTTS is no candidate.
    assert catalog.resolve("tts", s)["candidates"] == ["qwen3-tts"]
    # Enabled but its files are missing.
    embedding = catalog.resolve("embedding", s)
    assert not embedding["available"] and embedding["reason"] == "no_model"
    assert not catalog.resolve("rerank", s)["available"]
    assert catalog.resolve("moderation", s)["available"]
    assert catalog.resolve("realtime_voice", s)["model"] == "voice-cascade"

    chosen = settings(infra.base_url, tts={"model": "indextts"})
    tts = catalog.resolve("tts", chosen)
    assert not tts["available"] and tts["reason"] == "model_unavailable"

    off = settings(infra.base_url, stt={"enable": False})
    assert catalog.resolve("stt", off)["reason"] == "off"


@pytest.mark.asyncio
async def test_realtime_voice_hears_with_a_usable_chosen_asr_model(infra, monkeypatch):
    catalog = InfraCatalog()
    monkeypatch.setattr(catalog_module, "catalog", catalog)

    def chosen(model):
        return settings(infra.base_url, realtime_voice={"asr_model": model})

    # Not known yet: as chosen, the server decides.
    assert realtime_asr_model(chosen("sensevoice")) == "sensevoice"
    await catalog.refresh(settings(infra.base_url))
    assert realtime_asr_model(chosen("sensevoice")) == "sensevoice"
    assert realtime_asr_model(chosen("")) == ""
    # Not an ASR model, or not on the server: the server's default.
    assert realtime_asr_model(chosen("qwen3-tts")) == ""
    assert realtime_asr_model(chosen("gone")) == ""


@pytest.mark.asyncio
async def test_an_unreachable_server_leaves_services_to_try(infra):
    catalog = InfraCatalog()
    s = settings("http://127.0.0.1:9")
    await catalog.refresh(s)
    assert catalog.error and not catalog.known
    state = catalog.resolve("tts", settings("http://127.0.0.1:9", tts={"model": "x"}))
    assert state["available"] and state["model"] == "x"
    assert state["reason"] == "unknown"


@pytest.mark.asyncio
async def test_an_old_server_without_task_kinds_is_unknown(infra):
    infra.models = [{"id": "sensevoice"}]
    catalog = InfraCatalog()
    await catalog.refresh(settings(infra.base_url))
    assert catalog.models and not catalog.known
    assert catalog.resolve("stt", settings(infra.base_url))["reason"] == "unknown"


@pytest.mark.asyncio
async def test_listeners_hear_only_changes(infra):
    catalog = InfraCatalog()
    heard = []

    async def listener():
        heard.append(True)

    catalog.add_listener(listener)
    s = settings(infra.base_url)
    await catalog.refresh(s)
    await catalog.refresh(s)
    assert len(heard) == 1
    infra.models = MODELS[:1]
    await catalog.refresh(s)
    assert len(heard) == 2
    # Switched off: the models are forgotten.
    await catalog.refresh(infra_config.normalize_local_infra({}))
    assert catalog.models is None and len(heard) == 3


# ------------------------------------------------------------ client


@pytest.mark.asyncio
async def test_a_task_with_files_is_uploaded_started_and_downloaded(infra, tmp_path):
    client = LocalInfraClient(infra.base_url, token="tok", models={"tts": "qwen3-tts"})
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"REF")

    path = await client.synthesize(
        "你好", reference_audio=str(ref), reference_text="参考", dest_dir=tmp_path
    )

    assert open(path, "rb").read() == b"RIFFwav" and path.endswith(".wav")
    create, start = infra.rpc
    assert create["method"] == "create_task"
    assert create["params"]["task_kind"] == "tts.synthesize"
    assert create["params"]["model"] == "qwen3-tts"
    assert create["params"]["params"] == {"text": "你好", "reference_text": "参考"}
    assert create["params"]["files"] == [
        {
            "name": "ref.wav",
            "mime": "audio/wav",
            "role": "reference_audio",
            "required": True,
        }
    ]
    assert infra.uploads == {"reference_audio": b"REF"}
    assert start["params"]["task_id"] == "t1" and start["params"]["wait"] is True
    assert set(infra.auth) == {"Bearer tok"}


@pytest.mark.asyncio
async def test_a_task_without_files_runs_in_one_call(infra):
    client = LocalInfraClient(infra.base_url)
    output = await client.run_task("tts.synthesize", params={"text": "hi"})
    assert output["type"] == "tts_audio"
    assert [call["method"] for call in infra.rpc] == ["run_task"]
    assert "model" not in infra.rpc[0]["params"]
    assert infra.auth == [None]


@pytest.mark.asyncio
async def test_transcribe_sends_the_audio(infra, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"MP3")
    client = LocalInfraClient(infra.base_url, models={"stt": "sensevoice"})
    output = await client.transcribe(str(audio))
    assert output["text"] == "你好"
    assert infra.uploads == {"audio": b"MP3"}
    assert infra.rpc[0]["params"]["files"][0]["mime"] == "audio/mpeg"


@pytest.mark.asyncio
async def test_a_failed_rpc_raises(infra):
    client = LocalInfraClient(infra.base_url)
    with pytest.raises(LocalInfraError, match="boom"):
        await client.run_task("chat.complete", params={"fail": True})


@pytest.mark.asyncio
async def test_embeddings_and_rerank_use_the_openai_routes(infra):
    client = LocalInfraClient(infra.base_url, models={"embedding": "e5"})
    vectors = await client.embed(["a", "b"], input_type="query")
    # In input order whatever order the server answers in.
    assert vectors == [[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]]
    assert infra.embeddings == [
        {"input": ["a", "b"], "model": "e5", "input_type": "query"}
    ]
    results = await client.rerank("q", ["x", "y"], top_n=1, model="minilm")
    assert results == [{"index": 1, "relevance_score": 0.9}]


@pytest.mark.asyncio
async def test_without_a_known_model_the_server_picks_one_in_a_task(infra):
    # The OpenAI routes need a model; a task lets the server choose.
    client = LocalInfraClient(infra.base_url)
    assert await client.embed(["a"], input_type="query") == [[9.0, 9.0]]
    assert await client.rerank("q", ["x"], top_n=1) == [
        {"index": 0, "relevance_score": 0.5}
    ]
    calls = [(call["method"], call["params"]) for call in infra.rpc]
    assert calls == [
        (
            "run_task",
            {
                "task_kind": "text.embed",
                "params": {"input": ["a"], "input_type": "query"},
            },
        ),
        (
            "run_task",
            {
                "task_kind": "text.rerank",
                "params": {"query": "q", "documents": ["x"], "top_n": 1},
            },
        ),
    ]
    assert infra.embeddings == []


@pytest.mark.asyncio
async def test_a_settings_client_asks_for_the_resolved_models(infra, monkeypatch):
    from astrbot.core.local_infra import catalog as catalog_module

    catalog = InfraCatalog()
    monkeypatch.setattr(catalog_module, "catalog", catalog)
    s = settings(infra.base_url)
    await catalog.refresh(s)
    client = LocalInfraClient.from_settings(s)
    # Automatic choices included, so the OpenAI routes get a model.
    assert client.models["stt"] == "sensevoice"
    assert client.models["tts"] == "qwen3-tts"
    assert "embedding" not in client.models  # no usable embedding model


@pytest.mark.asyncio
async def test_http_errors_raise(infra):
    client = LocalInfraClient(infra.base_url + "/nowhere")
    with pytest.raises(LocalInfraError, match="HTTP 404"):
        await client.list_models()


@pytest.mark.asyncio
async def test_a_local_file_ref_is_used_in_place(tmp_path):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"W")
    client = LocalInfraClient("http://127.0.0.1:9")
    assert await client.download({"url": wav.as_uri()}) == str(wav)
    with pytest.raises(LocalInfraError):
        await client.download({"mime": "audio/wav"})


def test_get_local_infra_follows_the_setting(monkeypatch):
    from astrbot.core import local_infra

    monkeypatch.setattr(
        local_infra, "local_infra_settings", lambda: settings("http://h:1")
    )
    client = local_infra.get_local_infra()
    assert client is not None and client.base_url == "http://h:1"
    assert client.token == "tok"
    monkeypatch.setattr(
        local_infra,
        "local_infra_settings",
        lambda: infra_config.normalize_local_infra({"enable": False}),
    )
    assert local_infra.get_local_infra() is None


# ------------------------------------------------------------ providers


@pytest.mark.asyncio
async def test_the_provider_manager_follows_the_services(infra, monkeypatch):
    from astrbot.core.local_infra import catalog as catalog_module
    from astrbot.core.provider.manager import ProviderManager
    from astrbot.core.provider.provider import STTProvider, TTSProvider

    catalog = InfraCatalog()
    monkeypatch.setattr(catalog_module, "catalog", catalog)
    conf = {"local_infra": settings(infra.base_url, tts={"model": "qwen3-tts"})}
    monkeypatch.setattr(
        infra_config,
        "local_infra_settings",
        lambda config=None: infra_config.normalize_local_infra(conf["local_infra"]),
    )
    from astrbot.core.provider import manager as manager_module

    manager = ProviderManager.__new__(ProviderManager)
    manager.__dict__.update(
        {
            "reload_lock": __import__("asyncio").Lock(),
            "provider_insts": [],
            "stt_provider_insts": [],
            "tts_provider_insts": [],
            "embedding_provider_insts": [],
            "rerank_provider_insts": [],
            "inst_map": {},
            "provider_settings": {},
            "provider_stt_settings": {"provider_id": "local_infra/stt"},
            "provider_tts_settings": {},
            "provider_sources_config": [],
            "curr_provider_inst": None,
            "curr_stt_provider_inst": None,
            "curr_tts_provider_inst": None,
            "default_chat_provider_id": "",
            "local_infra_configs": {},
        }
    )
    assert manager_module  # imported for the providers' registration
    await catalog.refresh(conf["local_infra"])
    await manager.sync_local_infra()

    # No usable embedding (files missing) and no rerank model: no provider.
    assert set(manager.local_infra_configs) == {"local_infra/stt", "local_infra/tts"}
    assert isinstance(manager.inst_map["local_infra/stt"], STTProvider)
    assert isinstance(manager.inst_map["local_infra/tts"], TTSProvider)
    # The STT chosen as default is selected once it exists.
    assert manager.curr_stt_provider_inst is manager.inst_map["local_infra/stt"]
    assert manager.get_provider_config_by_id("local_infra/tts")["model"] == (
        "qwen3-tts"
    )

    # Switching a service off drops its provider; the others stay.
    tts = manager.inst_map["local_infra/tts"]
    stt = manager.inst_map["local_infra/stt"]
    conf["local_infra"]["services"]["stt"]["enable"] = False
    await manager.sync_local_infra()
    assert set(manager.local_infra_configs) == {"local_infra/tts"}
    assert manager.inst_map["local_infra/tts"] is tts
    assert stt not in manager.stt_provider_insts

    # A model that gets enabled on the server brings its provider.
    infra.models = [*MODELS, {**MODELS[3], "downloaded": True}][1:]
    infra.models = [m for m in infra.models if m["id"] != "e5" or m["downloaded"]]
    await catalog.refresh(conf["local_infra"])
    await manager.sync_local_infra()
    assert "local_infra/embedding" in manager.local_infra_configs
    assert manager.embedding_provider_insts

    # The server switched off: every provider goes.
    conf["local_infra"]["enable"] = False
    await catalog.refresh(conf["local_infra"])
    await manager.sync_local_infra()
    assert manager.local_infra_configs == {} and manager.inst_map == {}
    assert not manager.tts_provider_insts and not manager.embedding_provider_insts


@pytest.mark.asyncio
async def test_providers_call_the_server(infra, monkeypatch, tmp_path):
    from astrbot.core.local_infra import catalog as catalog_module
    from astrbot.core.provider.sources import local_infra_source as source

    # Each call asks for the model its service resolves to (here: chosen).
    monkeypatch.setattr(catalog_module, "catalog", InfraCatalog())
    chosen = settings(
        infra.base_url,
        tts={"model": "qwen3-tts"},
        embedding={"model": "e5"},
        rerank={"model": "minilm"},
    )
    monkeypatch.setattr(source, "local_infra_settings", lambda: chosen)
    monkeypatch.setattr(
        "astrbot.core.local_infra.client.LocalInfraClient.download",
        _fake_download(tmp_path),
    )
    tts = source.LocalInfraTTSProvider(
        {"id": "local_infra/tts", "type": "local_infra_tts"}, {}
    )
    path = await tts.get_audio("你好")
    assert open(path, "rb").read() == b"RIFFwav"
    assert infra.rpc[-1]["params"]["model"] == "qwen3-tts"

    embedding = source.LocalInfraEmbeddingProvider(
        {"id": "local_infra/embedding", "type": "local_infra_embedding"}, {}
    )
    await embedding.initialize()
    assert embedding.get_dim() == 3

    rerank = source.LocalInfraRerankProvider(
        {"id": "local_infra/rerank", "type": "local_infra_rerank"}, {}
    )
    results = await rerank.rerank("q", ["a", "b"])
    assert [(r.index, r.relevance_score) for r in results] == [(1, 0.9), (0, 0.1)]


def _fake_download(tmp_path):
    original = LocalInfraClient.download

    async def download(self, file_ref, dest_dir=None, timeout=None):
        return await original(self, file_ref, dest_dir or tmp_path, timeout)

    return download


# ------------------------------------------------------------ WebUI


class _Config(dict):
    saved = 0

    def save_config(self):
        self.saved += 1


@pytest.mark.asyncio
async def test_the_webui_saves_and_reports_the_services(infra, monkeypatch):
    from types import SimpleNamespace

    from astrbot.core.local_infra import catalog as catalog_module
    from astrbot.dashboard.services import local_infra_service as service_module
    from astrbot.dashboard.services.local_infra_service import (
        LocalInfraService,
        LocalInfraServiceError,
    )

    catalog = InfraCatalog()
    monkeypatch.setattr(catalog_module, "catalog", catalog)
    monkeypatch.setattr(service_module, "catalog", catalog)
    synced = []

    async def sync():
        synced.append(True)

    config = _Config()
    lifecycle = SimpleNamespace(
        astrbot_config=config,
        provider_manager=SimpleNamespace(inst_map={}, sync_local_infra=sync),
    )
    service = LocalInfraService(lifecycle)

    state = service.state()
    assert state["settings"]["enable"] is False
    assert state["services"]["stt"]["reason"] == "off"
    assert state["services"]["stt"]["provider_id"] == "local_infra/stt"

    with pytest.raises(LocalInfraServiceError):
        await service.save({"enable": True, "base_url": "ftp://x"})

    state = await service.save(
        {"enable": True, "base_url": infra.base_url + "/", "services": {}}
    )
    assert config.saved == 1 and synced == [True]
    assert config["local_infra"]["base_url"] == infra.base_url
    assert state["catalog"]["known"] is True
    assert state["services"]["tts"]["model"] == "qwen3-tts"
    assert state["realtime_url"].startswith("ws://")

    # A page opened before the embedding model was pinned keeps it.
    config["local_infra"]["services"]["embedding"]["model"] = "e5"
    await service.save({"enable": True, "base_url": infra.base_url, "services": {}})
    assert config["local_infra"]["services"]["embedding"]["model"] == "e5"

    probed = await service.probe({"base_url": infra.base_url})
    assert probed["known"] and len(probed["models"]) == len(MODELS)
    with pytest.raises(LocalInfraServiceError):
        await service.probe({"base_url": "http://127.0.0.1:9"})


@pytest.mark.asyncio
async def test_the_automatic_embedding_model_is_kept(infra, monkeypatch):
    """Knowledge bases keep one model's vectors: the first automatic choice
    is stored, so a model enabled later on the server does not replace it."""
    from astrbot.core.local_infra import catalog as catalog_module
    from astrbot.core.provider.manager import ProviderManager

    catalog = InfraCatalog()
    monkeypatch.setattr(catalog_module, "catalog", catalog)
    infra.models = [{**m, "downloaded": True} for m in MODELS]
    config = _Config(local_infra=settings(infra.base_url))
    manager = ProviderManager.__new__(ProviderManager)
    manager.acm = SimpleNamespace(default_conf=config)

    await catalog.refresh(config["local_infra"])
    manager._pin_local_infra_embedding_model()
    assert config["local_infra"]["services"]["embedding"]["model"] == "e5"
    assert config.saved == 1

    # A model listed before it on the server changes nothing now.
    infra.models = [
        {"id": "a-embed", "task_kinds": ["text.embed"], "enabled": True},
        *infra.models,
    ]
    await catalog.refresh(config["local_infra"])
    manager._pin_local_infra_embedding_model()
    state = catalog.resolve(
        "embedding", infra_config.normalize_local_infra(config["local_infra"])
    )
    assert state["model"] == "e5" and config.saved == 1


@pytest.mark.asyncio
async def test_moderation_needs_a_text_model(infra):
    # The server checks text (and OCR text of images) with a text model and
    # fails without one, so an image classifier alone does not serve it.
    infra.models = [
        {"id": "nsfw", "task_kinds": ["image.nsfw"], "enabled": True},
    ]
    catalog = InfraCatalog()
    s = settings(infra.base_url)
    await catalog.refresh(s)
    assert not catalog.resolve("moderation", s)["available"]


@pytest.mark.asyncio
async def test_a_chosen_model_the_server_dropped_is_still_asked_for(infra, monkeypatch):
    # Better the server's error than another model's embeddings.
    from astrbot.core.local_infra import catalog as catalog_module

    catalog = InfraCatalog()
    monkeypatch.setattr(catalog_module, "catalog", catalog)
    s = settings(infra.base_url, embedding={"model": "gone"})
    await catalog.refresh(s)
    assert catalog.resolve("embedding", s)["reason"] == "model_unavailable"
    client = LocalInfraClient.from_settings(s)
    assert client.models["embedding"] == "gone"
    await client.embed(["a"])
    assert infra.embeddings == [{"input": ["a"], "model": "gone"}]
    assert not infra.rpc
