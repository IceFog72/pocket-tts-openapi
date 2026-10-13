"""Cancellation, model ownership, encoder errors and shared cache regressions."""
import asyncio
import os
import shutil
import subprocess
import threading
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

from pocket_tts_server import api, audio
from pocket_tts_server.cache import CacheManager
from pocket_tts_server.model_manager import ModelManager
from pocket_tts_server.rate_limiter import CapacityLimiter, RateLimiter


class SilentModel:
    device = 'cpu'
    sample_rate = 24000

    def get_state_for_audio_prompt(self, voice: str) -> dict[str, torch.Tensor]:
        return {}

    def generate_audio_stream(self, **kwargs):
        yield torch.zeros(1920)


@pytest.fixture(autouse=True)
def isolated_admission(monkeypatch):
    monkeypatch.setattr(api, 'rate_limiter', RateLimiter(requests_per_window=1000))
    monkeypatch.setattr(api, 'websocket_capacity', CapacityLimiter(8))
    monkeypatch.setattr(audio, 'synthesis_capacity', CapacityLimiter(32))


def test_cancelled_waiter_skips_voice_preparation(monkeypatch):
    prepared = threading.Event()
    model = SilentModel()
    def record_voice_preparation(voice: str) -> dict[str, torch.Tensor]:
        prepared.set()
        return {}

    model.get_state_for_audio_prompt = record_voice_preparation
    manager = ModelManager(_model=model, _device='cpu', _language='english')
    monkeypatch.setattr(audio, 'model_manager', manager)
    queue = audio.AudioQueue()
    manager.acquire_lock()
    try:
        producer = audio._start_audio_producer(queue, 'alba', 'Hi.', model_tier='english-cpu')
        queue.abort = True
        producer.join(3)
        assert not producer.is_alive()
        assert not prepared.is_set()
    finally:
        manager.release_lock()


def test_cancelled_during_load_skips_voice_preparation(monkeypatch):
    import pocket_tts
    entered, release, prepared = threading.Event(), threading.Event(), threading.Event()
    model = SilentModel()
    def record_voice_preparation(voice: str) -> dict[str, torch.Tensor]:
        prepared.set()
        return {}

    model.get_state_for_audio_prompt = record_voice_preparation

    def load(language):
        entered.set()
        assert release.wait(3)
        return model

    monkeypatch.setattr(pocket_tts.TTSModel, 'load_model', staticmethod(load))
    monkeypatch.setattr(audio, 'model_manager', ModelManager(_model=SilentModel(), _language='english'))
    queue = audio.AudioQueue()
    producer = audio._start_audio_producer(queue, 'alba', 'Hi.', model_tier='french-cpu')
    try:
        assert entered.wait(3)
        queue.abort = True
    finally:
        release.set()
        producer.join(3)
    assert not producer.is_alive()
    assert not prepared.is_set()


def test_concurrent_language_requests_use_their_own_model(monkeypatch):
    import pocket_tts
    entered, release = threading.Event(), threading.Event()
    observed = []

    class Model(SilentModel):
        def __init__(self, language):
            self.language = language

        def generate_audio_stream(self, **kwargs):
            observed.append((self.language, kwargs['text_to_generate']))
            yield torch.zeros(240)

    def load(language):
        if language == 'french':
            entered.set()
            assert release.wait(3)
        return Model(language)

    monkeypatch.setattr(pocket_tts.TTSModel, 'load_model', staticmethod(load))
    monkeypatch.setattr(audio, 'model_manager', ModelManager(_model=Model('english'), _language='english', _device='cpu'))
    first = audio._start_audio_producer(audio.AudioQueue(), 'alba', 'French.', model_tier='french-cpu')
    assert entered.wait(3)
    second = audio._start_audio_producer(audio.AudioQueue(), 'alba', 'German.', model_tier='german-cpu')
    release.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert observed == [('french', 'French.'), ('german', 'German.')]


def test_failed_load_does_not_satisfy_a_waiter(monkeypatch):
    import pocket_tts
    entered, release = threading.Event(), threading.Event()
    calls, failures = [], []
    old_model = SilentModel()
    manager = ModelManager(_model=old_model, _language='english')

    def load(language):
        calls.append(language)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        raise ValueError('failed checkpoint')

    def request():
        try:
            manager.load(language='french')
        except ValueError as exc:
            failures.append(str(exc))

    monkeypatch.setattr(pocket_tts.TTSModel, 'load_model', staticmethod(load))
    first = threading.Thread(target=request)
    second = threading.Thread(target=request)
    first.start()
    assert entered.wait(3)
    second.start()
    release.set()
    first.join(3)
    second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert failures == ['failed checkpoint', 'failed checkpoint']
    assert calls == ['french', 'french']
    assert manager.model is old_model and manager.language == 'english'


def test_synthesis_capacity_rejects_and_releases(monkeypatch):
    capacity = CapacityLimiter(1)
    monkeypatch.setattr(audio, 'synthesis_capacity', capacity)
    manager = ModelManager(_model=SilentModel(), _language='english')
    monkeypatch.setattr(audio, 'model_manager', manager)
    queue = audio.AudioQueue()
    manager.acquire_lock()
    try:
        producer = audio._start_audio_producer(queue, 'alba', 'Hi.', model_tier='english-cpu')
        with pytest.raises(HTTPException) as error:
            audio._start_audio_producer(audio.AudioQueue(), 'alba', 'Hi.', model_tier='english-cpu')
        assert error.value.status_code == 503
        queue.abort = True
        producer.join(3)
        assert not producer.is_alive()
        assert capacity.acquire()
        capacity.release()
    finally:
        manager.release_lock()


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg unavailable')
@pytest.mark.parametrize('partial', [False, True])
def test_encoded_generation_failure_is_not_cached(monkeypatch, tmp_path, partial):
    class Model(SilentModel):
        def generate_audio_stream(self, **kwargs):
            if partial:
                yield torch.zeros(1920)
            raise ValueError('producer failed')

    monkeypatch.setattr(audio, 'model_manager', ModelManager(_model=Model(), _device='cpu', _language='english'))
    monkeypatch.setattr(audio.settings, 'audio_cache_dir', str(tmp_path))

    async def consume():
        with pytest.raises(ValueError, match='producer failed'):
            async for _ in audio.generate_audio('Hi.', voice='alba', format='webm', model_tier='english-cpu'):
                pass

    asyncio.run(consume())
    assert list(tmp_path.iterdir()) == []


def test_encoder_nonzero_exit_is_not_cached(monkeypatch, tmp_path):
    import sys

    def failing_encoder(format, speed):
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, 'rb') as source:
            process = subprocess.Popen([sys.executable, '-c',
                'import sys; sys.stdin.buffer.read(); sys.stderr.write("encoder failed"); sys.exit(7)'],
                stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process, write_fd

    monkeypatch.setattr(audio, '_start_ffmpeg_process', failing_encoder)
    monkeypatch.setattr(audio, '_ffmpeg_available', True)
    monkeypatch.setattr(audio, 'model_manager', ModelManager(_model=SilentModel(), _device='cpu', _language='english'))
    monkeypatch.setattr(audio.settings, 'audio_cache_dir', str(tmp_path))

    async def consume():
        with pytest.raises(HTTPException) as error:
            async for _ in audio.generate_audio('Hi.', voice='alba', format='webm', model_tier='english-cpu'):
                pass
        assert 'status 7' in error.value.detail
        assert 'encoder failed' in error.value.detail

    asyncio.run(consume())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('cancel', [False, True])
def test_failed_identical_request_preserves_published_cache(monkeypatch, tmp_path, cancel):
    monkeypatch.setattr(audio.settings, 'audio_cache_dir', str(tmp_path))
    monkeypatch.setattr(audio, 'model_manager', ModelManager(_model=SilentModel()))
    cache = CacheManager()
    monkeypatch.setattr(audio, 'cache_manager', cache)
    key = cache.generate_cache_key('Hi.', 'alba', 'pcm', 1, 0.3, 2, 0.95, 1.1, 'english-cpu')
    cache_path, meta_path = cache.get_cache_path(key, 'pcm')

    async def core(*args):
        # A concurrent request publishes after this request's cache miss.
        from pathlib import Path
        Path(cache_path).write_bytes(b'complete audio')
        Path(meta_path).write_text('{}')
        yield b'partial'
        if cancel:
            raise asyncio.CancelledError()
        raise RuntimeError('failed request')

    monkeypatch.setattr(audio, '_generate_audio_core', core)

    async def consume():
        with pytest.raises(asyncio.CancelledError if cancel else HTTPException):
            async for _ in audio.generate_audio('Hi.', 'alba', format='pcm', model_tier='english-cpu'):
                pass

    asyncio.run(consume())
    from pathlib import Path
    assert Path(cache_path).read_bytes() == b'complete audio'
    assert Path(meta_path).read_text() == '{}'
    assert not list(tmp_path.glob('*.tmp'))


def test_ignored_sampling_fields_share_cache_key():
    cache = CacheManager()
    first = cache.generate_cache_key('Hi.', 'alba', 'pcm', 1, 0.3, 2, 0.5, 1.0, 'english-cpu')
    second = cache.generate_cache_key('Hi.', 'alba', 'pcm', 1, 0.3, 2, 0.95, 1.9, 'english-cpu')
    assert first == second


def test_ignored_sampling_fields_are_not_set_on_model(monkeypatch):
    model = SilentModel()
    monkeypatch.setattr(audio, 'model_manager', ModelManager(_model=model, _device='cpu', _language='english'))
    producer = audio._start_audio_producer(audio.AudioQueue(), 'alba', 'Hi.', model_tier='english-cpu', top_p=0.5, repetition_penalty=1.5)
    producer.join(3)
    assert not producer.is_alive()
    assert not hasattr(model, 'top_p') and not hasattr(model, 'repetition_penalty')


def test_forwarded_header_does_not_change_rate_limit_identity(monkeypatch):
    seen = []
    monkeypatch.setattr(api, 'rate_limiter', SimpleNamespace(is_allowed=lambda ip: (seen.append(ip) or True, 0)))
    client = TestClient(api.app)
    for forwarded in ('first', 'second'):
        assert client.post('/set_tts_settings', headers={'X-Forwarded-For': forwarded}, json={}).status_code == 200
    assert seen == ['testclient', 'testclient']


def test_websocket_connections_are_globally_bounded(monkeypatch):
    monkeypatch.setattr(api, 'websocket_capacity', CapacityLimiter(1))
    monkeypatch.setattr(api, 'model_manager', ModelManager(_model=SilentModel()))
    client = TestClient(api.app)
    with client.websocket_connect('/v1/audio/stream') as first:
        with pytest.raises(WebSocketDenialResponse) as error:
            with client.websocket_connect('/v1/audio/live'):
                pytest.fail('Rejected connection completed its handshake')
        assert error.value.status_code == 503
        assert error.value.headers['retry-after'] == '5'
        first.send_json({'type': 'text.done'})
        assert first.receive_json()['status'] == 'session_ended'
    with client.websocket_connect('/v1/realtime') as recovered:
        recovered.send_json({'type': 'session.update'})
        recovered.send_json({'type': 'input_text.done'})
        assert recovered.receive_json()['type'] == 'response.done'


def test_websocket_phrases_are_rate_limited(monkeypatch):
    monkeypatch.setattr(api, 'rate_limiter', RateLimiter(requests_per_window=2))
    monkeypatch.setattr(api, 'model_manager', ModelManager(_model=SilentModel()))

    async def generate(**kwargs):
        yield b'\0\0'

    monkeypatch.setattr(api, 'generate_audio', generate)
    with TestClient(api.app).websocket_connect('/v1/audio/stream') as ws:
        ws.send_json({'type': 'text.append', 'text': 'One.', 'format': 'pcm', 'stream': True})
        assert ws.receive_json()['status'] == 'audio_start'
        assert ws.receive_bytes() == b'\0\0'
        assert ws.receive_json()['status'] == 'done'
        ws.send_json({'type': 'text.append', 'text': 'Two.'})
        error = ws.receive_json()
        assert error['status'] == 'error' and error['code'] == 'rate_limited'
        assert error['retryable'] is True and error['retry_after'] > 0
        ws.send_json({'type': 'text.done'})
        assert ws.receive_json()['status'] == 'session_ended'


@pytest.mark.parametrize('path,payload', [
    ('/v1/audio/speech', {'input': 'Hi.', 'voice': 'alba', 'response_format': 'pcm'}),
    ('/tts_to_audio/', {'text': 'Hi.', 'voice': 'alba', 'format': 'pcm'}),
    ('/tts_stream?text=Hi.&voice=alba&format=pcm', None),
])
def test_http_overload_returns_503_and_recovers(monkeypatch, tmp_path, path, payload):
    capacity = CapacityLimiter(1)
    monkeypatch.setattr(audio, 'synthesis_capacity', capacity)
    manager = ModelManager(_model=SilentModel(), _language='english', _device='cpu')
    monkeypatch.setattr(audio, 'model_manager', manager)
    monkeypatch.setattr(api, 'model_manager', manager)
    monkeypatch.setattr(audio.settings, 'audio_cache_dir', str(tmp_path))
    client = TestClient(api.app)
    assert capacity.acquire()
    try:
        response = client.get(path) if payload is None else client.post(path, json=payload)
        assert response.status_code == 503
        assert response.json()['detail'] == 'Synthesis capacity exhausted'
        assert list(tmp_path.iterdir()) == []
    finally:
        capacity.release()
    response = client.get(path) if payload is None else client.post(path, json=payload)
    assert response.status_code == 200
    assert len(response.content) > 0
    assert capacity.acquire()
    try:
        # A cached request needs no synthesis slot.
        cached = client.get(path) if payload is None else client.post(path, json=payload)
        assert cached.status_code == 200
        assert cached.content == response.content
    finally:
        capacity.release()


def test_http_response_closes_source_if_body_is_never_started():
    closed = []

    async def source():
        try:
            yield b'first'
            yield b'last'
        finally:
            closed.append(True)

    async def run():
        response = await api.audio_response(source(), media_type='audio/pcm')
        scope = {'type': 'http', 'method': 'GET', 'path': '/',
                 'asgi': {'spec_version': '2.4'}}

        async def send(message):
            raise OSError('disconnected before response body')

        async def receive():
            return {'type': 'http.disconnect'}

        with pytest.raises(Exception):
            await response(scope, receive, send)
        assert closed == [True]

    asyncio.run(run())


def test_closing_primed_response_aborts_generation_and_removes_temp(monkeypatch, tmp_path):
    monkeypatch.setattr(audio.settings, 'audio_cache_dir', str(tmp_path))
    monkeypatch.setattr(audio, 'model_manager', ModelManager(_model=SilentModel()))
    closed = []

    async def core(*args):
        try:
            yield b'partial'
            yield b'last'
        finally:
            closed.append(True)

    monkeypatch.setattr(audio, '_generate_audio_core', core)

    async def run():
        response = await api.audio_response(audio.generate_audio('Hi.', 'alba', format='pcm'))
        assert list(tmp_path.glob('*.tmp'))
        await response.source.aclose()
        assert closed == [True]
        assert list(tmp_path.iterdir()) == []

    asyncio.run(run())


def test_http_disconnect_before_first_audio_cancels_waiting_producer(monkeypatch, tmp_path):
    from starlette.requests import Request
    model = SilentModel()
    prepared = threading.Event()
    def record_voice_preparation(voice: str) -> dict[str, torch.Tensor]:
        prepared.set()
        return {}

    model.get_state_for_audio_prompt = record_voice_preparation
    manager = ModelManager(_model=model, _language='english', _device='cpu')
    monkeypatch.setattr(audio, 'model_manager', manager)
    monkeypatch.setattr(audio.settings, 'audio_cache_dir', str(tmp_path))
    producers = []
    original = audio._start_audio_producer

    def start(*args, **kwargs):
        producer = original(*args, **kwargs)
        producers.append(producer)
        return producer

    monkeypatch.setattr(audio, '_start_audio_producer', start)

    async def run():
        disconnect = asyncio.Event()

        body_sent = False
        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {'type': 'http.request', 'body': b'', 'more_body': False}
            await disconnect.wait()
            return {'type': 'http.disconnect'}

        request = Request({'type': 'http', 'method': 'GET', 'path': '/'}, receive)
        task = asyncio.create_task(api.audio_response(
            audio.generate_audio('Hi.', 'alba', format='pcm'), request=request))
        async with asyncio.timeout(3):
            while not producers:
                await asyncio.sleep(0.01)
        disconnect.set()
        with pytest.raises(HTTPException) as error:
            await asyncio.wait_for(task, timeout=2)
        assert error.value.status_code == 499
        await asyncio.to_thread(producers[0].join, 2)
        assert not producers[0].is_alive()
        assert not prepared.is_set()
        assert list(tmp_path.iterdir()) == []

    manager.acquire_lock()
    try:
        asyncio.run(run())
    finally:
        manager.release_lock()


def test_handshake_rate_limit_denies_open_but_keeps_discovery_available(monkeypatch):
    monkeypatch.setattr(api, 'rate_limiter', RateLimiter(requests_per_window=1))
    monkeypatch.setattr(api, 'model_manager', ModelManager(_model=SilentModel()))
    client = TestClient(api.app)
    assert client.post('/set_tts_settings', json={}).status_code == 200
    with pytest.raises(WebSocketDenialResponse) as error:
        with client.websocket_connect('/v1/audio/stream'):
            pytest.fail('Rate-limited connection completed its handshake')
    assert error.value.status_code == 429
    assert int(error.value.headers['retry-after']) > 0
    for path in ('/health', '/v1/voices', '/v1/models'):
        assert client.get(path).status_code == 200


def test_rate_limit_keeps_active_and_queued_speech(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(api, 'rate_limiter', RateLimiter(requests_per_window=3))
    monkeypatch.setattr(api, 'model_manager', ModelManager(_model=SilentModel()))

    async def generate(**kwargs):
        yield kwargs['text'].encode()
        if kwargs['text'] == 'One.':
            assert await asyncio.to_thread(release.wait, 3)

    monkeypatch.setattr(api, 'generate_audio', generate)
    try:
        with TestClient(api.app).websocket_connect('/v1/audio/stream') as ws:
            def send(text, request_id):
                ws.send_json({'type': 'text.append', 'text': text, 'request_id': request_id,
                              'format': 'pcm', 'stream': True})
            send('One.', '1')
            assert ws.receive_json()['status'] == 'audio_start'
            assert ws.receive_bytes() == b'One.'
            send('Two.', '2')
            send('Three.', '3')
            error = ws.receive_json()
            assert error['code'] == 'rate_limited' and error['request_id'] == '3'
            release.set()
            assert ws.receive_json()['request_id'] == ['1']
            assert ws.receive_json()['request_id'] == ['2']
            assert ws.receive_bytes() == b'Two.'
            assert ws.receive_json()['status'] == 'done'
    finally:
        release.set()
