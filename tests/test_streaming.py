import asyncio
import threading
from queue import Queue

import torch
from fastapi.testclient import TestClient

from pocket_tts_server import api, audio
from pocket_tts_server.model_manager import ModelManager


def test_websocket_sends_audio_before_synthesis_finishes(monkeypatch):
    release = threading.Event()

    async def generate(**kwargs):
        yield b'first pcm'
        assert await asyncio.to_thread(release.wait, 3)
        yield b'last pcm'

    monkeypatch.setattr(api, 'generate_audio', generate)
    monkeypatch.setattr(api, 'model_manager', ModelManager(_model=object(), _sample_rate=24000))
    try:
        with TestClient(api.app).websocket_connect('/v1/audio/stream') as ws:
            ws.send_json({'type': 'text.append', 'text': 'Hi.', 'voice': 'alba',
                          'format': 'pcm', 'stream': True, 'request_id': 'a'})
            assert ws.receive_json()['status'] == 'audio_start'
            assert ws.receive_bytes() == b'first pcm'
            release.set()
            assert ws.receive_bytes() == b'last pcm'
            assert ws.receive_json()['request_id'] == ['a']
            ws.send_json({'type': 'text.done'})
            assert ws.receive_json()['status'] == 'session_ended'
            ws.send_json({'type': 'text.append', 'text': 'Again.', 'voice': 'alba',
                          'format': 'pcm', 'stream': True, 'request_id': 'b'})
            assert ws.receive_json()['request_id'] == ['b']
            assert ws.receive_bytes() == b'first pcm'
            assert ws.receive_bytes() == b'last pcm'
            assert ws.receive_json()['request_id'] == ['b']
    finally:
        release.set()


def test_pcm_has_no_wav_header(monkeypatch):
    class Model:
        def get_state_for_audio_prompt(self, voice): return {}
        def generate_audio_stream(self, **kwargs): yield torch.full((240,), 0.25)

    monkeypatch.setattr(audio, 'model_manager', ModelManager(_model=Model(), _device='cpu', _language='english'))
    queue = Queue()
    producer = audio._start_audio_producer(queue, 'alba', 'Hi.', model_tier='english-cpu', raw_pcm=True)
    producer.join(5)
    assert not producer.is_alive()
    result = queue.get_nowait()
    assert len(result) == 480
    assert not result.startswith(b'RIFF')
    assert int.from_bytes(result[:2], 'little', signed=True) == 8191


def test_generation_errors_reach_consumer(monkeypatch):
    class Model:
        def get_state_for_audio_prompt(self, voice): raise ValueError('broken voice')

    monkeypatch.setattr(audio, 'model_manager', ModelManager(_model=Model(), _device='cpu', _language='english'))
    async def consume():
        import pytest
        with pytest.raises(ValueError, match='broken voice'):
            async for _ in audio._generate_audio_core('Hi.', 'alba', 1, 'pcm', 8192, model_tier='english-cpu'):
                pass
    asyncio.run(consume())


def test_live_text_fragments_flush_on_punctuation_and_text_done(monkeypatch):
    seen = []
    async def generate(**kwargs):
        seen.append(kwargs['text'])
        yield b'\0\0'
    monkeypatch.setattr(api, 'generate_audio', generate)
    monkeypatch.setattr(api, 'model_manager', ModelManager(_model=object(), _sample_rate=24000))
    with TestClient(api.app).websocket_connect('/v1/audio/live') as ws:
        ws.send_json({'type': 'text.append', 'text': 'Hel', 'request_id': 'a'})
        ws.send_json({'type': 'text.append', 'text': 'lo.', 'request_id': 'a'})
        assert ws.receive_json()['format'] == 'pcm'
        assert ws.receive_bytes() == b'\0\0'
        assert ws.receive_json()['status'] == 'done'
        ws.send_json({'type': 'text.append', 'text': 'Tail', 'request_id': 'b'})
        ws.send_json({'type': 'text.done'})
        assert ws.receive_json()['status'] == 'audio_start'
        assert ws.receive_bytes() == b'\0\0'
        assert ws.receive_json()['status'] == 'done'
        assert ws.receive_json()['status'] == 'session_ended'
    assert seen == ['Hello.', 'Tail']


def test_health_does_not_wait_for_generation_lock(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    manager = ModelManager(_model=object(), _device='cpu', _sample_rate=24000)
    monkeypatch.setattr(api, 'model_manager', manager)
    manager.acquire_lock()
    try:
        with ThreadPoolExecutor() as executor:
            result = executor.submit(api.health).result(timeout=0.5)
        assert result['model_loaded']
        assert result['sample_rate'] == 24000
    finally:
        manager.release_lock()


def test_ffmpeg_receives_every_generated_chunk(monkeypatch):
    import shutil
    import pytest
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg is unavailable')
    class Model:
        def get_state_for_audio_prompt(self, voice): return {}
        def generate_audio_stream(self, **kwargs):
            for _ in range(10):
                yield torch.zeros(1920)  # Ten 80ms chunks, not just the first one.
    monkeypatch.setattr(audio, 'model_manager', ModelManager(_model=Model(), _device='cpu',
                                                          _language='english', _sample_rate=24000))
    async def consume():
        return b''.join([chunk async for chunk in audio._generate_audio_core(
            'Hi.', 'alba', 2, 'pcm', 8192, model_tier='english-cpu')])
    data = asyncio.run(consume())
    # 0.8s generated + Pocket's 0.2s WAV tail, played at 2x, with atempo padding tolerance.
    assert 0.4 < len(data) / 48000 < 0.6
