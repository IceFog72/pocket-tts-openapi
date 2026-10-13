"""Pocket TTS 3.x compatibility at the server's generation boundary."""
from pathlib import Path
from queue import Queue

import torch
from fastapi.testclient import TestClient

from pocket_tts_server import audio
from pocket_tts_server.api import app
from pocket_tts_server.model_manager import ModelManager


def test_decode_steps_reach_model(monkeypatch):
    observed = []

    class Model:
        temp = 0.0
        sampler_decode_steps = 1

        def get_state_for_audio_prompt(self, voice):
            return {}

        def generate_audio_stream(self, model_state, text_to_generate):
            observed.append((self.temp, self.sampler_decode_steps, text_to_generate))
            yield torch.zeros(240)

    manager = ModelManager(_model=Model(), _device='cpu', _language='english', _sample_rate=24000)
    monkeypatch.setattr(audio, 'model_manager', manager)
    queue = Queue()
    thread = audio._start_audio_producer(queue, 'alba', 'Hello.', temperature=0.4,
                                        lsd_decode_steps=5, model_tier='english-cpu')
    thread.join(timeout=10)
    assert not thread.is_alive()
    chunks = []
    while not queue.empty():
        chunk = queue.get_nowait()
        if chunk is not None:
            chunks.append(chunk)
    assert observed == [(0.4, 5, 'Hello.')]
    assert b''.join(chunks).startswith(b'RIFF')
    assert len(b''.join(chunks)) > 44


def test_model_discovery_uses_installed_configs():
    from pocket_tts.models.tts_model import CONFIGS_DIR

    client = TestClient(app)
    response = client.get('/v1/models')
    assert response.status_code == 200
    ids = {model['id'] for model in response.json()['data']}
    assert ids == {f'{config.stem}-{device}' for config in CONFIGS_DIR.glob('*.yaml')
                   for device in ('cpu', 'gpu')}
    assert 'alba' in client.get('/v1/voices').json()['voices']


def test_voice_export_uses_sampler_steps(monkeypatch, tmp_path):
    import pocket_tts
    from pocket_tts_server import api, voices

    observed = []

    class Model:
        sampler_decode_steps = 1

        def get_state_for_audio_prompt(self, path, truncate=False):
            observed.append(self.sampler_decode_steps)
            return {}

    monkeypatch.setattr(api, 'model_manager', ModelManager(_model=Model()))
    monkeypatch.setattr(api.settings, 'voices_dir', str(tmp_path))
    monkeypatch.setattr(api.settings, 'embeddings_dir', str(tmp_path))
    monkeypatch.setattr(voices, 'load_custom_voices', lambda: set())
    monkeypatch.setattr(pocket_tts, 'export_model_state', lambda state, path: Path(path).write_bytes(b'export'))
    (tmp_path / 'test.wav').touch()
    response = TestClient(app).post('/v1/audio/export-voice', json={'voice': 'test', 'lsd_decode_steps': 7})
    assert response.status_code == 200
    assert observed == [7]
    assert (tmp_path / 'test.safetensors').read_bytes() == b'export'


def test_current_language_defaults_and_temperature():
    import yaml
    from pocket_tts.utils.config import CONFIGS_DIR, load_config
    from pocket_tts_server.config import settings
    from pocket_tts_server.models import SpeechRequest

    assert settings.temperature == 0.3
    assert SpeechRequest(input='Default settings.').temperature == 0.3
    for language in ('english', 'french', 'german', 'spanish', 'portuguese', 'italian', 'dutch'):
        config = load_config(CONFIGS_DIR / f'{language}.yaml')
        assert config.default_temperature == settings.temperature
        assert config.weights_path.startswith('hf://')
    english = yaml.safe_load((CONFIGS_DIR / 'english.yaml').read_text())
    september = yaml.safe_load((CONFIGS_DIR / 'english_2026-09.yaml').read_text())
    assert english['weights_path'] == september['weights_path'].replace('english_2026-09', 'english')
    assert english['flow_lm']['transformer'] == september['flow_lm']['transformer']
