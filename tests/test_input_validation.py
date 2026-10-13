"""Bad input fails before model/file access and sockets remain usable."""
import logging

import pytest
from fastapi.testclient import TestClient

from pocket_tts_server import api, audio
from pocket_tts_server.constants import VOICE_MAPPING
from pocket_tts_server.model_manager import ModelManager
from pocket_tts_server.rate_limiter import CapacityLimiter, RateLimiter
from pocket_tts_server.validation import resolve_voice


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, 'rate_limiter', RateLimiter(requests_per_window=1000))
    monkeypatch.setattr(api, 'websocket_capacity', CapacityLimiter(8))
    manager = ModelManager(_model=object(), _language='english', _device='cpu')
    monkeypatch.setattr(api, 'model_manager', manager)
    monkeypatch.setattr(audio, 'model_manager', manager)
    monkeypatch.setattr(audio.settings, 'audio_cache_dir', str(tmp_path))

    def unexpected(*args, **kwargs):
        pytest.fail('Invalid input reached model synthesis')

    monkeypatch.setattr(audio, '_start_audio_producer', unexpected)
    return TestClient(api.app)


@pytest.mark.parametrize('path,payload', [
    ('/v1/audio/speech', {'input': 'Hello.', 'voice': 'Nilou', 'response_format': 'pcm'}),
    ('/tts_to_audio/', {'text': 'Hello.', 'voice': 'Nilou', 'format': 'pcm'}),
    ('/tts_stream?text=Hello.&voice=Nilou&format=pcm', None),
])
def test_unknown_voice_is_client_error_without_traceback(client, caplog, path, payload):
    with caplog.at_level(logging.ERROR):
        response = client.get(path) if payload is None else client.post(path, json=payload)
    assert response.status_code == 404
    error = response.json()['error']
    assert error['code'] == 'unknown_voice'
    assert error['param'] == 'voice'
    assert error['retryable'] is False
    assert 'Nilou' in error['message'] and '/v1/voices' in error['message']
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.parametrize('fields,param', [
    ({'input': '   '}, 'input'),
    ({'input': '`"`'}, 'input'),
    ({'input': {'text': 'Hello'}}, 'input'),
    ({'voice': ['alba']}, 'voice'),
    ({'model': 'garbage-cpu'}, 'model'),
    ({'speed': 0}, 'speed'),
    ({'speed': -1}, 'speed'),
    ({'speed': 'nan'}, 'speed'),
    ({'speed': 'infinity'}, 'speed'),
    ({'speed': 'fast'}, 'speed'),
    ({'temperature': 9}, 'temperature'),
    ({'lsd_decode_steps': 0}, 'lsd_decode_steps'),
    ({'response_format': 'garbage'}, 'response_format'),
])
def test_openai_invalid_fields_return_safe_422(client, fields, param):
    payload = {'input': 'Hello.', 'voice': 'alba', **fields}
    response = client.post('/v1/audio/speech', json=payload)
    assert response.status_code == 422
    error = response.json()['error']
    assert error['param'] == param
    assert error['retryable'] is False
    assert 'Traceback' not in response.text
    assert set(error) == {'code', 'param', 'message', 'retryable'}


@pytest.mark.parametrize('payload', [None, [], 'hello', 123])
def test_xtts_requires_object_body(client, payload):
    response = client.post('/tts_to_audio/', content=__import__('json').dumps(payload),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 400
    assert response.json()['error']['retryable'] is False


def test_xtts_bad_json(client):
    response = client.post('/tts_to_audio/', content='{', headers={'Content-Type': 'application/json'})
    assert response.status_code == 400
    assert response.json()['error']['code'] == 'invalid_json'


@pytest.mark.parametrize('fields,param', [
    ({'text': []}, 'input'), ({'speed': 0}, 'speed'),
    ({'speed': 'fast'}, 'speed'), ({'format': 'bad'}, 'response_format'),
])
def test_xtts_fields_use_shared_validation(client, fields, param):
    response = client.post('/tts_to_audio/', json={'text': 'Hello.', 'voice': 'alba', **fields})
    assert response.status_code == 400
    assert response.json()['error']['param'] == param


@pytest.mark.parametrize('path', ['/v1/audio/stream', '/v1/audio/live'])
def test_unknown_voice_on_websocket_is_recoverable(client, caplog, monkeypatch, path):
    with caplog.at_level(logging.ERROR):
        with client.websocket_connect(path) as ws:
            ws.send_json({'type': 'text.append', 'text': 'Hello.', 'voice': 'Nilou',
                          'format': 'pcm', 'stream': True, 'request_id': 'wrong'})
            error = ws.receive_json()
            assert error['status'] == 'error'  # No preceding audio_start.
            assert error['request_id'] == 'wrong'
            assert error['code'] == 'unknown_voice'
            assert error['param'] == 'voice' and error['retryable'] is False
            assert isinstance(error['error'], str)
            ws.send_json({'type': 'text.done'})
            assert ws.receive_json()['status'] == 'session_ended'
            async def generate(**kwargs):
                yield b'\0\0'
            monkeypatch.setattr(api, 'generate_audio', generate)
            ws.send_json({'type': 'text.append', 'text': 'Hello.', 'voice': 'alba',
                          'format': 'pcm', 'stream': True, 'request_id': 'good'})
            assert ws.receive_json()['status'] == 'audio_start'
            assert ws.receive_bytes() == b'\0\0'
            assert ws.receive_json()['status'] == 'done'
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.parametrize('payload,param', [
    ([], None), ({'type': 'garbage'}, 'type'),
    ({'type': 'text.append', 'text': {}}, 'text'),
    ({'type': 'text.append', 'text': 'Hi.', 'voice': []}, 'voice'),
    ({'type': 'text.append', 'text': 'Hi.', 'model': 'bad'}, 'model'),
    ({'type': 'text.append', 'text': 'Hi.', 'speed': 0}, 'speed'),
    ({'type': 'text.append', 'text': 'Hi.', 'stream': 'garbage'}, 'stream'),
    ({'type': 'text.append', 'text': 'Hi.', 'request_id': {}}, 'request_id'),
])
def test_bad_socket_message_does_not_close_connection(client, payload, param):
    with client.websocket_connect('/v1/audio/stream') as ws:
        ws.send_json(payload)
        error = ws.receive_json()
        assert error['status'] == 'error' and error['retryable'] is False
        assert error['param'] == param
        ws.send_json({'type': 'text.done'})
        assert ws.receive_json()['status'] == 'session_ended'


def test_bad_socket_json_does_not_close_connection(client):
    with client.websocket_connect('/v1/audio/stream') as ws:
        ws.send_text('{')
        assert ws.receive_json()['code'] == 'invalid_json'
        ws.send_json({'type': 'text.done'})
        assert ws.receive_json()['status'] == 'session_ended'


def test_voice_resolution_preserves_aliases_case_and_file_paths(monkeypatch, tmp_path):
    assert resolve_voice('ALBA') == 'alba'
    assert resolve_voice('Alloy') == 'alba'
    embedding = tmp_path / 'TestVoice.safetensors'
    embedding.touch()
    monkeypatch.setitem(VOICE_MAPPING, 'TestVoice', str(embedding))
    assert resolve_voice('testvoice') == str(embedding)
    assert resolve_voice(str(embedding)) == str(embedding)
    embedding.unlink()
    with pytest.raises(api.InputError) as error:
        resolve_voice('TestVoice')
    assert error.value.code == 'voice_unavailable'
    assert str(tmp_path) not in error.value.detail


@pytest.mark.parametrize('session,code,param', [
    ({'voice': 'Nilou'}, 'unknown_voice', 'voice'),
    ({'voice': 'alba', 'speed': 0}, 'invalid_request', 'speed'),
    ({'voice': 'alba', 'format': 'garbage'}, 'invalid_request', 'response_format'),
])
def test_realtime_rejects_bad_speech_and_accepts_next_request(client, monkeypatch, session, code, param):
    with client.websocket_connect('/v1/realtime') as ws:
        ws.send_json({'type': 'session.update', 'session': session})
        ws.send_json({'type': 'input_text.append', 'text': 'Hello.'})
        error = ws.receive_json()['error']
        assert error['code'] == code and error['param'] == param
        assert error['retryable'] is False
        async def generate(**kwargs):
            yield b'\0\0'
        monkeypatch.setattr(api, 'generate_audio', generate)
        ws.send_json({'type': 'session.update', 'session': {'voice': 'alba', 'format': 'pcm', 'speed': 1}})
        ws.send_json({'type': 'input_text.append', 'text': 'Again.'})
        assert ws.receive_bytes() == b'\0\0'
        assert ws.receive_json()['type'] == 'response.audio.done'
        ws.send_json({'type': 'input_text.done'})
        assert ws.receive_json()['type'] == 'response.done'


def test_realtime_recovers_from_malformed_messages(client):
    with client.websocket_connect('/v1/realtime') as ws:
        ws.send_text('{')
        assert ws.receive_json()['error']['code'] == 'invalid_json'
        ws.send_json({'type': 'session.update', 'session': []})
        assert ws.receive_json()['error']['param'] == 'session'
        ws.send_json({'type': 'session.update', 'session': {'voice': 'alba'}})
        ws.send_json({'type': 'input_text.append', 'text': []})
        assert ws.receive_json()['error']['param'] == 'text'
        ws.send_json({'type': 'input_text.done'})
        assert ws.receive_json()['type'] == 'response.done'


@pytest.mark.parametrize('decoder', ['wav', 'safetensors'])
def test_corrupt_voice_reference_is_not_a_retryable_server_failure(monkeypatch, tmp_path, caplog, decoder):
    import soundfile
    import safetensors
    exception = soundfile.SoundFileError if decoder == 'wav' else safetensors.SafetensorError

    class Model:
        def get_state_for_audio_prompt(self, voice):
            raise exception('decoder detail with private file path')

    manager = ModelManager(_model=Model(), _language='english', _device='cpu')
    monkeypatch.setattr(api, 'model_manager', manager)
    monkeypatch.setattr(audio, 'model_manager', manager)
    monkeypatch.setattr(audio.settings, 'audio_cache_dir', str(tmp_path))
    monkeypatch.setattr(api, 'rate_limiter', RateLimiter(requests_per_window=1000))
    with caplog.at_level(logging.ERROR):
        response = TestClient(api.app).post('/v1/audio/speech', json={
            'input': 'Hello.', 'voice': 'alba', 'response_format': 'pcm'})
    assert response.status_code == 422
    error = response.json()['error']
    assert error['code'] == 'invalid_voice_file'
    assert error['retryable'] is False
    assert 'private file path' not in response.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert not list(tmp_path.iterdir())
