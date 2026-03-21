.PHONY: test test-server test-proxy integration clean

# Run all unit tests
test: test-server test-proxy

# Run server tests
test-server:
	venv/bin/python -m pytest test_pocketapi.py -v

# Run proxy tests
test-proxy:
	cd ice-open-tts-proxy && ../venv/bin/python -m pytest test_ice_open_tts_proxy.py -v

# Run integration tests (requires servers running)
integration:
	RUN_INTEGRATION_TESTS=1 venv/bin/python -m pytest test_pocketapi.py ice-open-tts-proxy/test_ice_open_tts_proxy.py -v -k integration

# Start servers for integration testing
start:
	venv/bin/python pocketapi.py &
	cd ice-open-tts-proxy && ../venv/bin/python ice_open_tts_proxy_cli.py --server --port 8181 &

# Stop servers
stop:
	pkill -f "python.*pocketapi" 2>/dev/null || true
	pkill -f "python.*ice_open_tts_proxy" 2>/dev/null || true

# Clean artifacts
clean:
	rm -rf audio_cache/ __pycache__ .pytest_cache ice-open-tts-proxy/__pycache__ ice-open-tts-proxy/.pytest_cache
