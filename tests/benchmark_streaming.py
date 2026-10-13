"""Run against a local, warmed model over a real WebSocket. No audio playback."""
import asyncio
import json
import resource
import socket
import time

import uvicorn
import websockets

from pocket_tts_server.api import app


async def main(voice="alba"):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(150):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError('Server exited before startup')
                await asyncio.sleep(0.05)
        async with websockets.connect(f'ws://127.0.0.1:{port}/v1/audio/stream') as ws:
            for index in range(3):
                started = time.monotonic()
                await ws.send(json.dumps({'type': 'text.append', 'text':
                    'Streaming starts while the rest of this sentence is still being generated.',
                    'voice': voice, 'model': 'english-cpu', 'format': 'pcm',
                    'stream': True, 'request_id': str(index)}))
                first = None
                count = 0
                size = 0
                async with asyncio.timeout(60):
                    while True:
                        item = await ws.recv()
                        if isinstance(item, bytes):
                            first = first if first is not None else time.monotonic() - started
                            count += 1
                            size += len(item)
                        else:
                            event = json.loads(item)
                            if event.get('status') == 'error':
                                raise RuntimeError(event)
                            if event.get('status') == 'done':
                                break
                elapsed = time.monotonic() - started
                assert first is not None and first < elapsed
                assert count > 1 and size > 0
                print(json.dumps({'run': index + 1, 'first_pcm_ms': round(first * 1000, 1),
                                  'total_ms': round(elapsed * 1000, 1), 'chunks': count,
                                  'audio_seconds': round(size / 48000, 3)}), flush=True)
    finally:
        server.should_exit = True
        await task
        sock.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="alba")
    asyncio.run(main(parser.parse_args().voice))
