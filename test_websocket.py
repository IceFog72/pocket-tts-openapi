#!/usr/bin/env python3
"""Test WebSocket TTS streaming endpoint."""
import asyncio
import json
import sys
import websockets

async def test_websocket(url: str = "ws://localhost:8005/v1/audio/stream"):
    """Connect to TTS WebSocket and stream audio."""
    print(f"Connecting to {url}")

    async with websockets.connect(url) as ws:
        # Send TTS request using text.append protocol
        request = {
            "type": "text.append",
            "text": "Hello! This is a WebSocket test. The audio is streaming in real time.",
            "voice": "nova",
            "format": "wav",
            "speed": 1.0,
            "request_id": "r0",
        }
        print(f"Sending: {json.dumps(request)}")
        await ws.send(json.dumps(request))

        # Signal end of text
        await ws.send(json.dumps({"type": "text.done"}))

        # Collect audio chunks
        total_bytes = 0
        chunk_count = 0

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                chunk_count += 1
                total_bytes += len(msg)
                print(f"  chunk {chunk_count}: {len(msg)} bytes (total: {total_bytes})", end="\r")
            else:
                data = json.loads(msg)
                if data.get("status") == "done":
                    print(f"\nDone! {chunk_count} chunks, {total_bytes} bytes")
                    break
                elif data.get("status") == "session_ended":
                    print(f"\nSession ended.")
                    break
                elif data.get("status") == "error":
                    print(f"\nError: {data.get('error')}")
                    break

        # Second request on same connection
        request2 = {
            "type": "text.append",
            "text": "Second request on the same connection.",
            "voice": "alloy",
            "format": "wav",
            "request_id": "r1",
        }
        print(f"\nSending second: {json.dumps(request2)}")
        await ws.send(json.dumps(request2))
        await ws.send(json.dumps({"type": "text.done"}))

        total_bytes = 0
        chunk_count = 0

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                chunk_count += 1
                total_bytes += len(msg)
                print(f"  chunk {chunk_count}: {len(msg)} bytes (total: {total_bytes})", end="\r")
            else:
                data = json.loads(msg)
                if data.get("status") == "done":
                    print(f"\nDone! {chunk_count} chunks, {total_bytes} bytes")
                    break
                elif data.get("status") == "session_ended":
                    print(f"\nSession ended.")
                    break
                elif data.get("status") == "error":
                    print(f"\nError: {data.get('error')}")
                    break

        # Save combined audio
        print("\nConnection closed.")

    # Test with mp3 format
    print(f"\n--- MP3 format test ---")
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({
            "type": "text.append",
            "text": "MP3 format test.",
            "voice": "shimmer",
            "format": "mp3",
            "request_id": "r2",
        }))
        await ws.send(json.dumps({"type": "text.done"}))

        audio_data = b""
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                audio_data += msg
            else:
                data = json.loads(msg)
                if data.get("status") == "done":
                    with open("/tmp/ws_test.mp3", "wb") as f:
                        f.write(audio_data)
                    print(f"Saved {len(audio_data)} bytes to /tmp/ws_test.mp3")
                    break
                elif data.get("status") == "session_ended":
                    break
                elif data.get("status") == "error":
                    print(f"Error: {data.get('error')}")
                    break


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8005/v1/audio/stream"
    asyncio.run(test_websocket(url))
