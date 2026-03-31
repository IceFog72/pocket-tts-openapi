#!/usr/bin/env python3
"""
Test script: TTS server WebSocket streaming behavior.

Sends a big multi-sentence text and measures when each chunk arrives.
Verifies if server streams audio AS sentences are generated, or waits for all.

Usage:
    python test_ws_streaming.py [url]
    python test_ws_streaming.py ws://localhost:8005/v1/audio/stream
"""

import asyncio
import json
import sys
import time
import websockets

DEFAULT_URL = "ws://localhost:8005/v1/audio/stream"

# Multi-sentence text — 5 sentences, ~8 seconds of expected audio
TEST_TEXT = (
    "The weather today is absolutely beautiful. "
    "I went for a long walk through the park this morning. "
    "The birds were singing and the flowers were blooming everywhere. "
    "It reminded me of spring days from my childhood. "
    "I hope tomorrow will be just as wonderful as today."
)


async def test_streaming(url: str, text: str, fmt: str = "wav"):
    """Connect to WS, send text, log when each chunk arrives."""

    print(f"Connecting to {url}")
    print(f"Format: {fmt}")
    print(f"Text: {len(text)} chars, ~{len(text.split())} words")
    print(f"Sentences: {len([s for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()])}")
    print("-" * 60)

    async with websockets.connect(url, max_size=50 * 1024 * 1024) as ws:
        # Send request immediately — server waits for client
        request = {
            "text": text,
            "voice": "nova",
            "format": fmt,
            "speed": 1.0,
            "temperature": 1.0,
            "top_p": 1.0,
        }
        t0 = time.monotonic()
        await ws.send(json.dumps(request))
        t_sent = time.monotonic() - t0
        print(f"Request sent in {t_sent * 1000:.0f}ms")
        print()

        # Collect chunks with timestamps
        chunks = []
        chunk_times = []
        total_bytes = 0
        done = False

        while not done:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
                t_now = time.monotonic() - t0

                if isinstance(msg, bytes):
                    total_bytes += len(msg)
                    chunks.append(len(msg))
                    chunk_times.append(t_now)

                    # Log every 10th chunk or first/last few
                    idx = len(chunks)
                    if idx <= 3 or idx % 10 == 0 or len(chunks) == len(chunks):
                        elapsed_from_prev = t_now - (chunk_times[-2] if len(chunk_times) > 1 else 0)
                        print(
                            f"  chunk #{idx:>4d}: {len(msg):>6d} bytes | "
                            f"t={t_now:>7.3f}s | "
                            f"delta={elapsed_from_prev * 1000:>6.1f}ms | "
                            f"total={total_bytes:>8d} bytes"
                        )

                elif isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("status") == "done":
                        done = True
                        t_total = time.monotonic() - t0
                        audio_dur = data.get("audio_duration", 0)
                        gen_time = data.get("gen_time", 0)

                        print()
                        print("=" * 60)
                        print(f"COMPLETE")
                        print(f"  Total time:       {t_total:.3f}s (wall clock)")
                        print(f"  Server gen_time:  {gen_time:.3f}s")
                        print(f"  Audio duration:   {audio_dur:.3f}s")
                        print(f"  Total chunks:     {len(chunks)}")
                        print(f"  Total bytes:      {total_bytes}")
                        print(f"  Speed:            {audio_dur / max(t_total, 0.01):.2f}x real-time")
                        print()

                        # Analyze streaming behavior
                        print("STREAMING ANALYSIS:")
                        if len(chunk_times) >= 2:
                            first_chunk_t = chunk_times[0]
                            last_chunk_t = chunk_times[-1]
                            stream_duration = last_chunk_t - first_chunk_t
                            print(f"  First chunk at:   {first_chunk_t:.3f}s")
                            print(f"  Last chunk at:    {last_chunk_t:.3f}s")
                            print(f"  Streaming window: {stream_duration:.3f}s")

                            # Check: are chunks spread out or bunched at the end?
                            midpoint = total_bytes / 2
                            cumulative = 0
                            t_half = 0
                            for i, size in enumerate(chunks):
                                cumulative += size
                                if cumulative >= midpoint:
                                    t_half = chunk_times[i]
                                    break
                            print(f"  50% bytes at:     {t_half:.3f}s ({t_half / last_chunk_t * 100:.0f}% of stream)")

                            if t_half / max(last_chunk_t, 0.001) < 0.3:
                                print(f"  ⚠️  Most data arrived late — likely buffering (FFmpeg or model)")
                            else:
                                print(f"  ✓  Data spread evenly — true streaming")

                            # Check for gaps > 500ms between chunks
                            gaps = []
                            for i in range(1, len(chunk_times)):
                                gap = chunk_times[i] - chunk_times[i - 1]
                                if gap > 0.5:
                                    gaps.append((i, gap))
                            if gaps:
                                print(f"  ⚠️  {len(gaps)} gaps > 500ms detected:")
                                for idx, gap in gaps[:5]:
                                    print(f"      after chunk #{idx}: {gap:.0f}ms pause")
                        else:
                            print(f"  ⚠️  Only {len(chunk_times)} chunk(s) — no streaming at all")

                    elif data.get("status") == "error":
                        print(f"\nERROR: {data.get('error')}")
                        done = True

            except asyncio.TimeoutError:
                print("\nTIMEOUT waiting for server response")
                done = True


async def test_compare_formats(url: str, text: str):
    """Test streaming with different formats to find the best one."""
    for fmt in ["wav", "mp3", "aac"]:
        print(f"\n{'=' * 60}")
        print(f"FORMAT: {fmt}")
        print(f"{'=' * 60}")
        try:
            await test_streaming(url, text, fmt)
        except Exception as e:
            print(f"  FAILED: {e}")
        await asyncio.sleep(1)


async def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    print("TTS WebSocket Streaming Test")
    print("=" * 60)

    # Test 1: Single format (default WAV for best streaming)
    await test_streaming(url, TEST_TEXT, "wav")

    # Uncomment to test all formats:
    # await test_compare_formats(url, TEST_TEXT)


if __name__ == "__main__":
    asyncio.run(main())
