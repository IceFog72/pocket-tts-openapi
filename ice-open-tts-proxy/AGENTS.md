# AGENTS.md - AI Agent Integration Guide

Instructions for AI coding agents (Claude, Codex, Gemini, etc.) to use Ice Open TTS Proxy as a tool.

## Quick Reference

### Check if Proxy is Running

```bash
curl -s http://127.0.0.1:8181/health
```

Expected response:
```json
{"status": "ok", "tts_connected": true, "audio_playing": false}
```

### Generate Speech

```bash
# Basic - generates and plays audio
curl -X POST http://127.0.0.1:8181/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Task completed successfully", "voice": "nova"}'
```

### Generate Audio File

```bash
# Save to file instead of playing
python tts_proxy_cli.py --text "Generating report summary" --voice nova --save /tmp/alert.mp3
```

## Integration Patterns

### Pattern 1: Notification Tool

Use when the agent needs to notify the user via audio:

```python
def speak_to_user(text: str) -> bool:
    """Speak text to user via TTS proxy."""
    try:
        response = requests.post(
            "http://127.0.0.1:8181/speak",
            json={"text": text, "voice": "nova", "speed": 1.0},
            timeout=10
        )
        return response.json().get("status") == "success"
    except:
        return False

# Usage
speak_to_user("Starting data analysis...")
# ... do work ...
speak_to_user("Analysis complete. Found 3 issues.")
```

### Pattern 2: Status Updates

For long-running tasks, provide progress updates:

```python
def report_progress(step: int, total: int, task: str):
    """Report progress via voice."""
    percent = int((step / total) * 100)
    speak_to_user(f"Step {step} of {total}: {task}. {percent} percent complete.")
```

### Pattern 3: Error Alerts

Alert user to errors or warnings:

```python
def alert_error(error_msg: str):
    """Alert user about an error."""
    speak_to_user(f"Error: {error_msg}. Please review.")

def alert_warning(warning_msg: str):
    """Alert user about a warning."""
    speak_to_user(f"Warning: {warning_msg}")
```

## Request Format

### Basic Request

```json
POST /speak
{
    "text": "Your message here",
    "voice": "nova",
    "speed": 1.0,
    "format": "wav"
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to speak (max 4096 chars) |
| `voice` | string | "nova" | Voice name |
| `speed` | float | 1.0 | Speed (0.25-4.0) |
| `format` | string | "wav" | Audio format: wav, mp3, opus, flac |

### Available Voices

**Standard:** nova, alloy, echo, fable, onyx, shimmer  
**Native:** alba, marius, javert, jean, fantine, cosette, eponine, azelma
**Custom:** Any .wav file in the server's voices/ folder

## Response Format

### Success

```json
{
    "status": "success",
    "message": "Playing: Your message...",
    "file": "/tmp/xxxxx.wav"
}
```

### Error

```json
{
    "error": "Error description"
}
```

## CLI Integration

### Using Python

```python
import subprocess

def speak_cli(text: str, voice: str = "nova"):
    """Use CLI to speak text."""
    subprocess.run([
        "python", "ice_open_tts_proxy_cli.py",
        "--speak", text,
        "--voice", voice
    ], capture_output=True)
```

### Using Bash

```bash
# One-liner to speak text
speak() {
    curl -s -X POST http://127.0.0.1:8181/speak \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$1\", \"voice\": \"${2:-nova}\"}"
}

# Usage
speak "Task complete" "nova"
```

## Error Handling

Always handle these cases:

1. **Proxy not running** - Connection refused
2. **TTS server not running** - Returns error in response
3. **Empty text** - Returns 400 error
4. **Long text** - Truncated at 4096 characters

```python
def safe_speak(text: str, voice: str = "nova") -> dict:
    """Speak with full error handling."""
    if not text:
        return {"status": "error", "error": "Empty text"}
    
    if len(text) > 4096:
        text = text[:4093] + "..."
    
    try:
        response = requests.post(
            "http://127.0.0.1:8181/speak",
            json={"text": text, "voice": voice},
            timeout=30
        )
        return response.json()
    except requests.ConnectionError:
        return {"status": "error", "error": "Proxy not running"}
    except requests.Timeout:
        return {"status": "error", "error": "Request timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

## Server Management

### Start Server (if needed)

```bash
# Check and start if not running
if ! curl -s http://127.0.0.1:8181/health > /dev/null 2>&1; then
    cd /path/to/test && ./start_ice_cli.sh 8181 &
    sleep 5
fi
```

### Server URLs

| Environment | URL |
|-------------|-----|
| Local | http://127.0.0.1:8181 |
| LAN | http://[your-ip]:8181 |

## Best Practices

1. **Check health** before sending requests
2. **Keep text short** (under 100 words)
3. **Use consistent voice** for same type of message
4. **Handle errors gracefully** - don't crash if TTS fails
5. **Don't spam** - rate limit is 120 requests/minute (increased for Live Mode)

## Example: Task Agent

```python
class TaskAgent:
    def __init__(self):
        self.proxy_url = "http://127.0.0.1:8181"
    
    def is_ready(self) -> bool:
        """Check if TTS proxy is available."""
        try:
            r = requests.get(f"{self.proxy_url}/health", timeout=2)
            return r.status_code == 200
        except:
            return False
    
    def notify(self, message: str, voice: str = "nova") -> bool:
        """Send voice notification."""
        try:
            requests.post(
                f"{self.proxy_url}/speak",
                json={"text": message, "voice": voice, "speed": 1.0},
                timeout=10
            )
            return True
        except:
            return False
    
    def task_start(self, task_name: str):
        self.notify(f"Starting {task_name}")
    
    def task_progress(self, current: int, total: int, task: str = ""):
        pct = int((current / total) * 100)
        self.notify(f"Progress: {pct} percent. {task}")
    
    def task_complete(self, task_name: str):
        self.notify(f"{task_name} complete")
    
    def task_error(self, task_name: str, error: str):
        self.notify(f"Error in {task_name}: {error}")
```

## Example: CI/CD Notification

```yaml
# GitHub Actions example
- name: Notify on completion
  if: always()
  run: |
    if [ "${{ job.status }}" == "success" ]; then
      curl -X POST http://your-server:8181/speak \
        -d '{"text": "Build successful", "voice": "nova"}'
    else
      curl -X POST http://your-server:8181/speak \
        -d '{"text": "Build failed", "voice": "nova"}'
    fi
```
