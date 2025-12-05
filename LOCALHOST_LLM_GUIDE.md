# Using MAKER with Localhost/Self-Hosted LLMs

## Issue Identified

The code **is working correctly** and reaching the LLM. The issue you experienced was:

### Problem: Response Timeout with Large Models
- **Model**: `gemma3:27b` (27 billion parameters)
- **Symptom**: Requests hang/timeout
- **Root Cause**: Large models are very slow to respond, especially when MAKER makes multiple calls for voting/consensus

### Test Results
✅ Direct API test works: `curl http://localhost:11434/v1/chat/completions` succeeds  
✅ OpenAI-compatible endpoint is functional  
⏱️ Response time is too long for the configured timeout  

## Solution

### 1. Use Smaller, Faster Models (Recommended)

```python
LLM_MODEL = "mistral:latest"      # Fast and capable (recommended)
LLM_MODEL = "qwen3:14b"           # Good balance of speed and quality
```

### 2. OR Increase Timeout for Large Models

```python
LLM_MODEL = "gemma3:27b"
LLM_TIMEOUT = 600  # 10 minutes for very large models
```

### 3. Code Updates Made

#### CustomLLMProvider (`src/maker_core/providers/custom_llm.py`)
- ✅ Added `"stream": False` to explicitly disable streaming
- ✅ This ensures cleaner responses from Ollama

#### Configuration Files
- ✅ Updated default model to `mistral:latest` (faster)
- ✅ Increased default timeout to 300 seconds
- ✅ Added performance tips and troubleshooting

## Available Models (from your system)

| Model | Size | Speed | Recommended For |
|-------|------|-------|-----------------|
| `mistral:latest` | ~7B | ⚡⚡⚡ Fast | **General use, development** |
| `qwen3:14b` | 14B | ⚡⚡ Medium | Good balance |
| `codestral:latest` | ~7B | ⚡⚡⚡ Fast | Code-related tasks |
| `gemma3:27b` | 27B | ⚡ Slow | High-quality output (needs timeout=600+) |
| `llama3.3:70b` | 70B | 🐌 Very Slow | Best quality (needs timeout=900+) |
| `gpt-oss:120b` | 120B | 🐌🐌 Extremely Slow | Research (needs timeout=1200+) |

## How MAKER Works with LLMs

MAKER uses the **Maximal Agentic Decomposition (MAD)** algorithm, which means:

1. **Classification**: Checks if question needs decomposition (~1 API call)
2. **Decomposition** (if needed): Breaks question into sub-questions (~1-2 API calls)
3. **Voting**: Generates multiple answers and votes for consensus (~3-5 API calls **per question**)
4. **Synthesis**: Combines sub-answers into final answer (~1 API call)

**Total API calls**: 5-20+ depending on complexity

**This is why model speed matters!** A slow 27B model making 20 calls can take 10+ minutes.

## Recommended Configuration

### For Development/Testing
```python
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "mistral:latest"  # Fast!
LLM_TIMEOUT = 300
```

### For Production/Quality
```python
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen3:14b"  # Good balance
LLM_TIMEOUT = 600
```

### For Maximum Quality (be patient!)
```python
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "gemma3:27b"
LLM_TIMEOUT = 900  # 15 minutes
```

## Quick Test

Run this to verify everything works:

```bash
# Test with fast model
cd /home/karsten/python_projects/maker_core
source venv/bin/activate

# Edit x_karsten_test.py to set:
# LLM_MODEL = "mistral:latest"
# LLM_TIMEOUT = 300

python x_karsten_test.py
# Choose example 1 and press Enter
```

You should see:
```
🔄 Connecting to LLM server...
✓ Provider configured
🤔 Asking LLM (this may take a moment)...
✓ Response received!
Answer: [The AI's answer]
```

## Troubleshooting

### If still hanging:
1. Check Ollama is running: `ollama list`
2. Test direct API: `curl http://localhost:11434/v1/models`
3. Try even smaller timeout test: Set `LLM_TIMEOUT = 60` with `mistral:latest`
4. Check Ollama logs: `journalctl -u ollama -f` (if running as service)

### If getting errors:
1. Verify model exists: `ollama list` should show your model
2. Pull model if needed: `ollama pull mistral`
3. Check port 11434 is accessible: `netstat -tlnp | grep 11434`

## Performance Optimization

### For MAKER with localhost LLMs:
- ✅ Use fastest model that meets your quality needs
- ✅ Keep documents/context under 4000 tokens
- ✅ Use simple questions when possible (avoids decomposition)
- ✅ Consider running on GPU for 10-20x speedup
- ✅ Use vLLM instead of Ollama for production (much faster)

### Hardware Considerations:
- **CPU only**: Use models ≤7B (mistral, codestral)
- **GPU (8GB VRAM)**: Can handle 14B models comfortably
- **GPU (16GB+ VRAM)**: Can handle 27B-70B models
- **GPU (24GB+ VRAM)**: Can handle 70B+ models

## Summary

**The code is working!** Your issue was using a 27B parameter model which is too slow for real-time use on most hardware. Switch to `mistral:latest` for fast responses, or increase timeout to 600+ seconds if you need the quality of larger models.
