# LMSupply Samples

This directory contains sample applications demonstrating how to use LMSupply packages.

## Available Samples

### TranscriberSample

Demonstrates speech-to-text transcription using Whisper models.

```bash
# Run with default settings
dotnet run --project TranscriberSample

# Transcribe an audio file
dotnet run --project TranscriberSample -- path/to/audio.wav

# Specify model and provider
dotnet run --project TranscriberSample -- path/to/audio.wav large cuda
```

**Features demonstrated:**
- Model loading with GPU/CPU provider selection
- Basic transcription
- Segment-level timestamps
- Language specification
- Real-time factor calculation

### EmbedderSample

Demonstrates text embedding and similarity comparison.

```bash
# Run with default settings
dotnet run --project EmbedderSample

# Specify model and provider
dotnet run --project EmbedderSample -- default cuda
```

**Features demonstrated:**
- Single and batch embedding
- Cosine similarity calculation
- GPU/CPU provider selection

## Running with GPU

### CUDA (NVIDIA)

```bash
# Enable CUDA package
dotnet run --project TranscriberSample -p:EnableCuda=true -- audio.wav default cuda
```

## Troubleshooting GPU Usage

If GPU is not being utilized:

1. Check the console output for "GPU Active" status
2. Verify GPU drivers are installed
3. For CUDA: Ensure CUDA toolkit is installed
4. AMD / Intel GPUs on Windows: no ONNX provider on ONNX Runtime 1.25+ (DirectML removed in 0.67.0) — these samples run on CPU there

The samples will display active providers in the console output to help diagnose GPU utilization issues.
