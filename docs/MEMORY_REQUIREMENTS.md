# Memory Requirements Guide

> LMSupply model memory requirements and OOM prevention guide

---

## Overview

LMSupply models run via ONNX Runtime. Memory usage is determined by:

- **Model Weights**: ~1.5–2× the model file size
- **Input/Output Tensors**: Proportional to batch size and sequence length
- **Intermediate Activations**: Temporary data generated during inference
- **Runtime Overhead**: Internal ONNX Runtime buffers

### Memory Estimation Formula

```
EstimatedMemory = ModelFileSize × 2
```

This formula is a rough estimate that includes model weight loading and runtime overhead.

---

## Memory by Domain

### Embedder (Text Embedding)

| Model | Parameters | ONNX Size | Est. Memory | Context |
|-------|------------|-----------|-------------|---------|
| multilingual-e5-small | 118M | ~470MB | ~940MB | 512 tokens |
| e5-small-v2 | 33M | ~130MB | ~260MB | 512 tokens |
| bge-base-en-v1.5 | 110M | ~440MB | ~880MB | 512 tokens |
| gte-base-en-v1.5 | 109M | ~440MB | ~880MB | 8K tokens |
| nomic-embed-text-v1.5 | 137M | ~550MB | ~1.1GB | 8K tokens |
| multilingual-e5-base | 278M | ~1.1GB | ~2.2GB | 512 tokens |
| bge-large-en-v1.5 | 335M | ~1.3GB | ~2.6GB | 512 tokens |
| gte-large-en-v1.5 | 434M | ~1.7GB | ~3.4GB | 8K tokens |
| multilingual-e5-large | 560M | ~2.2GB | ~4.4GB | 512 tokens |
| bge-m3 | 568M | ~2.3GB | ~4.5GB | 8K tokens |

**Recommended:**
- 💡 **Low Memory (< 4GB)**: `fast` (multilingual-e5-small)
- ⚖️ **Balanced (4-8GB)**: `quality` (gte-large-en-v1.5), English-first
- 🚀 **Quality (8GB+)**: `default` (bge-m3) — 100+ languages, 8K context

> These follow the aliases the embedder registry actually resolves. The table above lists other
> models that load by name; only the aliases carry the registry's tuning (pooling mode, dimensions,
> and the E5 query/passage prefixes).

---

### Reranker (Semantic Reranking)

| Model | Parameters | ONNX Size | Est. Memory | Context |
|-------|------------|-----------|-------------|---------|
| ms-marco-TinyBERT-L-2 | 4.4M | ~18MB | ~36MB | 512 tokens |
| ms-marco-MiniLM-L-6 | 22M | ~90MB | ~180MB | 512 tokens |
| ms-marco-MiniLM-L-12 | 33M | ~134MB | ~270MB | 512 tokens |
| bge-reranker-base | 278M | ~440MB | ~880MB | 512 tokens |
| bge-reranker-large | 560M | ~1.1GB | ~2.2GB | 512 tokens |
| bge-reranker-v2-m3 | 568M | ~1.1GB | ~2.2GB | 8K tokens |

**Recommended:**
- 💡 **Fast**: `ms-marco-TinyBERT-L-2` (ultra-lightweight)
- ⚖️ **Default**: `ms-marco-MiniLM-L-6` (balanced)
- 🚀 **Quality**: `bge-reranker-large`

---

### Generator (Text Generation)

Generator models require significantly more memory than other domains.

| Model | Parameters | ONNX Size | Est. Memory | Context |
|-------|------------|-----------|-------------|---------|
| Phi-4-mini-instruct | 3.8B | ~7.5GB | ~15GB | 16K tokens |
| Phi-3.5-mini-instruct | 3.8B | ~7.5GB | ~15GB | 128K tokens |
| Phi-4 | 14B | ~28GB | ~56GB | 16K tokens |

**GGUF Format (default backend, via llama-server):**

GGUF models are the default backend for Generator. They use significantly less memory via quantization:

| Model | Parameters | Q4_K_M Size | Est. Memory | Context |
|-------|------------|-------------|-------------|---------|
| Ministral 3 3B | 3B | ~1.8GB | ~3.6GB | 32K tokens |
| Hermes 3 Llama 3.1 8B | 8B | ~4.6GB | ~9.2GB | 8K tokens |
| Mistral Nemo 12B | 12B | ~7GB | ~14GB | 32K tokens |
| Qwen 3 32B | 32B | ~18GB | ~36GB | 32K tokens |
| Qwen 3.5 122B MoE (10B active) | 122B | ~70GB | ~140GB | 32K tokens |

**Quantization Options:**

| Quantization | Memory Reduction | Quality Impact |
|--------------|-----------------|----------------|
| Q8_0 | ~50% | Minimal |
| Q6_K | ~60% | Very Low |
| Q5_K_M | ~65% | Low |
| Q4_K_M | ~75% | Moderate |
| Q3_K_M | ~80% | Noticeable |
| Q2_K | ~87% | Significant |

**Automatic Quantization Selection (v0.16.0+):**

LMSupply automatically selects the **best quantization file that fits your available memory** from a GGUF repository:

- Measures available VRAM and RAM (subtracts 2GB GPU overhead and 4GB OS overhead)
- Among files that fit in memory, picks the highest-quality quantization (Q8 > Q6 > Q5 > Q4 …)
- To specify a file explicitly: `new GeneratorOptions { GgufFileName = "model-Q4_K_M.gguf" }`

**Recommended:**
- 💡 **Low Memory (4-8GB)**: `gguf:fast` (Ministral 3 3B Q4_K_M)
- ⚖️ **Balanced (8-16GB)**: `gguf:default` (Hermes 3 8B) or `Phi-4-mini` (ONNX)
- 🚀 **Quality (16GB+)**: `gguf:quality` (Mistral Nemo 12B) or `Phi-4` (ONNX)

---

### Transcriber (Speech Recognition)

| Model | Parameters | ONNX Size | Est. Memory | Languages |
|-------|------------|-----------|-------------|-----------|
| Whisper Tiny | 39M | ~150MB | ~300MB | Multi |
| Whisper Base | 74M | ~280MB | ~560MB | Multi |
| Whisper Small | 244M | ~950MB | ~1.9GB | Multi |
| Whisper Medium | 769M | ~3GB | ~6GB | Multi |
| Whisper Large V3 | 1.5B | ~6GB | ~12GB | Multi |
| Whisper Large V3 Turbo | 809M | ~3.2GB | ~6.4GB | Multi |

**Recommended:**
- 💡 **Fast**: `whisper-tiny-en` (English-only, ultra-fast)
- ⚖️ **Default**: `whisper-base` or `whisper-small`
- 🚀 **Quality**: `whisper-large-v3-turbo` (best quality/speed balance)

---

### Detector (Object Detection)

| Model | Parameters | ONNX Size | Est. Memory | Input Size |
|-------|------------|-----------|-------------|------------|
| EfficientDet-Lite0 | 3.9M | ~15MB | ~30MB | 320×320 |
| RT-DETR-R18 | - | ~80MB | ~160MB | 640×640 |
| RT-DETR-R34 | - | ~160MB | ~320MB | 640×640 |
| RT-DETR-R50 | - | ~200MB | ~400MB | 640×640 |
| RT-DETR-R101 | - | ~300MB | ~600MB | 640×640 |

**Recommended:**
- 💡 **Fast**: `efficientdet-lite0` (mobile/edge)
- ⚖️ **Default**: `rt-detr-r34`
- 🚀 **Quality**: `rt-detr-r101`

---

### Segmenter (Image Segmentation)

| Model | Parameters | ONNX Size | Est. Memory | Classes |
|-------|------------|-----------|-------------|---------|
| SegFormer-B0 | 3.8M | ~15MB | ~30MB | 150 |
| SegFormer-B1 | 13.7M | ~55MB | ~110MB | 150 |
| SegFormer-B2 | 27.4M | ~110MB | ~220MB | 150 |
| SegFormer-B3 | 47.3M | ~190MB | ~380MB | 150 |
| SegFormer-B4 | 64.1M | ~256MB | ~512MB | 150 |
| SegFormer-B5 | 84.7M | ~340MB | ~680MB | 150 |

**Recommended:**
- 💡 **Fast**: `segformer-b0`
- ⚖️ **Default**: `segformer-b2`
- 🚀 **Quality**: `segformer-b5`

---

### Synthesizer (Text-to-Speech)

Piper TTS models are lightweight:

| Voice | Quality | Size | Est. Memory |
|-------|---------|------|-------------|
| en_US-ryan-medium | Medium | ~20MB | ~40MB |
| en_US-lessac-medium | Medium | ~20MB | ~40MB |
| en_US-amy-low | Low | ~16MB | ~32MB |
| en_US-lessac-high | High | ~64MB | ~128MB |

**All Piper voices are lightweight with virtually no memory constraints.**

---

### Translator (Machine Translation)

| Model | Size | Est. Memory | Direction |
|-------|------|-------------|-----------|
| opus-mt-en-ko | ~300MB | ~600MB | EN→KO |
| opus-mt-ko-en | ~300MB | ~600MB | KO→EN |
| opus-mt-en-zh | ~300MB | ~600MB | EN→ZH |
| opus-mt-zh-en | ~300MB | ~600MB | ZH→EN |

---

### OCR (Optical Character Recognition)

OCR combines two models:

| Component | Model | Est. Memory |
|-----------|-------|-------------|
| Detection | DBNet | ~100MB |
| Recognition | CRNN | ~50MB |
| **Total** | - | **~150MB** |

---

### Captioner (Image Captioning)

| Model | Architecture | Est. Memory |
|-------|--------------|-------------|
| ViT-GPT2 | Encoder-Decoder | ~500MB-1GB |

---

## GPU VRAM vs System RAM

### With GPU (CUDA/DirectML/CoreML)

| VRAM | Recommended Models |
|------|-------------------|
| 4GB | Embedder (small), Reranker (mini), Detector (lite) |
| 6GB | Embedder (base), Transcriber (small), Segmenter (B2) |
| 8GB | Embedder (large), Generator (1-3B), Transcriber (medium) |
| 12GB | Generator (3-4B), Transcriber (large-turbo) |
| 16GB+ | Generator (7B+), multiple models simultaneously |

### CPU-only (System RAM)

| RAM | Recommended Usage |
|-----|-------------------|
| 8GB | One small model (Embedder/Reranker) |
| 16GB | One medium model or multiple small models |
| 32GB | Large model or multiple models simultaneously |
| 64GB+ | Large Generator + other domains combined |

---

## OOM Prevention Strategies

### 1. Choose the Right Model

```csharp
// Use "auto" alias for hardware-optimized selection
var embedder = await LocalEmbedder.LoadAsync("auto");
var generator = await LocalGenerator.LoadAsync("auto");
```

The `"auto"` alias automatically selects the optimal model based on `HardwareProfile`.

### 2. Use Lazy Loading

Models load when `LoadAsync` is called. Only load when needed:

```csharp
// ❌ Pre-load all models at startup
var embedder = await LocalEmbedder.LoadAsync("default");
var reranker = await LocalReranker.LoadAsync("default");
var generator = await LocalGenerator.LoadAsync("default"); // OOM risk!

// ✅ Load on demand
await using var embedder = await LocalEmbedder.LoadAsync("default");
// ... use embedder, then automatically released

await using var reranker = await LocalReranker.LoadAsync("default");
// ... use reranker, then automatically released
```

### 3. Explicit Disposal

Free memory explicitly with `DisposeAsync`:

```csharp
await using (var model = await LocalEmbedder.LoadAsync("large"))
{
    var embeddings = await model.EmbedAsync(texts);
} // automatically released here

// Or manual disposal
var model = await LocalEmbedder.LoadAsync("large");
try
{
    // use
}
finally
{
    await model.DisposeAsync();
}
```

### 4. Control Batch Size

Split large inputs into smaller batches:

```csharp
// ❌ Process large dataset at once
var embeddings = await embedder.EmbedAsync(thousandDocuments);

// ✅ Split into batches
var results = new List<float[]>();
foreach (var batch in thousandDocuments.Chunk(32))
{
    var batchEmbeddings = await embedder.EmbedAsync(batch);
    results.AddRange(batchEmbeddings);
}
```

### 5. Generator Special Considerations

Generator is memory-intensive:

```csharp
// Use GGUF quantized model (auto-selects best fit for your hardware)
var generator = await LocalGenerator.LoadAsync("model-q4_k_m.gguf");

// Or use a small ONNX model
var generator = await LocalGenerator.LoadAsync("microsoft/Phi-4-mini-instruct-onnx");
```

### 6. Check Estimated Memory

Use `EstimatedMemoryBytes` to verify expected memory usage:

```csharp
var model = await LocalEmbedder.LoadAsync("large");
var memoryMB = model.EstimatedMemoryBytes / (1024 * 1024);
Console.WriteLine($"Estimated memory: {memoryMB}MB");
```

---

## Performance Tiers Reference

LMSupply auto-selects models based on `HardwareProfile.Current.Tier`:

| Tier | GPU VRAM | System RAM | Recommended |
|------|----------|------------|-------------|
| Low | < 4GB or CPU only | < 16GB | Lightweight models only |
| Medium | 4-8GB | 16GB+ | Base models |
| High | 8-16GB | 32GB+ | Large models |
| Ultra | 16GB+ | 64GB+ | Maximum quality |

```csharp
var profile = HardwareProfile.Current;
Console.WriteLine($"Tier: {profile.Tier}");
Console.WriteLine($"GPU: {profile.GpuInfo?.Name ?? "None"}");
Console.WriteLine($"VRAM: {profile.GpuInfo?.VramBytes / (1024*1024*1024)}GB");
Console.WriteLine($"Recommended Provider: {profile.RecommendedProvider}");
```

---

## Troubleshooting

### Out of Memory Error

1. **Use a smaller model**: `"default"` → `"fast"`
2. **Change provider**: `ExecutionProvider.Cpu` (uses system RAM)
3. **Release other models**: `await otherModel.DisposeAsync()`
4. **Reduce batch size**: Process smaller amounts of data at once

### CUDA Out of Memory

```csharp
// Switch to DirectML (Windows)
var options = new EmbedderOptions { Provider = ExecutionProvider.DirectML };

// Or use CPU
var options = new EmbedderOptions { Provider = ExecutionProvider.Cpu };
```

### Suspected Memory Leak

Always release models with `DisposeAsync`:

```csharp
// ✅ Prefer the using statement
await using var model = await LocalEmbedder.LoadAsync("default");
```

---

## Related Documentation

- [GPU Providers Guide](GPU_PROVIDERS.md) - GPU provider selection
- [Model Lifecycle Guide](MODEL_LIFECYCLE.md) - Model lifecycle management
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Problem resolution
