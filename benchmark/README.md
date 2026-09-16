# LMSupply Benchmarks

Performance benchmarks for LMSupply packages using [BenchmarkDotNet](https://benchmarkdotnet.org/).

## Running Benchmarks

```bash
cd benchmark/LMSupply.Benchmarks

# Run all benchmarks (Release mode required)
dotnet run -c Release

# Run specific benchmark class
dotnet run -c Release -- --filter *Embedder*
dotnet run -c Release -- --filter *Transcriber*
dotnet run -c Release -- --filter *Provider*

# List all available benchmarks
dotnet run -c Release -- --list flat
```

## Available Benchmarks

### TranscriberBenchmarks

Tests Whisper transcription performance:
- Model: `fast`, `default`
- Timestamps: enabled/disabled

### EmbedderBenchmarks

Tests text embedding performance:
- Model: `fast`, `default`
- Batch sizes: 1, 8, 32

### ProviderBenchmarks

Compares CPU vs GPU/Auto provider performance:
- Useful for diagnosing GPU utilization issues
- Shows performance difference between providers

## Output

Benchmark results are saved to the `BenchmarkResults` directory:
- `*.html` - Interactive HTML report
- `*.md` - Markdown report (GitHub compatible)
- `*.csv` - CSV data for analysis

## Example Output

```
|           Method | ModelAlias | UseTimestamps |      Mean |    StdDev |
|----------------- |----------- |-------------- |----------:|----------:|
| TranscribeAudio  |       fast |         False |  523.4 ms |  12.3 ms  |
| TranscribeAudio  |       fast |          True |  567.8 ms |  15.1 ms  |
| TranscribeAudio  |    default |         False |  891.2 ms |  23.4 ms  |
| TranscribeAudio  |    default |          True |  934.5 ms |  18.7 ms  |
```

## Running with GPU

### CUDA

```bash
dotnet run -c Release -p:EnableCuda=true
```

## Interpreting Results

- **Mean**: Average execution time
- **StdDev**: Standard deviation (lower = more consistent)
- **Allocated**: Memory allocated per operation

### GPU Performance Tips

1. Check "GPU Active" in setup output
2. Compare CPU vs GPU benchmark results
3. For small workloads, CPU may be faster due to data transfer overhead
4. GPU benefits increase with larger batch sizes
