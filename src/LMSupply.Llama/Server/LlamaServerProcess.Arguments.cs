using System.Globalization;

namespace LMSupply.Llama.Server;

public sealed partial class LlamaServerProcess
{
    /// <summary>
    /// Translates the memory-map / memory-lock options into the flags the given build parses.
    /// llama.cpp b10105 replaced <c>--mmap</c>/<c>--no-mmap</c>/<c>--mlock</c> with one
    /// <c>--load-mode</c> (<c>none</c> | <c>mmap</c> | <c>mlock</c> = mmap + lock | <c>dio</c>);
    /// the old flags were removed in b11146 (v0.5.0), where passing them is a fatal "invalid
    /// argument" before the port is open. Builds below the gate get the legacy flags. A binary of
    /// unknown build (one whose <c>--version</c> could not be read) gets no flag for memory mapping
    /// on — that is llama.cpp's default on both sides of the gate, so omitting it is the only
    /// spelling that parses everywhere — and the legacy spelling for the non-default requests
    /// (<c>--no-mmap</c>, <c>--mlock</c>), which no spelling can make safe without the build.
    /// </summary>
    internal static IReadOnlyList<string> ResolveLoadModeArgs(bool? useMemoryMap, bool? useMemoryLock, string? serverVersion)
    {
        var args = new List<string>(2);
        var build = LlamaServerVersionRequirements.ParseBuildNumber(serverVersion);
        var gate = LlamaServerVersionRequirements.GetMinimumBuild("load-mode");

        if (build.HasValue && gate.HasValue && build.Value >= gate.Value)
        {
            // The new vocabulary has no "lock without mapping" mode: mlock implies mmap. Locking
            // is the stronger intent, so it wins over an explicit UseMemoryMap = false.
            string? mode = useMemoryLock == true ? "mlock"
                : useMemoryMap == true ? "mmap"
                : useMemoryMap == false ? "none"
                : null;

            if (mode != null)
            {
                args.Add("--load-mode");
                args.Add(mode);
            }

            return args;
        }

        if (useMemoryMap == false)
        {
            args.Add("--no-mmap");
        }
        else if (useMemoryMap == true && build.HasValue)
        {
            // A known pre-gate build: say what was asked. Unknown build: mmap is the default
            // everywhere, and "--mmap" no longer parses on b11146+.
            args.Add("--mmap");
        }

        if (useMemoryLock == true)
        {
            args.Add("--mlock");
        }

        return args;
    }

    /// <summary>
    /// The arguments <paramref name="mode"/> and <paramref name="pooling"/> add. An embedding or reranking server
    /// gets <c>--embedding</c> and no RAM prompt cache (<c>--cache-ram 0</c>): such a request always processes its
    /// whole input, so a cached prompt is never read back, and builds that cache it anyway hold up to the cache limit
    /// (8 GiB by default) for nothing. A reranking server pools by rank unless a pooling is given.
    /// </summary>
    internal static List<string> ModeArguments(ServerMode mode, PoolingType pooling)
    {
        var args = new List<string>();
        if (mode is ServerMode.Embedding or ServerMode.Reranking)
        {
            args.Add("--embedding");
            args.Add("--cache-ram");
            args.Add("0");
        }

        if (mode == ServerMode.Reranking && pooling == PoolingType.None)
        {
            pooling = PoolingType.Rank;
        }

        if (pooling != PoolingType.None)
        {
            args.Add("--pooling");
            args.Add(pooling.ToString().ToLowerInvariant());
        }

        return args;
    }

    /// <summary>The <c>--api-key</c>/<c>--api-key-file</c> argument in <paramref name="args"/>, if any.</summary>
    internal static string? FindApiKeyArgument(IReadOnlyList<string>? args)
    {
        if (args is null)
            return null;

        foreach (var arg in args)
        {
            var name = arg.Split('=', 2)[0].Trim();
            if (name is "--api-key" or "--api-key-file")
                return name;
        }

        return null;
    }

    private string BuildArguments()
    {
        var args = new List<string>
        {
            "--model", $"\"{_config.ModelPath}\"",
            "--port", _port.ToString(CultureInfo.InvariantCulture),
            "--ctx-size", _config.ContextSize.ToString(CultureInfo.InvariantCulture),
            "--n-gpu-layers", _config.GpuLayers.ToString(CultureInfo.InvariantCulture),
            "--batch-size", _config.BatchSize.ToString(CultureInfo.InvariantCulture),
            "--parallel", _config.Parallel.ToString(CultureInfo.InvariantCulture),
            "--host", LoopbackHost, // Only listen on loopback for security
            "--cont-batching",     // Enable continuous batching for better throughput
            "--jinja"              // Enable Jinja template processing for native tool calling support
        };

        // Physical batch size for VRAM efficiency
        if (_config.UBatchSize.HasValue)
        {
            args.Add("--ubatch-size");
            args.Add(_config.UBatchSize.Value.ToString(CultureInfo.InvariantCulture));
        }

        if (_config.FlashAttention)
        {
            // llama.cpp b8795+ requires an explicit value (on|off|auto).
            // Older builds that accepted the boolean form are no longer distributed via LMSupply.
            args.Add("--flash-attn");
            args.Add("on");
        }

        // KV cache quantization (Phase 1)
        if (!string.IsNullOrEmpty(_config.CacheTypeK))
        {
            args.Add("--cache-type-k");
            args.Add(_config.CacheTypeK);
        }

        if (!string.IsNullOrEmpty(_config.CacheTypeV))
        {
            args.Add("--cache-type-v");
            args.Add(_config.CacheTypeV);
        }

        // Memory options (Phase 1) — spelling depends on the build, see ResolveLoadModeArgs.
        args.AddRange(ResolveLoadModeArgs(_config.UseMemoryMap, _config.UseMemoryLock, _config.ServerVersion));

        // GPU options (Phase 1)
        if (_config.MainGpu.HasValue)
        {
            args.Add("--main-gpu");
            args.Add(_config.MainGpu.Value.ToString(CultureInfo.InvariantCulture));
        }

        // RoPE options (Phase 1)
        if (_config.RopeFreqBase.HasValue)
        {
            args.Add("--rope-freq-base");
            args.Add(_config.RopeFreqBase.Value.ToString("F1", CultureInfo.InvariantCulture));
        }

        if (_config.RopeFreqScale.HasValue)
        {
            args.Add("--rope-freq-scale");
            args.Add(_config.RopeFreqScale.Value.ToString("F4", CultureInfo.InvariantCulture));
        }

        // Speculative decoding
        if (!string.IsNullOrEmpty(_config.SpecType))
        {
            args.Add("--spec-type");
            args.Add(_config.SpecType);
        }

        if (!string.IsNullOrEmpty(_config.ModelDraft))
        {
            args.Add("--model-draft");
            args.Add($"\"{_config.ModelDraft}\"");
        }

        // YaRN RoPE scaling
        if (!string.IsNullOrEmpty(_config.RopeScaling))
        {
            args.Add("--rope-scaling");
            args.Add(_config.RopeScaling);
        }

        if (_config.YarnOriginalContext.HasValue)
        {
            args.Add("--yarn-orig-ctx");
            args.Add(_config.YarnOriginalContext.Value.ToString(CultureInfo.InvariantCulture));
        }

        if (_config.YarnExtensionFactor.HasValue)
        {
            args.Add("--yarn-ext-factor");
            args.Add(_config.YarnExtensionFactor.Value.ToString("F4", CultureInfo.InvariantCulture));
        }

        if (_config.YarnAttentionFactor.HasValue)
        {
            args.Add("--yarn-attn-factor");
            args.Add(_config.YarnAttentionFactor.Value.ToString("F4", CultureInfo.InvariantCulture));
        }

        if (_config.YarnBetaFast.HasValue)
        {
            args.Add("--yarn-beta-fast");
            args.Add(_config.YarnBetaFast.Value.ToString("F4", CultureInfo.InvariantCulture));
        }

        if (_config.YarnBetaSlow.HasValue)
        {
            args.Add("--yarn-beta-slow");
            args.Add(_config.YarnBetaSlow.Value.ToString("F4", CultureInfo.InvariantCulture));
        }

        // Multimodal projector (Phase 3)
        if (!string.IsNullOrEmpty(_config.MultimodalProjector))
        {
            args.Add("--mmproj");
            args.Add($"\"{_config.MultimodalProjector}\"");
        }

        // LoRA adapter (Phase 3)
        if (!string.IsNullOrEmpty(_config.LoraPath))
        {
            if (_config.LoraScale.HasValue)
            {
                args.Add("--lora-scaled");
                args.Add($"\"{_config.LoraPath}\"");
                args.Add(_config.LoraScale.Value.ToString("F2", CultureInfo.InvariantCulture));
            }
            else
            {
                args.Add("--lora");
                args.Add($"\"{_config.LoraPath}\"");
            }
        }

        args.AddRange(ModeArguments(_config.Mode, _config.Pooling));

        if (_config.AdditionalArgs != null)
        {
            args.AddRange(_config.AdditionalArgs);
        }

        return string.Join(" ", args);
    }
}
