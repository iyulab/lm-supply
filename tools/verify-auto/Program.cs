using System.Diagnostics;
using LMSupply;
using LMSupply.Diagnostics;
using LMSupply.Generator;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Gguf;
using LMSupply.Generator.Models;
using LMSupply.Hardware;

LMSupplyTraceListener.Attach((msg, sev) =>
{
    var tag = sev switch
    {
        TraceEventType.Warning => "WARN",
        TraceEventType.Error => "ERR ",
        _ => "INFO"
    };
    Console.WriteLine($"  [{tag}] {msg}");
});

// --dry-run: run only Phase 1 (hardware detection + selection), no download/load/generate.
// Lets an env-override matrix (LMSUPPLY_SYSTEM_RAM_MB / LMSUPPLY_VRAM_BUDGET_MB) exercise the
// real GetAutoSelection / LlamaBackendSelector path without pulling a model per scenario.
var dryRun = args.Contains("--dry-run");
var positional = args.Where(a => !a.StartsWith("--", StringComparison.Ordinal)).ToArray();
var targetAlias = positional.Length > 0 ? positional[0] : "default";
Console.WriteLine($"# Target alias: {targetAlias}{(dryRun ? " (dry-run)" : "")}\n");

// ───── Phase 1: dry-run selection (no downloads) ─────
Console.WriteLine("=== Phase 1: Selection dry-run ===");
var profile = HardwareProfile.Current;
Console.WriteLine($"GPU: {profile.GpuInfo.Vendor} {profile.GpuInfo.DeviceName} " +
                  $"total={Format(profile.GpuInfo.TotalMemoryBytes)} free={Format(profile.GpuInfo.FreeMemoryBytes)}");
Console.WriteLine($"RAM: {Format(profile.SystemMemoryBytes)}, Provider: {profile.RecommendedProvider}");

GgufModelInfo? target;
if (targetAlias is "default" or "auto")
{
    // RAM-aware selection (mirrors the real LocalGenerator.LoadAsync path), plus the auto backend
    // the generator would pick — so the dry-run reflects actual load behavior on this hardware.
    var selection = GgufModelRegistry.GetAutoSelection(
        profile.GpuInfo, profile.SystemMemoryBytes,
        GgufModelRegistry.DefaultBudgetContextLength, excludeKnownIssues: null);
    target = selection.Selected;
    var backend = LMSupply.Llama.LlamaBackendSelector.MapProvider(ExecutionProvider.Auto, profile.GpuInfo);
    Console.WriteLine($"Auto-selected: alias={target.AliasName} repo={target.RepoId} file={target.DefaultFile} reason={selection.Reason}");
    Console.WriteLine($"  budget: vram={Format(selection.AvailableVramBytes)} ram={Format(selection.AvailableSystemRamBytes)} -> backend={backend}");
}
else
{
    target = GgufModelRegistry.Resolve(targetAlias);
    if (target is null)
    {
        Console.Error.WriteLine($"FAIL: alias '{targetAlias}' not resolvable");
        return 1;
    }
    Console.WriteLine($"Resolved: alias={target.AliasName} repo={target.RepoId} file={target.DefaultFile}");
}
Console.WriteLine();

if (dryRun)
{
    Console.WriteLine("(dry-run: skipping Phase 2-4 download/load/generate)");
    return 0;
}

// ───── Phase 2: actual LoadAsync + download ─────
Console.WriteLine($"=== Phase 2: LocalGenerator.LoadAsync(\"{targetAlias}\") ===");

var lastReport = DateTime.UtcNow;
long lastBytes = 0;
string? currentFile = null;

var progress = new Progress<DownloadProgress>(p =>
{
    // Throttle: print every ~2s or when file changes
    var now = DateTime.UtcNow;
    if (p.FileName != currentFile)
    {
        currentFile = p.FileName;
        Console.WriteLine($"  download: {p.FileName}  ({Format(p.TotalBytes)} total)");
        lastBytes = p.BytesDownloaded;
        lastReport = now;
        return;
    }
    if ((now - lastReport).TotalSeconds < 2 && p.BytesDownloaded < p.TotalBytes)
        return;
    var rateMBs = (p.BytesDownloaded - lastBytes) / (now - lastReport).TotalSeconds / (1024 * 1024);
    var pct = p.TotalBytes > 0 ? (p.BytesDownloaded * 100.0 / p.TotalBytes) : 0;
    Console.WriteLine($"    {pct,5:F1}%  {Format(p.BytesDownloaded)}/{Format(p.TotalBytes)}  ~{rateMBs:F1} MB/s");
    lastBytes = p.BytesDownloaded;
    lastReport = now;
});

var sw = Stopwatch.StartNew();
IGeneratorModel model;
try
{
    model = await LocalGenerator.LoadAsync(targetAlias, progress: progress);
}
catch (Exception ex)
{
    Console.Error.WriteLine($"FAIL LoadAsync: {ex.GetType().Name}: {ex.Message}");
    if (ex.InnerException is not null)
        Console.Error.WriteLine($"  inner: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
    return 3;
}
sw.Stop();
Console.WriteLine($"LoadAsync OK in {sw.Elapsed.TotalSeconds:F1}s");

await using (model)
{
    var info = model.GetModelInfo();
    Console.WriteLine($"Model loaded: {info.ModelId} chatFormat={info.ChatFormat} provider={info.ExecutionProvider}");
    Console.WriteLine($"  path: {info.ModelPath}");
    Console.WriteLine($"  arch: {info.Architecture ?? "<n/a>"} quant={info.QuantizationType ?? "<n/a>"}");
    Console.WriteLine();

    // ───── Phase 3: warmup ─────
    Console.WriteLine("=== Phase 3: WarmupAsync ===");
    sw.Restart();
    await model.WarmupAsync();
    sw.Stop();
    Console.WriteLine($"Warmup OK in {sw.Elapsed.TotalSeconds:F1}s\n");

    // ───── Phase 4: text generation ─────
    Console.WriteLine("=== Phase 4: GenerateAsync ===");
    // Generous token budget so the smoke test sees real content. NOTE: Qwen3 is thinking-default-on
    // and EnableThinking=false does NOT suppress it (the flag only injects thinking for default-off
    // formatters like Gemma4),
    // so a <think> block may still appear. SUCCESS only asserts non-empty output, which is enough here.
    var genOpts = new GenerationOptions { MaxTokens = 128, Temperature = 0.2f };
    var prompt = "Say hello in three short sentences.";
    Console.WriteLine($"prompt: {prompt}");
    sw.Restart();

    Console.Write("--- output ---\n");
    var collected = new System.Text.StringBuilder();
    await foreach (var chunk in model.GenerateAsync(prompt, genOpts))
    {
        Console.Write(chunk);
        collected.Append(chunk);
    }
    sw.Stop();
    Console.WriteLine($"\n--- end ({sw.Elapsed.TotalSeconds:F1}s, {collected.Length} chars) ---");
    if (collected.Length == 0)
    {
        Console.Error.WriteLine("FAIL: empty generation output");
        return 4;
    }
}

Console.WriteLine("\nSUCCESS: end-to-end load + generate verified");
return 0;

static string Format(long? bytes)
{
    if (bytes is null or <= 0) return "n/a";
    const double gb = 1024.0 * 1024 * 1024;
    const double mb = 1024.0 * 1024;
    return bytes.Value >= (long)gb ? $"{bytes.Value / gb:F2}GB" : $"{bytes.Value / mb:F0}MB";
}
