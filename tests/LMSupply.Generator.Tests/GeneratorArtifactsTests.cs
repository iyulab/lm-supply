using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="GeneratorArtifacts"/>: a GGUF model is available offline only when its weights and the llama-server it runs on
/// are both cached; an ONNX model needs no server. Network-free — local model files and a private llama-server cache.
/// </summary>
public sealed class GeneratorArtifactsTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-genartifacts-" + Guid.NewGuid().ToString("N"));
    private readonly string _serverCache;
    private readonly string _ggufFile;

    public GeneratorArtifactsTests()
    {
        _serverCache = Path.Combine(_dir, "llama-server");
        Directory.CreateDirectory(_serverCache);
        _ggufFile = Path.Combine(_dir, "tiny-model.Q4_K_M.gguf");
        File.WriteAllBytes(_ggufFile, "GGUF"u8.ToArray());
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private GeneratorArtifacts Artifacts() => new(new GeneratorOptions
    {
        Provider = ExecutionProvider.Cpu,
        CacheDirectory = Path.Combine(_dir, "hub"),
        // No background update check: these tests must not reach the network.
        ServerUpdateOptions = new LlamaServerUpdateOptions { CacheDirectory = _serverCache, AutoDownloadUpdates = false },
    });

    private void SeedServer()
    {
        var versionDir = Path.Combine(_serverCache, "b7898", "cpu");
        Directory.CreateDirectory(versionDir);
        File.WriteAllBytes(Path.Combine(versionDir, OperatingSystem.IsWindows() ? "llama-server.exe" : "llama-server"), "binary"u8.ToArray());
    }

    [Fact]
    public async Task AGgufModelWithoutAServer_IsNotAvailableOffline_AndWithOneItIs()
    {
        var artifacts = Artifacts();
        (await artifacts.IsAvailableOfflineAsync(_ggufFile, Ct)).Should().BeFalse("a load with downloads disabled would find no llama-server");

        SeedServer();

        (await artifacts.IsAvailableOfflineAsync(_ggufFile, Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task AnOnnxModelDirectory_NeedsNoServer()
    {
        var onnxDir = Path.Combine(_dir, "onnx-model");
        Directory.CreateDirectory(onnxDir);

        (await Artifacts().IsAvailableOfflineAsync(onnxDir, Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task AModelThatIsNotThere_IsNotAvailableOffline()
    {
        SeedServer();

        (await Artifacts().IsAvailableOfflineAsync(Path.Combine(_dir, "missing.gguf"), Ct)).Should().BeFalse();
    }

    [Fact]
    public async Task Download_OfALocalGgufModel_WithTheServerCached_LeavesItAvailableOffline()
    {
        SeedServer();
        var artifacts = Artifacts();

        await artifacts.DownloadAsync(_ggufFile, cancellationToken: Ct);

        (await artifacts.IsAvailableOfflineAsync(_ggufFile, Ct)).Should().BeTrue();
        artifacts.Kind.Should().Be("generator");
    }
}
