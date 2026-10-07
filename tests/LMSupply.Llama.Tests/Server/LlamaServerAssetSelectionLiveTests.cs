using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Checks the asset patterns against the newest llama.cpp build release (excluded from CI via
/// <c>Category=Integration</c>). <see cref="LlamaServerAssetSelectionTests"/> pins the names llama.cpp used
/// when it was written; this is the test that notices a rename — which is how ROCm ("hip-radeon" →
/// "rocm-&lt;ver&gt;") and Linux SYCL ("sycl" → "sycl-fp16"/"sycl-fp32") once fell back to CPU unnoticed.
/// </summary>
[Trait("Category", "Integration")]
public sealed class LlamaServerAssetSelectionLiveTests
{
    /// <summary>Builds every release has carried for a long time; a miss here means upstream renamed one.</summary>
    public static TheoryData<LlamaServerPlatform, LlamaServerArchitecture, LlamaServerBackend> ShippedBuilds => new()
    {
        { LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cpu },
        { LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Vulkan },
        { LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda12 },
        { LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda13 },
        { LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Hip },
        { LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Sycl },
        { LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Cpu },
        { LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Vulkan },
        { LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda12 },
        { LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda13 },
        { LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Hip },
        { LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Sycl },
        { LlamaServerPlatform.MacOS, LlamaServerArchitecture.Arm64, LlamaServerBackend.Metal },
    };

    [Theory]
    [MemberData(nameof(ShippedBuilds))]
    public async Task Live_LatestRelease_HasExactlyOneArchivePerShippedBuild(
        LlamaServerPlatform platform, LlamaServerArchitecture arch, LlamaServerBackend backend)
    {
        var (tag, assets) = await LatestAssetsAsync(TestContext.Current.CancellationToken);

        var server = LlamaServerDownloader.GetAssetPattern(platform, arch, backend);
        assets.Where(name => server.IsMatch(name)).Should().ContainSingle(
            $"release {tag} should carry one {backend} server archive for {platform}/{arch}; assets: {string.Join(", ", assets)}");

        if (LlamaServerDownloader.GetCudartAssetPattern(platform, arch, backend) is { } companion)
        {
            assets.Where(name => companion.IsMatch(name)).Should().ContainSingle(
                $"release {tag} should carry one CUDA runtime companion for {backend} on {platform}/{arch}");
        }
    }

    private static async Task<(string Tag, string[] Assets)> LatestAssetsAsync(CancellationToken cancellationToken)
    {
        using var downloader = new LlamaServerDownloader(Path.Combine(Path.GetTempPath(), "lmsupply-live-" + Guid.NewGuid().ToString("N")));
        var tag = await downloader.GetLatestVersionAsync(cancellationToken)
            ?? throw new InvalidOperationException("no llama.cpp build release could be resolved");

        using var http = new HttpClient();
        http.DefaultRequestHeaders.UserAgent.ParseAdd("LMSupply-Tests");
        await using var body = await http.GetStreamAsync(
            $"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}", cancellationToken);
        using var release = await JsonDocument.ParseAsync(body, cancellationToken: cancellationToken);

        var assets = release.RootElement.GetProperty("assets").EnumerateArray()
            .Select(asset => asset.GetProperty("name").GetString())
            .OfType<string>()
            .ToArray();
        return (tag, assets);
    }
}
