using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Talks to the real GitHub Releases API (excluded from CI via <c>Category=Integration</c>). Exists
/// because the fake in <see cref="LlamaServerLatestResolutionTests"/> encodes an assumption about
/// upstream's release scheme — this is the one test that would notice llama.cpp changing it again.
/// Asserts only the invariant that must hold under any scheme: "latest" resolves to a build tag
/// whose release carries a platform asset.
/// </summary>
[Trait("Category", "Integration")]
public sealed class LlamaServerLatestResolutionLiveTests
{
    [Fact]
    public async Task Live_GetLatestVersionAsync_ResolvesToBuildTagWithPlatformAsset()
    {
        using var downloader = new LlamaServerDownloader(Path.Combine(Path.GetTempPath(), "lmsupply-live-" + Guid.NewGuid().ToString("N")));

        var latest = await downloader.GetLatestVersionAsync(TestContext.Current.CancellationToken);

        LlamaServerDownloader.IsBuildTag(latest).Should().BeTrue($"latest resolved to '{latest}', which is not a bNNNNN build tag");

        var asset = await downloader.GetAssetAsync(latest, LlamaServerBackend.Cpu, TestContext.Current.CancellationToken);
        asset.Should().NotBeNull($"build {latest} must carry a CPU asset for this platform");
        asset!.Version.Should().Be(latest);
    }
}
