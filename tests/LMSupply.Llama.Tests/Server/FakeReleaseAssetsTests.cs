using AwesomeAssertions;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A fake release listing declares an asset's size and the fake hub serves the asset's bytes — from two
/// separate builds of the same archive. Those bytes must be identical every time, or the download's
/// length check (the real one, 0.68.1) rejects the fixture's own asset: the CI run of 2026-09-22 failed
/// with "141 bytes but the repository listing says 139" on a test that waits 600 ms between the two
/// builds, because an archive entry stamped with "now" compresses to a different length across a second.
/// </summary>
public sealed class FakeReleaseAssetsTests
{
    [Fact]
    public async Task TheFakeArchive_IsByteForByteTheSame_AcrossBuildsSecondsApart()
    {
        var first = FakeReleaseAssets.ServerArchive();
        await Task.Delay(TimeSpan.FromMilliseconds(2100), TestContext.Current.CancellationToken);
        var second = FakeReleaseAssets.ServerArchive();

        second.Should().Equal(first, "the listing's size and the served bytes come from separate builds");
        FakeReleaseAssets.ServerArchiveSize.Should().Be(first.Length);
    }
}
