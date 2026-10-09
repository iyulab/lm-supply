using System.Runtime.InteropServices;
using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// A caller that cancels <see cref="RuntimeUpdateService.CheckAndApplyUpdateAsync"/> gets
/// <see cref="OperationCanceledException"/>. The version check's own timeout means "no update known", but the caller's
/// cancellation is not an answer about updates.
/// </summary>
[Trait("Category", "Unit")]
public sealed class RuntimeUpdateServiceCancellationTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), $"lmsupply-update-cancel-{Guid.NewGuid():N}");

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_cacheDir))
            {
                Directory.Delete(_cacheDir, recursive: true);
            }
        }
        catch (IOException)
        {
            // Best effort.
        }
    }

    [Fact]
    public async Task CheckAndApplyUpdateAsync_WhenTheCallerCancelsDuringTheVersionCheck_Throws()
    {
        using var caller = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        using var http = new HttpClient(new CancellingHandler(caller));
        await using var service = new RuntimeUpdateService(
            $"test-{Guid.NewGuid():N}",
            new RuntimeUpdateOptions { CacheDirectory = _cacheDir },
            new NuGetPackageResolver(http));
        var platform = new PlatformInfo { OS = OSPlatform.Windows, Architecture = Architecture.X64, RuntimeIdentifier = "win-x64" };

        var act = () => service.CheckAndApplyUpdateAsync(
            "Some.Runtime.Package",
            "cpu",
            platform,
            "1.0.0",
            (_, _, _) => throw new InvalidOperationException("No download is expected."),
            ct: caller.Token);

        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    /// <summary>Cancels the caller's token when the version request goes out, then observes the request's token.</summary>
    private sealed class CancellingHandler(CancellationTokenSource caller) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            caller.Cancel();
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(new HttpResponseMessage(System.Net.HttpStatusCode.NotFound));
        }
    }
}
