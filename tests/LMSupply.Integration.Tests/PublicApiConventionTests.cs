using Iyu.Conventions.Testing;
using Xunit;

namespace LMSupply.Integration.Tests;

/// <summary>
/// The public surface follows the two API rules of the ecosystem: every public async method takes a
/// <see cref="CancellationToken"/>, and failure is reported by an exception rather than by a returned object carrying a
/// success flag and an error. The scans are <c>Iyu.Conventions.Testing</c>'s, over the same assemblies as the
/// operational-language scan.
/// </summary>
/// <remarks>
/// The rosters are the methods that break a rule today. Shrink them; never grow them silently. A change to a listed
/// method's parameters changes its entry, which is a roster change on purpose.
/// </remarks>
public class PublicApiConventionTests
{
    private static readonly string[] KnownUncancellable =
    [
        // Disposal, like DisposeAsync: releasing the pool is not something a caller abandons halfway.
        "LMSupply.Llama.Server.LlamaServerPool.DisposeInstanceAsync()",
        // Fire-and-forget: returns at once; the background check is bounded by RuntimeUpdateOptions.VersionCheckTimeout, so a caller token would have nothing to cancel.
        "LMSupply.Runtime.RuntimeUpdateService.TriggerBackgroundCheckAsync(String, String, PlatformInfo, Func<String, IProgress<DownloadProgress>, CancellationToken, Task<String>>)",
    ];

    private static readonly string[] KnownResultReturns =
    [
        "LMSupply.Llama.Server.LlamaServerUpdateService.CheckAndApplyUpdateAsync(LlamaServerBackend, IProgress<DownloadProgress>, CancellationToken)",
        "LMSupply.Llama.Server.LlamaServerUpdateService.GetServerPathAsync(LlamaServerBackend, IProgress<DownloadProgress>, CancellationToken)",
        "LMSupply.Llama.Server.LlamaServerUpdateService.RollbackAsync(LlamaServerBackend, CancellationToken)",
    ];

    [Fact]
    public void PublicAsyncMethods_TakeACancellationToken() =>
        AsyncCancellation.Scan(OptionsReachabilityRosterTests.LibraryAssemblies()).ShouldMatchRoster(KnownUncancellable);

    [Fact]
    public void PublicMethods_DoNotReturnResultObjects() =>
        ResultReturns.Scan(OptionsReachabilityRosterTests.LibraryAssemblies()).ShouldMatchRoster(KnownResultReturns);

    // Positive control: an empty roster would also pass if the scan saw no public method at all.
    [Fact]
    public void Scan_SeesThePublicSurface() =>
        Assert.True(ResultReturns.Scan(OptionsReachabilityRosterTests.LibraryAssemblies()).MembersRead > 0, "the scan read too few public methods");
}
