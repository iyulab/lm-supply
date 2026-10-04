using System.Reflection;
using System.Runtime.Loader;
using Iyu.Conventions.Testing;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Every public option a caller can set must be read by the library. An option nothing reads is a
/// promise the library does not keep: setting it changes nothing and says nothing. In 0.63.0 three of
/// the transcriber's nine <c>TranscribeOptions</c> were never read, and <c>DisableAutoDownload</c> —
/// documented to throw when a model is not cached — was read by one module of the six that declare it.
/// </summary>
/// <remarks>
/// <para>
/// This scans the IL of every library assembly for a call to each option property's getter, outside
/// the options type itself, and pins the properties that have none. Wiring one up, or adding a new
/// option that nothing reads, then shows up here as a deliberate change to the roster.
/// </para>
/// <para>
/// It lives in this project because it is the one that references every module. It carries no
/// category trait, so it runs in CI.
/// </para>
/// <para>
/// Two limits, both deliberate. Reading is necessary, not sufficient — an option can be read and still
/// have no effect (<c>TranscribeOptions.Temperature</c> on a greedy decoder); that has no static signal.
/// And only properties an options type declares itself are checked: a property inherited from
/// <c>LMSupplyOptionsBase</c> may be honoured through a Core helper the module hands its options to,
/// which a direct-call scan cannot attribute to one module.
/// </para>
/// </remarks>
public class OptionsReachabilityRosterTests
{
    // Each entry has an open issue draft: wire the option, or remove it as a deliberate decision.
    private static readonly Dictionary<string, string[]> KnownUnread = new(StringComparer.Ordinal)
    {
        // The thirteen options this roster's first run found were each wired or removed (issue draft
        // "options nothing reads, across seven types"). A new entry needs its own draft.
        // Moving to Iyu.Conventions.Testing 0.3.0 (2026-10-03) found one more, GenerationOptions.IncludePromptInOutput
        // (every read copied it, no generator acted on it) — removed in 0.99.0.
    };

    private static readonly Lazy<OptionsReachabilityReport> Result = new(() =>
        OptionsReachability.Scan(LibraryAssemblies(), OptionsTypes.NamedWith("Options")));

    // The scan is Iyu.Conventions.Testing's, shared with the other repositories; a mismatch prints the roster it found.
    [Fact]
    public void EveryPublicOption_IsReadByTheLibrary_ExceptTheKnownRoster() =>
        Result.Value.ShouldMatchRoster(KnownUnread);

    // Positive controls: the scan must see reads it is known to have — same-assembly reads, reads that
    // live in async state machines, and a read from another assembly — or an empty roster above would
    // pass because the detector sees nothing.
    [Fact]
    public void Scan_SeesKnownReads()
    {
        var scan = Result.Value;

        scan.OptionTypes.Should().HaveCountGreaterThan(20, "the scan must find the library's options types");
        scan.Read.Should().Contain(
        [
            "LMSupply.Transcriber.TranscribeOptions.Language",
            "LMSupply.Transcriber.TranscribeOptions.MaxTokens",
            "LMSupply.Transcriber.TranscribeOptions.WordTimestamps",
            "LMSupply.Reranker.RerankerOptions.DisableAutoDownload",
        ]);
        scan.CrossAssemblyReads.Should().NotBeEmpty("some options are declared in one assembly and read in another");
    }

    // Every library assembly copied next to the tests — not the tests themselves, and not the console
    // host, which is a consumer: its reading an option does not make the library honour it. Loaded by
    // name because the compiler drops a project reference the test code never names.
    internal static List<Assembly> LibraryAssemblies()
        => [.. Directory.EnumerateFiles(AppContext.BaseDirectory, "LMSupply.*.dll")
            .Select(Path.GetFileNameWithoutExtension)
            .Where(name => name is not null
                && !name.EndsWith(".Tests", StringComparison.Ordinal)
                && !name.Contains(".Console", StringComparison.Ordinal))
            .Order(StringComparer.Ordinal)
            .Select(name => AssemblyLoadContext.Default.LoadFromAssemblyName(new AssemblyName(name!)))];
}
