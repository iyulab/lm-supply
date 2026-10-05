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
        "LMSupply.Download.DownloadManifest.ReadAsync(String)",
        "LMSupply.Download.DownloadManifest.WriteAsync(String, DownloadManifest)",
        "LMSupply.Generator.LocalGenerator.LoadFromPathAsync(String, GeneratorOptions)",
        "LMSupply.Llama.Server.LlamaServerPool.DisposeInstanceAsync()",
        "LMSupply.Llama.Server.LlamaServerPool.ReleaseIdleAsync()",
        "LMSupply.Runtime.RuntimeUpdateService.TriggerBackgroundCheckAsync(String, String, PlatformInfo, Func<String, IProgress<DownloadProgress>, CancellationToken, Task<String>>)",
        "LMSupply.Text.TokenizerFactory.CreateAutoAsync(String, Int32)",
        "LMSupply.Text.TokenizerFactory.CreateAutoPairAsync(String, Int32)",
        "LMSupply.Text.TokenizerFactory.CreateAutoSequenceAsync(String, Int32)",
        "LMSupply.Text.TokenizerFactory.CreateSentencePiecePairAsync(String, Int32)",
        "LMSupply.Text.TokenizerFactory.CreateSentencePieceSequenceAsync(String, Int32)",
        "LMSupply.Text.TokenizerFactory.CreateWordPieceAsync(String, Int32)",
        "LMSupply.Text.TokenizerFactory.CreateWordPiecePairAsync(String, Int32)",
        "LMSupply.Text.VocabularyLoader.LoadFromModelDirectoryAsync(String)",
        "LMSupply.Text.VocabularyLoader.LoadFromTokenizerJsonAsync(String)",
        "LMSupply.Text.VocabularyLoader.LoadFromVocabJsonAsync(String)",
        "LMSupply.Text.VocabularyLoader.LoadFromVocabTxtAsync(String)",
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
