using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests;

/// <summary>
/// GGUF registry freshness smoke test — HEAD-requests every registered alias's resolved
/// HuggingFace file URL and fails if the repo or file does not exist.
/// </summary>
/// <remarks>
/// Deliberately NOT wired into any scheduled workflow — mirrors this project's existing
/// <c>Category=Performance</c> precedent (a check worth having on demand, not worth scheduling
/// continuously against a third-party service's availability). Run manually with:
/// <c>dotnet test --filter "Category=Integration&amp;FullyQualifiedName~GgufRegistryFreshnessTests"</c>
///
/// This is the deliberately lighter option chosen over a full <c>check-catalog-staleness.ps1</c>-style
/// scheduled automation (ecosystem decision, 2026-08-18): the 2026-08-17 one-time registry audit
/// found 4/13 aliases pointing at a nonexistent repo or file — Gemma 4's <c>DefaultFile</c> mismatch
/// (cycle-267) and phi-4-mini's wrong <c>RepoId</c> (cycle-271) — both class-of-defect failures this
/// smoke test catches directly. It does not track gradual staleness (a file that still exists but
/// whose content silently changed) the way TokenMeter's pricing catalog check does; that broader
/// scope was explicitly deferred pending evidence it is needed.
///
/// HuggingFace returns different status codes depending on failure mode, all non-2xx and therefore
/// all caught by <c>IsSuccessStatusCode</c>: a nonexistent/private repo returns 401 (not 404 — HF
/// does not distinguish "doesn't exist" from "you can't see it"), an existing repo with a wrong
/// filename returns 404, and a resolvable file redirects to a signed CDN URL with 200. Verified live
/// against all three cases before writing this test.
/// </remarks>
[Trait("Category", "Integration")]
public class GgufRegistryFreshnessTests
{
    private const string HuggingFaceFileBase = "https://huggingface.co";
    private static readonly HttpClient Client = new() { Timeout = TimeSpan.FromSeconds(30) };

    public static IEnumerable<object[]> AllAliases() =>
        GgufModelRegistry.GetAliases()
            .Where(a => !a.Equals("gguf:auto", StringComparison.OrdinalIgnoreCase))
            .Select(a => new object[] { a });

    [Theory]
    [MemberData(nameof(AllAliases))]
    public async Task Alias_ResolvedFileUrl_Exists(string alias)
    {
        var model = GgufModelRegistry.Resolve(alias);
        model.Should().NotBeNull(
            because: $"'{alias}' is listed by GgufModelRegistry.GetAliases() so Resolve must find it");

        var url = $"{HuggingFaceFileBase}/{model!.RepoId}/resolve/main/{model.DefaultFile}";
        using var request = new HttpRequestMessage(HttpMethod.Head, url);
        using var response = await Client.SendAsync(request, TestContext.Current.CancellationToken);

        response.IsSuccessStatusCode.Should().BeTrue(
            because: $"'{alias}' -> {model.RepoId}/{model.DefaultFile} must resolve to an existing " +
                     $"HuggingFace file (got {(int)response.StatusCode} {response.StatusCode})");
    }
}
