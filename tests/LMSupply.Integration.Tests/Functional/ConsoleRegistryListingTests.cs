using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// The console's registry listing names what each domain actually loads. Six of its lists were hand-written and had drifted to
/// repositories the library never loads (Tesseract for OCR, YOLOv8 for the detector, SAM ViT for the segmenter, openai/whisper-*
/// for the transcriber, a missing opus-mt-en-ko for the translator).
/// </summary>
[Trait("Category", "Functional")]
[Trait("Domain", "ConsoleHost")]
public sealed class ConsoleRegistryListingTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>>
{
    public static TheoryData<string, string[]> RegistryDomains => new()
    {
        { "transcriber", [.. LMSupply.Transcriber.LocalTranscriber.Registry.GetAvailableModels().Select(m => m.Id)] },
        { "synthesizer", [.. LMSupply.Synthesizer.LocalSynthesizer.Registry.GetAvailableModels().Select(m => m.Id)] },
        { "translator", [.. LMSupply.Translator.LocalTranslator.Registry.GetAvailableModels().Select(m => m.Id)] },
        { "detector", [.. LMSupply.Detector.LocalDetector.Registry.GetAvailableModels().Select(m => m.Id.Split(':')[0])] },
        { "segmenter", [.. LMSupply.Segmenter.LocalSegmenter.Registry.GetAvailableModels().Select(m => m.Id)] },
    };

    [Theory]
    [MemberData(nameof(RegistryDomains))]
    public async Task EveryListedRepository_IsOneTheDomainLoads(string type, string[] registryRepos)
    {
        using var client = factory.CreateClient();
        var json = await client.GetFromJsonAsync<JsonElement>($"/api/registry/models/{type}", TestContext.Current.CancellationToken);

        var listed = json.GetProperty("models").EnumerateArray().Select(m => m.GetProperty("repoId").GetString()!).ToList();

        Assert.NotEmpty(listed);
        Assert.All(listed, repo => Assert.Contains(repo, registryRepos));
    }
}
