using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Internal.Llama;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// The model info of a GGUF load describes the file it opened, which is not the alias's default when a
/// smaller cached quantization stood in because the default did not fit the memory budget.
/// </summary>
public class GgufLoadIdentityTests
{
    private static string CachedPath(string file) => Path.Combine("cache", "models--org--repo", "snapshots", "main", file);

    [Fact]
    public void Describe_SubstitutedQuantization_NamesTheLoadedQuantization()
    {
        var balanced = GgufModelRegistry.Resolve("gguf:gemma4-balanced")!;
        balanced.DefaultFile.Should().Be("gemma-4-E4B-it-Q8_0.gguf", "the case this pins is Q8_0 requested, Q4_0 loaded");

        var identity = GgufLoadIdentity.Describe("gguf:gemma4-balanced", balanced, CachedPath("gemma-4-E4B-it-Q4_0.gguf"));

        identity.ModelId.Should().Be("Gemma 4 E4B Instruct (Q4_0)");
        identity.RequestedModelId.Should().Be("gguf:gemma4-balanced");
        identity.RequestedFile.Should().Be("gemma-4-E4B-it-Q8_0.gguf");
    }

    [Fact]
    public void Describe_DefaultLoaded_KeepsTheRegistryDisplayName()
    {
        var balanced = GgufModelRegistry.Resolve("gguf:gemma4-balanced")!;

        var identity = GgufLoadIdentity.Describe("gguf:gemma4-balanced", balanced, CachedPath(balanced.DefaultFile));

        identity.ModelId.Should().Be(balanced.DisplayName);
    }

    [Fact]
    public void Describe_DisplayNameWithoutQuantization_AppendsTheLoadedOne()
    {
        var model = GgufModelRegistry.Resolve("gguf:gemma4-default")!;
        model.DisplayName.Should().Be("Gemma 4 E4B Instruct");

        var identity = GgufLoadIdentity.Describe("gguf:gemma4-default", model, CachedPath("gemma-4-E4B-it-Q2_K.gguf"));

        identity.ModelId.Should().Be("Gemma 4 E4B Instruct (Q2_K)");
    }

    [Fact]
    public void Describe_RegistryAlias_CarriesTheAliasKnownIssues()
    {
        // The model used to look its known issues up by ModelId — the display name, which is not a registry
        // key — so every registry load reported none.
        GgufModelRegistry.Resolve("Gemma 4 E4B Instruct (Q8_0)").Should().BeNull();
        var balanced = GgufModelRegistry.Resolve("gguf:gemma4-balanced")!;
        balanced.KnownIssues.Should().NotBeEmpty();

        var identity = GgufLoadIdentity.Describe("gguf:gemma4-balanced", balanced, CachedPath(balanced.DefaultFile));

        identity.KnownIssues.Should().BeEquivalentTo(balanced.KnownIssues);
    }

    [Fact]
    public void Describe_NoRegistryAlias_UsesTheRequestedId()
    {
        var identity = GgufLoadIdentity.Describe("org/some-model-GGUF", registryInfo: null, CachedPath("some-model-Q4_K_M.gguf"));

        identity.ModelId.Should().Be("org/some-model-GGUF");
        identity.RequestedModelId.Should().Be("org/some-model-GGUF");
        identity.RequestedFile.Should().BeNull();
        identity.KnownIssues.Should().BeEmpty();
    }

    [Fact]
    public void ModelInfo_ReportsSubstitutionOnlyWhenTheLoadedFileDiffers()
    {
        var info = new GeneratorModelInfo("Gemma 4 E4B Instruct (Q4_0)", "p", 4096, "gemma4", "llama-server-cuda")
        {
            RequestedModelId = "gguf:gemma4-balanced",
            RequestedFile = "gemma-4-E4B-it-Q8_0.gguf",
            LoadedFile = "gemma-4-E4B-it-Q4_0.gguf",
        };
        info.IsQuantizationSubstituted.Should().BeTrue();
        ((IModelInfoBase)info).AliasName.Should().Be("gguf:gemma4-balanced");

        (info with { LoadedFile = "GEMMA-4-E4B-it-Q8_0.gguf" }).IsQuantizationSubstituted.Should().BeFalse();
        (info with { RequestedFile = null }).IsQuantizationSubstituted.Should().BeFalse("no alias was requested");
        (info with { RequestedModelId = null }).As<IModelInfoBase>().AliasName.Should().Be(info.ModelId);
    }

    [Theory]
    [InlineData("gemma-4-E4B-it-Q4_0.gguf", "Q4_0")]
    [InlineData("gemma-4-E4B-it-Q8_0.gguf", "Q8_0")]
    [InlineData("Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "IQ4_XS")]
    [InlineData("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "Q4_K_XL")]
    [InlineData("Meta-Llama-3-8B.Q4_K_M.gguf", "Q4_K_M")]
    [InlineData("model-q5_k_s-00001-of-00003.gguf", "Q5_K_S")]
    [InlineData("model-BF16.gguf", "BF16")]
    [InlineData("Qwen3-8B.gguf", null)]
    public void QuantizationLabel_FromFileName(string file, string? expected)
        => GgufQuantizationLabel.FromFileName(file).Should().Be(expected);
}
