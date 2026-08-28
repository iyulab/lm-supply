using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests.Internal.Llama;

/// <summary>
/// Tests for SelectReportedContextLength — ensures MaxContextLength reports GGUF model capability,
/// not the VRAM-capped session budget (regression: lm-supply context_length misreport 2026-05-02).
/// </summary>
public class LlamaServerGeneratorModelContextLengthTests
{
    private static GgufMetadata MakeGgufMeta(int contextLength) =>
        new() { ContextLength = contextLength };

    // ─── GgufMetadata wins over session budget (the primary fix) ───

    [Fact]
    public void SelectReportedContextLength_GgufMetadataPresent_ReturnsGgufCapability()
    {
        // Qwen2.5-7B: GGUF says 32768, session budget is 4096
        var options = new GeneratorOptions();  // MaxContextLength = null
        var meta = MakeGgufMeta(32768);

        var result = LlamaServerGeneratorModel.SelectReportedContextLength(options, meta, 4096);

        result.Should().Be(32768, because: "GGUF metadata capability trumps session budget");
    }

    [Fact]
    public void SelectReportedContextLength_GgufMetadataPresent_IgnoresVramBudget()
    {
        var options = new GeneratorOptions();
        var meta = MakeGgufMeta(131072);  // Gemma4 E4B

        var result = LlamaServerGeneratorModel.SelectReportedContextLength(options, meta, 4096);

        result.Should().Be(131072);
    }

    // ─── Explicit user cap wins over GGUF ───

    [Fact]
    public void SelectReportedContextLength_ExplicitUserCap_WinsOverGguf()
    {
        var options = new GeneratorOptions { MaxContextLength = 8192 };
        var meta = MakeGgufMeta(32768);

        var result = LlamaServerGeneratorModel.SelectReportedContextLength(options, meta, 4096);

        result.Should().Be(8192, because: "explicit user cap overrides GGUF metadata");
    }

    // ─── Fallback to VRAM-capped budget when no GGUF metadata ───

    [Fact]
    public void SelectReportedContextLength_NoGgufMetadata_FallsBackToVramBudget()
    {
        var options = new GeneratorOptions();  // MaxContextLength = null
        var vramBudget = 2048;

        var result = LlamaServerGeneratorModel.SelectReportedContextLength(options, null, vramBudget);

        result.Should().Be(vramBudget, because: "VRAM-capped budget is the only available fallback");
    }

    [Fact]
    public void SelectReportedContextLength_NullGgufContextLength_FallsBackToVramBudget()
    {
        var options = new GeneratorOptions();
        var meta = new GgufMetadata { ContextLength = null };  // metadata present but no ContextLength

        var result = LlamaServerGeneratorModel.SelectReportedContextLength(options, meta, 4096);

        result.Should().Be(4096);
    }

    // ─── Pre-fix regression guard ───

    [Fact]
    public void SelectReportedContextLength_DefaultOptions_Qwen2_5_7B_Returns32768()
    {
        // Exact Filer regression scenario: gguf:qwen2.5-7b loaded with default options
        // Pre-fix: returned 4096 → Filer routed to CompactSystemPrompt
        // Post-fix: returns 32768 → Filer routes to DefaultSystemPrompt
        var options = new GeneratorOptions();
        var meta = MakeGgufMeta(32768);

        var result = LlamaServerGeneratorModel.SelectReportedContextLength(options, meta, 4096);

        result.Should().Be(32768, because: "Filer must receive 32768 to route DefaultSystemPrompt for Qwen2.5-7B");
    }
}
