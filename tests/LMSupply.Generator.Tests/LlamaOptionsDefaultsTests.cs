using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Hardware;
using LMSupply.Llama.Server;

namespace LMSupply.Generator.Tests;

public class LlamaOptionsDefaultsTests
{
    [Fact]
    public void LlamaOptions_Default_TypeKIsAuto()
    {
        var opts = new LlamaOptions();
        opts.TypeK.Should().Be(KvCacheQuantizationType.Auto);
    }

    [Fact]
    public void LlamaOptions_Default_TypeVIsAuto()
    {
        var opts = new LlamaOptions();
        opts.TypeV.Should().Be(KvCacheQuantizationType.Auto);
    }

    [Fact]
    public void LlamaOptions_Default_SpeculativeDecodingIsAuto()
    {
        var opts = new LlamaOptions();
        opts.SpeculativeDecoding.Should().Be(SpeculativeDecodingMode.Auto);
    }

    [Fact]
    public void LlamaOptions_Default_RopeScalingIsDefault()
    {
        var opts = new LlamaOptions();
        opts.RopeScaling.Should().Be(RopeScalingMode.Default);
    }

    // ─── ResolveKvCacheType ───────────────────────────────────────────────
    [Theory]
    [InlineData(LlamaServerBackend.Cuda12,  "b8863", "q8_0")]
    [InlineData(LlamaServerBackend.Cuda13,  "b8863", "q8_0")]
    [InlineData(LlamaServerBackend.Metal,   "b8863", "q8_0")]
    [InlineData(LlamaServerBackend.Hip,     "b8863", "q8_0")]
    [InlineData(LlamaServerBackend.Vulkan,  "b8863", "q8_0")]   // ≥ 8500
    [InlineData(LlamaServerBackend.Vulkan,  "b8000", null)]     // < 8500 → F16 = null
    [InlineData(LlamaServerBackend.Cpu,     "b8863", null)]     // CPU → F16 = null
    [InlineData(LlamaServerBackend.Sycl,    "b8863", null)]     // SYCL → F16 = null
    public void ResolveKvCacheType_Auto_ReturnsExpectedString(
        LlamaServerBackend backend, string version, string? expected)
    {
        var result = LlamaServerGeneratorModel.ResolveKvCacheType(
            KvCacheQuantizationType.Auto, backend, version);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(KvCacheQuantizationType.F16,  null)]
    [InlineData(KvCacheQuantizationType.Q8_0, "q8_0")]
    [InlineData(KvCacheQuantizationType.Q4_0, "q4_0")]
    [InlineData(KvCacheQuantizationType.F32,  "f32")]
    public void ResolveKvCacheType_Explicit_ReturnsExpectedString(
        KvCacheQuantizationType type, string? expected)
    {
        var result = LlamaServerGeneratorModel.ResolveKvCacheType(
            type, LlamaServerBackend.Cpu, "b8863");
        result.Should().Be(expected);
    }

    // ─── ResolveSpecType ──────────────────────────────────────────────────
    // b8994 renamed --spec-type ngram → ngram-simple.
    [Theory]
    [InlineData(SpeculativeDecodingMode.Auto,  "b8994", "ngram-simple")]  // ≥ 8994 → new name
    [InlineData(SpeculativeDecodingMode.Auto,  "b8863", "ngram")]         // 8500–8993 → old name
    [InlineData(SpeculativeDecodingMode.Auto,  "b8000", null)]            // < 8500 → None
    [InlineData(SpeculativeDecodingMode.Auto,  null,    null)]            // unknown → None
    [InlineData(SpeculativeDecodingMode.Ngram, "b8994", "ngram-simple")]  // explicit, new server
    [InlineData(SpeculativeDecodingMode.Ngram, "b8000", "ngram")]         // explicit, old server
    [InlineData(SpeculativeDecodingMode.None,  "b8863", null)]
    public void ResolveSpecType_ReturnsExpectedString(
        SpeculativeDecodingMode mode, string? version, string? expected)
    {
        var result = LlamaServerGeneratorModel.ResolveSpecType(mode, version);
        result.Should().Be(expected);
    }

    // ─── BuildAdditionalArgs ──────────────────────────────────────────────
    [Fact]
    public void BuildAdditionalArgs_NoOptionsSet_ReturnsEmpty()
    {
        var args = LlamaServerGeneratorModel.BuildAdditionalArgs(new LlamaOptions());
        args.Should().BeEmpty();
    }

    [Fact]
    public void BuildAdditionalArgs_ThreadsSet_EmitsThreadsFlag()
    {
        var args = LlamaServerGeneratorModel.BuildAdditionalArgs(new LlamaOptions { Threads = 8 });
        args.Should().Equal("--threads", "8");
    }

    [Fact]
    public void BuildAdditionalArgs_AdditionalArgsSet_AppendsVerbatim()
    {
        var args = LlamaServerGeneratorModel.BuildAdditionalArgs(
            new LlamaOptions { AdditionalArgs = ["--verbose", "--log-colors"] });
        args.Should().Equal("--verbose", "--log-colors");
    }

    [Fact]
    public void BuildAdditionalArgs_ThreadsAndAdditionalArgsSet_AppendsAdditionalArgsAfterThreads()
    {
        var args = LlamaServerGeneratorModel.BuildAdditionalArgs(
            new LlamaOptions { Threads = 4, AdditionalArgs = ["--verbose"] });
        args.Should().Equal("--threads", "4", "--verbose");
    }

    [Fact]
    public void BuildAdditionalArgs_EmptyAdditionalArgsList_ReturnsEmpty()
    {
        var args = LlamaServerGeneratorModel.BuildAdditionalArgs(
            new LlamaOptions { AdditionalArgs = [] });
        args.Should().BeEmpty();
    }

    // ggml-org/llama.cpp#24343 — "Gemma4Assistant requires ctx_other to be set" during
    // llama.cpp's automatic memory-fitting. Workaround: -fit off, gated to this architecture only.
    [Fact]
    public void BuildAdditionalArgs_Gemma4AssistantArchitecture_EmitsFitOff()
    {
        var args = LlamaServerGeneratorModel.BuildAdditionalArgs(
            new LlamaOptions(), "gemma4_assistant");
        args.Should().Equal("-fit", "off");
    }

    [Theory]
    [InlineData("Gemma4_Assistant")]
    [InlineData("GEMMA4_ASSISTANT")]
    public void BuildAdditionalArgs_Gemma4AssistantArchitecture_IsCaseInsensitive(string architecture)
    {
        var args = LlamaServerGeneratorModel.BuildAdditionalArgs(new LlamaOptions(), architecture);
        args.Should().Equal("-fit", "off");
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("llama")]
    [InlineData("gemma3")]
    [InlineData("gemma4")]
    public void BuildAdditionalArgs_OtherArchitectures_DoesNotEmitFitOff(string? architecture)
    {
        var args = LlamaServerGeneratorModel.BuildAdditionalArgs(new LlamaOptions(), architecture);
        args.Should().BeEmpty();
    }

    [Fact]
    public void BuildAdditionalArgs_Gemma4AssistantArchitecture_FitOffPrecedesUserAdditionalArgs()
    {
        var args = LlamaServerGeneratorModel.BuildAdditionalArgs(
            new LlamaOptions { Threads = 4, AdditionalArgs = ["--verbose"] }, "gemma4_assistant");
        args.Should().Equal("--threads", "4", "-fit", "off", "--verbose");
    }
}
