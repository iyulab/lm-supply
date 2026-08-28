using AwesomeAssertions;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;
using Xunit;

namespace LMSupply.Generator.Tests;

public class ContextLengthDiagnosticsTests
{
    [Fact]
    public void AdjustedContextLength_IsNullByDefault()
    {
        var info = new GeneratorModelInfo("id", "/path/model.gguf", 4096, "chatml", "llama-server-cpu");

        info.AdjustedContextLength.Should().BeNull();
    }

    [Fact]
    public void AdjustedContextLength_ReflectsSetValue()
    {
        var info = new GeneratorModelInfo("id", "/path/model.gguf", 16384, "gemma4", "llama-server-cuda")
        {
            AdjustedContextLength = 3530
        };

        info.AdjustedContextLength.Should().Be(3530);
        info.MaxContextLength.Should().Be(16384);
    }

    [Fact]
    public void AdjustedContextLength_NullWhenNotSet()
    {
        var info = new GeneratorModelInfo("id", "/path/model.gguf", 4096, "chatml", "llama-server-cpu")
        {
            AdjustedContextLength = null
        };

        info.AdjustedContextLength.Should().BeNull();
    }

    [Fact]
    public void ResolveAdjustedContextLength_ReturnsNullWhenEqual()
    {
        var result = LlamaServerGeneratorModel.ResolveAdjustedContextLength(4096, 4096);

        result.Should().BeNull();
    }

    [Fact]
    public void ResolveAdjustedContextLength_ReturnsEffectiveWhenDiffers()
    {
        var result = LlamaServerGeneratorModel.ResolveAdjustedContextLength(16384, 3530);

        result.Should().Be(3530);
    }

    [Fact]
    public void SelectReportedContextLength_PrefersExplicitUserSetting()
    {
        // SelectReportedContextLength returns user's explicit cap even after VRAM adjustment
        var options = new GeneratorOptions { MaxContextLength = 16384 };

        var reported = LlamaServerGeneratorModel.SelectReportedContextLength(options, ggufMetadata: null, vramCappedBudget: 3530);

        reported.Should().Be(16384);
    }
}
