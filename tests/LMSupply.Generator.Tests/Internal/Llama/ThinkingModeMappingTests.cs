using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;
using Xunit;

namespace LMSupply.Generator.Tests.Internal.Llama;

/// <summary>
/// Tests the mapping from the public <see cref="ThinkingMode"/> to the server-level
/// <c>enable_thinking</c> flag forwarded via chat_template_kwargs. Auto must map to null so the
/// model's own template default is preserved (Qwen3 thinks, Gemma does not) — the non-negotiable
/// "don't change current default" constraint. HW-free.
/// </summary>
public class ThinkingModeMappingTests
{
    [Theory]
    [InlineData(ThinkingMode.Auto, null)]
    [InlineData(ThinkingMode.On, true)]
    [InlineData(ThinkingMode.Off, false)]
    public void ThinkingModeToEnableFlag_Maps(ThinkingMode mode, bool? expected)
        => LlamaServerGeneratorModel.ThinkingModeToEnableFlag(mode).Should().Be(expected);

    [Fact]
    public void GenerationOptions_DefaultThinking_IsAuto_PreservesModelDefault()
        => new GenerationOptions().Thinking.Should().Be(ThinkingMode.Auto,
            "the default must not change a model's built-in thinking behavior");
}
