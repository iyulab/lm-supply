using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;
using Xunit;

namespace LMSupply.Generator.Tests.Internal.Llama;

/// <summary>
/// Tests the mapping from the public <see cref="ToolChoice"/> to the server-level
/// <see cref="LMSupply.Llama.Server.LlamaToolChoice"/>. Unset/Auto must map to null so the
/// server's own "auto" default is preserved. HW-free.
/// </summary>
public class ToolChoiceMappingTests
{
    [Fact]
    public void ToToolChoice_Null_MapsToNull()
        => LlamaServerGeneratorModel.ToToolChoice(null).Should().BeNull();

    [Fact]
    public void ToToolChoice_Auto_MapsToNull_PreservesServerDefault()
        => LlamaServerGeneratorModel.ToToolChoice(ToolChoice.Auto).Should().BeNull();

    [Fact]
    public void ToToolChoice_None_MapsToLlamaNone()
        => LlamaServerGeneratorModel.ToToolChoice(ToolChoice.None)
            .Should().BeSameAs(LMSupply.Llama.Server.LlamaToolChoice.None);

    [Fact]
    public void ToToolChoice_Required_MapsToLlamaRequired()
        => LlamaServerGeneratorModel.ToToolChoice(ToolChoice.Required)
            .Should().BeSameAs(LMSupply.Llama.Server.LlamaToolChoice.Required);

    [Fact]
    public void ToToolChoice_Function_MapsToLlamaFunction_PreservesName()
    {
        var mapped = LlamaServerGeneratorModel.ToToolChoice(ToolChoice.Function("get_weather"));

        mapped.Should().NotBeNull();
        mapped!.Mode.Should().Be(LMSupply.Llama.Server.LlamaToolChoiceMode.Function);
        mapped.FunctionName.Should().Be("get_weather");
    }

    [Fact]
    public void GenerationOptions_DefaultToolChoice_IsNull_PreservesModelDefault()
        => new GenerationOptions().ToolChoice.Should().BeNull(
            "the default must not change a model's built-in tool-choice behavior (auto)");
}
