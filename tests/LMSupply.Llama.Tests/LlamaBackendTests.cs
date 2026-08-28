using AwesomeAssertions;
using LMSupply.Llama;

namespace LMSupply.Llama.Tests;

public class LlamaBackendTests
{
    [Fact]
    public void LlamaBackend_ShouldHaveExpectedValues()
    {
        // Verify all expected backend types exist
        Enum.GetValues<LlamaBackend>().Should().Contain(LlamaBackend.Cpu);
        Enum.GetValues<LlamaBackend>().Should().Contain(LlamaBackend.Cuda12);
        Enum.GetValues<LlamaBackend>().Should().Contain(LlamaBackend.Cuda13);
        Enum.GetValues<LlamaBackend>().Should().Contain(LlamaBackend.Vulkan);
        Enum.GetValues<LlamaBackend>().Should().Contain(LlamaBackend.Metal);
        Enum.GetValues<LlamaBackend>().Should().Contain(LlamaBackend.Rocm);
    }

    [Theory]
    [InlineData(LlamaBackend.Cpu, "Cpu")]
    [InlineData(LlamaBackend.Cuda12, "Cuda12")]
    [InlineData(LlamaBackend.Cuda13, "Cuda13")]
    [InlineData(LlamaBackend.Vulkan, "Vulkan")]
    [InlineData(LlamaBackend.Metal, "Metal")]
    [InlineData(LlamaBackend.Rocm, "Rocm")]
    public void LlamaBackend_ShouldHaveCorrectNames(LlamaBackend backend, string expectedName)
    {
        backend.ToString().Should().Be(expectedName);
    }

    [Fact]
    public void LlamaBackend_Count_ShouldBeSix()
    {
        Enum.GetValues<LlamaBackend>().Length.Should().Be(6);
    }
}
