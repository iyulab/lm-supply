using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests;

/// <summary>
/// Tests for LlamaServerConfig argument building.
/// </summary>
public class LlamaServerConfigArgsTests
{
    [Theory]
    [InlineData("spec-ngram",   8500)]
    [InlineData("kv-q8-vulkan", 8500)]
    public void GetMinimumBuild_FeatureKey_ReturnsExpectedBuild(string key, int expected)
    {
        LlamaServerVersionRequirements.GetMinimumBuild(key).Should().Be(expected);
    }

    [Fact]
    public void LlamaServerConfig_HasSpecTypeProperty()
    {
        var config = new LlamaServerConfig { ModelPath = "model.gguf", SpecType = "ngram" };
        config.SpecType.Should().Be("ngram");
    }

    [Fact]
    public void LlamaServerConfig_HasYarnProperties()
    {
        var config = new LlamaServerConfig
        {
            ModelPath = "model.gguf",
            RopeScaling = "yarn",
            YarnOriginalContext = 4096,
            YarnExtensionFactor = -1f,
            YarnAttentionFactor = 1.0f,
            YarnBetaFast = 32f,
            YarnBetaSlow = 1f
        };
        config.RopeScaling.Should().Be("yarn");
        config.YarnOriginalContext.Should().Be(4096u);
    }
}
