using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using Xunit;

namespace LMSupply.Generator.Tests;

public class GgufChatFormatDetectorTests
{
    [Theory]
    [InlineData("Llama-3.2-3B-Instruct-Q4_K_M.gguf", "llama3")]
    [InlineData("llama-2-7b-chat-Q4_K_M.gguf", "llama3")]
    [InlineData("Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", "llama3")]
    public void DetectFromFilename_LlamaModels_ReturnsLlama3(string filename, string expected)
    {
        var result = GgufChatFormatDetector.DetectFromFilename(filename);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("Qwen2.5-7B-Instruct-Q4_K_M.gguf", "chatml")]
    [InlineData("qwen-1.5-14b-chat.Q4_K_M.gguf", "chatml")]
    public void DetectFromFilename_QwenModels_ReturnsChatML(string filename, string expected)
    {
        var result = GgufChatFormatDetector.DetectFromFilename(filename);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("gemma-2-9b-it-Q4_K_M.gguf", "gemma")]
    [InlineData("gemma-7b-it.Q4_K_M.gguf", "gemma")]
    public void DetectFromFilename_GemmaModels_ReturnsGemma(string filename, string expected)
    {
        var result = GgufChatFormatDetector.DetectFromFilename(filename);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("gemma-4-E4B-it-Q4_K_M.gguf", "gemma4")]
    [InlineData("gemma-4-E2B-it-Q4_K_M.gguf", "gemma4")]
    [InlineData("gemma-4-26B-A4B-it-Q4_K_M.gguf", "gemma4")]
    [InlineData("gemma-4-31B-it-Q4_K_M.gguf", "gemma4")]
    public void DetectFromFilename_Gemma4Models_ReturnsGemma4(string filename, string expected)
    {
        var result = GgufChatFormatDetector.DetectFromFilename(filename);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf", "exaone")]
    [InlineData("exaone-3-7b-instruct.Q4_K_M.gguf", "exaone")]
    public void DetectFromFilename_ExaoneModels_ReturnsExaone(string filename, string expected)
    {
        var result = GgufChatFormatDetector.DetectFromFilename(filename);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf", "deepseek")]
    [InlineData("deepseek-coder-6.7b-instruct.Q4_K_M.gguf", "deepseek")]
    public void DetectFromFilename_DeepSeekModels_ReturnsDeepSeek(string filename, string expected)
    {
        var result = GgufChatFormatDetector.DetectFromFilename(filename);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "mistral")]
    [InlineData("mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", "mistral")]
    public void DetectFromFilename_MistralModels_ReturnsMistral(string filename, string expected)
    {
        var result = GgufChatFormatDetector.DetectFromFilename(filename);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("phi-3-mini-4k-instruct.Q4_K_M.gguf", "phi3")]
    [InlineData("Phi-4-mini-instruct.Q4_K_M.gguf", "phi3")]
    public void DetectFromFilename_PhiModels_ReturnsPhi3(string filename, string expected)
    {
        var result = GgufChatFormatDetector.DetectFromFilename(filename);
        result.Should().Be(expected);
    }

    [Theory]
    [InlineData("unknown-model.gguf", "chatml")]
    [InlineData("custom-model-v1.Q4_K_M.gguf", "chatml")]
    public void DetectFromFilename_UnknownModels_DefaultsToChatML(string filename, string expected)
    {
        var result = GgufChatFormatDetector.DetectFromFilename(filename);
        result.Should().Be(expected);
    }

    [Fact]
    public void DetectFromFilename_WithFullPath_ExtractsFilename()
    {
        var fullPath = "/home/user/.cache/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

        var result = GgufChatFormatDetector.DetectFromFilename(fullPath);

        result.Should().Be("llama3");
    }

    [Theory]
    [InlineData("llama3", true)]
    [InlineData("chatml", true)]
    [InlineData("gemma", true)]
    [InlineData("gemma4", true)]
    [InlineData("exaone", true)]
    [InlineData("deepseek", true)]
    [InlineData("mistral", true)]
    [InlineData("phi3", true)]
    [InlineData("unknown_format", false)]
    public void IsSupported_ReturnsCorrectResult(string format, bool expected)
    {
        var result = GgufChatFormatDetector.IsSupported(format);
        result.Should().Be(expected);
    }
}
