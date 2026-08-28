using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Tests for the CUDA-runtime companion asset matcher. llama.cpp ships the CUDA runtime
/// (cudart/cublas/cublasLt) in a SEPARATE release asset, e.g. "cudart-llama-bin-win-cuda-12.4-x64.zip",
/// not inside the main "llama-b&lt;n&gt;-bin-win-cuda-12.4-x64.zip" archive. Without it the cuda binary
/// silently falls back to CPU. <see cref="LlamaServerDownloader.GetCudartAssetPattern"/> finds the
/// matching cudart asset (any cuda minor for the backend major), and returns null for non-CUDA
/// backends (no companion needed). HW-free / network-free: pure pattern matching.
/// </summary>
public class LlamaServerCudartAssetTests
{
    [Theory]
    [InlineData("cudart-llama-bin-win-cuda-12.4-x64.zip", true)]
    [InlineData("cudart-llama-bin-win-cuda-12.9-x64.zip", true)]
    [InlineData("cudart-llama-bin-win-cuda-13.3-x64.zip", false)] // wrong major
    [InlineData("llama-b9692-bin-win-cuda-12.4-x64.zip", false)]   // the main asset, not cudart
    [InlineData("cudart-llama-bin-win-cuda-12.4-arm64.zip", false)] // wrong arch
    public void Cuda12_Windows_X64_MatchesOnlyCudart12(string assetName, bool expected)
    {
        var pattern = LlamaServerDownloader.GetCudartAssetPattern(
            LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda12);

        pattern.Should().NotBeNull("CUDA backends need a cudart companion");
        pattern!.IsMatch(assetName).Should().Be(expected);
    }

    [Fact]
    public void Cuda13_Windows_X64_MatchesCudart13()
    {
        var pattern = LlamaServerDownloader.GetCudartAssetPattern(
            LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda13);

        pattern.Should().NotBeNull();
        pattern!.IsMatch("cudart-llama-bin-win-cuda-13.3-x64.zip").Should().BeTrue();
        pattern.IsMatch("cudart-llama-bin-win-cuda-12.4-x64.zip").Should().BeFalse("wrong major");
    }

    [Theory]
    [InlineData(LlamaServerBackend.Cpu)]
    [InlineData(LlamaServerBackend.Vulkan)]
    [InlineData(LlamaServerBackend.Metal)]
    [InlineData(LlamaServerBackend.Hip)]
    public void NonCudaBackend_HasNoCompanion(LlamaServerBackend backend)
    {
        LlamaServerDownloader.GetCudartAssetPattern(
            LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, backend)
            .Should().BeNull("only CUDA backends need a separate runtime package");
    }
}
