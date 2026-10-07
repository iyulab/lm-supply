using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Pins the asset selection to the asset lists llama.cpp actually publishes. Each platform × architecture
/// × backend must match exactly one archive — the server build, never the CUDA runtime companion that
/// shares its name as a suffix — so the choice does not depend on the order the releases API lists assets in.
/// HW-free / network-free: the lists are copied from the GitHub releases API.
/// </summary>
public class LlamaServerAssetSelectionTests
{
    /// <summary>Release b11459 (2026-10): current naming — Linux CUDA, ROCm, split SYCL builds, build-numbered Linux cudart.</summary>
    private static readonly string[] B11459 =
    [
        "cudart-llama-b11459-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "cudart-llama-b11459-bin-ubuntu-cuda-13.4-arm64.tar.gz",
        "cudart-llama-b11459-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.4-arm64.zip",
        "cudart-llama-bin-win-cuda-13.4-x64.zip",
        "llama-b11459-bin-android-arm64-snapdragon.tar.gz",
        "llama-b11459-bin-android-arm64.tar.gz",
        "llama-b11459-bin-linux-arm64-snapdragon.tar.gz",
        "llama-b11459-bin-macos-arm64.tar.gz",
        "llama-b11459-bin-macos-x64.tar.gz",
        "llama-b11459-bin-ubuntu-arm64.tar.gz",
        "llama-b11459-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "llama-b11459-bin-ubuntu-cuda-13.4-arm64.tar.gz",
        "llama-b11459-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "llama-b11459-bin-ubuntu-openvino-2026.4.1-x64.tar.gz",
        "llama-b11459-bin-ubuntu-rocm-10.0-x64.tar.gz",
        "llama-b11459-bin-ubuntu-s390x.tar.gz",
        "llama-b11459-bin-ubuntu-sycl-fp16-x64.tar.gz",
        "llama-b11459-bin-ubuntu-sycl-fp32-x64.tar.gz",
        "llama-b11459-bin-ubuntu-vulkan-arm64.tar.gz",
        "llama-b11459-bin-ubuntu-vulkan-x64.tar.gz",
        "llama-b11459-bin-ubuntu-x64.tar.gz",
        "llama-b11459-bin-win-cpu-arm64.zip",
        "llama-b11459-bin-win-cpu-x64.zip",
        "llama-b11459-bin-win-cuda-12.4-x64.zip",
        "llama-b11459-bin-win-cuda-13.4-arm64.zip",
        "llama-b11459-bin-win-cuda-13.4-x64.zip",
        "llama-b11459-bin-win-opencl-adreno-arm64.zip",
        "llama-b11459-bin-win-openvino-2026.4.1-x64.zip",
        "llama-b11459-bin-win-rocm-10.0-x64.zip",
        "llama-b11459-bin-win-sycl-x64.zip",
        "llama-b11459-bin-win-vulkan-arm64.zip",
        "llama-b11459-bin-win-vulkan-x64.zip",
        "llama-b11459-ui.tar.gz",
        "llama-b11459-xcframework.zip",
    ];

    /// <summary>Release b7902: older naming a pinned version can still resolve — "hip-radeon", Windows-only CUDA.</summary>
    private static readonly string[] B7902 =
    [
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.1-x64.zip",
        "llama-b7902-bin-310p-openEuler-aarch64.tar.gz",
        "llama-b7902-bin-310p-openEuler-x86.tar.gz",
        "llama-b7902-bin-910b-openEuler-aarch64-aclgraph.tar.gz",
        "llama-b7902-bin-910b-openEuler-x86-aclgraph.tar.gz",
        "llama-b7902-bin-macos-arm64.tar.gz",
        "llama-b7902-bin-macos-x64.tar.gz",
        "llama-b7902-bin-ubuntu-s390x.tar.gz",
        "llama-b7902-bin-ubuntu-vulkan-x64.tar.gz",
        "llama-b7902-bin-ubuntu-x64.tar.gz",
        "llama-b7902-bin-win-cpu-arm64.zip",
        "llama-b7902-bin-win-cpu-x64.zip",
        "llama-b7902-bin-win-cuda-12.4-x64.zip",
        "llama-b7902-bin-win-cuda-13.1-x64.zip",
        "llama-b7902-bin-win-hip-radeon-x64.zip",
        "llama-b7902-bin-win-opencl-adreno-arm64.zip",
        "llama-b7902-bin-win-sycl-x64.zip",
        "llama-b7902-bin-win-vulkan-x64.zip",
        "llama-b7902-xcframework.zip",
    ];

    private static string[] Release(string tag) => tag switch
    {
        "b11459" => B11459,
        "b7902" => B7902,
        _ => throw new ArgumentOutOfRangeException(nameof(tag))
    };

    private static string[] ServerMatches(string tag, LlamaServerPlatform platform, LlamaServerArchitecture arch, LlamaServerBackend backend)
    {
        var pattern = LlamaServerDownloader.GetAssetPattern(platform, arch, backend);
        return Release(tag).Where(name => pattern.IsMatch(name)).ToArray();
    }

    [Theory]
    // Linux — the reported failure: the cudart companion was taken as the server archive
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda12, "llama-b11459-bin-ubuntu-cuda-12.8-x64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda13, "llama-b11459-bin-ubuntu-cuda-13.4-x64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.Arm64, LlamaServerBackend.Cuda13, "llama-b11459-bin-ubuntu-cuda-13.4-arm64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Cpu, "llama-b11459-bin-ubuntu-x64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.Arm64, LlamaServerBackend.Cpu, "llama-b11459-bin-ubuntu-arm64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Vulkan, "llama-b11459-bin-ubuntu-vulkan-x64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.Arm64, LlamaServerBackend.Vulkan, "llama-b11459-bin-ubuntu-vulkan-arm64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Hip, "llama-b11459-bin-ubuntu-rocm-10.0-x64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Sycl, "llama-b11459-bin-ubuntu-sycl-fp32-x64.tar.gz")]
    // Windows
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda12, "llama-b11459-bin-win-cuda-12.4-x64.zip")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda13, "llama-b11459-bin-win-cuda-13.4-x64.zip")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.Arm64, LlamaServerBackend.Cuda13, "llama-b11459-bin-win-cuda-13.4-arm64.zip")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cpu, "llama-b11459-bin-win-cpu-x64.zip")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.Arm64, LlamaServerBackend.Cpu, "llama-b11459-bin-win-cpu-arm64.zip")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Vulkan, "llama-b11459-bin-win-vulkan-x64.zip")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.Arm64, LlamaServerBackend.Vulkan, "llama-b11459-bin-win-vulkan-arm64.zip")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Hip, "llama-b11459-bin-win-rocm-10.0-x64.zip")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Sycl, "llama-b11459-bin-win-sycl-x64.zip")]
    // macOS
    [InlineData("b11459", LlamaServerPlatform.MacOS, LlamaServerArchitecture.Arm64, LlamaServerBackend.Metal, "llama-b11459-bin-macos-arm64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.MacOS, LlamaServerArchitecture.X64, LlamaServerBackend.Cpu, "llama-b11459-bin-macos-x64.tar.gz")]
    // Older naming
    [InlineData("b7902", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Hip, "llama-b7902-bin-win-hip-radeon-x64.zip")]
    [InlineData("b7902", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda13, "llama-b7902-bin-win-cuda-13.1-x64.zip")]
    [InlineData("b7902", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Vulkan, "llama-b7902-bin-ubuntu-vulkan-x64.tar.gz")]
    public void Backend_PicksExactlyItsServerArchive(
        string tag, LlamaServerPlatform platform, LlamaServerArchitecture arch, LlamaServerBackend backend, string expected)
    {
        ServerMatches(tag, platform, arch, backend).Should().Equal(expected);
    }

    [Theory]
    // No build for these: resolution moves on to the CPU fallback instead of taking a near-miss
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.Arm64, LlamaServerBackend.Cuda12)]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Metal)]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Metal)]
    [InlineData("b7902", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda12)]
    public void Backend_WithoutABuild_MatchesNothing(
        string tag, LlamaServerPlatform platform, LlamaServerArchitecture arch, LlamaServerBackend backend)
    {
        ServerMatches(tag, platform, arch, backend).Should().BeEmpty();
    }

    [Theory]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda12, "cudart-llama-b11459-bin-ubuntu-cuda-12.8-x64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda13, "cudart-llama-b11459-bin-ubuntu-cuda-13.4-x64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Linux, LlamaServerArchitecture.Arm64, LlamaServerBackend.Cuda13, "cudart-llama-b11459-bin-ubuntu-cuda-13.4-arm64.tar.gz")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda12, "cudart-llama-bin-win-cuda-12.4-x64.zip")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda13, "cudart-llama-bin-win-cuda-13.4-x64.zip")]
    [InlineData("b11459", LlamaServerPlatform.Windows, LlamaServerArchitecture.Arm64, LlamaServerBackend.Cuda13, "cudart-llama-bin-win-cuda-13.4-arm64.zip")]
    [InlineData("b7902", LlamaServerPlatform.Windows, LlamaServerArchitecture.X64, LlamaServerBackend.Cuda13, "cudart-llama-bin-win-cuda-13.1-x64.zip")]
    public void CudaBackend_PicksExactlyItsRuntimeCompanion(
        string tag, LlamaServerPlatform platform, LlamaServerArchitecture arch, LlamaServerBackend backend, string expected)
    {
        var pattern = LlamaServerDownloader.GetCudartAssetPattern(platform, arch, backend);

        pattern.Should().NotBeNull();
        Release(tag).Where(name => pattern!.IsMatch(name)).Should().Equal(expected);
    }
}
