using System.Runtime.InteropServices;
using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

public class RuntimePackageRegistryTests
{
    // --- GetPackageConfig: OnnxRuntime ---

    [Fact]
    public void GetPackageConfig_OnnxRuntime_Cpu_ReturnsCorrectPackage()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntime, "cpu");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntime");
        config.NativeLibraryName.Should().Be("onnxruntime");
        config.AdditionalLibraries.Should().BeEmpty();
    }

    [Fact]
    public void GetPackageConfig_OnnxRuntime_DirectML_ReturnsCorrectPackage()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntime, "directml");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntime.DirectML");
    }

    [Fact]
    public void GetPackageConfig_OnnxRuntime_Cuda_ReturnsGpuPackage()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntime, "cuda");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntime.Gpu");
        config.AdditionalLibraries.Should().Contain("onnxruntime_providers_cuda");
        config.AdditionalLibraries.Should().Contain("onnxruntime_providers_shared");
    }

    [Fact]
    public void GetPackageConfig_OnnxRuntime_Cuda_WinX64_ReturnsPlatformSpecificPackage()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntime, "cuda", "win-x64");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntime.Gpu.Windows");
    }

    [Fact]
    public void GetPackageConfig_OnnxRuntime_Cuda_LinuxX64_ReturnsLinuxPackage()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntime, "cuda", "linux-x64");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntime.Gpu");
    }

    [Fact]
    public void GetPackageConfig_OnnxRuntime_Cuda12_ReturnsGpuPackage()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntime, "cuda12");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntime.Gpu");
    }

    // --- GetPackageConfig: OnnxRuntimeGenAI ---

    [Fact]
    public void GetPackageConfig_GenAI_Cpu_ReturnsCorrectPackage()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntimeGenAI, "cpu");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntimeGenAI");
        config.NativeLibraryName.Should().Be("onnxruntime-genai");
    }

    [Fact]
    public void GetPackageConfig_GenAI_DirectML_ReturnsCorrectPackage()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntimeGenAI, "directml");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntimeGenAI.DirectML");
    }

    [Fact]
    public void GetPackageConfig_GenAI_Cuda_ReturnsCudaPackage()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntimeGenAI, "cuda");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntimeGenAI.Cuda");
        config.AdditionalLibraries.Should().Contain("onnxruntime-genai-cuda");
    }

    [Fact]
    public void GetPackageConfig_GenAI_Cuda_NoPlatformOverride()
    {
        // GenAI CUDA does NOT have platform-specific overrides
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntimeGenAI, "cuda", "win-x64");

        config.Should().NotBeNull();
        // Should NOT return a Windows-specific package (that's OnnxRuntime-only)
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntimeGenAI.Cuda");
    }

    // --- Provider normalization ---

    [Fact]
    public void GetPackageConfig_AutoProvider_FallsToCpu()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntime, "auto");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntime");
    }

    [Fact]
    public void GetPackageConfig_CaseInsensitive()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntime, "DIRECTML");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntime.DirectML");
    }

    [Fact]
    public void GetPackageConfig_UnknownProvider_FallsToCpu()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(
            RuntimePackageRegistry.PackageTypes.OnnxRuntime, "unknown-provider");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntime");
    }

    [Fact]
    public void GetPackageConfig_PackageTypeCaseInsensitive()
    {
        var config = RuntimePackageRegistry.GetPackageConfig("ONNXRUNTIME-GENAI", "cpu");

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Microsoft.ML.OnnxRuntimeGenAI");
    }

    // --- IsCudaProvider ---

    [Theory]
    [InlineData("cuda", true)]
    [InlineData("cuda11", true)]
    [InlineData("cuda12", true)]
    [InlineData("CUDA", true)]
    [InlineData("Cuda12", true)]
    [InlineData("cpu", false)]
    [InlineData("directml", false)]
    [InlineData("coreml", false)]
    public void IsCudaProvider_ReturnsExpected(string provider, bool expected)
    {
        RuntimePackageRegistry.IsCudaProvider(provider).Should().Be(expected);
    }

    // --- GetSupportedProviders ---

    [Fact]
    public void GetSupportedProviders_OnnxRuntime_ReturnsKnownProviders()
    {
        var providers = RuntimePackageRegistry.GetSupportedProviders(
            RuntimePackageRegistry.PackageTypes.OnnxRuntime).ToList();

        providers.Should().Contain("cpu");
        providers.Should().Contain("directml");
        providers.Should().Contain("cuda");
    }

    [Fact]
    public void GetSupportedProviders_GenAI_ReturnsKnownProviders()
    {
        var providers = RuntimePackageRegistry.GetSupportedProviders(
            RuntimePackageRegistry.PackageTypes.OnnxRuntimeGenAI).ToList();

        providers.Should().Contain("cpu");
        providers.Should().Contain("directml");
        providers.Should().Contain("cuda");
    }

    // --- GetNativeLibraryFileName ---

    [Fact]
    public void GetNativeLibraryFileName_Windows_ReturnsDll()
    {
        var platform = new PlatformInfo
        {
            OS = OSPlatform.Windows,
            Architecture = Architecture.X64,
            RuntimeIdentifier = "win-x64"
        };

        var fileName = RuntimePackageRegistry.GetNativeLibraryFileName("onnxruntime", platform);

        fileName.Should().Be("onnxruntime.dll");
    }

    [Fact]
    public void GetNativeLibraryFileName_Linux_ReturnsSo()
    {
        var platform = new PlatformInfo
        {
            OS = OSPlatform.Linux,
            Architecture = Architecture.X64,
            RuntimeIdentifier = "linux-x64"
        };

        var fileName = RuntimePackageRegistry.GetNativeLibraryFileName("onnxruntime", platform);

        fileName.Should().Be("libonnxruntime.so");
    }

    [Fact]
    public void GetNativeLibraryFileName_MacOS_ReturnsDylib()
    {
        var platform = new PlatformInfo
        {
            OS = OSPlatform.OSX,
            Architecture = Architecture.Arm64,
            RuntimeIdentifier = "osx-arm64"
        };

        var fileName = RuntimePackageRegistry.GetNativeLibraryFileName("onnxruntime", platform);

        fileName.Should().Be("libonnxruntime.dylib");
    }

    // --- PackageConfig record ---

    [Fact]
    public void PackageConfig_DefaultAdditionalLibraries_IsEmptyArray()
    {
        var config = new RuntimePackageRegistry.PackageConfig("Test.Package", "testlib");

        config.AdditionalLibraries.Should().NotBeNull();
        config.AdditionalLibraries.Should().BeEmpty();
    }

    [Fact]
    public void PackageConfig_WithAdditionalLibraries_SetCorrectly()
    {
        var config = new RuntimePackageRegistry.PackageConfig(
            "Test.Package", "testlib", ["extra1", "extra2"]);

        config.AdditionalLibraries.Should().HaveCount(2);
        config.AdditionalLibraries.Should().Contain("extra1");
    }
}
