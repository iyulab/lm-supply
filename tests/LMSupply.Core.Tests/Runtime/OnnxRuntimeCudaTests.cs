using AwesomeAssertions;
using LMSupply.Runtime;
using Xunit;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// ONNX Runtime 1.27 and later publish their NuGet GPU package built against CUDA 13; the provider then needs CUDA 13
/// libraries and a driver that supports CUDA 13. The check used to accept any installed CUDA and told the user to install
/// CUDA 12, which cannot load the 1.30 provider (it depends on cublasLt64_13.dll).
/// </summary>
public sealed class OnnxRuntimeCudaTests
{
    [Theory]
    [InlineData("1.26.0", 12)]
    [InlineData("1.27.0", 13)]
    [InlineData("1.30.0", 13)]
    [InlineData(null, 13)]
    [InlineData("not-a-version", 13)]
    public void RequiredCudaMajor_FollowsTheOnnxRuntimeVersion(string? version, int expected)
        => OnnxRuntimeCuda.RequiredCudaMajor(version).Should().Be(expected);

    [Fact]
    public void TheLoadedRuntime_NeedsCuda13()
        // This repository pins ONNX Runtime 1.30.0; a downgrade below 1.27 should change this deliberately.
        => OnnxRuntimeCuda.RequiredCudaMajorForLoadedRuntime.Should().Be(13);

    [Fact]
    public void ADriverForCuda12_CannotRunTheCuda13Build_AndSaysWhy()
    {
        var shortfall = OnnxRuntimeCuda.DriverShortfall(12, 9, 13);

        shortfall.Should().Contain("CUDA up to 12.9").And.Contain("needs CUDA 13");
    }

    [Theory]
    [InlineData(13, 0)]
    [InlineData(13, 2)]
    [InlineData(14, 0)]
    public void ADriverForTheRequiredMajorOrNewer_HasNoShortfall(int major, int minor)
        => OnnxRuntimeCuda.DriverShortfall(major, minor, 13).Should().BeNull();
}
