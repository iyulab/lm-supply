using System.Runtime.InteropServices;
using FluentAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

public class PlatformInfoTests
{
    [Fact]
    public void DetectPlatform_ShouldReturnValidPlatformInfo()
    {
        // Act
        var platform = EnvironmentDetector.DetectPlatform();

        // Assert
        platform.Should().NotBeNull();
        platform.RuntimeIdentifier.Should().NotBeNullOrEmpty();
        platform.Is64Bit.Should().Be(Environment.Is64BitProcess);
    }

    [Fact]
    public void DetectPlatform_ShouldDetectOperatingSystem()
    {
        // Act
        var platform = EnvironmentDetector.DetectPlatform();

        // Assert
        if (OperatingSystem.IsWindows())
            platform.OS.Should().Be(OSPlatform.Windows);
        else if (OperatingSystem.IsLinux())
            platform.OS.Should().Be(OSPlatform.Linux);
        else if (OperatingSystem.IsMacOS())
            platform.OS.Should().Be(OSPlatform.OSX);
    }

    [Fact]
    public void DetectPlatform_ShouldDetectArchitecture()
    {
        // Act
        var platform = EnvironmentDetector.DetectPlatform();

        // Assert
        platform.Architecture.Should().BeOneOf(
            Architecture.X64,
            Architecture.Arm64,
            Architecture.X86,
            Architecture.Arm);
    }

    [Fact]
    public void RuntimeIdentifier_ShouldFollowDotNetConvention()
    {
        // Act
        var platform = EnvironmentDetector.DetectPlatform();

        // Assert
        // RID format: {os}-{arch} e.g., "win-x64", "linux-arm64"
        platform.RuntimeIdentifier.Should().Contain("-");
        platform.RuntimeIdentifier.Should().MatchRegex(@"^[a-z]+-[a-z0-9]+$");
    }

    [Fact]
    public void IsWindows_ShouldMatchOperatingSystem()
    {
        // Act
        var platform = EnvironmentDetector.DetectPlatform();

        // Assert
        platform.IsWindows.Should().Be(OperatingSystem.IsWindows());
    }

    [Fact]
    public void IsLinux_ShouldMatchOperatingSystem()
    {
        // Act
        var platform = EnvironmentDetector.DetectPlatform();

        // Assert
        platform.IsLinux.Should().Be(OperatingSystem.IsLinux());
    }

    [Fact]
    public void IsMacOS_ShouldMatchOperatingSystem()
    {
        // Act
        var platform = EnvironmentDetector.DetectPlatform();

        // Assert
        platform.IsMacOS.Should().Be(OperatingSystem.IsMacOS());
    }

    [Fact]
    public void NativeLibraryExtension_ShouldBeCorrectForPlatform()
    {
        // Act
        var platform = EnvironmentDetector.DetectPlatform();

        // Assert
        if (platform.IsWindows)
            platform.NativeLibraryExtension.Should().Be(".dll");
        else if (platform.IsLinux)
            platform.NativeLibraryExtension.Should().Be(".so");
        else if (platform.IsMacOS)
            platform.NativeLibraryExtension.Should().Be(".dylib");
    }
}
