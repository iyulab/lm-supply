using System.Runtime.InteropServices;

namespace LMSupply.Runtime;

/// <summary>
/// Contains information about the current platform environment.
/// </summary>
public sealed record PlatformInfo
{
    /// <summary>
    /// Gets the operating system (Windows, Linux, OSX).
    /// </summary>
    public required OSPlatform OS { get; init; }

    /// <summary>
    /// Gets the processor architecture (X64, Arm64, etc.).
    /// </summary>
    public required Architecture Architecture { get; init; }

    /// <summary>
    /// Gets the .NET Runtime Identifier (e.g., win-x64, linux-arm64, osx-arm64).
    /// </summary>
    public required string RuntimeIdentifier { get; init; }

    /// <summary>
    /// Gets whether the platform is 64-bit.
    /// </summary>
    public bool Is64Bit => Architecture is Architecture.X64 or Architecture.Arm64;

    /// <summary>
    /// Gets whether the platform is Windows.
    /// </summary>
    public bool IsWindows => OS == OSPlatform.Windows;

    /// <summary>
    /// Gets whether the platform is Linux.
    /// </summary>
    public bool IsLinux => OS == OSPlatform.Linux;

    /// <summary>
    /// Gets whether the platform is macOS.
    /// </summary>
    public bool IsMacOS => OS == OSPlatform.OSX;

    /// <summary>
    /// Gets whether the platform is ARM-based.
    /// </summary>
    public bool IsArm => Architecture is Architecture.Arm or Architecture.Arm64;

    /// <summary>
    /// Gets the native library file extension for the current OS.
    /// </summary>
    public string NativeLibraryExtension => OS switch
    {
        _ when OS == OSPlatform.Windows => ".dll",
        _ when OS == OSPlatform.Linux => ".so",
        _ when OS == OSPlatform.OSX => ".dylib",
        _ => ".so"
    };

    /// <summary>
    /// Gets the native library prefix for the current OS.
    /// </summary>
    public string NativeLibraryPrefix => OS switch
    {
        _ when OS == OSPlatform.Windows => "",
        _ => "lib"
    };

    public override string ToString() => $"{OS} {Architecture} ({RuntimeIdentifier})";
}
