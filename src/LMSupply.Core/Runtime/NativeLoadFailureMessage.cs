namespace LMSupply.Runtime;

/// <summary>The platform families the native-load error text distinguishes.</summary>
internal enum NativeLoadPlatform
{
    Windows,
    Linux,
    MacOS,
    Other,
}

/// <summary>
/// The error a caller gets when the native ONNX Runtime cannot be loaded: what that usually means on the platform, the
/// loader's own explanation when there is one, and — for the one refusal that has a different remedy from «install a
/// system library» — what to do about it.
/// </summary>
/// <remarks>
/// macOS library validation (a hardened-runtime app without the <c>disable-library-validation</c> entitlement) refuses a
/// native library signed by another team. The runtime LMSupply downloads into its cache at first use is signed by Microsoft,
/// so every model load in such an app fails — and the generic text («required system libraries») sends the reader to look for a
/// missing package. The loader's text says «code signature … not valid for use in process» / «different Team IDs», and the
/// remedy is <see cref="RuntimeManagerOptions.RuntimeDirectory"/> (the runtime shipped and signed with the app).
/// </remarks>
internal static class NativeLoadFailureMessage
{
    private static readonly string[] CodeSigningRefusalMarkers =
    [
        "code signature",
        "different Team IDs",
        "library load disallowed by system policy",
        "not valid for use in process",
    ];

    public static NativeLoadPlatform CurrentPlatform =>
        OperatingSystem.IsWindows() ? NativeLoadPlatform.Windows
        : OperatingSystem.IsLinux() ? NativeLoadPlatform.Linux
        : OperatingSystem.IsMacOS() ? NativeLoadPlatform.MacOS
        : NativeLoadPlatform.Other;

    /// <summary>True when the loader's text says the binary's code signature was refused (macOS library validation).</summary>
    public static bool IsCodeSigningRefusal(string? loaderReason) =>
        !string.IsNullOrEmpty(loaderReason)
        && CodeSigningRefusalMarkers.Any(marker => loaderReason.Contains(marker, StringComparison.OrdinalIgnoreCase));

    /// <summary>The message for a native ONNX Runtime that failed to load; <paramref name="loaderReason"/> is the loader's own text, if known.</summary>
    public static string ForOnnxRuntime(NativeLoadPlatform platform, string? loaderReason)
    {
        var codeSigningRefusal = platform == NativeLoadPlatform.MacOS && IsCodeSigningRefusal(loaderReason);

        // A recognised library-validation refusal has its own remedy below; the generic «install system libraries» hint would
        // send the reader to the wrong fix first.
        var message = codeSigningRefusal
            ? "ONNX Runtime native library failed to load."
            : platform switch
        {
            NativeLoadPlatform.Windows =>
                "ONNX Runtime native library failed to load. This usually means Visual C++ Redistributable 2015-2022 is not installed. "
                + "Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe",
            NativeLoadPlatform.Linux =>
                "ONNX Runtime native library failed to load. Ensure required shared libraries are installed (libstdc++, libgomp).",
            _ =>
                "ONNX Runtime native library failed to load. Ensure required system libraries are installed.",
        };

        if (!string.IsNullOrWhiteSpace(loaderReason))
            message += $" The loader said: {loaderReason}";

        if (codeSigningRefusal)
        {
            message += " macOS library validation refuses a native library that is not signed by the same team as the application - "
                + "which is what happens to the ONNX Runtime LMSupply downloads into its cache at first use. In a signed (hardened-runtime) "
                + "app, ship the native ONNX Runtime with the application, signed with it, and set RuntimeManagerOptions.RuntimeDirectory to "
                + "its folder (see the README, 'Signed macOS apps'); the alternative is the com.apple.security.cs.disable-library-validation entitlement.";
        }

        return message;
    }
}
