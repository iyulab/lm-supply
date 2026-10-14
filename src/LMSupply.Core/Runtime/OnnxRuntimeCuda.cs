namespace LMSupply.Runtime;

/// <summary>
/// The CUDA major version the ONNX Runtime GPU build in use needs. ONNX Runtime 1.27 and later publish their NuGet GPU
/// package built against CUDA 13.0 (it runs on any CUDA 13.x runtime, with cuDNN 9); earlier versions were built against
/// CUDA 12. The provider dll then depends on that major's libraries (<c>cublasLt64_13.dll</c>), so a CUDA 12 install,
/// or a driver that only supports CUDA 12, cannot run it.
/// </summary>
internal static class OnnxRuntimeCuda
{
    private static readonly Version FirstCuda13Release = new(1, 27);

    /// <summary>The CUDA major for an ONNX Runtime version; an unreadable version is taken as current (13).</summary>
    internal static int RequiredCudaMajor(string? onnxRuntimeVersion)
        => Version.TryParse(onnxRuntimeVersion, out var version) && version < FirstCuda13Release ? 12 : 13;

    /// <summary>The CUDA major the loaded ONNX Runtime assembly's GPU build needs.</summary>
    internal static int RequiredCudaMajorForLoadedRuntime => RequiredCudaMajor(RuntimeManager.TryGetOnnxRuntimeVersion());

    /// <summary>
    /// Why an NVIDIA driver cannot run a CUDA <paramref name="requiredMajor"/> build, or <see langword="null"/> when it can.
    /// The driver reports the newest CUDA version it supports.
    /// </summary>
    internal static string? DriverShortfall(int? driverCudaMajor, int? driverCudaMinor, int requiredMajor)
        => driverCudaMajor is { } major && major >= requiredMajor
            ? null
            : $"the NVIDIA driver supports CUDA up to {driverCudaMajor ?? 0}.{driverCudaMinor ?? 0} and the ONNX Runtime GPU build needs CUDA {requiredMajor}; " +
              $"update the driver to one that supports CUDA {requiredMajor}";
}
