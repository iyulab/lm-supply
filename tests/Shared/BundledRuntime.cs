using LMSupply.Lifecycle;
using LMSupply.Runtime;

namespace LMSupply.Testing;

/// <summary>
/// An ONNX Runtime an application ships with itself, for tests of what a model kind reports about its runtime: a runtime
/// manager whose <see cref="RuntimeManagerOptions.RuntimeDirectory"/> holds the CPU package's native libraries by name, used
/// by <see cref="OnnxRuntimeArtifacts"/> in the calling flow until disposed — so the answer reads neither this machine's
/// runtime cache nor the package feed. Without it, a test would pass or fail by what the machine happens to have cached.
/// </summary>
internal sealed class BundledRuntime : IDisposable
{
    private readonly IDisposable _scope;

    private BundledRuntime(IDisposable scope) => _scope = scope;

    /// <param name="directory">Where the libraries go; <paramref name="present"/> false leaves it empty, a runtime not shipped.</param>
    /// <param name="present">Whether the libraries are there.</param>
    /// <remarks>
    /// Synchronous on purpose: the scope is an async-local value, and one set inside an awaited method does not flow back
    /// to its caller.
    /// </remarks>
    public static BundledRuntime Use(string directory, bool present = true)
    {
        Directory.CreateDirectory(directory);
        if (present)
        {
            var platform = EnvironmentDetector.DetectPlatform();
            var config = RuntimePackageRegistry.GetPackageConfig("onnxruntime", "cpu", platform.RuntimeIdentifier)!;
            foreach (var library in new[] { config.NativeLibraryName }.Concat(config.AdditionalLibraries))
                File.WriteAllBytes(Path.Combine(directory, RuntimePackageRegistry.GetNativeLibraryFileName(library, platform)), []);
        }
        else
        {
            foreach (var file in Directory.EnumerateFiles(directory))
                File.Delete(file);
        }

        var manager = new RuntimeManager(new RuntimeManagerOptions { RuntimeDirectory = directory, CacheDirectory = Path.Combine(directory, "cache") });
        return new BundledRuntime(OnnxRuntimeArtifacts.UseManager(manager));
    }

    public void Dispose() => _scope.Dispose();
}
