using LMSupply.Generator.Abstractions;

namespace LMSupply.Generator;

/// <summary>
/// Registration point for the optional ONNX Runtime GenAI backend (the LMSupply.Generator.Onnx
/// package). A consumer that needs ONNX model loading references that package and calls
/// <c>LMSupply.Generator.Onnx.OnnxGeneratorBackend.Register()</c> once at startup, before loading
/// an ONNX model. Auto-registration (e.g. via <c>[ModuleInitializer]</c>) is deliberately not
/// used: a package reference alone does not guarantee the assembly is loaded before this
/// registry is consulted, and a silent no-op registration would be worse than the explicit
/// <see cref="NotSupportedException"/> below.
/// </summary>
public static class OnnxGeneratorBackendRegistry
{
    private static IOnnxGeneratorBackend? _backend;

    /// <summary>
    /// True when an ONNX backend has been registered. Consulted by hardware auto-selection
    /// (<c>GeneratorRoutingPolicy</c>) so it never picks ONNX for a consumer that only referenced
    /// the GGUF/llama-server path.
    /// </summary>
    public static bool IsRegistered => Volatile.Read(ref _backend) is not null;

    /// <summary>
    /// Registers the ONNX backend implementation. Idempotent — a later call replaces the
    /// previously registered backend.
    /// </summary>
    public static void Register(IOnnxGeneratorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        Volatile.Write(ref _backend, backend);
    }

    /// <summary>
    /// Returns the registered backend, or throws a <see cref="NotSupportedException"/> naming the
    /// package to add when none is registered.
    /// </summary>
    internal static IOnnxGeneratorBackend Require()
    {
        return Volatile.Read(ref _backend) ?? throw new NotSupportedException(
            "Loading an ONNX Runtime GenAI model requires the LMSupply.Generator.Onnx package. " +
            "Add a PackageReference to LMSupply.Generator.Onnx and call " +
            "LMSupply.Generator.Onnx.OnnxGeneratorBackend.Register() during startup.");
    }

    /// <summary>
    /// Test-only reset back to unregistered. Production code never unregisters — a process either
    /// referenced LMSupply.Generator.Onnx and registered once at startup, or it never did.
    /// </summary>
    internal static void ResetForTests() => Volatile.Write(ref _backend, null);
}
