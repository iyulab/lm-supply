using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Tests for <see cref="LlamaServerDownloader.CudaRuntimePresent"/>. The CUDA runtime is three
/// libraries (cudart, cublas, cublasLt). A partial extraction (e.g. small cudart present but the
/// 473 MB cublasLt truncated/missing after an interrupt or concurrent write) must NOT count as
/// "present" — otherwise the cache is permanently poisoned: ggml-cuda fails to load, the server
/// silently runs on CPU, and a re-run never re-provisions because the early-return sees cudart.
/// Requiring all three makes a partial state self-heal (re-provision) on the next load.
///
/// HW-free: creates dummy files in a temp dir. Uses Windows-style names, which the matcher also
/// accepts on Linux (it checks both naming conventions), so the test is OS-agnostic.
/// </summary>
public sealed class LlamaServerCudaRuntimePresentTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-cudart-test-" + Guid.NewGuid().ToString("N"));

    public LlamaServerCudaRuntimePresentTests() => Directory.CreateDirectory(_dir);
    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    private void Touch(string name) => File.WriteAllText(Path.Combine(_dir, name), "x");

    [Fact]
    public void AllThreeRuntimeDlls_IsPresent()
    {
        Touch("cudart64_12.dll");
        Touch("cublas64_12.dll");
        Touch("cublasLt64_12.dll");

        LlamaServerDownloader.CudaRuntimePresent(_dir).Should().BeTrue(
            "the full CUDA runtime is present");
    }

    [Fact]
    public void OnlyCudart_IsNotPresent_PoisonGuard()
    {
        // The durable-poison case: a partial extraction left only the small cudart.
        Touch("cudart64_12.dll");

        LlamaServerDownloader.CudaRuntimePresent(_dir).Should().BeFalse(
            "a partial runtime must re-provision, not be treated as complete");
    }

    [Fact]
    public void MissingCublasLt_IsNotPresent()
    {
        Touch("cudart64_12.dll");
        Touch("cublas64_12.dll");
        // cublasLt (the largest, widest write window) missing

        LlamaServerDownloader.CudaRuntimePresent(_dir).Should().BeFalse(
            "every required runtime library must be present");
    }

    [Fact]
    public void EmptyDir_IsNotPresent()
    {
        LlamaServerDownloader.CudaRuntimePresent(_dir).Should().BeFalse();
    }

    [Fact]
    public void NonexistentDir_IsNotPresent()
    {
        LlamaServerDownloader.CudaRuntimePresent(Path.Combine(_dir, "does-not-exist"))
            .Should().BeFalse();
    }
}
