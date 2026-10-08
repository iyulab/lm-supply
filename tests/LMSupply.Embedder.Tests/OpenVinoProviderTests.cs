using AwesomeAssertions;
using Xunit;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// <see cref="ExecutionProvider.OpenVino"/> end to end: the plugin is downloaded into the runtime cache, registered from
/// there (not from the application directory), and the session runs on the Intel GPU with the same vectors the CPU
/// session produces. Needs an Intel GPU, the model download and the ~120 MB plugin download; LocalOnly keeps it out of CI.
/// </summary>
[Trait("Category", "LocalOnly")]
public sealed class OpenVinoProviderTests
{
    [Fact]
    public async Task OpenVino_RunsOnTheIntelGpu_WithTheCpuVectors()
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            Assert.Skip("The OpenVINO plugin ships for Windows x64 and Linux x64 only.");

        var ct = TestContext.Current.CancellationToken;
        await using var gpu = await LocalEmbedder.LoadAsync("default", new EmbedderOptions { Provider = ExecutionProvider.OpenVino },
            cancellationToken: ct);
        if (!gpu.ActiveProviders.Contains("OpenVINOExecutionProvider"))
            Assert.Skip($"No Intel GPU exposed by the OpenVINO plugin on this host (active: {string.Join(", ", gpu.ActiveProviders)}).");

        gpu.IsGpuActive.Should().BeTrue();

        await using var cpu = await LocalEmbedder.LoadAsync("default", new EmbedderOptions { Provider = ExecutionProvider.Cpu },
            cancellationToken: ct);
        const string text = "The quick brown fox jumps over the lazy dog.";
        var a = await gpu.EmbedAsync(text, ct);
        var b = await cpu.EmbedAsync(text, ct);

        a.Length.Should().Be(b.Length);
        Cosine(a, b).Should().BeGreaterThan(0.999, "a GPU kernel differs from the CPU one by float rounding, not in meaning");
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / Math.Sqrt(na * nb);
    }
}
