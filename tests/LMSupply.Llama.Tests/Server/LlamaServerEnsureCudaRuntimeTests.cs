using System.IO.Compression;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// AC#3 wiring test for <see cref="LlamaServerDownloader.EnsureCudaRuntimeAsync"/>: concurrent callers
/// for the SAME versionDir (Filer loads embedder + generator into one cuda12 dir via Task.WhenAll)
/// must converge to a single release fetch + single cudart download/extract — proving the method wires
/// the gate, the <c>CudaRuntimePresent</c> predicate, and <c>ProvisionCudaRuntimeCoreAsync</c> together
/// correctly (the <c>RunGatedOnceAsync</c> primitive is unit-tested separately).
///
/// Network-free: a fake HttpMessageHandler returns a release manifest + an in-memory cudart zip holding
/// the three runtime DLLs. Skipped on macOS (no CUDA companion). Uses the process-wide static gate, so
/// it lives in its own collection to avoid parallel pollution with any other EnsureCudaRuntime caller.
/// </summary>
[Collection("CudartGate")]
public sealed class LlamaServerEnsureCudaRuntimeTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "lmsupply-ensure-cudart-" + Guid.NewGuid().ToString("N"));

    public LlamaServerEnsureCudaRuntimeTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task ConcurrentEnsure_SameVersionDir_FetchesAndExtractsOnce()
    {
        Assert.SkipWhen(RuntimeInformation.IsOSPlatform(OSPlatform.OSX),
            "macOS has no CUDA cudart companion; EnsureCudaRuntimeAsync is a no-op there");

        var os = OperatingSystem.IsWindows() ? "win" : "ubuntu";
        var assetName = $"cudart-llama-bin-{os}-cuda-12.4-x64.zip";
        var cudartUrl = "https://fake.local/" + assetName;
        var releaseJson =
            $$"""{ "assets": [ { "name": "{{assetName}}", "browser_download_url": "{{cudartUrl}}" } ] }""";

        var handler = new CountingHandler(releaseJson, cudartUrl, BuildCudartZip());
        using var http = new HttpClient(handler);
        var downloader = new LlamaServerDownloader(_dir, http);

        var tasks = Enumerable.Range(0, 5).Select(_ =>
            downloader.EnsureCudaRuntimeAsync(_dir, LlamaServerBackend.Cuda12, "b9692"));
        await Task.WhenAll(tasks);

        handler.ReleaseFetches.Should().Be(1,
            "serialization + the post-acquire re-check must fetch the release manifest exactly once");
        handler.CudartFetches.Should().Be(1, "and download the cudart asset exactly once");
        LlamaServerDownloader.CudaRuntimePresent(_dir).Should().BeTrue(
            "the complete runtime must be extracted into the versionDir");
    }

    private static byte[] BuildCudartZip()
    {
        using var ms = new MemoryStream();
        using (var zip = new ZipArchive(ms, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (var name in new[] { "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll" })
            {
                using var entry = zip.CreateEntry(name).Open();
                entry.Write("x"u8);
            }
        }
        return ms.ToArray();
    }

    private sealed class CountingHandler(string releaseJson, string cudartUrl, byte[] zipBytes)
        : HttpMessageHandler
    {
        private int _releaseFetches;
        private int _cudartFetches;
        public int ReleaseFetches => _releaseFetches;
        public int CudartFetches => _cudartFetches;

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            if (url.Contains("/tags/"))
            {
                Interlocked.Increment(ref _releaseFetches);
                return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(releaseJson) };
            }
            if (url == cudartUrl)
            {
                Interlocked.Increment(ref _cudartFetches);
                await Task.Delay(150, cancellationToken); // widen the window so a missing gate would race
                return new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(zipBytes) };
            }
            return new HttpResponseMessage(HttpStatusCode.NotFound);
        }
    }
}
