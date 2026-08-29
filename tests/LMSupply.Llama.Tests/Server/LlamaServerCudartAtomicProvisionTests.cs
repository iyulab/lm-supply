using System.IO.Compression;
using System.Net;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// #3 atomicity for <see cref="LlamaServerDownloader.EnsureCudaRuntimeAsync"/>: cudart provisioning
/// must be failure/crash-atomic with respect to the versionDir. The download + extract happen in an
/// isolated staging directory; only a fully extracted runtime is moved into the versionDir. A failed
/// or interrupted extract (corrupt/truncated archive) must therefore leave NO partial state in the
/// versionDir — neither the downloaded archive nor half-extracted DLLs. This keeps the
/// <see cref="LlamaServerDownloader.CudaRuntimePresent"/> completeness check from ever seeing a poisoned
/// versionDir and stops a truncated cublasLt from being load-attempted by ggml.
///
/// Network-free (fake HttpMessageHandler) and gate-isolated (process-wide static gate → own collection).
/// Skipped on macOS (no CUDA companion).
/// </summary>
[Collection("CudartGate")]
public sealed class LlamaServerCudartAtomicProvisionTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "lmsupply-cudart-atomic-" + Guid.NewGuid().ToString("N"));

    public LlamaServerCudartAtomicProvisionTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task FailedExtract_LeavesVersionDirClean()
    {
        Assert.SkipWhen(RuntimeInformation.IsOSPlatform(OSPlatform.OSX),
            "macOS has no CUDA cudart companion; EnsureCudaRuntimeAsync is a no-op there");

        var (releaseJson, cudartUrl) = BuildRelease();
        // Corrupt archive: the download succeeds but the extract throws (invalid zip/tar).
        var handler = new StubHandler(releaseJson, cudartUrl, "this is not a valid archive"u8.ToArray());
        using var http = new HttpClient(handler);
        var downloader = new LlamaServerDownloader(_dir, http);

        // Best-effort: EnsureCudaRuntimeAsync swallows the extract failure (must not throw).
        await downloader.EnsureCudaRuntimeAsync(_dir, LlamaServerBackend.Cuda12, "b9692", cancellationToken: TestContext.Current.CancellationToken);

        Directory.EnumerateFileSystemEntries(_dir).Should().BeEmpty(
            "a failed extract must leave neither the archive nor partial DLLs in the versionDir");
        LlamaServerDownloader.CudaRuntimePresent(_dir).Should().BeFalse();
    }

    [Fact]
    public async Task SuccessfulProvision_LeavesOnlyRuntime_NoStagingOrArchive()
    {
        Assert.SkipWhen(RuntimeInformation.IsOSPlatform(OSPlatform.OSX),
            "macOS has no CUDA cudart companion; EnsureCudaRuntimeAsync is a no-op there");

        var (releaseJson, cudartUrl) = BuildRelease();
        var handler = new StubHandler(releaseJson, cudartUrl, BuildCudartZip());
        using var http = new HttpClient(handler);
        var downloader = new LlamaServerDownloader(_dir, http);

        await downloader.EnsureCudaRuntimeAsync(_dir, LlamaServerBackend.Cuda12, "b9692", cancellationToken: TestContext.Current.CancellationToken);

        LlamaServerDownloader.CudaRuntimePresent(_dir).Should().BeTrue(
            "the complete runtime must be extracted into the versionDir");
        Directory.EnumerateFiles(_dir, "*.zip").Should().BeEmpty("the archive must not be left behind");
        Directory.EnumerateDirectories(_dir).Should().BeEmpty(
            "the staging directory must be cleaned up after a successful move");
    }

    [Theory]
    [InlineData("cublasLt64_12.dll", 1)]
    [InlineData("libcublasLt.so.12", 1)]
    [InlineData("cudart64_12.dll", 0)]
    [InlineData("cublas64_12.dll", 0)]
    [InlineData("libcudart.so.12", 0)]
    public void CudartMoveOrder_RanksCublasLtLast(string fileName, int expectedKey)
    {
        // Pins the crash-safety invariant: cublasLt (the last family CudaRuntimePresent requires) must
        // sort after every other runtime file so the completeness check only flips true once the whole
        // runtime is in place. Without this, a crash mid-move could leave a versionDir that reports
        // present=true while cublasLt is still missing.
        LlamaServerDownloader.CudartMoveOrder(Path.Combine("any", "dir", fileName)).Should().Be(expectedKey);
    }

    [Fact]
    public void CudartMoveOrder_SortsCublasLtToEnd()
    {
        var files = new[] { "cublasLt64_12.dll", "cudart64_12.dll", "cublas64_12.dll" };
        files.OrderBy(LlamaServerDownloader.CudartMoveOrder).Last()
            .Should().Be("cublasLt64_12.dll", "cublasLt must be the final file moved into the versionDir");
    }

    private static (string releaseJson, string cudartUrl) BuildRelease()
    {
        var os = OperatingSystem.IsWindows() ? "win" : "ubuntu";
        var assetName = $"cudart-llama-bin-{os}-cuda-12.4-x64.zip";
        var cudartUrl = "https://fake.local/" + assetName;
        var releaseJson =
            $$"""{ "assets": [ { "name": "{{assetName}}", "browser_download_url": "{{cudartUrl}}" } ] }""";
        return (releaseJson, cudartUrl);
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

    private sealed class StubHandler(string releaseJson, string cudartUrl, byte[] archiveBytes)
        : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            if (url.Contains("/tags/"))
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                { Content = new StringContent(releaseJson) });
            if (url == cudartUrl)
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                { Content = new ByteArrayContent(archiveBytes) });
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
