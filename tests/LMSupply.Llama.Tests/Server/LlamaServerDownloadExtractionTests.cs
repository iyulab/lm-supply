using System.IO.Compression;
using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Regression coverage for the Linux e2e failure investigated 2026-08-17
/// (<c>ISSUE-lm-supply-20260806-062000-llama-server-binary-absent-on-linux-runner.md</c>):
/// "llama-server did not launch" / "The process was never created" on ubuntu runners.
///
/// Root cause confirmed against the actual b10290 release: llama.cpp's Linux/macOS tar.gz assets
/// unpack into a top-level wrapper directory (<c>llama-b10290/llama-server</c>), while the Windows
/// zip for the same release is flat (<c>llama-server.exe</c> at the archive root) — so the
/// flat-layout assumption in <see cref="LlamaServerDownloader"/> only ever broke on Linux/macOS.
/// <see cref="LlamaServerDownloader.DownloadAsync"/> also returned the expected-but-wrong path
/// unconditionally, with no check that extraction actually produced it, so the failure surfaced far
/// downstream as an opaque process-start error instead of at the point it actually happened.
/// </summary>
public sealed class LlamaServerDownloadExtractionTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "lmsupply-extract-" + Guid.NewGuid().ToString("N"));

    public LlamaServerDownloadExtractionTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    // ---- FlattenSingleTopLevelDirectory (pure filesystem logic, format-agnostic) ----

    [Fact]
    public void Flatten_MovesNestedFilesUp_AndRemovesWrapper()
    {
        var wrapper = Directory.CreateDirectory(Path.Combine(_dir, "llama-b10290"));
        File.WriteAllBytes(Path.Combine(wrapper.FullName, "llama-server"), "bin"u8.ToArray());
        File.WriteAllBytes(Path.Combine(wrapper.FullName, "libggml.so"), "lib"u8.ToArray());

        LlamaServerDownloader.FlattenSingleTopLevelDirectory(_dir);

        File.Exists(Path.Combine(_dir, "llama-server")).Should().BeTrue(
            "the binary must land directly in the version directory, not under the archive's wrapper folder");
        File.Exists(Path.Combine(_dir, "libggml.so")).Should().BeTrue(
            "sibling shared libraries must move with the binary so runtime library resolution still finds them");
        Directory.Exists(wrapper.FullName).Should().BeFalse("the now-empty wrapper directory must be removed");
    }

    [Fact]
    public void Flatten_NoOpWhenArchiveWasAlreadyFlat()
    {
        // Windows zip layout: files sit directly at the extraction root, no single wrapper directory.
        File.WriteAllBytes(Path.Combine(_dir, "llama-server.exe"), "bin"u8.ToArray());
        File.WriteAllBytes(Path.Combine(_dir, "ggml-base.dll"), "lib"u8.ToArray());

        LlamaServerDownloader.FlattenSingleTopLevelDirectory(_dir);

        Directory.GetFileSystemEntries(_dir).Select(Path.GetFileName)
            .Should().BeEquivalentTo(["llama-server.exe", "ggml-base.dll"],
                "a flat archive must be left untouched");
    }

    [Fact]
    public void Flatten_NoOpWhenSingleTopLevelEntryIsAFile()
    {
        File.WriteAllBytes(Path.Combine(_dir, "only-file.txt"), "x"u8.ToArray());

        LlamaServerDownloader.FlattenSingleTopLevelDirectory(_dir);

        File.Exists(Path.Combine(_dir, "only-file.txt")).Should().BeTrue(
            "a single top-level FILE (not a directory) is not an archive wrapper and must be left alone");
    }

    // ---- End-to-end via DownloadAsync (the actual bug: wrong path returned, never verified) ----

    [Fact]
    public async Task DownloadAsync_NestedArchive_FlattensAndReturnsAnExistingPath()
    {
        // Simulates the real b10290 ubuntu-x64.tar.gz layout using a zip container (the flatten step
        // operates on the extracted directory tree and is agnostic to the original archive format —
        // the existing cudart tests in this file use the same substitution). Platform is forced to
        // Linux via the asset regardless of the OS actually running this test.
        var archiveBytes = BuildZipWithWrapperDirectory("llama-b10290", ("llama-server", "bin"), ("libggml.so", "lib"));
        using var http = new HttpClient(new StubHandler("https://fake.local/asset.zip", archiveBytes));
        using var downloader = new LlamaServerDownloader(_dir, http);

        var asset = new LlamaServerAsset
        {
            Name = "llama-b10290-bin-ubuntu-x64.zip",
            DownloadUrl = "https://fake.local/asset.zip",
            Version = "b10290",
            Platform = LlamaServerPlatform.Linux,
            Backend = LlamaServerBackend.Cpu,
            Architecture = LlamaServerArchitecture.X64
        };

        var serverPath = await downloader.DownloadAsync(asset, cancellationToken: TestContext.Current.CancellationToken);

        File.Exists(serverPath).Should().BeTrue(
            "DownloadAsync must return a path that actually exists after extraction, not the archive's nested location");
        Path.GetFileName(serverPath).Should().Be("llama-server");
        File.Exists(Path.Combine(Path.GetDirectoryName(serverPath)!, "libggml.so")).Should().BeTrue(
            "sibling shared libraries must be flattened alongside the executable");
    }

    [Fact]
    public async Task DownloadAsync_ArchiveMissingExpectedExecutable_ThrowsWithDiagnostic()
    {
        // The archive downloads and extracts cleanly, but never contained "llama-server" at all
        // (e.g. a mismatched asset). Before this fix, DownloadAsync returned the expected-but-absent
        // path anyway; the failure would only surface later as an opaque process-start error.
        var archiveBytes = BuildZipWithWrapperDirectory("llama-b10290", ("some-other-tool", "bin"));
        using var http = new HttpClient(new StubHandler("https://fake.local/asset.zip", archiveBytes));
        using var downloader = new LlamaServerDownloader(_dir, http);

        var asset = new LlamaServerAsset
        {
            Name = "llama-b10290-bin-ubuntu-x64.zip",
            DownloadUrl = "https://fake.local/asset.zip",
            Version = "b10290",
            Platform = LlamaServerPlatform.Linux,
            Backend = LlamaServerBackend.Cpu,
            Architecture = LlamaServerArchitecture.X64
        };

        var act = async () => await downloader.DownloadAsync(asset);

        (await act.Should().ThrowAsync<InvalidOperationException>())
            .WithMessage("*llama-server*not found*")
            .Which.Message.Should().Contain("some-other-tool",
                "the error must show what was actually extracted, not just that the binary is missing");
    }

    private static byte[] BuildZipWithWrapperDirectory(string wrapperName, params (string name, string content)[] entries)
    {
        using var ms = new MemoryStream();
        using (var zip = new ZipArchive(ms, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (var (name, content) in entries)
            {
                using var entry = zip.CreateEntry($"{wrapperName}/{name}").Open();
                entry.Write(System.Text.Encoding.UTF8.GetBytes(content));
            }
        }
        return ms.ToArray();
    }

    private sealed class StubHandler(string downloadUrl, byte[] archiveBytes) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (request.RequestUri!.ToString() == downloadUrl)
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                { Content = new ByteArrayContent(archiveBytes) });
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
