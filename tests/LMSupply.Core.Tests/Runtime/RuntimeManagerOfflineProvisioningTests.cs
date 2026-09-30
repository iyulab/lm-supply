using System.IO.Compression;
using System.Net;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// An application that promises to install and run with no network must be able to keep the native ONNX Runtime off
/// the network too, not only the model files: a bundled runtime directory, a local-only (cache or fail) mode, and a
/// pinned version that is never resolved against nuget.org. Every test counts the feed requests the manager makes.
/// </summary>
[Trait("Category", "Unit")]
public sealed class RuntimeManagerOfflineProvisioningTests : IDisposable
{
    private const string OnnxRuntime = RuntimePackageRegistry.PackageTypes.OnnxRuntime;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "lmsupply-ort-offline-" + Guid.NewGuid().ToString("N"));

    private string CacheDir => Path.Combine(_root, "cache");

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    private static async Task<(RuntimeManager Manager, CountingFeed Feed)> CreateAsync(RuntimeManagerOptions options)
    {
        var feed = new CountingFeed();
        var manager = new RuntimeManager(options, updateOptions: null, feed);
        await manager.InitializeAsync(TestContext.Current.CancellationToken);
        return (manager, feed);
    }

    private static string NativeFile(RuntimeManager manager, string library = "onnxruntime") =>
        RuntimePackageRegistry.GetNativeLibraryFileName(library, manager.Platform);

    private string SeedCache(RuntimeManager manager, string version, string provider = "cpu")
    {
        var dir = Path.Combine(CacheDir, OnnxRuntime, provider, version, manager.Platform.RuntimeIdentifier);
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, NativeFile(manager)), [1, 2, 3]);
        return dir;
    }

    [Fact]
    public async Task LocalOnly_CachedVersion_IsUsed_WithNoFeedRequest()
    {
        var (manager, feed) = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, DisableAutoDownload = true });
        var cached = SeedCache(manager, "1.30.0");

        var (path, version, _) = await manager.ResolveRuntimeForProviderAsync(
            "cpu", OnnxRuntime, "1.30.0", progress: null, TestContext.Current.CancellationToken);

        path.Should().Be(cached);
        version.Should().Be("1.30.0");
        feed.Requests.Should().BeEmpty("a local-only manager never asks nuget.org");
    }

    [Fact]
    public async Task LocalOnly_NothingCached_Throws_InsteadOfDownloading()
    {
        var (manager, feed) = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, DisableAutoDownload = true });

        var act = () => manager.ResolveRuntimeForProviderAsync(
            "cpu", OnnxRuntime, "1.30.0", progress: null, TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<ModelLoadException>())
            .WithMessage("*DisableAutoDownload*RuntimeDirectory*");
        feed.Requests.Should().BeEmpty();
    }

    [Fact]
    public async Task LocalOnly_WithoutPin_OlderCachedVersionStandsIn()
    {
        var (manager, feed) = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, DisableAutoDownload = true });
        var older = SeedCache(manager, "1.24.4");

        var (path, version, _) = await manager.ResolveRuntimeForProviderAsync(
            "cpu", OnnxRuntime, "1.30.0", progress: null, TestContext.Current.CancellationToken);

        path.Should().Be(older, "with downloads off, the cached copy is the only thing that can run (same policy as an unreachable feed)");
        version.Should().Be("1.24.4");
        feed.Requests.Should().BeEmpty();
    }

    [Fact]
    public async Task LocalOnly_WithPin_OnlyThePinnedVersionIsAccepted()
    {
        var (manager, feed) = await CreateAsync(new RuntimeManagerOptions
        {
            CacheDirectory = CacheDir, DisableAutoDownload = true, PinnedVersion = "1.30.0",
        });
        SeedCache(manager, "1.24.4");

        var act = () => manager.ResolveRuntimeForProviderAsync(
            "cpu", OnnxRuntime, version: null, progress: null, TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<ModelLoadException>()).WithMessage("*1.30.0*");
        feed.Requests.Should().BeEmpty();
    }

    [Fact]
    public async Task RuntimeDirectory_IsUsedAsIs_WithNoFeedRequest()
    {
        var bundle = Path.Combine(_root, "bundle");
        var (manager, feed) = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, RuntimeDirectory = bundle });
        Directory.CreateDirectory(bundle);
        File.WriteAllBytes(Path.Combine(bundle, NativeFile(manager)), [1]);

        var (path, _, _) = await manager.ResolveRuntimeForProviderAsync(
            "cpu", OnnxRuntime, version: null, progress: null, TestContext.Current.CancellationToken);

        path.Should().Be(Path.GetFullPath(bundle));
        feed.Requests.Should().BeEmpty();
        manager.Options.AllowsRuntimeUpdates.Should().BeFalse("a bundled runtime is never replaced");
    }

    [Fact]
    public async Task RuntimeDirectory_WithoutTheProvidersLibraries_RefusesThatProvider()
    {
        // A CPU-only bundle cannot serve CUDA: the request fails (the Auto chain then moves on to CPU) rather than
        // handing back a directory whose runtime has no CUDA provider.
        var bundle = Path.Combine(_root, "bundle");
        var (manager, feed) = await CreateAsync(new RuntimeManagerOptions { RuntimeDirectory = bundle });
        Directory.CreateDirectory(bundle);
        File.WriteAllBytes(Path.Combine(bundle, NativeFile(manager)), [1]);

        var act = () => manager.ResolveRuntimeForProviderAsync(
            "cuda12", OnnxRuntime, version: null, progress: null, TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<ModelLoadException>())
            .WithMessage($"*{NativeFile(manager, "onnxruntime_providers_cuda")}*");
        feed.Requests.Should().BeEmpty();
    }

    [Fact]
    public async Task RuntimeDirectory_Missing_NamesTheExpectedFile()
    {
        var (manager, _) = await CreateAsync(new RuntimeManagerOptions { RuntimeDirectory = Path.Combine(_root, "absent") });

        var act = () => manager.ResolveRuntimeForProviderAsync(
            "cpu", OnnxRuntime, version: null, progress: null, TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<ModelLoadException>()).WithMessage($"*{NativeFile(manager)}*");
    }

    [Fact]
    public async Task PinnedVersion_DownloadsExactlyThatVersion_AndNeverAsksForTheLatest()
    {
        var (manager, feed) = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, PinnedVersion = "1.29.0" });
        feed.Nupkg = Nupkg(manager);

        var (path, version, _) = await manager.ResolveRuntimeForProviderAsync(
            "cpu", OnnxRuntime, version: null, progress: null, TestContext.Current.CancellationToken);

        version.Should().Be("1.29.0");
        File.Exists(Path.Combine(path, NativeFile(manager))).Should().BeTrue();
        feed.Requests.Should().OnlyContain(u => u.Contains("/1.29.0/"),
            "a pinned version is downloaded by name; no version index or 'latest' lookup is made");
        manager.Options.AllowsRuntimeUpdates.Should().BeFalse();
    }

    [Fact]
    public async Task DefaultOptions_StillAllowUpdates()
    {
        var (manager, _) = await CreateAsync(new RuntimeManagerOptions());
        manager.Options.AllowsRuntimeUpdates.Should().BeTrue("the default behaviour is unchanged: provisioned from nuget.org, kept up to date");
    }

    [Fact]
    public void Configure_AfterTheProcessWideInstanceExists_Throws()
    {
        _ = RuntimeManager.Instance;

        var act = () => RuntimeManager.Configure(new RuntimeManagerOptions { DisableAutoDownload = true });

        act.Should().Throw<InvalidOperationException>().WithMessage("*before the first model load*");
    }

    private static byte[] Nupkg(RuntimeManager manager)
    {
        using var stream = new MemoryStream();
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
        {
            var rid = manager.Platform.RuntimeIdentifier;
            using var entry = zip.CreateEntry($"runtimes/{rid}/native/{NativeFile(manager)}").Open();
            entry.Write([9, 9, 9]);
        }
        return stream.ToArray();
    }

    /// <summary>Records every request; serves <see cref="Nupkg"/> for a .nupkg and 404 for anything else.</summary>
    private sealed class CountingFeed : HttpMessageHandler
    {
        public List<string> Requests { get; } = [];
        public byte[]? Nupkg { get; set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            lock (Requests)
                Requests.Add(url);

            if (Nupkg is not null && url.EndsWith(".nupkg", StringComparison.OrdinalIgnoreCase))
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Nupkg) });

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
