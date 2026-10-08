using System.IO.Compression;
using System.Net;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Inference;
using LMSupply.Runtime;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// <see cref="ExecutionProvider.OpenVino"/> is an ONNX Runtime plugin with its own package and version line. It is
/// provisioned at the version this library pins, never "the latest on the feed" and never the ONNX Runtime version, and
/// a plugin request never turns into the CPU package.
/// </summary>
[Trait("Category", "Unit")]
public sealed class ExecutionProviderPluginTests : IDisposable
{
    private const string Plugin = RuntimePackageRegistry.PackageTypes.ExecutionProviderPlugin;
    private const string OpenVino = RuntimePackageRegistry.Providers.OpenVino;
    private const string PluginLibrary = "onnxruntime_providers_openvino_plugin";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "lmsupply-ep-plugin-" + Guid.NewGuid().ToString("N"));

    private string CacheDir => Path.Combine(_root, "cache");

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Registry_OpenVino_IsThePinnedIntelPluginPackage()
    {
        var config = RuntimePackageRegistry.GetPackageConfig(Plugin, OpenVino);

        config.Should().NotBeNull();
        config!.PackageId.Should().Be("Intel.ML.OnnxRuntime.EP.OpenVINO");
        config.NativeLibraryName.Should().Be(PluginLibrary);
        config.PinnedVersion.Should().Be(RuntimePackageRegistry.OpenVinoPluginVersion);
    }

    [Fact]
    public void Registry_UnknownPlugin_IsNull_NotTheCpuPackage()
    {
        RuntimePackageRegistry.GetPackageConfig(Plugin, "no-such-plugin").Should().BeNull(
            "a plugin is its provider; the CPU package cannot stand in for it");
    }

    [Fact]
    public void Registry_OnnxRuntimePackages_FollowTheAssemblyVersion()
    {
        RuntimePackageRegistry.GetPackageConfig(RuntimePackageRegistry.PackageTypes.OnnxRuntime, "cpu")!
            .PinnedVersion.Should().BeNull();
    }

    [Fact]
    public async Task Provisioning_DownloadsThePinnedVersion_EvenUnderAGlobalOnnxRuntimePin()
    {
        // The global pin names an ONNX Runtime version; the plugin's own pin is the only version it has.
        var (manager, feed) = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, PinnedVersion = "1.29.0" });
        feed.Nupkg = Nupkg(manager, includePlugin: true);

        var library = await manager.EnsureExecutionProviderPluginAsync(OpenVino, progress: null, TestContext.Current.CancellationToken);

        Path.GetFileName(library).Should().Be(RuntimePackageRegistry.GetNativeLibraryFileName(PluginLibrary, manager.Platform));
        File.Exists(library).Should().BeTrue();
        library.Should().Contain(Path.Combine(Plugin, OpenVino, RuntimePackageRegistry.OpenVinoPluginVersion));
        feed.Requests.Should().NotBeEmpty().And.OnlyContain(u => u.Contains($"/{RuntimePackageRegistry.OpenVinoPluginVersion}/"),
            "the pinned version is downloaded by name; no version index or 'latest' lookup is made");
    }

    [Fact]
    public async Task Provisioning_APackageWithoutThePluginLibrary_Throws()
    {
        var (manager, feed) = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir });
        feed.Nupkg = Nupkg(manager, includePlugin: false);

        var act = () => manager.EnsureExecutionProviderPluginAsync(OpenVino, progress: null, TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<ModelLoadException>()).WithMessage($"*{PluginLibrary}*");
    }

    [Fact]
    public async Task LocalOnly_AnotherCachedPluginVersion_DoesNotStandIn()
    {
        var (manager, feed) = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, DisableAutoDownload = true });
        var other = Path.Combine(CacheDir, Plugin, OpenVino, "1.6.1", manager.Platform.RuntimeIdentifier);
        Directory.CreateDirectory(other);
        File.WriteAllBytes(Path.Combine(other, RuntimePackageRegistry.GetNativeLibraryFileName(PluginLibrary, manager.Platform)), [1]);

        var act = () => manager.EnsureExecutionProviderPluginAsync(OpenVino, progress: null, TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<ModelLoadException>()).WithMessage($"*{RuntimePackageRegistry.OpenVinoPluginVersion}*");
        feed.Requests.Should().BeEmpty();
    }

    [Fact]
    public void Configure_BeforeThePluginIsRegistered_AppendsNothing()
    {
        if (OpenVinoExecutionProvider.IsRegistered)
            Assert.Skip("The plugin is registered in this process already.");

        using var options = new SessionOptions();
        OnnxSessionFactory.ConfigureExecutionProvider(options, ExecutionProvider.OpenVino).Should().BeFalse(
            "the session runs on CPU, as an explicit CUDA request does on a host without CUDA");
    }

    [Fact]
    public void ActiveProviders_ForAnAppendedOpenVinoProvider_NameIt_AndCountAsGpu()
    {
        var active = OnnxSessionFactory.ResolveActiveProviders(ExecutionProvider.OpenVino, gpuEpAppended: true);

        active.Should().Equal("OpenVINOExecutionProvider", "CPUExecutionProvider");
        OnnxSessionFactory.ResolveActiveProviders(ExecutionProvider.OpenVino, gpuEpAppended: false)
            .Should().Equal("CPUExecutionProvider");
    }

    [Fact]
    public void GpuLoadConfig_CarriesTheCacheDirectory_AsOpenVinoJson()
    {
        // The plugin ignores a bare cache_dir option; the GPU model cache is an OpenVINO device property in load_config.
        var dir = Path.Combine(_root, "a dir with spaces", "model-cache");

        using var json = System.Text.Json.JsonDocument.Parse(OpenVinoExecutionProvider.GpuLoadConfig(dir));

        json.RootElement.GetProperty("GPU").GetProperty("CACHE_DIR").GetString().Should().Be(dir);
    }

    private static async Task<(RuntimeManager Manager, CountingFeed Feed)> CreateAsync(RuntimeManagerOptions options)
    {
        var feed = new CountingFeed();
        var manager = new RuntimeManager(options, updateOptions: null, feed);
        await manager.InitializeAsync(TestContext.Current.CancellationToken);
        return (manager, feed);
    }

    private static byte[] Nupkg(RuntimeManager manager, bool includePlugin)
    {
        using var stream = new MemoryStream();
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
        {
            var native = $"runtimes/{manager.Platform.RuntimeIdentifier}/native/";
            var files = new List<string> { RuntimePackageRegistry.GetNativeLibraryFileName("openvino", manager.Platform) };
            if (includePlugin)
                files.Add(RuntimePackageRegistry.GetNativeLibraryFileName(PluginLibrary, manager.Platform));
            foreach (var file in files)
            {
                using var entry = zip.CreateEntry(native + file).Open();
                entry.Write([7, 7, 7]);
            }
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
