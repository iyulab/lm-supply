using System.Text.Json;
using System.Text.Json.Serialization;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// The state file records builds by version and backend directory, resolved against the cache root in use, so a
/// relocated cache root (<c>LMSUPPLY_CACHE_DIR</c>, a moved profile, a renamed drive) keeps its builds and its
/// rollback history. A file written by a release that recorded absolute paths is migrated on read, and the file this
/// release writes still parses for such a release sharing the cache.
/// Network-free: <see cref="LlamaServerUpdateOptions.AutoDownloadUpdates"/> is off and every build is on disk.
/// </summary>
public sealed partial class LlamaServerCacheRelocationTests : IDisposable
{
    private static readonly string Exe = OperatingSystem.IsWindows() ? "llama-server.exe" : "llama-server";
    private static readonly string Platform = LlamaServerUpdateService.GetCurrentPlatform();

    private readonly string _base;
    private readonly string _oldRoot;
    private readonly string _newRoot;

    public LlamaServerCacheRelocationTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "lmsupply-tests", Guid.NewGuid().ToString("N"));
        _oldRoot = Path.Combine(_base, "old", "llama-server");
        _newRoot = Path.Combine(_base, "new", "llama-server");
        Directory.CreateDirectory(_oldRoot);
        Directory.CreateDirectory(_newRoot);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
    }

    private static void Build(string root, string version, string buildBackend)
    {
        var dir = Path.Combine(root, version, buildBackend);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, Exe), "stub");
    }

    private static string J(string value) => JsonSerializer.Serialize(value, V1Context.Default.String);

    /// <summary>
    /// A state file as LMSupply 0.121 and earlier wrote it: absolute paths under <paramref name="pathRoot"/>.
    /// cpu: b3 installed, previous b2 and b1; cuda12: b3; vulkan: b9 (its only build).
    /// </summary>
    private static void WriteV1State(string fileRoot, string pathRoot)
    {
        string P(string version, string backend) => J(Path.Combine(pathRoot, version, backend));
        var json = $$"""
            {
              "schemaVersion": 1,
              "entries": {
                "cpu|{{Platform}}": {
                  "installedVersion": "b3", "installedPath": {{P("b3", "cpu")}},
                  "lastVersionCheck": "2026-10-14T00:00:00+00:00", "updateReady": false,
                  "previousVersions": [
                    { "version": "b2", "path": {{P("b2", "cpu")}}, "installedAt": "2026-10-10T00:00:00+00:00" },
                    { "version": "b1", "path": {{P("b1", "cpu")}}, "installedAt": "2026-10-01T00:00:00+00:00" }
                  ],
                  "failedVersions": ["b4"], "backend": "cpu"
                },
                "cuda12|{{Platform}}": {
                  "installedVersion": "b3", "installedPath": {{P("b3", "cuda12")}},
                  "lastVersionCheck": "2026-10-14T00:00:00+00:00", "updateReady": false,
                  "previousVersions": [], "failedVersions": [], "backend": "cuda12"
                },
                "vulkan|{{Platform}}": {
                  "installedVersion": "b9", "installedPath": {{P("b9", "vulkan")}},
                  "lastVersionCheck": "2026-10-14T00:00:00+00:00", "updateReady": false,
                  "previousVersions": [], "failedVersions": [], "backend": "vulkan"
                }
              }
            }
            """;
        File.WriteAllText(Path.Combine(fileRoot, "llama-server-state.json"), json);
    }

    private static void CreateBuilds(string root)
    {
        Build(root, "b3", "cpu");
        Build(root, "b3", "cuda12");
        Build(root, "b2", "cpu");
        Build(root, "b1", "cpu");
        Build(root, "b9", "vulkan");
    }

    private LlamaServerUpdateService Service() => new(new LlamaServerUpdateOptions
    {
        CacheDirectory = _newRoot,
        AutoDownloadUpdates = false,
        VersionCheckInterval = TimeSpan.FromDays(365)
    });

    private static string[] BuildsUnder(string root) => Directory.GetDirectories(root)
        .SelectMany(Directory.GetDirectories)
        .Select(dir => $"{Path.GetFileName(Path.GetDirectoryName(dir))}/{Path.GetFileName(dir)}")
        .Order(StringComparer.Ordinal)
        .ToArray();

    [Fact]
    public async Task MovedRoot_OldRootStillPresent_ServesAndKeepsTheBuildsUnderTheNewRoot()
    {
        // The moment right after copying the cache: the old root still exists, the copied state names it.
        CreateBuilds(_oldRoot);
        CreateBuilds(_newRoot);
        WriteV1State(_newRoot, pathRoot: _oldRoot);
        var ct = TestContext.Current.CancellationToken;

        await using (var service = Service())
        {
            var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: ct);
            result.ServerPath.Should().Be(Path.Combine(_newRoot, "b3", "cpu", Exe), "the configured root is the one in use");
        }

        using var manager = new LlamaServerStateManager(_newRoot);
        var deleted = await manager.CleanupUnreferencedVersionsAsync(ct);

        deleted.Should().Be(0, "every build under the new root is installed or a rollback build");
        BuildsUnder(_newRoot).Should().Equal("b1/cpu", "b2/cpu", "b3/cpu", "b3/cuda12", "b9/vulkan");
    }

    [Fact]
    public async Task MovedRoot_OldRootGone_KeepsRollbackHistoryAndEveryBackendsInstall()
    {
        CreateBuilds(_newRoot);
        WriteV1State(_newRoot, pathRoot: Path.Combine(_base, "gone", "llama-server"));
        var ct = TestContext.Current.CancellationToken;

        await using (var service = Service())
        {
            var info = await service.GetUpdateInfoAsync(LlamaServerBackend.Cpu, ct);
            info.Should().NotBeNull();
            info!.ServerPath.Should().Be(Path.Combine(_newRoot, "b3", "cpu", Exe));
            File.Exists(info.ServerPath).Should().BeTrue();

            var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: ct);
            result.ServerPath.Should().Be(Path.Combine(_newRoot, "b3", "cpu", Exe));
        }

        using var manager = new LlamaServerStateManager(_newRoot);
        var cpu = await manager.GetStateAsync("cpu", Platform, ct);
        cpu!.PreviousVersions.Select(p => p.Version).Should().Equal("b2", "b1");
        cpu.FailedVersions.Should().Contain("b4");

        (await manager.CleanupUnreferencedVersionsAsync(ct)).Should().Be(0);
        BuildsUnder(_newRoot).Should().Contain("b9/vulkan", "vulkan's installed build is referenced by its own entry");
    }

    [Fact]
    public async Task Rollback_AfterMove_ReturnsTheBuildUnderTheNewRoot()
    {
        CreateBuilds(_newRoot);
        WriteV1State(_newRoot, pathRoot: Path.Combine(_base, "gone", "llama-server"));

        await using var service = Service();
        var result = await service.RollbackAsync(LlamaServerBackend.Cpu, TestContext.Current.CancellationToken);

        result.Success.Should().BeTrue();
        result.ServerPath.Should().Be(Path.Combine(_newRoot, "b2", "cpu", Exe));
    }

    [Fact]
    public async Task Rollback_SkipsPreviousBuildsThatAreGone()
    {
        CreateBuilds(_newRoot);
        Directory.Delete(Path.Combine(_newRoot, "b2"), recursive: true);
        WriteV1State(_newRoot, pathRoot: _newRoot);

        await using var service = Service();
        var result = await service.RollbackAsync(LlamaServerBackend.Cpu, TestContext.Current.CancellationToken);

        result.ServerPath.Should().Be(Path.Combine(_newRoot, "b1", "cpu", Exe), "b2's build is gone, b1's is not");
    }

    [Fact]
    public async Task InstalledBuildGone_AdoptsACachedBuildAndKeepsThePreviousBuildsThatExist()
    {
        // The recorded install (b3) is gone; b2 is cached. Adoption must not reset the entry.
        Build(_newRoot, "b2", "cpu");
        WriteV1State(_newRoot, pathRoot: _newRoot);
        Build(_newRoot, "b1", "cpu");
        var ct = TestContext.Current.CancellationToken;

        await using (var service = Service())
        {
            var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: ct);
            result.ServerPath.Should().Be(Path.Combine(_newRoot, "b2", "cpu", Exe));
        }

        using var manager = new LlamaServerStateManager(_newRoot);
        var cpu = await manager.GetStateAsync("cpu", Platform, ct);
        cpu!.InstalledVersion.Should().Be("b2");
        cpu.PreviousVersions.Select(p => p.Version).Should().Equal(["b1"], "b2 is now installed and b1's build exists");
        cpu.FailedVersions.Should().Contain("b4", "adoption keeps the versions known to fail");
    }

    [Fact]
    public async Task GetUpdateInfo_RecordedBuildGone_ReturnsNull()
    {
        WriteV1State(_newRoot, pathRoot: _newRoot); // no builds on disk at all

        await using var service = Service();

        (await service.GetUpdateInfoAsync(LlamaServerBackend.Cpu, TestContext.Current.CancellationToken))
            .Should().BeNull("nothing is installed under the cache root in use");
    }

    [Fact]
    public async Task Cleanup_ReclaimsABackendDirectoryNoEntryReferences_KeepsItsSibling()
    {
        Build(_newRoot, "b1", "cpu");
        Build(_newRoot, "b1", "cuda12"); // orphan: no entry references b1/cuda12
        Build(_newRoot, "b2", "cpu");
        var ct = TestContext.Current.CancellationToken;

        using var manager = new LlamaServerStateManager(_newRoot);
        await manager.CreateInitialStateAsync("cpu", Platform, "b1", "cpu", ct);
        await manager.MarkUpdateReadyAsync("cpu", Platform, "b2", "cpu", ct);
        await manager.ActivateUpdateAsync("cpu", Platform, cancellationToken: ct);

        (await manager.CleanupUnreferencedVersionsAsync(ct)).Should().Be(1);
        BuildsUnder(_newRoot).Should().Equal("b1/cpu", "b2/cpu");
    }

    [Fact]
    public async Task FallbackBuild_ResolvesUnderTheBackendDirectoryItWasExtractedTo()
    {
        // A CUDA request that fell back to the CPU build: the cuda12 entry's build lives under b5/cpu.
        Build(_newRoot, "b5", "cpu");
        var ct = TestContext.Current.CancellationToken;
        using (var manager = new LlamaServerStateManager(_newRoot))
            await manager.CreateInitialStateAsync("cuda12", Platform, "b5", "cpu", ct);

        await using var service = Service();
        var result = await service.GetServerPathAsync(LlamaServerBackend.Cuda12, cancellationToken: ct);

        result.ServerPath.Should().Be(Path.Combine(_newRoot, "b5", "cpu", Exe));
    }

    [Fact]
    public async Task V1File_IsMigratedToVersionAndBuildBackend()
    {
        WriteV1State(_newRoot, pathRoot: Path.Combine(_base, "elsewhere", "llama-server"));

        using var manager = new LlamaServerStateManager(_newRoot);
        var cpu = await manager.GetStateAsync("cpu", Platform, TestContext.Current.CancellationToken);

        cpu!.InstalledBuildBackend.Should().Be("cpu");
        cpu.PreviousVersions.Select(p => p.BuildBackend).Should().Equal("cpu", "cpu");
        manager.GetBuildDirectory(cpu.InstalledVersion, cpu.InstalledBuildBackend!)
            .Should().Be(Path.Combine(_newRoot, "b3", "cpu"));
    }

    [Fact]
    public async Task WrittenFile_ParsesForAReleaseThatRequiresAbsolutePaths_PointingUnderTheCurrentRoot()
    {
        // LMSupply 0.121 and earlier require installedPath / previousVersions[].path and resolve builds from them; a
        // file they cannot parse is discarded whole (every other backend's entry and rollback history with it).
        CreateBuilds(_newRoot);
        WriteV1State(_newRoot, pathRoot: Path.Combine(_base, "gone", "llama-server"));
        using (var manager = new LlamaServerStateManager(_newRoot))
            await manager.CreateInitialStateAsync("cpu", Platform, "b3", "cpu", TestContext.Current.CancellationToken);

        var json = File.ReadAllText(Path.Combine(_newRoot, "llama-server-state.json"));
        var file = JsonSerializer.Deserialize(json, V1Context.Default.V1File)!;

        file.Entries.Should().HaveCount(3);
        file.Entries[$"cpu|{Platform}"].InstalledPath.Should().Be(Path.Combine(_newRoot, "b3", "cpu"));
        file.Entries[$"cpu|{Platform}"].PreviousVersions.Select(p => p.Path)
            .Should().Equal(Path.Combine(_newRoot, "b2", "cpu"), Path.Combine(_newRoot, "b1", "cpu"));
        file.Entries[$"vulkan|{Platform}"].InstalledPath.Should().Be(Path.Combine(_newRoot, "b9", "vulkan"));
    }

    // The schema LMSupply 0.121 reads (required members as it declares them).
    [JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
    [JsonSerializable(typeof(V1File))]
    [JsonSerializable(typeof(string))]
    internal sealed partial class V1Context : JsonSerializerContext;

    internal sealed class V1File
    {
        public Dictionary<string, V1State> Entries { get; set; } = [];
    }

    internal sealed class V1State
    {
        public required string InstalledVersion { get; set; }
        public required string InstalledPath { get; set; }
        public List<V1Entry> PreviousVersions { get; set; } = [];
    }

    internal sealed class V1Entry
    {
        public required string Version { get; set; }
        public required string Path { get; set; }
    }
}
