using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Tests for <see cref="LlamaServerStateManager.CleanupUnreferencedVersionsAsync"/>, the retention
/// mechanism for the llama-server binary cache. Superseded builds accumulate forever without it
/// (observed in the field: 59 versioned build dirs / 24 GB on a machine that lived through many
/// LMSupply versions) because <c>ActivateUpdateAsync</c> trims the rollback list in state only and
/// never touches the disk.
///
/// The mechanism is deliberately single: trimming state makes a version unreferenced, and cleanup
/// deletes any version directory (<c>b*</c>) under the cache root that no state entry references
/// via InstalledPath, PendingPath, or PreviousVersions[].Path.
///
/// Network-free / HW-free: state manager + real temp directories only.
/// </summary>
public sealed class LlamaServerCacheRetentionTests : IDisposable
{
    private readonly string _root;

    public LlamaServerCacheRetentionTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "lmsupply-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    private string CreateVersionDir(string version, string backend = "vulkan")
    {
        var dir = Path.Combine(_root, version, backend);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "llama-server.exe"), "stub");
        return dir;
    }

    private static LlamaServerVersionState State(string version, string path) => new()
    {
        InstalledVersion = version,
        InstalledPath = path,
        Backend = "vulkan"
    };

    [Fact]
    public async Task Cleanup_DeletesVersionDirectoriesNotReferencedByAnyStateEntry()
    {
        var referenced = CreateVersionDir("b200");
        CreateVersionDir("b100"); // orphan — never referenced by state

        using var manager = new LlamaServerStateManager(_root);
        await manager.UpdateStateAsync("vulkan", "win-x64", State("b200", referenced));

        var deleted = await manager.CleanupUnreferencedVersionsAsync();

        deleted.Should().Be(1, "exactly the orphaned b100 must be swept");
        Directory.Exists(Path.Combine(_root, "b100")).Should().BeFalse("orphan is unreferenced");
        Directory.Exists(Path.Combine(_root, "b200")).Should().BeTrue("installed version stays");
    }

    [Fact]
    public async Task Cleanup_KeepsInstalledPendingAndPreviousVersionDirectories()
    {
        var installed = CreateVersionDir("b300");
        var pending = CreateVersionDir("b400");
        var previous = CreateVersionDir("b200");
        CreateVersionDir("b100"); // orphan

        var state = State("b300", installed);
        state.PendingVersion = "b400";
        state.PendingPath = pending;
        state.UpdateReady = true;
        state.PreviousVersions.Add(new VersionEntry
        {
            Version = "b200",
            Path = previous,
            InstalledAt = DateTimeOffset.UtcNow
        });

        using var manager = new LlamaServerStateManager(_root);
        await manager.UpdateStateAsync("vulkan", "win-x64", state);

        var deleted = await manager.CleanupUnreferencedVersionsAsync();

        deleted.Should().Be(1);
        Directory.Exists(Path.Combine(_root, "b300")).Should().BeTrue("installed is referenced");
        Directory.Exists(Path.Combine(_root, "b400")).Should().BeTrue("pending is referenced");
        Directory.Exists(Path.Combine(_root, "b200")).Should().BeTrue("rollback previous is referenced");
        Directory.Exists(Path.Combine(_root, "b100")).Should().BeFalse("orphan must go");
    }

    [Fact]
    public async Task Cleanup_IgnoresNonVersionDirectoriesAndTheStateFile()
    {
        var referenced = CreateVersionDir("b200");
        Directory.CreateDirectory(Path.Combine(_root, "not-a-version"));
        Directory.CreateDirectory(Path.Combine(_root, "backup")); // no b\d+ shape

        using var manager = new LlamaServerStateManager(_root);
        await manager.UpdateStateAsync("vulkan", "win-x64", State("b200", referenced));

        var deleted = await manager.CleanupUnreferencedVersionsAsync();

        deleted.Should().Be(0);
        Directory.Exists(Path.Combine(_root, "not-a-version")).Should().BeTrue();
        Directory.Exists(Path.Combine(_root, "backup")).Should().BeTrue();
        File.Exists(Path.Combine(_root, "llama-server-state.json")).Should().BeTrue(
            "the persisted state file must survive cleanup");
    }

    [Fact]
    public async Task Cleanup_DoesNotDeletePrefixSiblingOfReferencedVersion()
    {
        // Path-prefix trap: "b90" is a string prefix of "b900" — matching must be
        // separator-aware or b900 would wrongly count as referenced (or vice versa).
        var referenced = CreateVersionDir("b90");
        CreateVersionDir("b900"); // orphan despite sharing the b90 prefix

        using var manager = new LlamaServerStateManager(_root);
        await manager.UpdateStateAsync("vulkan", "win-x64", State("b90", referenced));

        var deleted = await manager.CleanupUnreferencedVersionsAsync();

        deleted.Should().Be(1);
        Directory.Exists(Path.Combine(_root, "b90")).Should().BeTrue("referenced version stays");
        Directory.Exists(Path.Combine(_root, "b900")).Should().BeFalse("prefix sibling is an orphan");
    }

    [Fact]
    public async Task Cleanup_WithNoStateEntries_DeletesNothing()
    {
        // A machine can have cached builds but no state yet (pre-state caches are adopted via
        // GetCachedVersions discovery). Zero entries must be treated as "unknown", not "orphaned".
        CreateVersionDir("b100");
        CreateVersionDir("b200");

        using var manager = new LlamaServerStateManager(_root);

        var deleted = await manager.CleanupUnreferencedVersionsAsync();

        deleted.Should().Be(0, "an empty state must never be interpreted as everything-is-orphaned");
        Directory.Exists(Path.Combine(_root, "b100")).Should().BeTrue();
        Directory.Exists(Path.Combine(_root, "b200")).Should().BeTrue();
    }

    [Fact]
    public async Task Cleanup_ReadsStateFreshFromDisk_NotFromInMemoryCache()
    {
        // Cleanup is the destructive operation, and the cache directory is shared across
        // processes (consumer apps ship several desktop apps on one machine). A cleanup decided
        // from hours-stale in-memory state could delete a directory another process just
        // referenced as pending — so cleanup must re-read the state file from disk.
        var installed = CreateVersionDir("b200");
        var adopted = CreateVersionDir("b300"); // referenced only by the "other process" write

        using var manager = new LlamaServerStateManager(_root);
        await manager.UpdateStateAsync("vulkan", "win-x64", State("b200", installed)); // populates cache

        // Simulate another process updating the shared state file on disk.
        using (var external = new LlamaServerStateManager(_root))
        {
            var state = State("b200", installed);
            state.PendingVersion = "b300";
            state.PendingPath = adopted;
            state.UpdateReady = true;
            await external.UpdateStateAsync("vulkan", "win-x64", state);
        }

        var deleted = await manager.CleanupUnreferencedVersionsAsync();

        deleted.Should().Be(0, "the on-disk state references b300 as pending");
        Directory.Exists(Path.Combine(_root, "b300")).Should().BeTrue(
            "cleanup must honor the freshest on-disk state, not the process-local cache");
    }

    [Fact]
    public async Task ActivateUpdate_ThenCleanup_RemovesVersionTrimmedOutOfRollbackWindow()
    {
        // Full retention flow: install b1 → update to b2 → update to b3 with maxVersionsToKeep=1.
        // After the second activation b1 falls out of the rollback window, so cleanup must
        // reclaim its directory while b2 (rollback) and b3 (installed) stay.
        var dir1 = CreateVersionDir("b1");
        var dir2 = CreateVersionDir("b2");
        var dir3 = CreateVersionDir("b3");

        using var manager = new LlamaServerStateManager(_root);
        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b1", dir1);

        await manager.MarkUpdateReadyAsync("vulkan", "win-x64", "b2", dir2);
        (await manager.ActivateUpdateAsync("vulkan", "win-x64", maxVersionsToKeep: 1)).Should().NotBeNull();

        await manager.MarkUpdateReadyAsync("vulkan", "win-x64", "b3", dir3);
        (await manager.ActivateUpdateAsync("vulkan", "win-x64", maxVersionsToKeep: 1)).Should().NotBeNull();

        var deleted = await manager.CleanupUnreferencedVersionsAsync();

        deleted.Should().Be(1, "b1 was trimmed out of the rollback window");
        Directory.Exists(Path.Combine(_root, "b1")).Should().BeFalse("superseded build is reclaimed");
        Directory.Exists(Path.Combine(_root, "b2")).Should().BeTrue("rollback candidate stays");
        Directory.Exists(Path.Combine(_root, "b3")).Should().BeTrue("installed version stays");
    }
}
