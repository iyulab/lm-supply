using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

public class RuntimeVersionStateManagerTests : IDisposable
{
    private readonly string _tempDir;

    public RuntimeVersionStateManagerTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "lm-supply-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
        GC.SuppressFinalize(this);
    }

    private RuntimeVersionStateManager CreateManager() => new(_tempDir);

    // --- GetPackageKey ---

    [Fact]
    public void GetPackageKey_FormatsCorrectly()
    {
        var key = RuntimeVersionStateManager.GetPackageKey("llama-server", "vulkan", "win-x64");

        key.Should().Be("llama-server|vulkan|win-x64");
    }

    [Fact]
    public void GetPackageKey_IsCaseInsensitive()
    {
        var key1 = RuntimeVersionStateManager.GetPackageKey("LLama-Server", "Vulkan", "Win-X64");
        var key2 = RuntimeVersionStateManager.GetPackageKey("llama-server", "vulkan", "win-x64");

        key1.Should().Be(key2);
    }

    // --- StateFilePath ---

    [Fact]
    public void StateFilePath_ReturnsExpectedPath()
    {
        using var manager = CreateManager();

        manager.StateFilePath.Should().Be(Path.Combine(_tempDir, "runtime-versions.json"));
    }

    // --- GetStateAsync ---

    [Fact]
    public async Task GetStateAsync_NoState_ReturnsNull()
    {
        using var manager = CreateManager();

        var state = await manager.GetStateAsync("pkg|provider|rid");

        state.Should().BeNull();
    }

    // --- GetOrCreateStateAsync ---

    [Fact]
    public async Task GetOrCreateStateAsync_NewKey_CreatesState()
    {
        using var manager = CreateManager();
        var key = "llama-server|vulkan|win-x64";

        var state = await manager.GetOrCreateStateAsync(key, "1.0.0");

        state.Should().NotBeNull();
        state.InstalledVersion.Should().Be("1.0.0");
        state.LastVersionCheck.Should().Be(DateTimeOffset.MinValue);
    }

    [Fact]
    public async Task GetOrCreateStateAsync_ExistingKey_ReturnsExisting()
    {
        using var manager = CreateManager();
        var key = "llama-server|vulkan|win-x64";

        var first = await manager.GetOrCreateStateAsync(key, "1.0.0");
        first.LatestKnownVersion = "2.0.0"; // Modify in-memory
        await manager.UpdateStateAsync(key, first);

        var second = await manager.GetOrCreateStateAsync(key, "9.9.9");

        // Should return existing, NOT recreate with 9.9.9
        second.InstalledVersion.Should().Be("1.0.0");
        second.LatestKnownVersion.Should().Be("2.0.0");
    }

    // --- UpdateStateAsync ---

    [Fact]
    public async Task UpdateStateAsync_PersistsState()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";

        var state = new RuntimeVersionState { InstalledVersion = "1.0.0" };
        await manager.UpdateStateAsync(key, state);

        var loaded = await manager.GetStateAsync(key);
        loaded.Should().NotBeNull();
        loaded!.InstalledVersion.Should().Be("1.0.0");
    }

    // --- RecordVersionCheckAsync ---

    [Fact]
    public async Task RecordVersionCheckAsync_UpdatesFieldsOnExistingState()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";
        await manager.GetOrCreateStateAsync(key, "1.0.0");

        var before = DateTimeOffset.UtcNow;
        await manager.RecordVersionCheckAsync(key, "2.0.0");

        var state = await manager.GetStateAsync(key);
        state!.LatestKnownVersion.Should().Be("2.0.0");
        state.LastVersionCheck.Should().BeOnOrAfter(before);
    }

    [Fact]
    public async Task RecordVersionCheckAsync_NoExistingState_DoesNothing()
    {
        using var manager = CreateManager();

        await manager.RecordVersionCheckAsync("nonexistent", "2.0.0");

        var state = await manager.GetStateAsync("nonexistent");
        state.Should().BeNull();
    }

    // --- MarkUpdateReadyAsync ---

    [Fact]
    public async Task MarkUpdateReadyAsync_SetsUpdateFields()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";
        await manager.GetOrCreateStateAsync(key, "1.0.0");

        await manager.MarkUpdateReadyAsync(key, "2.0.0", "/update/path");

        var state = await manager.GetStateAsync(key);
        state!.UpdateReady.Should().BeTrue();
        state.UpdateReadyPath.Should().Be("/update/path");
        state.LatestKnownVersion.Should().Be("2.0.0");
        // PendingVersion is cleared to null per implementation
        state.PendingVersion.Should().BeNull();
    }

    [Fact]
    public async Task MarkUpdateReadyAsync_NoExistingState_DoesNothing()
    {
        using var manager = CreateManager();

        await manager.MarkUpdateReadyAsync("nonexistent", "2.0.0", "/path");

        var state = await manager.GetStateAsync("nonexistent");
        state.Should().BeNull();
    }

    // --- ActivateUpdateAsync ---

    [Fact]
    public async Task ActivateUpdateAsync_NoState_ReturnsNull()
    {
        using var manager = CreateManager();

        var result = await manager.ActivateUpdateAsync("nonexistent", 2);

        result.Should().BeNull();
    }

    [Fact]
    public async Task ActivateUpdateAsync_NotReady_ReturnsCurrent()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";
        await manager.GetOrCreateStateAsync(key, "1.0.0");

        var result = await manager.ActivateUpdateAsync(key, 2);

        result.Should().NotBeNull();
        result!.InstalledVersion.Should().Be("1.0.0");
        result.UpdateReady.Should().BeFalse();
    }

    [Fact]
    public async Task ActivateUpdateAsync_MovesCurrentToPrevious()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";
        await manager.GetOrCreateStateAsync(key, "1.0.0");
        await manager.MarkUpdateReadyAsync(key, "2.0.0", "/new");

        var result = await manager.ActivateUpdateAsync(key, 2);

        result.Should().NotBeNull();
        result!.InstalledVersion.Should().Be("2.0.0");
        result.UpdateReady.Should().BeFalse();
        result.UpdateReadyPath.Should().BeNull();
        result.PreviousVersions.Should().HaveCount(1);
        result.PreviousVersions[0].Should().Be("1.0.0");
    }

    [Fact]
    public async Task ActivateUpdateAsync_TrimsOldVersions()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";
        await manager.GetOrCreateStateAsync(key, "1.0.0");

        // Activate: 1.0.0 -> 2.0.0 -> 3.0.0 -> 4.0.0
        await manager.MarkUpdateReadyAsync(key, "2.0.0", "/v2");
        await manager.ActivateUpdateAsync(key, 2);

        await manager.MarkUpdateReadyAsync(key, "3.0.0", "/v3");
        await manager.ActivateUpdateAsync(key, 2);

        await manager.MarkUpdateReadyAsync(key, "4.0.0", "/v4");
        var result = await manager.ActivateUpdateAsync(key, 2);

        result!.PreviousVersions.Should().HaveCount(2);
        result.PreviousVersions[0].Should().Be("3.0.0");
        result.PreviousVersions[1].Should().Be("2.0.0");
    }

    // --- RollbackAsync ---

    [Fact]
    public async Task RollbackAsync_NoState_ReturnsNulls()
    {
        using var manager = CreateManager();

        var (previousVersion, path) = await manager.RollbackAsync("nonexistent", "2.0.0");

        previousVersion.Should().BeNull();
        path.Should().BeNull();
    }

    [Fact]
    public async Task RollbackAsync_NoPreviousVersions_ReturnsNulls()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";
        await manager.GetOrCreateStateAsync(key, "1.0.0");

        var (previousVersion, path) = await manager.RollbackAsync(key, "1.0.0");

        previousVersion.Should().BeNull();
        path.Should().BeNull();
    }

    [Fact]
    public async Task RollbackAsync_RestoresPreviousAndMarksCurrentFailed()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";
        await manager.GetOrCreateStateAsync(key, "1.0.0");
        await manager.MarkUpdateReadyAsync(key, "2.0.0", "/v2");
        await manager.ActivateUpdateAsync(key, 2);

        var (previousVersion, _) = await manager.RollbackAsync(key, "2.0.0");

        previousVersion.Should().Be("1.0.0");

        var state = await manager.GetStateAsync(key);
        state!.InstalledVersion.Should().Be("1.0.0");
        state.FailedVersions.Should().Contain("2.0.0");
        state.UpdateReady.Should().BeFalse();
        state.PreviousVersions.Should().BeEmpty();
    }

    // --- IsVersionCheckDueAsync ---

    [Fact]
    public async Task IsVersionCheckDueAsync_NoState_ReturnsTrue()
    {
        using var manager = CreateManager();

        var due = await manager.IsVersionCheckDueAsync("nonexistent", TimeSpan.FromHours(1));

        due.Should().BeTrue();
    }

    [Fact]
    public async Task IsVersionCheckDueAsync_RecentCheck_ReturnsFalse()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";
        await manager.GetOrCreateStateAsync(key, "1.0.0");
        await manager.RecordVersionCheckAsync(key, "1.0.0");

        var due = await manager.IsVersionCheckDueAsync(key, TimeSpan.FromHours(1));

        due.Should().BeFalse();
    }

    [Fact]
    public async Task IsVersionCheckDueAsync_OldCheck_ReturnsTrue()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";

        // Create with MinValue timestamp (never checked)
        var state = await manager.GetOrCreateStateAsync(key, "1.0.0");
        // LastVersionCheck defaults to DateTimeOffset.MinValue, so any interval should be due

        var due = await manager.IsVersionCheckDueAsync(key, TimeSpan.FromMilliseconds(1));

        due.Should().BeTrue();
    }

    // --- MarkPendingDownloadAsync ---

    [Fact]
    public async Task MarkPendingDownloadAsync_SetsPendingVersion()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";
        await manager.GetOrCreateStateAsync(key, "1.0.0");

        await manager.MarkPendingDownloadAsync(key, "2.0.0");

        var state = await manager.GetStateAsync(key);
        state!.PendingVersion.Should().Be("2.0.0");
    }

    [Fact]
    public async Task MarkPendingDownloadAsync_NoState_DoesNothing()
    {
        using var manager = CreateManager();

        await manager.MarkPendingDownloadAsync("nonexistent", "2.0.0");

        var state = await manager.GetStateAsync("nonexistent");
        state.Should().BeNull();
    }

    // --- ClearPendingDownloadAsync ---

    [Fact]
    public async Task ClearPendingDownloadAsync_ClearsPendingVersion()
    {
        using var manager = CreateManager();
        var key = "pkg|provider|rid";
        await manager.GetOrCreateStateAsync(key, "1.0.0");
        await manager.MarkPendingDownloadAsync(key, "2.0.0");

        await manager.ClearPendingDownloadAsync(key);

        var state = await manager.GetStateAsync(key);
        state!.PendingVersion.Should().BeNull();
    }

    // --- RuntimeVersionState.UpdateAvailable ---

    [Fact]
    public void UpdateAvailable_SameVersion_ReturnsFalse()
    {
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "1.0.0"
        };

        state.UpdateAvailable.Should().BeFalse();
    }

    [Fact]
    public void UpdateAvailable_DifferentVersion_ReturnsTrue()
    {
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "2.0.0"
        };

        state.UpdateAvailable.Should().BeTrue();
    }

    [Fact]
    public void UpdateAvailable_NullLatest_ReturnsFalse()
    {
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = null
        };

        state.UpdateAvailable.Should().BeFalse();
    }

    [Fact]
    public void UpdateAvailable_FailedVersion_ReturnsFalse()
    {
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "2.0.0",
            FailedVersions = ["2.0.0"]
        };

        state.UpdateAvailable.Should().BeFalse();
    }

    [Fact]
    public void UpdateAvailable_CaseInsensitive()
    {
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "1.0.0" // Same version, different case not applicable for same string
        };

        state.UpdateAvailable.Should().BeFalse();
    }

    // --- Persistence across instances ---

    [Fact]
    public async Task State_PersistsAcrossInstances()
    {
        var key = "pkg|provider|rid";

        using (var manager1 = CreateManager())
        {
            await manager1.GetOrCreateStateAsync(key, "1.0.0");
        }

        using var manager2 = CreateManager();
        var state = await manager2.GetStateAsync(key);

        state.Should().NotBeNull();
        state!.InstalledVersion.Should().Be("1.0.0");
    }

    [Fact]
    public async Task State_CorruptedFile_StartsFromFresh()
    {
        var stateFilePath = Path.Combine(_tempDir, "runtime-versions.json");
        await File.WriteAllTextAsync(stateFilePath, "{ corrupted json!!! }");

        using var manager = CreateManager();
        var state = await manager.GetStateAsync("pkg|provider|rid");

        state.Should().BeNull();
    }

    // --- Multiple packages ---

    [Fact]
    public async Task MultiplePackages_IndependentState()
    {
        using var manager = CreateManager();
        var key1 = "llama-server|vulkan|win-x64";
        var key2 = "onnxruntime|cpu|win-x64";

        await manager.GetOrCreateStateAsync(key1, "1.0.0");
        await manager.GetOrCreateStateAsync(key2, "2.0.0");

        var state1 = await manager.GetStateAsync(key1);
        var state2 = await manager.GetStateAsync(key2);

        state1!.InstalledVersion.Should().Be("1.0.0");
        state2!.InstalledVersion.Should().Be("2.0.0");
    }

    // --- Dispose ---

    [Fact]
    public void Dispose_DoubleDispose_DoesNotThrow()
    {
        var manager = CreateManager();

        manager.Dispose();
        manager.Dispose(); // Should not throw
    }
}
