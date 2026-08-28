using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

public class LlamaServerStateManagerTests : IDisposable
{
    private readonly string _tempDir;

    public LlamaServerStateManagerTests()
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

    private LlamaServerStateManager CreateManager() => new(_tempDir);

    // --- Constructor ---

    [Fact]
    public void Constructor_CreatesDirectory()
    {
        var dir = Path.Combine(_tempDir, "sub");
        using var manager = new LlamaServerStateManager(dir);

        Directory.Exists(dir).Should().BeTrue();
    }

    // --- GetStateAsync ---

    [Fact]
    public async Task GetStateAsync_NoState_ReturnsNull()
    {
        using var manager = CreateManager();

        var state = await manager.GetStateAsync("vulkan", "win-x64");

        state.Should().BeNull();
    }

    [Fact]
    public async Task GetStateAsync_AfterCreate_ReturnsState()
    {
        using var manager = CreateManager();

        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/path/to/server");

        var state = await manager.GetStateAsync("vulkan", "win-x64");

        state.Should().NotBeNull();
        state!.InstalledVersion.Should().Be("b7898");
        state.InstalledPath.Should().Be("/path/to/server");
        state.Backend.Should().Be("vulkan");
    }

    [Fact]
    public async Task GetStateAsync_DifferentKey_ReturnsNull()
    {
        using var manager = CreateManager();

        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/path");

        var state = await manager.GetStateAsync("cuda12", "win-x64");

        state.Should().BeNull();
    }

    [Fact]
    public async Task GetStateAsync_KeyIsCaseInsensitive()
    {
        using var manager = CreateManager();

        await manager.CreateInitialStateAsync("Vulkan", "Win-X64", "b7898", "/path");

        var state = await manager.GetStateAsync("vulkan", "win-x64");

        state.Should().NotBeNull();
        state!.InstalledVersion.Should().Be("b7898");
    }

    // --- UpdateStateAsync ---

    [Fact]
    public async Task UpdateStateAsync_OverwritesExistingState()
    {
        using var manager = CreateManager();

        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/old");

        var newState = new LlamaServerVersionState
        {
            InstalledVersion = "b7900",
            InstalledPath = "/new",
            Backend = "vulkan"
        };
        await manager.UpdateStateAsync("vulkan", "win-x64", newState);

        var loaded = await manager.GetStateAsync("vulkan", "win-x64");
        loaded.Should().NotBeNull();
        loaded!.InstalledVersion.Should().Be("b7900");
        loaded.InstalledPath.Should().Be("/new");
    }

    // --- RecordVersionCheckAsync ---

    [Fact]
    public async Task RecordVersionCheckAsync_UpdatesVersionAndTimestamp()
    {
        using var manager = CreateManager();
        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/path");

        var before = DateTimeOffset.UtcNow;
        await manager.RecordVersionCheckAsync("vulkan", "win-x64", "b7900");

        var state = await manager.GetStateAsync("vulkan", "win-x64");
        state!.LatestKnownVersion.Should().Be("b7900");
        state.LastVersionCheck.Should().BeOnOrAfter(before);
    }

    [Fact]
    public async Task RecordVersionCheckAsync_NoExistingState_DoesNothing()
    {
        using var manager = CreateManager();

        // Should not throw
        await manager.RecordVersionCheckAsync("vulkan", "win-x64", "b7900");

        var state = await manager.GetStateAsync("vulkan", "win-x64");
        state.Should().BeNull();
    }

    // --- MarkUpdateReadyAsync ---

    [Fact]
    public async Task MarkUpdateReadyAsync_SetsUpdateFields()
    {
        using var manager = CreateManager();
        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/old");

        await manager.MarkUpdateReadyAsync("vulkan", "win-x64", "b7900", "/new");

        var state = await manager.GetStateAsync("vulkan", "win-x64");
        state!.PendingVersion.Should().Be("b7900");
        state.PendingPath.Should().Be("/new");
        state.UpdateReady.Should().BeTrue();
    }

    [Fact]
    public async Task MarkUpdateReadyAsync_NoExistingState_DoesNothing()
    {
        using var manager = CreateManager();

        await manager.MarkUpdateReadyAsync("vulkan", "win-x64", "b7900", "/path");

        var state = await manager.GetStateAsync("vulkan", "win-x64");
        state.Should().BeNull();
    }

    // --- ActivateUpdateAsync ---

    [Fact]
    public async Task ActivateUpdateAsync_NoState_ReturnsNull()
    {
        using var manager = CreateManager();

        var result = await manager.ActivateUpdateAsync("vulkan", "win-x64");

        result.Should().BeNull();
    }

    [Fact]
    public async Task ActivateUpdateAsync_NoUpdateReady_ReturnsNull()
    {
        using var manager = CreateManager();
        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/path");

        var result = await manager.ActivateUpdateAsync("vulkan", "win-x64");

        result.Should().BeNull();
    }

    [Fact]
    public async Task ActivateUpdateAsync_MovesCurrentToPrevious()
    {
        using var manager = CreateManager();
        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/old");
        await manager.MarkUpdateReadyAsync("vulkan", "win-x64", "b7900", "/new");

        var result = await manager.ActivateUpdateAsync("vulkan", "win-x64");

        result.Should().NotBeNull();
        result!.InstalledVersion.Should().Be("b7900");
        result.InstalledPath.Should().Be("/new");
        result.UpdateReady.Should().BeFalse();
        result.PendingVersion.Should().BeNull();
        result.PendingPath.Should().BeNull();
        result.PreviousVersions.Should().HaveCount(1);
        result.PreviousVersions[0].Version.Should().Be("b7898");
        result.PreviousVersions[0].Path.Should().Be("/old");
    }

    [Fact]
    public async Task ActivateUpdateAsync_TrimsOldVersions()
    {
        using var manager = CreateManager();
        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7896", "/v1");

        // Activate updates: b7896 -> b7897 -> b7898 -> b7899
        await manager.MarkUpdateReadyAsync("vulkan", "win-x64", "b7897", "/v2");
        await manager.ActivateUpdateAsync("vulkan", "win-x64", maxVersionsToKeep: 2);

        await manager.MarkUpdateReadyAsync("vulkan", "win-x64", "b7898", "/v3");
        await manager.ActivateUpdateAsync("vulkan", "win-x64", maxVersionsToKeep: 2);

        await manager.MarkUpdateReadyAsync("vulkan", "win-x64", "b7899", "/v4");
        var result = await manager.ActivateUpdateAsync("vulkan", "win-x64", maxVersionsToKeep: 2);

        result!.PreviousVersions.Should().HaveCount(2);
        result.PreviousVersions[0].Version.Should().Be("b7898"); // most recent previous
        result.PreviousVersions[1].Version.Should().Be("b7897");
    }

    // --- RollbackAsync ---

    [Fact]
    public async Task RollbackAsync_NoState_ReturnsNull()
    {
        using var manager = CreateManager();

        var result = await manager.RollbackAsync("vulkan", "win-x64");

        result.Should().BeNull();
    }

    [Fact]
    public async Task RollbackAsync_NoPreviousVersions_ReturnsNull()
    {
        using var manager = CreateManager();
        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/path");

        var result = await manager.RollbackAsync("vulkan", "win-x64");

        result.Should().BeNull();
    }

    [Fact]
    public async Task RollbackAsync_RestoresPreviousAndMarksCurrentFailed()
    {
        using var manager = CreateManager();
        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/old");
        await manager.MarkUpdateReadyAsync("vulkan", "win-x64", "b7900", "/new");
        await manager.ActivateUpdateAsync("vulkan", "win-x64");

        var result = await manager.RollbackAsync("vulkan", "win-x64");

        result.Should().NotBeNull();
        result!.InstalledVersion.Should().Be("b7898");
        result.InstalledPath.Should().Be("/old");
        result.FailedVersions.Should().Contain("b7900");
        result.UpdateReady.Should().BeFalse();
        result.PendingVersion.Should().BeNull();
        result.PendingPath.Should().BeNull();
        result.PreviousVersions.Should().BeEmpty();
    }

    // --- MarkVersionFailedAsync ---

    [Fact]
    public async Task MarkVersionFailedAsync_AddsToFailedSet()
    {
        using var manager = CreateManager();
        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/path");

        await manager.MarkVersionFailedAsync("vulkan", "win-x64", "b7900");
        await manager.MarkVersionFailedAsync("vulkan", "win-x64", "b7901");

        var state = await manager.GetStateAsync("vulkan", "win-x64");
        state!.FailedVersions.Should().HaveCount(2);
        state.FailedVersions.Should().Contain("b7900");
        state.FailedVersions.Should().Contain("b7901");
    }

    [Fact]
    public async Task MarkVersionFailedAsync_NoExistingState_DoesNothing()
    {
        using var manager = CreateManager();

        await manager.MarkVersionFailedAsync("vulkan", "win-x64", "b7900");

        var state = await manager.GetStateAsync("vulkan", "win-x64");
        state.Should().BeNull();
    }

    // --- CreateInitialStateAsync ---

    [Fact]
    public async Task CreateInitialStateAsync_SetsAllFields()
    {
        using var manager = CreateManager();

        var before = DateTimeOffset.UtcNow;
        var state = await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/path");

        state.InstalledVersion.Should().Be("b7898");
        state.InstalledPath.Should().Be("/path");
        state.Backend.Should().Be("vulkan");
        state.LatestKnownVersion.Should().Be("b7898");
        state.LastVersionCheck.Should().BeOnOrAfter(before);
        state.PreviousVersions.Should().BeEmpty();
        state.FailedVersions.Should().BeEmpty();
        state.UpdateReady.Should().BeFalse();
    }

    // --- Persistence across instances ---

    [Fact]
    public async Task State_PersistsAcrossInstances()
    {
        using (var manager1 = CreateManager())
        {
            await manager1.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/path");
        }

        using var manager2 = CreateManager();
        var state = await manager2.GetStateAsync("vulkan", "win-x64");

        state.Should().NotBeNull();
        state!.InstalledVersion.Should().Be("b7898");
    }

    [Fact]
    public async Task State_CorruptedFile_StartsFromFresh()
    {
        // Create a corrupted state file
        var stateFilePath = Path.Combine(_tempDir, "llama-server-state.json");
        await File.WriteAllTextAsync(stateFilePath, "{ corrupted json data !@#$");

        using var manager = CreateManager();
        var state = await manager.GetStateAsync("vulkan", "win-x64");

        state.Should().BeNull(); // Fresh state has no entries
    }

    // --- Multiple backends ---

    [Fact]
    public async Task MultipleBackends_IndependentState()
    {
        using var manager = CreateManager();

        await manager.CreateInitialStateAsync("vulkan", "win-x64", "b7898", "/vulkan");
        await manager.CreateInitialStateAsync("cuda12", "win-x64", "b7899", "/cuda");

        var vulkan = await manager.GetStateAsync("vulkan", "win-x64");
        var cuda = await manager.GetStateAsync("cuda12", "win-x64");

        vulkan!.InstalledVersion.Should().Be("b7898");
        cuda!.InstalledVersion.Should().Be("b7899");
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
