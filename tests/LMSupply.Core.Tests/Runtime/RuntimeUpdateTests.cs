using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// Tests for the runtime auto-update system.
/// </summary>
[Trait("Category", "Unit")]
public class RuntimeUpdateTests : IDisposable
{
    private readonly string _testCacheDir;

    public RuntimeUpdateTests()
    {
        _testCacheDir = Path.Combine(Path.GetTempPath(), $"lmsupply-update-test-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testCacheDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_testCacheDir))
            {
                Directory.Delete(_testCacheDir, recursive: true);
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
        GC.SuppressFinalize(this);
    }

    #region RuntimeVersionState Tests

    [Fact]
    public void RuntimeVersionState_UpdateAvailable_WhenVersionsDiffer()
    {
        // Arrange
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "1.1.0"
        };

        // Assert
        state.UpdateAvailable.Should().BeTrue();
    }

    [Fact]
    public void RuntimeVersionState_UpdateAvailable_FalseWhenSameVersion()
    {
        // Arrange
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "1.0.0"
        };

        // Assert
        state.UpdateAvailable.Should().BeFalse();
    }

    [Fact]
    public void RuntimeVersionState_UpdateAvailable_FalseWhenVersionFailed()
    {
        // Arrange
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = "1.1.0",
            FailedVersions = ["1.1.0"]
        };

        // Assert
        state.UpdateAvailable.Should().BeFalse("failed versions should be skipped");
    }

    [Fact]
    public void RuntimeVersionState_UpdateAvailable_FalseWhenNoLatestVersion()
    {
        // Arrange
        var state = new RuntimeVersionState
        {
            InstalledVersion = "1.0.0",
            LatestKnownVersion = null
        };

        // Assert
        state.UpdateAvailable.Should().BeFalse();
    }

    #endregion

    #region RuntimeUpdateOptions Tests

    [Fact]
    public void RuntimeUpdateOptions_Default_HasReasonableDefaults()
    {
        // Arrange
        var options = RuntimeUpdateOptions.Default;

        // Assert
        options.VersionCheckInterval.Should().Be(TimeSpan.FromHours(24));
        options.AutoDownloadUpdates.Should().BeTrue();
        options.UpdateOnWarmup.Should().BeTrue();
        options.IncludePrerelease.Should().BeFalse();
        options.MaxVersionsToKeep.Should().Be(2);
        options.VersionCheckTimeout.Should().Be(TimeSpan.FromSeconds(30));
    }

    [Fact]
    public void RuntimeUpdateOptions_CanBeCustomized()
    {
        // Arrange
        var options = new RuntimeUpdateOptions
        {
            VersionCheckInterval = TimeSpan.FromHours(1),
            AutoDownloadUpdates = false,
            UpdateOnWarmup = false,
            IncludePrerelease = true,
            MaxVersionsToKeep = 5
        };

        // Assert
        options.VersionCheckInterval.Should().Be(TimeSpan.FromHours(1));
        options.AutoDownloadUpdates.Should().BeFalse();
        options.UpdateOnWarmup.Should().BeFalse();
        options.IncludePrerelease.Should().BeTrue();
        options.MaxVersionsToKeep.Should().Be(5);
    }

    #endregion

    #region RuntimeUpdateInfo Tests

    [Fact]
    public void RuntimeUpdateInfo_UpdateAvailable_WhenVersionsDiffer()
    {
        // Arrange
        var info = new RuntimeUpdateInfo
        {
            InstalledVersion = "1.0.0",
            LatestVersion = "1.1.0"
        };

        // Assert
        info.UpdateAvailable.Should().BeTrue();
    }

    [Fact]
    public void RuntimeUpdateInfo_UpdateAvailable_FalseWhenSameVersion()
    {
        // Arrange
        var info = new RuntimeUpdateInfo
        {
            InstalledVersion = "1.0.0",
            LatestVersion = "1.0.0"
        };

        // Assert
        info.UpdateAvailable.Should().BeFalse();
    }

    #endregion

    #region RuntimeUpdateResult Tests

    [Fact]
    public void RuntimeUpdateResult_NoUpdateNeeded_CreatesCorrectResult()
    {
        // Arrange & Act
        var result = RuntimeUpdateResult.NoUpdateNeeded("1.0.0", "/path/to/runtime");

        // Assert
        result.Updated.Should().BeFalse();
        result.NewVersion.Should().Be("1.0.0");
        result.RuntimePath.Should().Be("/path/to/runtime");
        result.WasRollback.Should().BeFalse();
    }

    [Fact]
    public void RuntimeUpdateResult_UpdateApplied_CreatesCorrectResult()
    {
        // Arrange & Act
        var result = RuntimeUpdateResult.UpdateApplied("1.0.0", "1.1.0", "/path/to/runtime");

        // Assert
        result.Updated.Should().BeTrue();
        result.PreviousVersion.Should().Be("1.0.0");
        result.NewVersion.Should().Be("1.1.0");
        result.RuntimePath.Should().Be("/path/to/runtime");
        result.WasRollback.Should().BeFalse();
    }

    [Fact]
    public void RuntimeUpdateResult_Rollback_CreatesCorrectResult()
    {
        // Arrange & Act
        var result = RuntimeUpdateResult.Rollback("1.1.0", "1.0.0", "/path/to/runtime");

        // Assert
        result.Updated.Should().BeTrue();
        result.WasRollback.Should().BeTrue();
        result.PreviousVersion.Should().Be("1.1.0");
        result.NewVersion.Should().Be("1.0.0");
        result.RuntimePath.Should().Be("/path/to/runtime");
    }

    [Fact]
    public void RuntimeUpdateResult_Failed_CreatesCorrectResult()
    {
        // Arrange & Act
        var result = RuntimeUpdateResult.Failed("Download failed");

        // Assert
        result.Updated.Should().BeFalse();
        result.ErrorMessage.Should().Be("Download failed");
    }

    #endregion

    #region RuntimeVersionStateManager Tests

    [Fact]
    public async Task RuntimeVersionStateManager_GetPackageKey_FormatsCorrectly()
    {
        // Arrange & Act
        var key = RuntimeVersionStateManager.GetPackageKey("llama-server", "vulkan", "win-x64");

        // Assert
        key.Should().Be("llama-server|vulkan|win-x64");
    }

    [Fact]
    public async Task RuntimeVersionStateManager_GetOrCreateState_CreatesNewState()
    {
        // Arrange
        using var manager = new RuntimeVersionStateManager(_testCacheDir);

        // Act
        var state = await manager.GetOrCreateStateAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);

        // Assert
        state.InstalledVersion.Should().Be("1.0.0");
        state.LastVersionCheck.Should().Be(DateTimeOffset.MinValue);
    }

    [Fact]
    public async Task RuntimeVersionStateManager_GetOrCreateState_ReturnsExistingState()
    {
        // Arrange
        using var manager = new RuntimeVersionStateManager(_testCacheDir);
        var initialState = await manager.GetOrCreateStateAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);

        // Act - try to create again with different version
        var state = await manager.GetOrCreateStateAsync("test|cpu|win-x64", "2.0.0", TestContext.Current.CancellationToken);

        // Assert - should return the original state
        state.InstalledVersion.Should().Be("1.0.0");
    }

    [Fact]
    public async Task RuntimeVersionStateManager_RecordVersionCheck_UpdatesState()
    {
        // Arrange
        using var manager = new RuntimeVersionStateManager(_testCacheDir);
        await manager.GetOrCreateStateAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);

        // Act
        await manager.RecordVersionCheckAsync("test|cpu|win-x64", "1.1.0", TestContext.Current.CancellationToken);
        var state = await manager.GetStateAsync("test|cpu|win-x64", TestContext.Current.CancellationToken);

        // Assert
        state.Should().NotBeNull();
        state!.LatestKnownVersion.Should().Be("1.1.0");
        state.LastVersionCheck.Should().BeCloseTo(DateTimeOffset.UtcNow, TimeSpan.FromSeconds(5));
    }

    [Fact]
    public async Task RuntimeVersionStateManager_MarkUpdateReady_SetsFlags()
    {
        // Arrange
        using var manager = new RuntimeVersionStateManager(_testCacheDir);
        await manager.GetOrCreateStateAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);

        // Act
        await manager.MarkUpdateReadyAsync("test|cpu|win-x64", "1.1.0", "/path/to/update", TestContext.Current.CancellationToken);
        var state = await manager.GetStateAsync("test|cpu|win-x64", TestContext.Current.CancellationToken);

        // Assert
        state.Should().NotBeNull();
        state!.UpdateReady.Should().BeTrue();
        state.UpdateReadyPath.Should().Be("/path/to/update");
        state.LatestKnownVersion.Should().Be("1.1.0");
    }

    [Fact]
    public async Task RuntimeVersionStateManager_ActivateUpdate_MovesPreviousVersion()
    {
        // Arrange
        using var manager = new RuntimeVersionStateManager(_testCacheDir);
        await manager.GetOrCreateStateAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);
        await manager.MarkUpdateReadyAsync("test|cpu|win-x64", "1.1.0", "/path/to/update", TestContext.Current.CancellationToken);

        // Act
        var state = await manager.ActivateUpdateAsync("test|cpu|win-x64", maxVersionsToKeep: 2, ct: TestContext.Current.CancellationToken);

        // Assert
        state.Should().NotBeNull();
        state!.InstalledVersion.Should().Be("1.1.0");
        state.UpdateReady.Should().BeFalse();
        state.PreviousVersions.Should().Contain("1.0.0");
    }

    [Fact]
    public async Task RuntimeVersionStateManager_ActivateUpdate_TrimsOldVersions()
    {
        // Arrange
        using var manager = new RuntimeVersionStateManager(_testCacheDir);
        await manager.GetOrCreateStateAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);

        // Simulate multiple updates
        await manager.MarkUpdateReadyAsync("test|cpu|win-x64", "1.1.0", "/path/1.1.0", TestContext.Current.CancellationToken);
        await manager.ActivateUpdateAsync("test|cpu|win-x64", maxVersionsToKeep: 2, ct: TestContext.Current.CancellationToken);

        await manager.MarkUpdateReadyAsync("test|cpu|win-x64", "1.2.0", "/path/1.2.0", TestContext.Current.CancellationToken);
        await manager.ActivateUpdateAsync("test|cpu|win-x64", maxVersionsToKeep: 2, ct: TestContext.Current.CancellationToken);

        await manager.MarkUpdateReadyAsync("test|cpu|win-x64", "1.3.0", "/path/1.3.0", TestContext.Current.CancellationToken);
        var state = await manager.ActivateUpdateAsync("test|cpu|win-x64", maxVersionsToKeep: 2, ct: TestContext.Current.CancellationToken);

        // Assert - only 2 previous versions should be kept
        state.Should().NotBeNull();
        state!.InstalledVersion.Should().Be("1.3.0");
        state.PreviousVersions.Should().HaveCount(2);
        state.PreviousVersions.Should().Contain("1.2.0");
        state.PreviousVersions.Should().Contain("1.1.0");
        state.PreviousVersions.Should().NotContain("1.0.0");
    }

    [Fact]
    public async Task RuntimeVersionStateManager_Rollback_RevertsToPreviousVersion()
    {
        // Arrange
        using var manager = new RuntimeVersionStateManager(_testCacheDir);
        await manager.GetOrCreateStateAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);
        await manager.MarkUpdateReadyAsync("test|cpu|win-x64", "1.1.0", "/path/1.1.0", TestContext.Current.CancellationToken);
        await manager.ActivateUpdateAsync("test|cpu|win-x64", maxVersionsToKeep: 2, ct: TestContext.Current.CancellationToken);

        // Act
        var (previousVersion, _) = await manager.RollbackAsync("test|cpu|win-x64", "1.1.0", TestContext.Current.CancellationToken);
        var state = await manager.GetStateAsync("test|cpu|win-x64", TestContext.Current.CancellationToken);

        // Assert
        previousVersion.Should().Be("1.0.0");
        state.Should().NotBeNull();
        state!.InstalledVersion.Should().Be("1.0.0");
        state.FailedVersions.Should().Contain("1.1.0");
    }

    [Fact]
    public async Task RuntimeVersionStateManager_IsVersionCheckDue_ReturnsTrueWhenExpired()
    {
        // Arrange
        using var manager = new RuntimeVersionStateManager(_testCacheDir);
        await manager.GetOrCreateStateAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);

        // Act - check with 1 hour interval (state was just created with MinValue timestamp)
        var isDue = await manager.IsVersionCheckDueAsync("test|cpu|win-x64", TimeSpan.FromHours(1), TestContext.Current.CancellationToken);

        // Assert
        isDue.Should().BeTrue("LastVersionCheck was MinValue, so it should be due");
    }

    [Fact]
    public async Task RuntimeVersionStateManager_IsVersionCheckDue_ReturnsFalseWhenRecent()
    {
        // Arrange
        using var manager = new RuntimeVersionStateManager(_testCacheDir);
        await manager.GetOrCreateStateAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);
        await manager.RecordVersionCheckAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);

        // Act - check with 1 hour interval (just recorded check)
        var isDue = await manager.IsVersionCheckDueAsync("test|cpu|win-x64", TimeSpan.FromHours(1), TestContext.Current.CancellationToken);

        // Assert
        isDue.Should().BeFalse("LastVersionCheck was just updated");
    }

    [Fact]
    public async Task RuntimeVersionStateManager_PersistsState()
    {
        // Arrange - create state with first manager
        var stateFile = Path.Combine(_testCacheDir, "runtime-versions.json");

        using (var manager1 = new RuntimeVersionStateManager(_testCacheDir))
        {
            await manager1.GetOrCreateStateAsync("test|cpu|win-x64", "1.0.0", TestContext.Current.CancellationToken);
            await manager1.RecordVersionCheckAsync("test|cpu|win-x64", "1.1.0", TestContext.Current.CancellationToken);
        }

        // Act - load with second manager
        using var manager2 = new RuntimeVersionStateManager(_testCacheDir);
        var state = await manager2.GetStateAsync("test|cpu|win-x64", TestContext.Current.CancellationToken);

        // Assert
        File.Exists(stateFile).Should().BeTrue();
        state.Should().NotBeNull();
        state!.InstalledVersion.Should().Be("1.0.0");
        state.LatestKnownVersion.Should().Be("1.1.0");
    }

    #endregion
}
