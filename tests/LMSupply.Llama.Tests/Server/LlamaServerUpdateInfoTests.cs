using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

public class LlamaServerUpdateInfoTests
{
    // --- UpdateAvailable ---

    [Fact]
    public void UpdateAvailable_DifferentVersion_ReturnsTrue()
    {
        var info = CreateInfo(installed: "b1234", latest: "b5678");

        info.UpdateAvailable.Should().BeTrue();
    }

    [Fact]
    public void UpdateAvailable_SameVersion_ReturnsFalse()
    {
        var info = CreateInfo(installed: "b1234", latest: "b1234");

        info.UpdateAvailable.Should().BeFalse();
    }

    [Fact]
    public void UpdateAvailable_SameVersionDifferentCase_ReturnsFalse()
    {
        var info = CreateInfo(installed: "B1234", latest: "b1234");

        info.UpdateAvailable.Should().BeFalse();
    }

    [Fact]
    public void UpdateAvailable_NullLatest_ReturnsFalse()
    {
        var info = CreateInfo(installed: "b1234", latest: null);

        info.UpdateAvailable.Should().BeFalse();
    }

    // --- GetSummary ---

    [Fact]
    public void GetSummary_UpToDate_ContainsUpToDate()
    {
        var info = CreateInfo(installed: "b1234", latest: "b1234");

        info.GetSummary().Should().Contain("Up to date");
        info.GetSummary().Should().Contain("b1234");
    }

    [Fact]
    public void GetSummary_UpdateAvailable_ContainsLatestVersion()
    {
        var info = CreateInfo(installed: "b1234", latest: "b5678");

        info.GetSummary().Should().Contain("Update available");
        info.GetSummary().Should().Contain("b5678");
    }

    [Fact]
    public void GetSummary_UpdateReady_ContainsUpdateReady()
    {
        var info = CreateInfo(installed: "b1234", latest: "b5678", updateReady: true);

        info.GetSummary().Should().Contain("Update ready");
    }

    [Fact]
    public void GetSummary_ContainsBackend()
    {
        var info = CreateInfo(installed: "b1234", backend: LlamaServerBackend.Cuda12);

        info.GetSummary().Should().Contain("Cuda12");
    }

    // --- Record properties ---

    [Fact]
    public void Properties_SetCorrectly()
    {
        var now = DateTimeOffset.UtcNow;
        var info = new LlamaServerUpdateInfo
        {
            InstalledVersion = "b1234",
            LatestVersion = "b5678",
            UpdateReady = true,
            PendingVersion = "b5678",
            LastChecked = now,
            Backend = LlamaServerBackend.Vulkan,
            ServerPath = "/usr/bin/llama-server"
        };

        info.InstalledVersion.Should().Be("b1234");
        info.LatestVersion.Should().Be("b5678");
        info.UpdateReady.Should().BeTrue();
        info.PendingVersion.Should().Be("b5678");
        info.LastChecked.Should().Be(now);
        info.Backend.Should().Be(LlamaServerBackend.Vulkan);
        info.ServerPath.Should().Be("/usr/bin/llama-server");
    }

    private static LlamaServerUpdateInfo CreateInfo(
        string installed = "b1234",
        string? latest = null,
        bool updateReady = false,
        LlamaServerBackend backend = LlamaServerBackend.Cpu)
    {
        return new LlamaServerUpdateInfo
        {
            InstalledVersion = installed,
            LatestVersion = latest,
            UpdateReady = updateReady,
            Backend = backend,
            ServerPath = "/path/llama-server"
        };
    }
}

public class LlamaServerUpdateResultTests
{
    // --- Success property ---

    [Fact]
    public void Success_NoError_ReturnsTrue()
    {
        var result = new LlamaServerUpdateResult
        {
            ServerPath = "/path",
            Backend = LlamaServerBackend.Cpu,
            Error = null
        };

        result.Success.Should().BeTrue();
    }

    [Fact]
    public void Success_WithError_ReturnsFalse()
    {
        var result = new LlamaServerUpdateResult
        {
            ServerPath = "/path",
            Backend = LlamaServerBackend.Cpu,
            Error = "Something failed"
        };

        result.Success.Should().BeFalse();
    }

    // --- NoUpdate factory ---

    [Fact]
    public void NoUpdate_CreatesCorrectResult()
    {
        var result = LlamaServerUpdateResult.NoUpdate("/path/llama", LlamaServerBackend.Cuda12, "b1234");

        result.Updated.Should().BeFalse();
        result.ServerPath.Should().Be("/path/llama");
        result.Backend.Should().Be(LlamaServerBackend.Cuda12);
        result.PreviousVersion.Should().Be("b1234");
        result.NewVersion.Should().Be("b1234");
        result.Error.Should().BeNull();
        result.Success.Should().BeTrue();
    }

    // --- WithUpdate factory ---

    [Fact]
    public void WithUpdate_CreatesCorrectResult()
    {
        var result = LlamaServerUpdateResult.WithUpdate("/path/llama", LlamaServerBackend.Vulkan, "b1000", "b2000");

        result.Updated.Should().BeTrue();
        result.ServerPath.Should().Be("/path/llama");
        result.Backend.Should().Be(LlamaServerBackend.Vulkan);
        result.PreviousVersion.Should().Be("b1000");
        result.NewVersion.Should().Be("b2000");
        result.Error.Should().BeNull();
        result.Success.Should().BeTrue();
    }

    // --- Failed factory ---

    [Fact]
    public void Failed_CreatesCorrectResult()
    {
        var result = LlamaServerUpdateResult.Failed("/path/llama", LlamaServerBackend.Cpu, "Download failed");

        result.Updated.Should().BeFalse();
        result.ServerPath.Should().Be("/path/llama");
        result.Backend.Should().Be(LlamaServerBackend.Cpu);
        result.Error.Should().Be("Download failed");
        result.Success.Should().BeFalse();
        result.PreviousVersion.Should().BeNull();
        result.NewVersion.Should().BeNull();
    }

    // --- Default properties ---

    [Fact]
    public void DefaultProperties()
    {
        var result = new LlamaServerUpdateResult
        {
            ServerPath = "/path",
            Backend = LlamaServerBackend.Cpu
        };

        result.Updated.Should().BeFalse();
        result.PreviousVersion.Should().BeNull();
        result.NewVersion.Should().BeNull();
        result.Error.Should().BeNull();
        result.Success.Should().BeTrue();
    }
}

public class LlamaServerVersionStateModelTests
{
    // --- LlamaServerVersionState defaults ---

    [Fact]
    public void LlamaServerVersionState_RequiredAndDefaults()
    {
        var state = new LlamaServerVersionState
        {
            InstalledVersion = "b1000",
            InstalledPath = "/path/v1"
        };

        state.InstalledVersion.Should().Be("b1000");
        state.InstalledPath.Should().Be("/path/v1");
        state.LatestKnownVersion.Should().BeNull();
        state.PendingVersion.Should().BeNull();
        state.PendingPath.Should().BeNull();
        state.UpdateReady.Should().BeFalse();
        state.PreviousVersions.Should().NotBeNull();
        state.PreviousVersions.Should().BeEmpty();
        state.FailedVersions.Should().NotBeNull();
        state.FailedVersions.Should().BeEmpty();
        state.Backend.Should().Be("vulkan"); // Default
    }

    [Fact]
    public void LlamaServerVersionState_PreviousVersions_CanAdd()
    {
        var state = new LlamaServerVersionState
        {
            InstalledVersion = "b3000",
            InstalledPath = "/path/v3",
            PreviousVersions =
            [
                new VersionEntry { Version = "b1000", Path = "/old1" },
                new VersionEntry { Version = "b2000", Path = "/old2" }
            ]
        };

        state.PreviousVersions.Should().HaveCount(2);
        state.PreviousVersions[0].Version.Should().Be("b1000");
    }

    [Fact]
    public void LlamaServerVersionState_FailedVersions_TracksUnique()
    {
        var state = new LlamaServerVersionState
        {
            InstalledVersion = "b3000",
            InstalledPath = "/path/v3",
            FailedVersions = ["b1000", "b2000", "b1000"]
        };

        // HashSet deduplicates
        state.FailedVersions.Should().HaveCount(2);
    }

    // --- VersionEntry ---

    [Fact]
    public void VersionEntry_SetsProperties()
    {
        var now = DateTimeOffset.UtcNow;
        var entry = new VersionEntry
        {
            Version = "b3000",
            Path = "/versions/b3000",
            InstalledAt = now
        };

        entry.Version.Should().Be("b3000");
        entry.Path.Should().Be("/versions/b3000");
        entry.InstalledAt.Should().Be(now);
    }

    // --- LlamaServerStateFile ---

    [Fact]
    public void LlamaServerStateFile_Defaults()
    {
        var file = new LlamaServerStateFile();

        file.SchemaVersion.Should().Be(1);
        file.Entries.Should().NotBeNull();
        file.Entries.Should().BeEmpty();
    }

    [Fact]
    public void LlamaServerStateFile_MultipleEntries()
    {
        var file = new LlamaServerStateFile();
        file.Entries["cuda12|win-x64"] = new LlamaServerVersionState
        {
            InstalledVersion = "b3000",
            InstalledPath = "/path/cuda12"
        };
        file.Entries["cpu|linux-x64"] = new LlamaServerVersionState
        {
            InstalledVersion = "b2500",
            InstalledPath = "/path/cpu"
        };

        file.Entries.Should().HaveCount(2);
        file.Entries["cuda12|win-x64"].InstalledVersion.Should().Be("b3000");
    }
}
