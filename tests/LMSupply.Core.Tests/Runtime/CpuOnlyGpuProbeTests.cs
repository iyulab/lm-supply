using AwesomeAssertions;
using LMSupply.Hardware;
using LMSupply.Pool;
using LMSupply.Runtime;
using Xunit;

namespace LMSupply.Core.Tests.Runtime;

[CollectionDefinition(nameof(GpuProbeSerialGroup), DisableParallelization = true)]
public sealed class GpuProbeSerialGroup;

/// <summary>
/// A load that asked for the CPU does not probe the GPU. The probe loads the vendor driver libraries (NVML, and the
/// CUDA driver with it) into the process; before this, every Local* entry point's pool, the runtime manager's
/// initialization and the load paths' hardware reads probed regardless of the provider. Run without parallel
/// neighbours so the probe count only moves for the code under test.
/// </summary>
[Collection(nameof(GpuProbeSerialGroup))]
public sealed class CpuOnlyGpuProbeTests
{
    [Fact]
    public async Task RuntimeManager_ProbesTheGpuOnFirstNeed_NotAtInitialization()
    {
        EnvironmentDetector.ClearCache();
        await using var manager = new RuntimeManager();
        var before = GpuDetector.ProbeCount;

        await manager.InitializeAsync(TestContext.Current.CancellationToken);

        GpuDetector.ProbeCount.Should().Be(before, "initialization detects the platform only");
        manager.IsGpuDetected.Should().BeFalse();

        // Positive control: the Auto choice is what the probe is for.
        _ = manager.RecommendedProvider;
        GpuDetector.ProbeCount.Should().Be(before + 1);
        manager.IsGpuDetected.Should().BeTrue();
    }

    [Fact]
    public void HardwareProfileForCpu_DoesNotProbe()
    {
        var before = GpuDetector.ProbeCount;

        var profile = HardwareProfile.For(ExecutionProvider.Cpu);

        GpuDetector.ProbeCount.Should().Be(before);
        profile.RecommendedProvider.Should().Be(ExecutionProvider.Cpu);
        profile.GpuInfo.Vendor.Should().Be(GpuVendor.Unknown);
        profile.SystemMemoryBytes.Should().BePositive();
    }

    [Fact]
    public async Task ModelPool_ReadsTheHardwareOnFirstUse_NotAtConstruction()
    {
        var before = GpuDetector.ProbeCount;

        await using var pool = new ModelPool<FakeModel, object>(new FakeLoader());

        GpuDetector.ProbeCount.Should().Be(before, "a pool is a static of every Local* entry point");
    }

    [Fact]
    public async Task ModelPool_WithAConfiguredBudget_NeverReadsTheHardware()
    {
        var before = GpuDetector.ProbeCount;
        await using var pool = new ModelPool<FakeModel, object>(new FakeLoader(), new ModelPoolOptions { MaxMemoryBytes = 1024 });

        pool.AvailableMemoryBytes.Should().Be(1024);
        GpuDetector.ProbeCount.Should().Be(before);
    }

    private sealed class FakeModel : IAsyncDisposable
    {
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class FakeLoader : IModelLoader<FakeModel, object>
    {
        public Task<FakeModel> LoadAsync(string modelId, object? options, CancellationToken cancellationToken)
            => Task.FromResult(new FakeModel());

        public long EstimateMemoryBytes(string modelId, object? options) => 0;
    }
}
