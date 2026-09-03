using AwesomeAssertions;
using LMSupply.Generator.Abstractions;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Tests for the ONNX backend registration seam (docket iyulab/lm-supply#164 — the ONNX Runtime
/// GenAI backend split into the optional LMSupply.Generator.Onnx package). This assembly disables
/// test parallelization (AssemblyInfo.cs), which is what makes mutating the process-global
/// <see cref="OnnxGeneratorBackendRegistry"/> in these tests safe.
/// </summary>
public sealed class OnnxGeneratorBackendRegistryTests
{
    [Fact]
    public void IsRegistered_Initially_IsFalse()
    {
        OnnxGeneratorBackendRegistry.ResetForTests();

        OnnxGeneratorBackendRegistry.IsRegistered.Should().BeFalse();
    }

    [Fact]
    public void Register_ThenIsRegistered_IsTrue()
    {
        OnnxGeneratorBackendRegistry.ResetForTests();
        try
        {
            OnnxGeneratorBackendRegistry.Register(new StubOnnxGeneratorBackend());

            OnnxGeneratorBackendRegistry.IsRegistered.Should().BeTrue();
        }
        finally
        {
            OnnxGeneratorBackendRegistry.ResetForTests();
        }
    }

    [Fact]
    public void Register_NullBackend_ThrowsArgumentNullException()
    {
        var action = () => OnnxGeneratorBackendRegistry.Register(null!);

        action.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public async Task BuildAsync_ExplicitOnnxRepo_WithoutBackendRegistered_ThrowsNamingThePackage()
    {
        OnnxGeneratorBackendRegistry.ResetForTests();

        var action = () => TextGeneratorBuilder.Create()
            .WithHuggingFaceModel("microsoft/Phi-4-mini-instruct-onnx")
            .BuildAsync();

        await action.Should().ThrowAsync<NotSupportedException>()
            .WithMessage("*LMSupply.Generator.Onnx*");
    }

    private sealed class StubOnnxGeneratorBackend : IOnnxGeneratorBackend
    {
        public IGeneratorModel CreateModel(string modelId, string modelPath, IChatFormatter chatFormatter, GeneratorOptions options, string? configBasePath = null)
            => throw new NotSupportedException("stub — not exercised by these tests");

        public IOnnxGeneratorModelFactory CreateFactory(string cacheDirectory, ExecutionProvider provider)
            => throw new NotSupportedException("stub — not exercised by these tests");

        public Task EnsureRuntimeAsync(ExecutionProvider provider, IProgress<LMSupply.DownloadProgress>? progress, CancellationToken cancellationToken)
            => throw new NotSupportedException("stub — not exercised by these tests");
    }
}
