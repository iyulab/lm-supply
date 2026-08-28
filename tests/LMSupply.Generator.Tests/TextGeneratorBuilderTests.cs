using System.Diagnostics;
using AwesomeAssertions;

namespace LMSupply.Generator.Tests;

[Collection("TraceListeners")]
public class TextGeneratorBuilderTests
{
    [Fact]
    public void Create_ReturnsNewBuilder()
    {
        // Act
        var builder = TextGeneratorBuilder.Create();

        // Assert
        builder.Should().NotBeNull();
    }

    [Fact]
    public void WithModelPath_SetsModelPath()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithModelPath("C:/models/phi3");
    }

    [Fact]
    public void WithHuggingFaceModel_SetsModelId()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithHuggingFaceModel("microsoft/Phi-3.5-mini-instruct-onnx");
    }

    [Fact]
    public void WithDefaultModel_UsesPhi35Mini()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithDefaultModel();
    }

    [Theory]
    [InlineData(GeneratorModelPreset.Default)]
    [InlineData(GeneratorModelPreset.Fast)]
    [InlineData(GeneratorModelPreset.Quality)]
    [InlineData(GeneratorModelPreset.Small)]
    public void WithModel_AcceptsAllPresets(GeneratorModelPreset preset)
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithModel(preset);
    }

    [Fact]
    public void FluentChaining_Works()
    {
        // Act
        var builder = TextGeneratorBuilder.Create()
            .WithDefaultModel()
            .WithProvider(ExecutionProvider.Cpu)
            .WithMaxContextLength(4096)
            .WithConcurrency(2)
            .WithVerboseLogging()
            .ForCreativeGeneration();

        // Assert
        builder.Should().NotBeNull();
    }

    [Fact]
    public async Task BuildAsync_WithoutModel_ThrowsInvalidOperationException()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert
        var action = () => builder.BuildAsync();
        await action.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*Model path or ID is required*");
    }

    [Fact]
    public void WithModelPath_NullPath_ThrowsArgumentNullException()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert
        var action = () => builder.WithModelPath(null!);
        action.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void WithHuggingFaceModel_NullId_ThrowsArgumentNullException()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert
        var action = () => builder.WithHuggingFaceModel(null!);
        action.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void WithMemoryManagement_NullOptions_ThrowsArgumentNullException()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert
        var action = () => builder.WithMemoryManagement(null!);
        action.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void WithCacheDirectory_SetsDirectory()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithCacheDirectory("C:/cache");
    }

    [Fact]
    public void WithChatFormat_SetsFormat()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithChatFormat("phi3");
    }

    [Fact]
    public void WithLlamaOptions_SetsOptions()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithLlamaOptions(new LlamaOptions { AdditionalArgs = ["--verbose"] });
    }

    [Fact]
    public void WithLlamaOptions_NullOptions_ThrowsArgumentNullException()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert
        var action = () => builder.WithLlamaOptions(null!);
        action.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void WithServerUpdateOptions_SetsOptions()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert - should not throw
        builder.WithServerUpdateOptions(new LMSupply.Llama.Server.LlamaServerUpdateOptions { PinnedVersion = "b7898" });
    }

    [Fact]
    public void WithServerUpdateOptions_NullOptions_ThrowsArgumentNullException()
    {
        // Arrange
        var builder = TextGeneratorBuilder.Create();

        // Act & Assert
        var action = () => builder.WithServerUpdateOptions(null!);
        action.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void BuildLocalGeneratorOptions_MirrorsServerUpdateOptions()
    {
        // Regression guard for the completeness-gap class this field belongs to: a prior field
        // (LlamaOptions.AdditionalArgs, 2026-08-09) was set via the builder but silently dropped by
        // this exact private mirror, because BuildLocalGeneratorOptions() is a hand-maintained field
        // list rather than a full copy. Reflection is used deliberately — the point is to catch a
        // missing line in that private method, which a "should not throw" test on the public
        // With*() method alone cannot see.
        var pinned = new LMSupply.Llama.Server.LlamaServerUpdateOptions { PinnedVersion = "b7898" };
        var builder = TextGeneratorBuilder.Create()
            .WithDefaultModel()
            .WithServerUpdateOptions(pinned);

        var method = typeof(TextGeneratorBuilder).GetMethod(
            "BuildLocalGeneratorOptions",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        method.Should().NotBeNull("BuildLocalGeneratorOptions must still exist under this name");

        var mirrored = (GeneratorOptions)method!.Invoke(builder, null)!;

        mirrored.ServerUpdateOptions.Should().BeSameAs(pinned,
            "the GGUF/LocalGenerator routing path must receive the same ServerUpdateOptions set via the builder");
    }

    // Note: Integration test for runtime loading is in the sample project
    // TextGeneratorBuilderSample validates that EnsureGenAiRuntimeAsync is called
    // before OnnxGeneratorModel creation, fixing the DllNotFoundException issue.

    private sealed class CapturingListener : TraceListener
    {
        private readonly List<string> _lines = new();
        public IReadOnlyList<string> Lines => _lines;

        public override void Write(string? message) { if (message != null) _lines.Add(message); }
        public override void WriteLine(string? message) { if (message != null) _lines.Add(message); }
    }

    private static async Task<(IReadOnlyList<string> Lines, Exception? Error)> CaptureBuildAsync(
        Func<TextGeneratorBuilder> configure)
    {
        var listener = new CapturingListener();
        Trace.Listeners.Add(listener);
        Exception? error = null;
        try
        {
            // Pre-canceled token: the auto-selection trace fires synchronously inside
            // LoadAutoAsync before the cancellation check inside the downstream loader,
            // so the trace line is observable deterministically without downloading
            // any weights or hitting the network.
            using var cts = new CancellationTokenSource();
            cts.Cancel();
            try
            {
                await configure().BuildAsync(cts.Token);
            }
            catch (Exception ex)
            {
                error = ex;
            }
        }
        finally
        {
            Trace.Listeners.Remove(listener);
        }
        return (listener.Lines, error);
    }

    [Fact]
    public async Task BuildAsync_WithDefaultModel_RoutesThroughLocalGeneratorAuto()
    {
        // Regression test for ISSUE-LMSupply.Generator-20260429-000657:
        // WithDefaultModel() previously stamped _modelId = "default" and BuildAsync
        // routed it through OnnxGeneratorModelFactory, which 401'd against
        // huggingface.co/default. The fix delegates to LocalGenerator.LoadAsync
        // for hardware-aware GGUF/ONNX dispatch — observable via the
        // "[LocalGenerator.auto]" trace line emitted by LoadAutoAsync.
        var (lines, error) = await CaptureBuildAsync(
            () => TextGeneratorBuilder.Create().WithDefaultModel());

        lines.Should().Contain(l => l.Contains("[LocalGenerator.auto]"),
            because: "WithDefaultModel must dispatch through LocalGenerator.LoadAutoAsync, " +
                     "not the ONNX factory which would try to download a HuggingFace repo named 'default'");

        // The error message (if any) must NOT mention the broken HF "default" repo path.
        if (error is not null)
        {
            error.Message.Should().NotContain("from 'default'",
                because: "the broken path was downloading 'genai_config.json' from 'default'");
            error.Message.Should().NotContain("huggingface.co/default",
                because: "the broken path was 401'ing against huggingface.co/default");
        }
    }

    [Theory]
    [InlineData(GeneratorModelPreset.Default)]
    public async Task BuildAsync_WithModelDefaultPreset_RoutesThroughLocalGeneratorAuto(GeneratorModelPreset preset)
    {
        // The Default preset shares the underlying alias with WithDefaultModel(),
        // so it must follow the same routing.
        var (lines, _) = await CaptureBuildAsync(
            () => TextGeneratorBuilder.Create().WithModel(preset));

        lines.Should().Contain(l => l.Contains("[LocalGenerator.auto]"));
    }

    [Theory]
    [InlineData("gguf:gemma4-fast")]
    [InlineData("gguf:gemma4-default")]
    public async Task BuildAsync_WithGgufAlias_RoutesThroughLocalGenerator(string alias)
    {
        // GGUF aliases must also bypass the ONNX factory. The negative signal is that
        // the failure path must not contain a 401 against a literal "gguf" / alias-stripped
        // HuggingFace repo, since LocalGenerator routes them to the GGUF loader.
        var (_, error) = await CaptureBuildAsync(
            () => TextGeneratorBuilder.Create().WithHuggingFaceModel(alias));

        if (error is not null)
        {
            error.Message.Should().NotContain("from 'gguf'",
                because: "the alias must not be split into a literal 'gguf' repo download");
        }
    }

    [Fact]
    public async Task BuildAsync_WithExplicitOnnxRepo_DoesNotRouteThroughLocalGeneratorAuto()
    {
        // Negative regression: an explicit ONNX HuggingFace repo id must continue to
        // flow through the ONNX factory path, not the LocalGenerator auto trace.
        var (lines, _) = await CaptureBuildAsync(() =>
            TextGeneratorBuilder.Create()
                .WithHuggingFaceModel("microsoft/Phi-4-mini-instruct-onnx"));

        lines.Should().NotContain(l => l.Contains("[LocalGenerator.auto]"),
            because: "an explicit ONNX repo id must still go through OnnxGeneratorModelFactory, " +
                     "not LocalGenerator.LoadAutoAsync");
    }
}
