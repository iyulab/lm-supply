using AwesomeAssertions;

namespace LMSupply.Generator.Onnx.Tests;

/// <summary>
/// Regression tests for OnnxGeneratorModelFactory path resolution,
/// including the 2-level nested variant layout used by Phi-4 Mini ONNX.
/// </summary>
public class OnnxGeneratorPathResolutionTests
{
    private static string CreateHfCacheLayout(
        string root,
        string org,
        string name,
        string? variantRelativePath,
        bool includeConfig = true,
        bool includeModelOnnx = true)
    {
        var snapshotDir = Path.Combine(root, $"models--{org}--{name}", "snapshots", "main");
        var modelDir = variantRelativePath is null
            ? snapshotDir
            : Path.Combine(snapshotDir, variantRelativePath);
        Directory.CreateDirectory(modelDir);
        if (includeConfig)
            File.WriteAllText(Path.Combine(modelDir, "genai_config.json"), "{}");
        if (includeModelOnnx)
            File.WriteAllText(Path.Combine(modelDir, "model.onnx"), "dummy");
        return snapshotDir;
    }

    [Fact]
    public void IsModelAvailable_OneLevelVariant_ReturnsTrue()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        try
        {
            CreateHfCacheLayout(tempDir, "acme", "test-model", "cpu-int4");
            using var factory = new OnnxGeneratorModelFactory(tempDir, ExecutionProvider.Cpu);

            factory.IsModelAvailable("acme/test-model").Should().BeTrue();
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void IsModelAvailable_TwoLevelNestedVariant_ReturnsTrue()
    {
        // Phi-4 Mini layout: cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        try
        {
            CreateHfCacheLayout(
                tempDir,
                "microsoft",
                "Phi-4-mini-instruct-onnx",
                Path.Combine("cpu_and_mobile", "cpu-int4-rtn-block-32-acc-level-4"));

            using var factory = new OnnxGeneratorModelFactory(tempDir, ExecutionProvider.Cpu);

            factory.IsModelAvailable("microsoft/Phi-4-mini-instruct-onnx").Should().BeTrue();
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void IsModelAvailable_NoValidLayout_ReturnsFalse()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        try
        {
            // Directory exists but no genai_config.json anywhere
            var snapshot = Path.Combine(tempDir, "models--acme--empty", "snapshots", "main", "something");
            Directory.CreateDirectory(snapshot);

            using var factory = new OnnxGeneratorModelFactory(tempDir, ExecutionProvider.Cpu);

            factory.IsModelAvailable("acme/empty").Should().BeFalse();
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void IsModelAvailable_SnapshotRoot_ReturnsTrue()
    {
        // Layout where genai_config.json sits at the snapshot root directly.
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        try
        {
            CreateHfCacheLayout(tempDir, "acme", "flat-model", variantRelativePath: null);

            using var factory = new OnnxGeneratorModelFactory(tempDir, ExecutionProvider.Cpu);

            factory.IsModelAvailable("acme/flat-model").Should().BeTrue();
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }
}
