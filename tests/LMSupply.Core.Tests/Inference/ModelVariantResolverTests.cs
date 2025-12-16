using FluentAssertions;
using LMSupply.Inference;

namespace LMSupply.Core.Tests.Inference;

public class ModelVariantResolverTests : IDisposable
{
    private readonly string _testDirectory;

    public ModelVariantResolverTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"ModelVariantTest_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDirectory);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDirectory))
        {
            Directory.Delete(_testDirectory, recursive: true);
        }
    }

    #region ResolveModelPath Tests

    [Fact]
    public void ResolveModelPath_NonExistentDirectory_ReturnsNull()
    {
        var result = ModelVariantResolver.ResolveModelPath(
            Path.Combine(_testDirectory, "nonexistent"),
            ModelPrecision.FP16);

        result.Should().BeNull();
    }

    [Fact]
    public void ResolveModelPath_EmptyDirectory_ReturnsNull()
    {
        var result = ModelVariantResolver.ResolveModelPath(_testDirectory, ModelPrecision.FP16);

        result.Should().BeNull();
    }

    [Fact]
    public void ResolveModelPath_ExactMatch_ReturnsPath()
    {
        var fp16Path = CreateModelFile("model_fp16.onnx");

        var result = ModelVariantResolver.ResolveModelPath(_testDirectory, ModelPrecision.FP16);

        result.Should().Be(fp16Path);
    }

    [Fact]
    public void ResolveModelPath_FallbackToOtherPrecision_WhenExactNotFound()
    {
        var fp32Path = CreateModelFile("model_fp32.onnx");

        var result = ModelVariantResolver.ResolveModelPath(_testDirectory, ModelPrecision.FP16);

        result.Should().Be(fp32Path);
    }

    [Fact]
    public void ResolveModelPath_Auto_PrioritizesFP16()
    {
        CreateModelFile("model_fp32.onnx");
        var fp16Path = CreateModelFile("model_fp16.onnx");
        CreateModelFile("model_int8.onnx");

        var result = ModelVariantResolver.ResolveModelPath(_testDirectory, ModelPrecision.Auto);

        result.Should().Be(fp16Path);
    }

    [Fact]
    public void ResolveModelPath_Auto_FallsBackToINT8WhenNoFP16()
    {
        CreateModelFile("model_fp32.onnx");
        var int8Path = CreateModelFile("model_int8.onnx");

        var result = ModelVariantResolver.ResolveModelPath(_testDirectory, ModelPrecision.Auto);

        result.Should().Be(int8Path);
    }

    [Fact]
    public void ResolveModelPath_FallbackToDefault_WhenNoPrecisionSpecificFound()
    {
        var defaultPath = CreateModelFile("model.onnx");

        var result = ModelVariantResolver.ResolveModelPath(
            _testDirectory,
            ModelPrecision.FP16,
            fallbackToDefault: true);

        result.Should().Be(defaultPath);
    }

    [Fact]
    public void ResolveModelPath_NoFallbackToDefault_WhenDisabled()
    {
        CreateModelFile("model.onnx");

        var result = ModelVariantResolver.ResolveModelPath(
            _testDirectory,
            ModelPrecision.FP16,
            fallbackToDefault: false);

        result.Should().BeNull();
    }

    [Fact]
    public void ResolveModelPath_OnnxSubdirectory_FindsModels()
    {
        var onnxDir = Path.Combine(_testDirectory, "onnx");
        Directory.CreateDirectory(onnxDir);
        var fp16Path = CreateModelFile(Path.Combine("onnx", "model_fp16.onnx"));

        var result = ModelVariantResolver.ResolveModelPath(_testDirectory, ModelPrecision.FP16);

        result.Should().Be(fp16Path);
    }

    [Fact]
    public void ResolveModelPath_OnnxSubdirectory_FallbackToDefaultModel()
    {
        var onnxDir = Path.Combine(_testDirectory, "onnx");
        Directory.CreateDirectory(onnxDir);
        var defaultPath = CreateModelFile(Path.Combine("onnx", "model.onnx"));

        var result = ModelVariantResolver.ResolveModelPath(
            _testDirectory,
            ModelPrecision.FP16,
            fallbackToDefault: true);

        result.Should().Be(defaultPath);
    }

    #endregion

    #region DiscoverModelVariants Tests

    [Fact]
    public void DiscoverModelVariants_EmptyDirectory_ReturnsEmpty()
    {
        var variants = ModelVariantResolver.DiscoverModelVariants(_testDirectory);

        variants.Should().BeEmpty();
    }

    [Fact]
    public void DiscoverModelVariants_FindsAllVariants()
    {
        var fp32Path = CreateModelFile("model_fp32.onnx");
        var fp16Path = CreateModelFile("model_fp16.onnx");
        var int8Path = CreateModelFile("model_int8.onnx");
        var int4Path = CreateModelFile("model_int4.onnx");

        var variants = ModelVariantResolver.DiscoverModelVariants(_testDirectory);

        variants.Should().HaveCount(4);
        variants[ModelPrecision.FP32].Should().Be(fp32Path);
        variants[ModelPrecision.FP16].Should().Be(fp16Path);
        variants[ModelPrecision.INT8].Should().Be(int8Path);
        variants[ModelPrecision.INT4].Should().Be(int4Path);
    }

    [Fact]
    public void DiscoverModelVariants_QuantizedModel_MapsToINT8()
    {
        var quantizedPath = CreateModelFile("model_quantized.onnx");

        var variants = ModelVariantResolver.DiscoverModelVariants(_testDirectory);

        variants.Should().ContainKey(ModelPrecision.INT8);
        variants[ModelPrecision.INT8].Should().Be(quantizedPath);
    }

    [Fact]
    public void DiscoverModelVariants_DefaultModel_InfersPrecisionFromName()
    {
        var defaultPath = CreateModelFile("model.onnx");

        var variants = ModelVariantResolver.DiscoverModelVariants(_testDirectory);

        variants.Should().HaveCount(1);
        variants.Should().ContainKey(ModelPrecision.FP32); // default infers to FP32
        variants[ModelPrecision.FP32].Should().Be(defaultPath);
    }

    [Fact]
    public void DiscoverModelVariants_OnnxSubdirectory_FindsVariants()
    {
        var onnxDir = Path.Combine(_testDirectory, "onnx");
        Directory.CreateDirectory(onnxDir);
        var fp16Path = CreateModelFile(Path.Combine("onnx", "model_fp16.onnx"));
        var int8Path = CreateModelFile(Path.Combine("onnx", "model_int8.onnx"));

        var variants = ModelVariantResolver.DiscoverModelVariants(_testDirectory);

        variants.Should().HaveCount(2);
        variants[ModelPrecision.FP16].Should().Be(fp16Path);
        variants[ModelPrecision.INT8].Should().Be(int8Path);
    }

    [Fact]
    public void DiscoverModelVariants_MainDirectoryTakesPrecedence()
    {
        var mainFp16 = CreateModelFile("model_fp16.onnx");

        var onnxDir = Path.Combine(_testDirectory, "onnx");
        Directory.CreateDirectory(onnxDir);
        CreateModelFile(Path.Combine("onnx", "model_fp16.onnx"));

        var variants = ModelVariantResolver.DiscoverModelVariants(_testDirectory);

        variants[ModelPrecision.FP16].Should().Be(mainFp16);
    }

    #endregion

    #region GetVariantInfo Tests

    [Fact]
    public void GetVariantInfo_ReturnsCompleteInfo()
    {
        CreateModelFile("model_fp32.onnx");
        CreateModelFile("model_fp16.onnx");
        CreateModelFile("model_int8.onnx");

        var info = ModelVariantResolver.GetVariantInfo(_testDirectory);

        info.ModelDirectory.Should().Be(_testDirectory);
        info.VariantCount.Should().Be(3);
        info.HasFP32.Should().BeTrue();
        info.HasFP16.Should().BeTrue();
        info.HasINT8.Should().BeTrue();
        info.HasINT4.Should().BeFalse();
        info.RecommendedPrecision.Should().Be(ModelPrecision.FP16); // FP16 has priority
    }

    [Fact]
    public void GetVariantInfo_EmptyDirectory_ReturnsEmptyInfo()
    {
        var info = ModelVariantResolver.GetVariantInfo(_testDirectory);

        info.VariantCount.Should().Be(0);
        info.HasFP32.Should().BeFalse();
        info.HasFP16.Should().BeFalse();
        info.HasINT8.Should().BeFalse();
        info.HasINT4.Should().BeFalse();
    }

    [Fact]
    public void GetVariantInfo_OnlyINT8_RecommendsINT8()
    {
        CreateModelFile("model_int8.onnx");

        var info = ModelVariantResolver.GetVariantInfo(_testDirectory);

        info.RecommendedPrecision.Should().Be(ModelPrecision.INT8);
    }

    [Fact]
    public void GetVariantInfo_ToString_FormatsCorrectly()
    {
        CreateModelFile("model_fp16.onnx");
        CreateModelFile("model_int8.onnx");

        var info = ModelVariantResolver.GetVariantInfo(_testDirectory);
        var toString = info.ToString();

        toString.Should().Contain("ModelVariants(2)");
        toString.Should().Contain("FP16");
        toString.Should().Contain("INT8");
        toString.Should().Contain("Recommended: FP16");
    }

    #endregion

    #region Fallback Order Tests

    [Fact]
    public void ResolveModelPath_FP32Requested_FallbackOrder()
    {
        // FP32 fallback: FP16 → INT8 → INT4
        var int8Path = CreateModelFile("model_int8.onnx");

        var result = ModelVariantResolver.ResolveModelPath(_testDirectory, ModelPrecision.FP32);

        result.Should().Be(int8Path); // Falls back through FP16 (not found) to INT8
    }

    [Fact]
    public void ResolveModelPath_INT4Requested_FallbackOrder()
    {
        // INT4 fallback: INT8 → FP16 → FP32
        var fp32Path = CreateModelFile("model_fp32.onnx");

        var result = ModelVariantResolver.ResolveModelPath(_testDirectory, ModelPrecision.INT4);

        result.Should().Be(fp32Path); // Falls back through INT8, FP16 (not found) to FP32
    }

    [Fact]
    public void ResolveModelPath_INT8Requested_PrefersFP16Fallback()
    {
        // INT8 fallback: FP16 → FP32 → INT4
        var fp16Path = CreateModelFile("model_fp16.onnx");
        CreateModelFile("model_fp32.onnx");

        var result = ModelVariantResolver.ResolveModelPath(_testDirectory, ModelPrecision.INT8);

        result.Should().Be(fp16Path);
    }

    #endregion

    #region Helper Methods

    private string CreateModelFile(string relativePath)
    {
        var fullPath = Path.Combine(_testDirectory, relativePath);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        File.WriteAllBytes(fullPath, [0x4F, 0x4E, 0x4E, 0x58]); // "ONNX" magic bytes
        return fullPath;
    }

    #endregion
}
