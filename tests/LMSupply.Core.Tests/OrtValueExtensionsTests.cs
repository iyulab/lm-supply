using System.Runtime.InteropServices;
using FluentAssertions;
using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Core.Tests;

/// <summary>
/// Tests for OrtValue extensions.
/// Note: Tests that require ONNX Runtime native binaries are marked with [Trait("Category", "Integration")]
/// and will be skipped if native libraries are not available.
/// </summary>
[Collection("OrtValue")]
public class OrtValueExtensionsTests
{
    /// <summary>
    /// Checks if ONNX Runtime native library is likely available by checking for DLL existence.
    /// This avoids calling ORT code which can cause fatal crashes on missing native libs.
    /// </summary>
    private static bool IsOrtAvailable()
    {
        // Check common locations for ONNX Runtime native library
        var assemblyLocation = typeof(OrtValue).Assembly.Location;
        var assemblyDir = Path.GetDirectoryName(assemblyLocation) ?? ".";

        // Check for the native library in the assembly directory
        var dllName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "onnxruntime.dll"
            : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                ? "libonnxruntime.dylib"
                : "libonnxruntime.so";

        var nativePath = Path.Combine(assemblyDir, dllName);
        if (File.Exists(nativePath)) return true;

        // Check in runtimes folder structure
        var rid = RuntimeInformation.RuntimeIdentifier;
        var runtimesPath = Path.Combine(assemblyDir, "runtimes", rid, "native", dllName);
        if (File.Exists(runtimesPath)) return true;

        // Check parent directories (for test execution from bin folder)
        var parentDir = Path.GetDirectoryName(assemblyDir);
        if (parentDir != null)
        {
            var parentNativePath = Path.Combine(parentDir, dllName);
            if (File.Exists(parentNativePath)) return true;
        }

        return false;
    }

    #region Float Array Tests - Null Validation (No ORT required)

    [Fact]
    public void CreateTensorFromArray_Float_NullData_ShouldThrow()
    {
        var act = () => OrtValueExtensions.CreateTensorFromArray((float[])null!, [1, 2]);

        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void CreateTensorFromArray_Float_NullShape_ShouldThrow()
    {
        var act = () => OrtValueExtensions.CreateTensorFromArray(new float[] { 1, 2 }, null!);

        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void CreateTensorFromArray_Long_NullData_ShouldThrow()
    {
        var act = () => OrtValueExtensions.CreateTensorFromArray((long[])null!, [1, 2]);

        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void CreateTensorFromArray_Long_NullShape_ShouldThrow()
    {
        var act = () => OrtValueExtensions.CreateTensorFromArray(new long[] { 1, 2 }, null!);

        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void CreateTensorFromArray_Int_NullData_ShouldThrow()
    {
        var act = () => OrtValueExtensions.CreateTensorFromArray((int[])null!, [1, 2]);

        act.Should().Throw<ArgumentNullException>();
    }

    #endregion

    #region Float Array Tests - ORT Required

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTensorFromArray_Float_ShouldCreateValidTensor()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var data = new float[] { 1, 2, 3, 4, 5, 6 };
        var shape = new long[] { 2, 3 };

        using var tensor = OrtValueExtensions.CreateTensorFromArray(data, shape);

        tensor.Should().NotBeNull();
        tensor.GetShape().Should().Equal([2, 3]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTensorFromArray_Float_DataShouldBeAccessible()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var data = new float[] { 1.5f, 2.5f, 3.5f };
        var shape = new long[] { 1, 3 };

        using var tensor = OrtValueExtensions.CreateTensorFromArray(data, shape);
        var result = tensor.ToFloatArray();

        result.Should().Equal([1.5f, 2.5f, 3.5f]);
    }

    #endregion

    #region Long Array Tests - ORT Required

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTensorFromArray_Long_ShouldCreateValidTensor()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var data = new long[] { 1, 2, 3, 4 };
        var shape = new long[] { 2, 2 };

        using var tensor = OrtValueExtensions.CreateTensorFromArray(data, shape);

        tensor.Should().NotBeNull();
        tensor.GetShape().Should().Equal([2, 2]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTensorFromArray_Long_DataShouldBeAccessible()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var data = new long[] { 100, 200, 300 };
        var shape = new long[] { 3 };

        using var tensor = OrtValueExtensions.CreateTensorFromArray(data, shape);
        var result = tensor.ToLongArray();

        result.Should().Equal([100, 200, 300]);
    }

    #endregion

    #region Int Array Tests - ORT Required

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTensorFromArray_Int_ShouldConvertToLong()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var data = new int[] { 1, 2, 3 };
        var shape = new long[] { 1, 3 };

        using var tensor = OrtValueExtensions.CreateTensorFromArray(data, shape);
        var result = tensor.ToLongArray();

        result.Should().Equal([1, 2, 3]);
    }

    #endregion

    #region Image Tensor Tests - ORT Required

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateImageTensor_ShouldHaveCorrectShape()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var imageData = new float[1 * 3 * 224 * 224];

        using var tensor = OrtValueExtensions.CreateImageTensor(imageData, 1, 3, 224, 224);

        tensor.GetShape().Should().Equal([1, 3, 224, 224]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateImageTensor_WithBatch_ShouldHaveCorrectShape()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var imageData = new float[4 * 3 * 128 * 128];

        using var tensor = OrtValueExtensions.CreateImageTensor(imageData, 4, 3, 128, 128);

        tensor.GetShape().Should().Equal([4, 3, 128, 128]);
    }

    #endregion

    #region Token Tensor Tests - ORT Required

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTokenTensor_ShouldHaveCorrectShape()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var tokenIds = new long[] { 1, 2, 3, 4, 5 };

        using var tensor = OrtValueExtensions.CreateTokenTensor(tokenIds, 1);

        tensor.GetShape().Should().Equal([1, 5]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTokenTensor_WithBatch_ShouldHaveCorrectShape()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var tokenIds = new long[] { 1, 2, 3, 4, 5, 6 };

        using var tensor = OrtValueExtensions.CreateTokenTensor(tokenIds, 2);

        tensor.GetShape().Should().Equal([2, 3]);
    }

    #endregion

    #region Attention Mask Tests - ORT Required

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateAttentionMask_ShouldCreateAllOnes()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        using var tensor = OrtValueExtensions.CreateAttentionMask(5, 1);

        var mask = tensor.ToLongArray();
        mask.Should().Equal([1, 1, 1, 1, 1]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateAttentionMask_WithBatch_ShouldHaveCorrectShape()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        using var tensor = OrtValueExtensions.CreateAttentionMask(3, 2);

        tensor.GetShape().Should().Equal([2, 3]);
        var mask = tensor.ToLongArray();
        mask.Should().OnlyContain(x => x == 1);
    }

    #endregion

    #region Position IDs Tests - ORT Required

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreatePositionIds_ShouldCreateSequentialIds()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        using var tensor = OrtValueExtensions.CreatePositionIds(5, 1);

        var posIds = tensor.ToLongArray();
        posIds.Should().Equal([0, 1, 2, 3, 4]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreatePositionIds_WithBatch_ShouldRepeatSequence()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        using var tensor = OrtValueExtensions.CreatePositionIds(3, 2);

        var posIds = tensor.ToLongArray();
        posIds.Should().Equal([0, 1, 2, 0, 1, 2]);
    }

    #endregion

    #region Memory Tests - ORT Required

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTensorFromMemory_Float_ShouldWork()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var data = new float[] { 1, 2, 3, 4 };
        var memory = data.AsMemory();

        using var tensor = OrtValueExtensions.CreateTensorFromMemory(memory, [2, 2]);

        tensor.GetShape().Should().Equal([2, 2]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTensorFromMemory_Long_ShouldWork()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var data = new long[] { 100, 200 };
        var memory = data.AsMemory();

        using var tensor = OrtValueExtensions.CreateTensorFromMemory(memory, [1, 2]);

        tensor.GetShape().Should().Equal([1, 2]);
    }

    #endregion
}

[Collection("OrtValue")]
public class InferenceBufferPoolTests
{
    private static bool IsOrtAvailable()
    {
        var assemblyLocation = typeof(OrtValue).Assembly.Location;
        var assemblyDir = Path.GetDirectoryName(assemblyLocation) ?? ".";

        var dllName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "onnxruntime.dll"
            : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                ? "libonnxruntime.dylib"
                : "libonnxruntime.so";

        var nativePath = Path.Combine(assemblyDir, dllName);
        return File.Exists(nativePath);
    }

    #region Buffer Pool Tests - No ORT Required

    [Fact]
    public void RentFloat_ShouldReturnArray()
    {
        using var pool = new InferenceBufferPool();

        var array = pool.RentFloat(100);

        array.Should().NotBeNull();
        array.Length.Should().BeGreaterThanOrEqualTo(100);
    }

    [Fact]
    public void RentLong_ShouldReturnArray()
    {
        using var pool = new InferenceBufferPool();

        var array = pool.RentLong(50);

        array.Should().NotBeNull();
        array.Length.Should().BeGreaterThanOrEqualTo(50);
    }

    [Fact]
    public void RentFloat_MultipleCalls_ShouldTrackArrays()
    {
        using var pool = new InferenceBufferPool();

        var array1 = pool.RentFloat(100);
        var array2 = pool.RentFloat(200);
        var array3 = pool.RentFloat(50);

        array1.Should().NotBeNull();
        array2.Should().NotBeNull();
        array3.Should().NotBeNull();
        array1.Length.Should().BeGreaterThanOrEqualTo(100);
        array2.Length.Should().BeGreaterThanOrEqualTo(200);
        array3.Length.Should().BeGreaterThanOrEqualTo(50);
    }

    [Fact]
    public void Dispose_ShouldNotThrow()
    {
        var pool = new InferenceBufferPool();
        pool.RentFloat(100);
        pool.RentLong(50);

        var act = () => pool.Dispose();

        act.Should().NotThrow();
    }

    [Fact]
    public void MultipleDispose_ShouldNotThrow()
    {
        var pool = new InferenceBufferPool();
        pool.RentFloat(100);

        pool.Dispose();
        var act = () => pool.Dispose();

        act.Should().NotThrow();
    }

    #endregion

    #region Tensor Creation Tests - ORT Required

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateImageTensor_ShouldCreateValidTensor()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        using var pool = new InferenceBufferPool();
        using var tensor = pool.CreateImageTensor(1, 3, 64, 64);

        tensor.GetShape().Should().Equal([1, 3, 64, 64]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTokenTensor_ShouldCreateValidTensor()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        using var pool = new InferenceBufferPool();
        using var tensor = pool.CreateTokenTensor(2, 10);

        tensor.GetShape().Should().Equal([2, 10]);
    }

    #endregion
}

[Collection("OrtValue")]
public class OrtValueInputTests
{
    private static bool IsOrtAvailable()
    {
        var assemblyLocation = typeof(OrtValue).Assembly.Location;
        var assemblyDir = Path.GetDirectoryName(assemblyLocation) ?? ".";

        var dllName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "onnxruntime.dll"
            : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                ? "libonnxruntime.dylib"
                : "libonnxruntime.so";

        var nativePath = Path.Combine(assemblyDir, dllName);
        return File.Exists(nativePath);
    }

    #region OrtValueInput Tests - ORT Required

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateFloat_ShouldCreateValidInput()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        using var input = OrtValueInput.CreateFloat("test", [1, 2, 3], [1, 3]);

        input.Name.Should().Be("test");
        input.Value.Should().NotBeNull();
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateLong_ShouldCreateValidInput()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        using var input = OrtValueInput.CreateLong("input_ids", new long[] { 101, 102 }, [1, 2]);

        input.Name.Should().Be("input_ids");
        input.Value.GetShape().Should().Equal([1, 2]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateImage_ShouldCreateValidInput()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var data = new float[1 * 3 * 32 * 32];
        using var input = OrtValueInput.CreateImage("pixel_values", data, 1, 3, 32, 32);

        input.Name.Should().Be("pixel_values");
        input.Value.GetShape().Should().Equal([1, 3, 32, 32]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateTokenIds_ShouldCreateValidInput()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        using var input = OrtValueInput.CreateTokenIds("input_ids", new long[] { 1, 2, 3, 4 }, 1);

        input.Name.Should().Be("input_ids");
        input.Value.GetShape().Should().Equal([1, 4]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void CreateAttentionMask_ShouldCreateValidInput()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        using var input = OrtValueInput.CreateAttentionMask("attention_mask", 5, 1);

        input.Name.Should().Be("attention_mask");
        input.Value.GetShape().Should().Equal([1, 5]);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void Dispose_ShouldDisposeOrtValue()
    {
        Skip.IfNot(IsOrtAvailable(), "ONNX Runtime native library not available");

        var input = OrtValueInput.CreateFloat("test", [1, 2], [1, 2]);

        var act = () => input.Dispose();

        act.Should().NotThrow();
    }

    #endregion
}
