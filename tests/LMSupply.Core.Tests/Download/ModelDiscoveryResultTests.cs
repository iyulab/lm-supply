using AwesomeAssertions;
using LMSupply.Core.Download;

namespace LMSupply.Core.Tests.Download;

public class ModelDiscoveryResultTests
{
    // --- Computed properties ---

    [Fact]
    public void IsEncoderDecoder_EncoderDecoderArchitecture_ReturnsTrue()
    {
        var result = CreateResult(architecture: ModelArchitecture.EncoderDecoder);

        result.IsEncoderDecoder.Should().BeTrue();
    }

    [Fact]
    public void IsEncoderDecoder_SingleModel_ReturnsFalse()
    {
        var result = CreateResult(architecture: ModelArchitecture.SingleModel);

        result.IsEncoderDecoder.Should().BeFalse();
    }

    [Fact]
    public void IsEncoderDecoder_Unknown_ReturnsFalse()
    {
        var result = CreateResult();

        result.IsEncoderDecoder.Should().BeFalse();
    }

    [Fact]
    public void HasExternalData_WithFiles_ReturnsTrue()
    {
        var result = CreateResult(externalDataFiles: ["model.onnx_data"]);

        result.HasExternalData.Should().BeTrue();
    }

    [Fact]
    public void HasExternalData_Empty_ReturnsFalse()
    {
        var result = CreateResult();

        result.HasExternalData.Should().BeFalse();
    }

    // --- PrimaryEncoderFile / PrimaryDecoderFile ---

    [Fact]
    public void PrimaryEncoderFile_WithEncoderFiles_ReturnsFirst()
    {
        var result = CreateResult(encoderFiles: ["onnx/encoder.onnx", "onnx/encoder_fp16.onnx"]);

        result.PrimaryEncoderFile.Should().Be("onnx/encoder.onnx");
    }

    [Fact]
    public void PrimaryEncoderFile_Empty_ReturnsNull()
    {
        var result = CreateResult();

        result.PrimaryEncoderFile.Should().BeNull();
    }

    [Fact]
    public void PrimaryDecoderFile_WithDecoderFiles_ReturnsFirst()
    {
        var result = CreateResult(decoderFiles: ["onnx/decoder.onnx"]);

        result.PrimaryDecoderFile.Should().Be("onnx/decoder.onnx");
    }

    [Fact]
    public void PrimaryDecoderFile_Empty_ReturnsNull()
    {
        var result = CreateResult();

        result.PrimaryDecoderFile.Should().BeNull();
    }

    // --- GetAllFiles ---

    [Fact]
    public void GetAllFiles_ConcatenatesAllFileLists()
    {
        var result = CreateResult(
            onnxFiles: ["model.onnx"],
            externalDataFiles: ["model.onnx_data"],
            configFiles: ["config.json", "tokenizer.json"]);

        var allFiles = result.GetAllFiles().ToList();

        allFiles.Should().HaveCount(4);
        allFiles.Should().ContainInOrder("model.onnx", "model.onnx_data", "config.json", "tokenizer.json");
    }

    [Fact]
    public void GetAllFiles_EmptyLists_ReturnsOnlyOnnxFiles()
    {
        var result = CreateResult(onnxFiles: ["model.onnx"]);

        var allFiles = result.GetAllFiles().ToList();

        allFiles.Should().HaveCount(1);
        allFiles.Should().Contain("model.onnx");
    }

    // --- GetFileNames ---

    [Fact]
    public void GetFileNames_RootLevelFiles_ReturnsAsIs()
    {
        var result = CreateResult(
            onnxFiles: ["model.onnx"],
            configFiles: ["config.json"]);

        var names = result.GetFileNames().ToList();

        names.Should().Contain("model.onnx");
        names.Should().Contain("config.json");
    }

    [Fact]
    public void GetFileNames_SubfolderFiles_StripsPath()
    {
        var result = CreateResult(
            onnxFiles: ["onnx/model.onnx"],
            configFiles: ["onnx/config.json"]);

        var names = result.GetFileNames().ToList();

        names.Should().Contain("model.onnx");
        names.Should().Contain("config.json");
    }

    [Fact]
    public void GetFileNames_DeepSubfolder_StripsFullPath()
    {
        var result = CreateResult(
            onnxFiles: ["models/fp16/encoder.onnx"]);

        var names = result.GetFileNames().ToList();

        names.Should().Contain("encoder.onnx");
    }

    // --- GetOnnxDirectory ---

    [Fact]
    public void GetOnnxDirectory_NoSubfolder_ReturnsBaseDir()
    {
        var result = CreateResult(subfolder: null);

        var dir = result.GetOnnxDirectory("/models/my-model");

        dir.Should().Be("/models/my-model");
    }

    [Fact]
    public void GetOnnxDirectory_EmptySubfolder_ReturnsBaseDir()
    {
        var result = CreateResult(subfolder: "");

        var dir = result.GetOnnxDirectory("/models/my-model");

        dir.Should().Be("/models/my-model");
    }

    [Fact]
    public void GetOnnxDirectory_WithSubfolder_CombinesPath()
    {
        var result = CreateResult(subfolder: "onnx");

        var dir = result.GetOnnxDirectory("/models/my-model");

        dir.Should().Be(Path.Combine("/models/my-model", "onnx"));
    }

    [Fact]
    public void GetOnnxDirectory_SubfolderWithSlash_ConvertsSeparator()
    {
        var result = CreateResult(subfolder: "models/fp16");

        var dir = result.GetOnnxDirectory("/base");

        var expected = Path.Combine("/base", "models" + Path.DirectorySeparatorChar + "fp16");
        dir.Should().Be(expected);
    }

    // --- GetFilePath (static) ---

    [Fact]
    public void GetFilePath_SimpleFile_CombinesPath()
    {
        var path = ModelDiscoveryResult.GetFilePath("/models/test", "model.onnx");

        path.Should().Be(Path.Combine("/models/test", "model.onnx"));
    }

    [Fact]
    public void GetFilePath_SubfolderFile_ConvertsSeparator()
    {
        var path = ModelDiscoveryResult.GetFilePath("/models/test", "onnx/model.onnx");

        var expected = Path.Combine("/models/test", "onnx" + Path.DirectorySeparatorChar + "model.onnx");
        path.Should().Be(expected);
    }

    // --- GetEncoderPath / GetDecoderPath ---

    [Fact]
    public void GetEncoderPath_WithEncoderFiles_ReturnsFirstFilePath()
    {
        var result = CreateResult(encoderFiles: ["onnx/encoder.onnx"]);

        var path = result.GetEncoderPath("/base");

        var expected = Path.Combine("/base", "onnx" + Path.DirectorySeparatorChar + "encoder.onnx");
        path.Should().Be(expected);
    }

    [Fact]
    public void GetEncoderPath_NoEncoderFiles_ReturnsNull()
    {
        var result = CreateResult();

        result.GetEncoderPath("/base").Should().BeNull();
    }

    [Fact]
    public void GetDecoderPath_WithDecoderFiles_ReturnsFirstFilePath()
    {
        var result = CreateResult(decoderFiles: ["decoder_model.onnx"]);

        var path = result.GetDecoderPath("/base");

        path.Should().Be(Path.Combine("/base", "decoder_model.onnx"));
    }

    [Fact]
    public void GetDecoderPath_NoDecoderFiles_ReturnsNull()
    {
        var result = CreateResult();

        result.GetDecoderPath("/base").Should().BeNull();
    }

    // --- Default values ---

    [Fact]
    public void DefaultArchitecture_IsUnknown()
    {
        var result = CreateResult();

        result.Architecture.Should().Be(ModelArchitecture.Unknown);
    }

    [Fact]
    public void DefaultDecoderVariant_IsStandard()
    {
        var result = CreateResult();

        result.DetectedDecoderVariant.Should().Be(DecoderVariant.Standard);
    }

    [Fact]
    public void DefaultLists_AreEmpty()
    {
        var result = CreateResult();

        result.ExternalDataFiles.Should().BeEmpty();
        result.ConfigFiles.Should().BeEmpty();
        result.EncoderFiles.Should().BeEmpty();
        result.DecoderFiles.Should().BeEmpty();
    }

    // --- ModelVariants ---

    [Fact]
    public void ModelVariants_DefaultLists_AreEmpty()
    {
        var variants = new ModelVariants();

        variants.Default.Should().BeEmpty();
        variants.Fp16.Should().BeEmpty();
        variants.Quant8.Should().BeEmpty();
        variants.Quant4.Should().BeEmpty();
        variants.Cpu.Should().BeEmpty();
        variants.Cuda.Should().BeEmpty();
    }

    [Fact]
    public void ModelVariants_WithData_RetainsValues()
    {
        var variants = new ModelVariants
        {
            Default = ["model.onnx"],
            Fp16 = ["model_fp16.onnx"],
            Quant4 = ["model_q4.onnx"]
        };

        variants.Default.Should().HaveCount(1);
        variants.Fp16.Should().HaveCount(1);
        variants.Quant4.Should().HaveCount(1);
        variants.Quant8.Should().BeEmpty();
    }

    [Fact]
    public void AvailableVariants_Null_ByDefault()
    {
        var result = CreateResult();

        result.AvailableVariants.Should().BeNull();
    }

    // --- Helper ---

    private static ModelDiscoveryResult CreateResult(
        string repoId = "test/model",
        string? subfolder = null,
        string[]? onnxFiles = null,
        string[]? externalDataFiles = null,
        string[]? configFiles = null,
        string[]? encoderFiles = null,
        string[]? decoderFiles = null,
        ModelArchitecture architecture = ModelArchitecture.Unknown)
    {
        return new ModelDiscoveryResult
        {
            RepoId = repoId,
            Subfolder = subfolder,
            OnnxFiles = onnxFiles ?? ["model.onnx"],
            ExternalDataFiles = externalDataFiles ?? [],
            ConfigFiles = configFiles ?? [],
            EncoderFiles = encoderFiles ?? [],
            DecoderFiles = decoderFiles ?? [],
            Architecture = architecture
        };
    }
}
