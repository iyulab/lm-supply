using System.Text.Json;
using AwesomeAssertions;
using LMSupply.Core.Download;

namespace LMSupply.Core.Tests.Download;

public class RepoFileTests
{
    [Fact]
    public void IsFile_ShouldBeTrue_WhenTypeIsFile()
    {
        var file = new RepoFile { Path = "model.onnx", Type = "file" };

        file.IsFile.Should().BeTrue();
        file.IsDirectory.Should().BeFalse();
    }

    [Fact]
    public void IsDirectory_ShouldBeTrue_WhenTypeIsDirectory()
    {
        var file = new RepoFile { Path = "onnx", Type = "directory" };

        file.IsDirectory.Should().BeTrue();
        file.IsFile.Should().BeFalse();
    }

    [Theory]
    [InlineData("FILE")]
    [InlineData("File")]
    [InlineData("file")]
    public void IsFile_ShouldBeCaseInsensitive(string type)
    {
        var file = new RepoFile { Path = "test.bin", Type = type };

        file.IsFile.Should().BeTrue();
    }

    [Theory]
    [InlineData("DIRECTORY")]
    [InlineData("Directory")]
    [InlineData("directory")]
    public void IsDirectory_ShouldBeCaseInsensitive(string type)
    {
        var file = new RepoFile { Path = "subdir", Type = type };

        file.IsDirectory.Should().BeTrue();
    }

    [Fact]
    public void FileName_ShouldExtractFromPath()
    {
        var file = new RepoFile { Path = "onnx/fp16/model.onnx", Type = "file" };

        file.FileName.Should().Be("model.onnx");
    }

    [Fact]
    public void FileName_ShouldReturnFullPath_WhenNoSlash()
    {
        var file = new RepoFile { Path = "config.json", Type = "file" };

        file.FileName.Should().Be("config.json");
    }

    [Fact]
    public void Directory_ShouldExtractParentPath()
    {
        var file = new RepoFile { Path = "onnx/fp16/model.onnx", Type = "file" };

        file.Directory.Should().Be("onnx/fp16");
    }

    [Fact]
    public void Directory_ShouldReturnNull_WhenInRoot()
    {
        var file = new RepoFile { Path = "model.onnx", Type = "file" };

        file.Directory.Should().BeNull();
    }

    [Fact]
    public void Directory_SingleLevel_ShouldReturnParent()
    {
        var file = new RepoFile { Path = "onnx/model.onnx", Type = "file" };

        file.Directory.Should().Be("onnx");
    }

    [Fact]
    public void ShouldDeserialize_FromJson()
    {
        var json = """{"path":"onnx/model.onnx","type":"file","size":1048576,"oid":"abc123"}""";

        var file = JsonSerializer.Deserialize<RepoFile>(json);

        file.Should().NotBeNull();
        file!.Path.Should().Be("onnx/model.onnx");
        file.Type.Should().Be("file");
        file.Size.Should().Be(1048576);
        file.Oid.Should().Be("abc123");
    }

    [Fact]
    public void ShouldDeserialize_WithoutOptionalFields()
    {
        var json = """{"path":"readme.md","type":"file","size":512}""";

        var file = JsonSerializer.Deserialize<RepoFile>(json);

        file!.Oid.Should().BeNull();
    }

    [Fact]
    public void Size_ShouldDefaultToZero()
    {
        var file = new RepoFile { Path = "test", Type = "file" };

        file.Size.Should().Be(0);
    }
}

public class ModelPreferencesTests
{
    [Fact]
    public void Default_ShouldHaveExpectedValues()
    {
        var prefs = ModelPreferences.Default;

        prefs.PreferLowMemory.Should().BeFalse();
        prefs.PreferredProvider.Should().Be(ExecutionProvider.Cpu);
        prefs.QuantizationPriority.Should().Equal(
            Quantization.Default, Quantization.Fp16, Quantization.Quant8, Quantization.Quant4);
        prefs.PreferredSubfolder.Should().BeNull();
        prefs.PreferredOnnxFiles.Should().BeEmpty();
        prefs.RequireMatchedQuantization.Should().BeTrue();
    }

    [Fact]
    public void LowMemory_ShouldPreferQuantizedFirst()
    {
        var prefs = ModelPreferences.LowMemory;

        prefs.PreferLowMemory.Should().BeTrue();
        prefs.QuantizationPriority.Should().Equal(
            Quantization.Quant4, Quantization.Quant8, Quantization.Fp16, Quantization.Default);
    }

    [Fact]
    public void HighQuality_ShouldPreferFullPrecision()
    {
        var prefs = ModelPreferences.HighQuality;

        prefs.PreferLowMemory.Should().BeFalse();
        prefs.QuantizationPriority.Should().Equal(
            Quantization.Default, Quantization.Fp16, Quantization.Quant8, Quantization.Quant4);
    }

    [Fact]
    public void Default_ShouldReturnSameInstance()
    {
        var a = ModelPreferences.Default;
        var b = ModelPreferences.Default;

        a.Should().BeSameAs(b);
    }

    [Fact]
    public void DecoderVariantPriority_ShouldHaveDefaultOrder()
    {
        var prefs = new ModelPreferences();

        prefs.DecoderVariantPriority.Should().Equal(
            DecoderVariant.Merged, DecoderVariant.Standard, DecoderVariant.WithPast);
    }

    [Fact]
    public void CustomPreferences_ShouldOverrideDefaults()
    {
        var prefs = new ModelPreferences
        {
            PreferLowMemory = true,
            PreferredProvider = ExecutionProvider.Cuda,
            PreferredSubfolder = "fp16",
            PreferredOnnxFiles = ["encoder.onnx", "decoder.onnx"],
            RequireMatchedQuantization = false
        };

        prefs.PreferLowMemory.Should().BeTrue();
        prefs.PreferredProvider.Should().Be(ExecutionProvider.Cuda);
        prefs.PreferredSubfolder.Should().Be("fp16");
        prefs.PreferredOnnxFiles.Should().HaveCount(2);
        prefs.RequireMatchedQuantization.Should().BeFalse();
    }
}

public class QuantizationEnumTests
{
    [Fact]
    public void ShouldHaveFourValues()
    {
        Enum.GetValues<Quantization>().Should().HaveCount(4);
    }

    [Theory]
    [InlineData(Quantization.Default, 0)]
    [InlineData(Quantization.Fp16, 1)]
    [InlineData(Quantization.Quant8, 2)]
    [InlineData(Quantization.Quant4, 3)]
    public void ShouldHaveExpectedIntValues(Quantization q, int expected)
    {
        ((int)q).Should().Be(expected);
    }
}

public class DecoderVariantEnumTests
{
    [Fact]
    public void ShouldHaveFourValues()
    {
        Enum.GetValues<DecoderVariant>().Should().HaveCount(4);
    }

    [Theory]
    [InlineData(DecoderVariant.Standard, 0)]
    [InlineData(DecoderVariant.Merged, 1)]
    [InlineData(DecoderVariant.WithPast, 2)]
    [InlineData(DecoderVariant.DecoderOnly, 3)]
    public void ShouldHaveExpectedIntValues(DecoderVariant v, int expected)
    {
        ((int)v).Should().Be(expected);
    }
}

public class ExecutionProviderEnumTests
{
    [Fact]
    public void ShouldHaveFiveValues()
    {
        Enum.GetValues<ExecutionProvider>().Should().HaveCount(5);
    }

    [Theory]
    [InlineData(ExecutionProvider.Auto, 0)]
    [InlineData(ExecutionProvider.Cuda, 1)]
    [InlineData(ExecutionProvider.DirectML, 2)]
    [InlineData(ExecutionProvider.CoreML, 3)]
    [InlineData(ExecutionProvider.Cpu, 4)]
    public void ShouldHaveExpectedIntValues(ExecutionProvider p, int expected)
    {
        ((int)p).Should().Be(expected);
    }
}
