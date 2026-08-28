using System.Text;
using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;
using Xunit;

namespace LMSupply.Generator.Tests.Gguf;

/// <summary>
/// Tests for GgufMetadataReader.
/// </summary>
public class GgufMetadataReaderTests
{
    [Fact]
    public async Task IsGgufFileAsync_WithNonExistentFile_ReturnsFalse()
    {
        var result = await GgufMetadataReader.IsGgufFileAsync("/non/existent/path.gguf");

        result.Should().BeFalse();
    }

    [Fact]
    public async Task ReadAsync_WithNonExistentFile_ReturnsNull()
    {
        var result = await GgufMetadataReader.ReadAsync("/non/existent/path.gguf");

        result.Should().BeNull();
    }

    [Fact]
    public async Task IsGgufFileAsync_WithInvalidMagic_ReturnsFalse()
    {
        // Create a temporary file with wrong magic number
        var tempPath = Path.GetTempFileName();
        try
        {
            await File.WriteAllBytesAsync(tempPath, "NOTGGUF"u8.ToArray());

            var result = await GgufMetadataReader.IsGgufFileAsync(tempPath);

            result.Should().BeFalse();
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Fact]
    public async Task IsGgufFileAsync_WithValidMagic_ReturnsTrue()
    {
        // Create a temporary file with valid GGUF magic number
        var tempPath = Path.GetTempFileName();
        try
        {
            // GGUF magic: 0x46554747 (little-endian: "GGUF")
            var magic = new byte[] { 0x47, 0x47, 0x55, 0x46 };
            await File.WriteAllBytesAsync(tempPath, magic);

            var result = await GgufMetadataReader.IsGgufFileAsync(tempPath);

            result.Should().BeTrue();
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Fact]
    public async Task ReadAsync_WithInvalidVersion_ReturnsNull()
    {
        // Create a minimal GGUF header with invalid version
        var tempPath = Path.GetTempFileName();
        try
        {
            await using var stream = File.Create(tempPath);
            await using var writer = new BinaryWriter(stream);

            // Magic
            writer.Write(0x46554747u);
            // Version 99 (unsupported)
            writer.Write(99u);

            stream.Close();

            var result = await GgufMetadataReader.ReadAsync(tempPath);

            result.Should().BeNull();
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Fact]
    public async Task ReadAsync_WithMinimalValidGgufV3_ParsesCorrectly()
    {
        // Create a minimal valid GGUF v3 file
        var tempPath = Path.GetTempFileName();
        try
        {
            await CreateMinimalGgufFileAsync(tempPath, version: 3);

            var result = await GgufMetadataReader.ReadAsync(tempPath);

            result.Should().NotBeNull();
            result!.Version.Should().Be(3);
            result.TensorCount.Should().Be(0);
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Fact]
    public async Task ReadAsync_WithMinimalValidGgufV2_ParsesCorrectly()
    {
        // Create a minimal valid GGUF v2 file
        var tempPath = Path.GetTempFileName();
        try
        {
            await CreateMinimalGgufFileAsync(tempPath, version: 2);

            var result = await GgufMetadataReader.ReadAsync(tempPath);

            result.Should().NotBeNull();
            result!.Version.Should().Be(2);
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Fact]
    public async Task ReadAsync_WithArchitectureMetadata_ParsesArchitecture()
    {
        var tempPath = Path.GetTempFileName();
        try
        {
            var metadata = new Dictionary<string, object>
            {
                { "general.architecture", "llama" }
            };
            await CreateGgufFileWithMetadataAsync(tempPath, metadata);

            var result = await GgufMetadataReader.ReadAsync(tempPath);

            result.Should().NotBeNull();
            result!.Architecture.Should().Be("llama");
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Fact]
    public async Task ReadAsync_WithFullModelMetadata_ParsesAllFields()
    {
        var tempPath = Path.GetTempFileName();
        try
        {
            var metadata = new Dictionary<string, object>
            {
                { "general.architecture", "gemma" },
                { "general.name", "gemma-2b-it" },
                { "general.file_type", 15 }, // Q4_K_M
                { "gemma.block_count", 18 },
                { "gemma.context_length", 8192 },
                { "gemma.embedding_length", 2048 },
                { "gemma.attention.head_count", 8 },
                { "gemma.attention.head_count_kv", 1 },
                { "gemma.feed_forward_length", 16384 }
            };
            await CreateGgufFileWithMetadataAsync(tempPath, metadata);

            var result = await GgufMetadataReader.ReadAsync(tempPath);

            result.Should().NotBeNull();
            result!.Architecture.Should().Be("gemma");
            result.Name.Should().Be("gemma-2b-it");
            result.LayerCount.Should().Be(18);
            result.ContextLength.Should().Be(8192);
            result.EmbeddingLength.Should().Be(2048);
            result.HeadCount.Should().Be(8);
            result.HeadCountKv.Should().Be(1);
            result.FeedForwardLength.Should().Be(16384);
            result.FileType.Should().Be(15);
            result.QuantizationType.Should().Be("Q4_K_M");
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Fact]
    public void GgufMetadata_ToMemoryConfig_CreatesValidConfig()
    {
        var metadata = new GgufMetadata
        {
            Version = 3,
            Architecture = "llama",
            LayerCount = 32,
            EmbeddingLength = 4096,
            ContextLength = 4096,
            FileType = 15 // Q4_K_M
        };

        var config = metadata.ToMemoryConfig();

        config.Should().NotBeNull();
        config!.NumLayers.Should().Be(32);
        config.HiddenSize.Should().Be(4096);
        config.ContextLength.Should().Be(4096);
        config.Quantization.Should().Be(Quantization.Quant4);
    }

    [Fact]
    public void GgufMetadata_ToMemoryConfig_WithContextOverride_UsesOverride()
    {
        var metadata = new GgufMetadata
        {
            Version = 3,
            Architecture = "llama",
            LayerCount = 32,
            EmbeddingLength = 4096,
            ContextLength = 4096
        };

        var config = metadata.ToMemoryConfig(contextLength: 8192);

        config.Should().NotBeNull();
        config!.ContextLength.Should().Be(8192);
    }

    [Fact]
    public void GgufMetadata_ToMemoryConfig_WithInsufficientData_ReturnsNull()
    {
        var metadata = new GgufMetadata
        {
            Version = 3,
            Architecture = "llama"
            // Missing LayerCount and EmbeddingLength
        };

        var config = metadata.ToMemoryConfig();

        config.Should().BeNull();
    }

    [Fact]
    public void GgufMetadata_GetSummary_ReturnsFormattedString()
    {
        var metadata = new GgufMetadata
        {
            Version = 3,
            Name = "Test Model",
            Architecture = "llama",
            LayerCount = 32,
            EmbeddingLength = 4096,
            ContextLength = 8192,
            HeadCount = 32,
            QuantizationType = "Q4_K_M"
        };

        var summary = metadata.GetSummary();

        summary.Should().Contain("Name: Test Model");
        summary.Should().Contain("Architecture: llama");
        summary.Should().Contain("Layers: 32");
        summary.Should().Contain("Hidden Size: 4096");
        summary.Should().Contain("Context: 8192");
        summary.Should().Contain("Heads: 32");
        summary.Should().Contain("Quantization: Q4_K_M");
    }

    [Theory]
    [InlineData(0, "F32")]
    [InlineData(1, "F16")]
    [InlineData(2, "Q4_0")]
    [InlineData(8, "Q8_0")]
    [InlineData(15, "Q4_K_M")]
    [InlineData(17, "Q5_K_M")]
    [InlineData(18, "Q6_K")]
    public async Task ReadAsync_InfersQuantizationFromFileType(int fileType, string expectedQuantization)
    {
        var tempPath = Path.GetTempFileName();
        try
        {
            var metadata = new Dictionary<string, object>
            {
                { "general.file_type", fileType }
            };
            await CreateGgufFileWithMetadataAsync(tempPath, metadata);

            var result = await GgufMetadataReader.ReadAsync(tempPath);

            result.Should().NotBeNull();
            result!.QuantizationType.Should().Be(expectedQuantization);
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    #region Helper Methods

    private static async Task CreateMinimalGgufFileAsync(string path, uint version = 3)
    {
        await using var stream = File.Create(path);
        await using var writer = new BinaryWriter(stream);

        // Magic: "GGUF" in little-endian
        writer.Write(0x46554747u);
        // Version
        writer.Write(version);
        // Tensor count
        writer.Write(0UL);
        // Metadata KV count
        writer.Write(0UL);
    }

    private static async Task CreateGgufFileWithMetadataAsync(
        string path,
        Dictionary<string, object> metadata)
    {
        await using var stream = File.Create(path);
        await using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // Magic: "GGUF"
        writer.Write(0x46554747u);
        // Version 3
        writer.Write(3u);
        // Tensor count
        writer.Write(0UL);
        // Metadata KV count
        writer.Write((ulong)metadata.Count);

        // Write metadata
        foreach (var (key, value) in metadata)
        {
            WriteString(writer, key);
            WriteValue(writer, value);
        }
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static void WriteValue(BinaryWriter writer, object value)
    {
        switch (value)
        {
            case int i:
                writer.Write(5u); // GGUF_TYPE_INT32
                writer.Write(i);
                break;
            case uint u:
                writer.Write(4u); // GGUF_TYPE_UINT32
                writer.Write(u);
                break;
            case float f:
                writer.Write(6u); // GGUF_TYPE_FLOAT32
                writer.Write(f);
                break;
            case string s:
                writer.Write(8u); // GGUF_TYPE_STRING
                WriteString(writer, s);
                break;
            case bool b:
                writer.Write(7u); // GGUF_TYPE_BOOL
                writer.Write((byte)(b ? 1 : 0));
                break;
            default:
                throw new NotSupportedException($"Unsupported type: {value.GetType()}");
        }
    }

    #endregion
}
