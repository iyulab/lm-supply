using AwesomeAssertions;
using LMSupply.Transcriber.Internal;

namespace LMSupply.Transcriber.Tests;

public class WhisperConfigReaderTests
{
    [Fact]
    public void ParseConfig_LargeV3Turbo_ShouldReturn128MelBins()
    {
        var json = """
        {
            "model_type": "whisper",
            "num_mel_bins": 128,
            "d_model": 1280,
            "decoder_layers": 4,
            "encoder_layers": 32
        }
        """;

        var config = WhisperConfigReader.ParseConfig(json);

        config.Should().NotBeNull();
        config!.NumMelBins.Should().Be(128);
        config.HiddenSize.Should().Be(1280);
    }

    [Fact]
    public void ParseConfig_TinyModel_ShouldReturn80MelBins()
    {
        var json = """
        {
            "model_type": "whisper",
            "num_mel_bins": 80,
            "d_model": 384
        }
        """;

        var config = WhisperConfigReader.ParseConfig(json);

        config.Should().NotBeNull();
        config!.NumMelBins.Should().Be(80);
        config.HiddenSize.Should().Be(384);
    }

    [Fact]
    public void ParseConfig_BaseModel_ShouldReturn512HiddenSize()
    {
        var json = """
        {
            "model_type": "whisper",
            "num_mel_bins": 80,
            "d_model": 512
        }
        """;

        var config = WhisperConfigReader.ParseConfig(json);

        config.Should().NotBeNull();
        config!.HiddenSize.Should().Be(512);
    }

    [Fact]
    public void ParseConfig_MissingModelType_ShouldReturnNull()
    {
        // Without explicit model_type: "whisper", config is rejected
        // to prevent applying Whisper params to non-Whisper models
        var json = """
        {
            "num_mel_bins": 128,
            "d_model": 1280
        }
        """;

        var config = WhisperConfigReader.ParseConfig(json);

        config.Should().BeNull();
    }

    [Fact]
    public void ParseConfig_NonWhisperModelType_ShouldReturnNull()
    {
        var json = """
        {
            "model_type": "bert",
            "hidden_size": 768
        }
        """;

        var config = WhisperConfigReader.ParseConfig(json);

        config.Should().BeNull();
    }

    [Fact]
    public void ParseConfig_OnlyNumMelBins_ShouldReturnPartialConfig()
    {
        var json = """
        {
            "model_type": "whisper",
            "num_mel_bins": 128
        }
        """;

        var config = WhisperConfigReader.ParseConfig(json);

        config.Should().NotBeNull();
        config!.NumMelBins.Should().Be(128);
        config.HiddenSize.Should().BeNull();
    }

    [Fact]
    public void ParseConfig_OnlyDModel_ShouldReturnPartialConfig()
    {
        var json = """
        {
            "model_type": "whisper",
            "d_model": 768
        }
        """;

        var config = WhisperConfigReader.ParseConfig(json);

        config.Should().NotBeNull();
        config!.NumMelBins.Should().BeNull();
        config.HiddenSize.Should().Be(768);
    }

    [Fact]
    public void ParseConfig_NoRelevantFields_ShouldReturnNull()
    {
        var json = """
        {
            "model_type": "whisper",
            "vocab_size": 51865,
            "encoder_layers": 4
        }
        """;

        var config = WhisperConfigReader.ParseConfig(json);

        config.Should().BeNull();
    }

    [Fact]
    public void ParseConfig_FloatValues_ShouldNotThrow()
    {
        // Some HF configs may have float values for numeric fields.
        // TryGetInt32 returns false for numbers with fractional parts,
        // so these are safely skipped without throwing.
        var json = """
        {
            "model_type": "whisper",
            "num_mel_bins": 128.0,
            "d_model": 1280.5
        }
        """;

        // Should not throw (previously GetInt32 would throw InvalidOperationException)
        var config = WhisperConfigReader.ParseConfig(json);

        // Both values have decimal points, so TryGetInt32 skips them → returns null
        config.Should().BeNull();
    }

    [Fact]
    public void ParseConfig_InvalidJson_ShouldThrowJsonException()
    {
        var json = "not valid json{";

        var act = () => WhisperConfigReader.ParseConfig(json);

        act.Should().Throw<System.Text.Json.JsonException>();
    }

    [Fact]
    public void ReadConfig_NonExistentDirectory_ShouldReturnNull()
    {
        var config = WhisperConfigReader.ReadConfig(Path.Combine(Path.GetTempPath(), $"nonexistent-{Guid.NewGuid():N}"));

        config.Should().BeNull();
    }

    [Fact]
    public void ReadConfig_DirectoryWithConfigJson_ShouldParse()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"whisper-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            File.WriteAllText(Path.Combine(tempDir, "config.json"), """
            {
                "model_type": "whisper",
                "num_mel_bins": 128,
                "d_model": 1280
            }
            """);

            var config = WhisperConfigReader.ReadConfig(tempDir);

            config.Should().NotBeNull();
            config!.NumMelBins.Should().Be(128);
            config.HiddenSize.Should().Be(1280);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void ParseConfig_RealKoreanModelConfig_ShouldParseCorrectly()
    {
        // Realistic config.json from onnx-community/whisper-large-v3-turbo-korean-ggml-ONNX
        var json = """
        {
            "architectures": ["WhisperForConditionalGeneration"],
            "d_model": 1280,
            "decoder_attention_heads": 20,
            "decoder_ffn_dim": 5120,
            "decoder_layers": 4,
            "encoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "encoder_layers": 32,
            "model_type": "whisper",
            "num_mel_bins": 128,
            "vocab_size": 51866
        }
        """;

        var config = WhisperConfigReader.ParseConfig(json);

        config.Should().NotBeNull();
        config!.NumMelBins.Should().Be(128);
        config.HiddenSize.Should().Be(1280);
    }
}
