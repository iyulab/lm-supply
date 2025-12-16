using FluentAssertions;
using LMSupply.Generator.Internal;

namespace LMSupply.Generator.Tests;

public class GenAiConfigReaderTests
{
    [Fact]
    public void ReadMaxContextLength_NonExistentPath_ReturnsDefault()
    {
        // Arrange
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());

        // Act
        var result = GenAiConfigReader.ReadMaxContextLength(path);

        // Assert
        result.Should().Be(4096);
    }

    [Fact]
    public void ReadMaxContextLength_ValidConfig_ReturnsContextLength()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var configPath = Path.Combine(tempDir, "genai_config.json");
            File.WriteAllText(configPath, """
                {
                    "model": {
                        "context_length": 8192,
                        "vocab_size": 32000
                    }
                }
                """);

            // Act
            var result = GenAiConfigReader.ReadMaxContextLength(tempDir);

            // Assert
            result.Should().Be(8192);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void ReadMaxContextLength_MaxPositionEmbeddings_ReturnsValue()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var configPath = Path.Combine(tempDir, "genai_config.json");
            File.WriteAllText(configPath, """
                {
                    "model": {
                        "max_position_embeddings": 131072
                    }
                }
                """);

            // Act
            var result = GenAiConfigReader.ReadMaxContextLength(tempDir);

            // Assert
            result.Should().Be(131072);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void ReadMaxContextLength_SearchMaxLength_ReturnsValue()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var configPath = Path.Combine(tempDir, "genai_config.json");
            File.WriteAllText(configPath, """
                {
                    "search": {
                        "max_length": 2048
                    }
                }
                """);

            // Act
            var result = GenAiConfigReader.ReadMaxContextLength(tempDir);

            // Assert
            result.Should().Be(2048);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void ReadMaxContextLength_InvalidJson_ReturnsDefault()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var configPath = Path.Combine(tempDir, "genai_config.json");
            File.WriteAllText(configPath, "{ invalid json }");

            // Act
            var result = GenAiConfigReader.ReadMaxContextLength(tempDir);

            // Assert
            result.Should().Be(4096);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void ReadModelType_ValidConfig_ReturnsType()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var configPath = Path.Combine(tempDir, "genai_config.json");
            File.WriteAllText(configPath, """
                {
                    "model": {
                        "type": "phi3"
                    }
                }
                """);

            // Act
            var result = GenAiConfigReader.ReadModelType(tempDir);

            // Assert
            result.Should().Be("phi3");
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void ReadVocabSize_ValidConfig_ReturnsSize()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var configPath = Path.Combine(tempDir, "genai_config.json");
            File.WriteAllText(configPath, """
                {
                    "model": {
                        "vocab_size": 32064
                    }
                }
                """);

            // Act
            var result = GenAiConfigReader.ReadVocabSize(tempDir);

            // Assert
            result.Should().Be(32064);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }
}
