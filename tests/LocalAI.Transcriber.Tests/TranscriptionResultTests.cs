using FluentAssertions;

namespace LocalAI.Transcriber.Tests;

public class TranscriptionResultTests
{
    [Fact]
    public void TranscriptionResult_ShouldStoreRequiredProperties()
    {
        var result = new TranscriptionResult
        {
            Text = "Test transcription",
            Language = "en"
        };

        result.Text.Should().Be("Test transcription");
        result.Language.Should().Be("en");
        result.Segments.Should().BeEmpty();
        result.DurationSeconds.Should().BeNull();
        result.InferenceTimeMs.Should().BeNull();
    }

    [Fact]
    public void TranscriptionResult_ShouldStoreAllProperties()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 5, Text = "Hello" },
            new() { Id = 1, Start = 5, End = 10, Text = "World" }
        };

        var result = new TranscriptionResult
        {
            Text = "Hello World",
            Language = "en",
            LanguageProbability = 0.95f,
            Segments = segments,
            DurationSeconds = 10.0,
            InferenceTimeMs = 500.0
        };

        result.Text.Should().Be("Hello World");
        result.Language.Should().Be("en");
        result.LanguageProbability.Should().Be(0.95f);
        result.Segments.Should().HaveCount(2);
        result.DurationSeconds.Should().Be(10.0);
        result.InferenceTimeMs.Should().Be(500.0);
    }

    [Fact]
    public void TranscriptionResult_RealTimeFactor_ShouldCalculateCorrectly()
    {
        var result = new TranscriptionResult
        {
            Text = "Test",
            Language = "en",
            DurationSeconds = 10.0,
            InferenceTimeMs = 5000.0
        };

        // 10 seconds audio / 5 seconds processing = 2x real-time
        result.RealTimeFactor.Should().Be(2.0);
    }

    [Fact]
    public void TranscriptionSegment_ShouldStoreRequiredProperties()
    {
        var segment = new TranscriptionSegment
        {
            Text = "Test segment"
        };

        segment.Id.Should().Be(0);
        segment.Start.Should().Be(0);
        segment.End.Should().Be(0);
        segment.Text.Should().Be("Test segment");
    }

    [Fact]
    public void TranscriptionSegment_ShouldStoreAllProperties()
    {
        var segment = new TranscriptionSegment
        {
            Id = 1,
            Start = 10.5,
            End = 15.3,
            Text = "Test segment",
            AvgLogProb = -0.5f,
            NoSpeechProb = 0.1f,
            CompressionRatio = 1.2f
        };

        segment.Id.Should().Be(1);
        segment.Start.Should().Be(10.5);
        segment.End.Should().Be(15.3);
        segment.Text.Should().Be("Test segment");
        segment.AvgLogProb.Should().Be(-0.5f);
        segment.NoSpeechProb.Should().Be(0.1f);
        segment.CompressionRatio.Should().Be(1.2f);
    }

    [Fact]
    public void TranscriptionSegment_Duration_ShouldCalculateCorrectly()
    {
        var segment = new TranscriptionSegment
        {
            Start = 10.0,
            End = 15.5,
            Text = "Test"
        };

        segment.Duration.Should().Be(5.5);
    }

    [Fact]
    public void TranscriptionSegment_ToString_ShouldFormatCorrectly()
    {
        var segment = new TranscriptionSegment
        {
            Start = 65.5,
            End = 70.0,
            Text = "Hello world"
        };

        var str = segment.ToString();

        str.Should().Contain("01:05");
        str.Should().Contain("01:10");
        str.Should().Contain("Hello world");
    }

    [Fact]
    public void TranscribeOptions_ShouldHaveExpectedDefaults()
    {
        var options = new TranscribeOptions();

        options.Language.Should().BeNull();
        options.Translate.Should().BeFalse();
        options.WordTimestamps.Should().BeFalse();
    }

    [Fact]
    public void TranscribeOptions_Translate_ShouldToggle()
    {
        var transcribeOptions = new TranscribeOptions { Translate = false };
        var translateOptions = new TranscribeOptions { Translate = true };

        transcribeOptions.Translate.Should().BeFalse();
        translateOptions.Translate.Should().BeTrue();
    }
}
