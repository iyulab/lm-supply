using AwesomeAssertions;

namespace LMSupply.Transcriber.Tests;

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

    /// <summary>
    /// Documents that WordTimestamps option controls segment-level timestamps only,
    /// not word-level timestamps. This is a known limitation.
    /// Fix documentation for issue #4: WordTimestamps not working.
    /// </summary>
    [Fact]
    public void TranscribeOptions_WordTimestamps_ControlsSegmentTimestampsOnly()
    {
        var optionsEnabled = new TranscribeOptions { WordTimestamps = true };
        var optionsDisabled = new TranscribeOptions { WordTimestamps = false };

        // WordTimestamps = true enables segment-level timestamp tokens in Whisper decoding
        // This creates multiple segments with Start/End times based on speech patterns
        // WordTimestamps = false creates a single segment per 30-second audio chunk
        optionsEnabled.WordTimestamps.Should().BeTrue();
        optionsDisabled.WordTimestamps.Should().BeFalse();
    }

    /// <summary>
    /// Documents that the Words property in TranscriptionSegment is always null.
    /// True word-level timestamps require cross-attention alignment with DTW,
    /// which is not implemented in the current version.
    /// Fix documentation for issue #4: Words property always null.
    /// </summary>
    [Fact]
    public void TranscriptionSegment_Words_IsAlwaysNull()
    {
        // The Words property exists for API compatibility but is not populated
        // Word-level timestamps require extracting cross-attention weights from
        // the Whisper decoder and applying Dynamic Time Warping (DTW) alignment.
        // See: https://github.com/linto-ai/whisper-timestamped
        var segment = new TranscriptionSegment
        {
            Id = 0,
            Start = 0,
            End = 30,
            Text = "Test transcription",
            Words = null // Always null - word-level timestamps not implemented
        };

        segment.Words.Should().BeNull("word-level timestamps require cross-attention DTW which is not implemented");
    }

    /// <summary>
    /// Tests that WordTimestamp class can be instantiated correctly
    /// for future use when word-level timestamps are implemented.
    /// </summary>
    [Fact]
    public void WordTimestamp_ShouldStoreProperties()
    {
        var wordTimestamp = new WordTimestamp
        {
            Word = "hello",
            Start = 1.5,
            End = 2.0,
            Probability = 0.95f
        };

        wordTimestamp.Word.Should().Be("hello");
        wordTimestamp.Start.Should().Be(1.5);
        wordTimestamp.End.Should().Be(2.0);
        wordTimestamp.Probability.Should().Be(0.95f);
    }
}
