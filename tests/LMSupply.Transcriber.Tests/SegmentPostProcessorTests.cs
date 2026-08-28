using AwesomeAssertions;
using LMSupply.Transcriber.Internal;

namespace LMSupply.Transcriber.Tests;

public class SegmentPostProcessorTests
{
    [Fact]
    public void RemovesConsecutiveDuplicateSegments()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 3, Text = "Hello world" },
            new() { Id = 1, Start = 3, End = 6, Text = "예수님의 이름으로 기도하옵나이다 아멘" },
            new() { Id = 2, Start = 6, End = 9, Text = "예수님의 이름으로 기도하옵나이다 아멘" },
            new() { Id = 3, Start = 9, End = 12, Text = "예수님의 이름으로 기도하옵나이다 아멘" },
            new() { Id = 4, Start = 12, End = 15, Text = "Goodbye" }
        };

        var (result, _) = SegmentPostProcessor.Process(segments, null);

        result.Should().HaveCount(3);
        result[0].Text.Should().Be("Hello world");
        result[1].Text.Should().Be("예수님의 이름으로 기도하옵나이다 아멘");
        result[2].Text.Should().Be("Goodbye");
    }

    [Fact]
    public void KeepsNonConsecutiveDuplicates()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 3, Text = "Repeated phrase" },
            new() { Id = 1, Start = 3, End = 6, Text = "Different text" },
            new() { Id = 2, Start = 6, End = 9, Text = "Another line" },
            new() { Id = 3, Start = 9, End = 12, Text = "More content" },
            new() { Id = 4, Start = 12, End = 15, Text = "Repeated phrase" }
        };

        var (result, _) = SegmentPostProcessor.Process(segments, null);

        result.Should().HaveCount(5);
        result[0].Text.Should().Be("Repeated phrase");
        result[4].Text.Should().Be("Repeated phrase");
    }

    [Fact]
    public void FiltersHighCompressionRatio()
    {
        // Highly repetitive text will have high compression ratio
        var repetitiveText = string.Join(" ", Enumerable.Repeat("hello", 50));
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 5, Text = "Normal varied text with different words" },
            new() { Id = 1, Start = 5, End = 10, Text = repetitiveText }
        };

        var (result, _) = SegmentPostProcessor.Process(segments, null);

        result.Should().HaveCount(1);
        result[0].Text.Should().Be("Normal varied text with different words");
    }

    [Fact]
    public void KeepsNormalCompressionRatio()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 5, Text = "The quick brown fox jumps over the lazy dog near the river bank" },
            new() { Id = 1, Start = 5, End = 10, Text = "A completely different sentence with varied vocabulary and structure" }
        };

        var (result, _) = SegmentPostProcessor.Process(segments, null);

        result.Should().HaveCount(2);
    }

    [Fact]
    public void FiltersHighNoSpeechProb()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 5, Text = "Real speech", NoSpeechProb = 0.1f },
            new() { Id = 1, Start = 5, End = 10, Text = "Probably silence", NoSpeechProb = 0.9f }
        };

        var (result, _) = SegmentPostProcessor.Process(segments, null);

        result.Should().HaveCount(1);
        result[0].Text.Should().Be("Real speech");
    }

    [Fact]
    public void KeepsLowNoSpeechProb()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 5, Text = "Speech one", NoSpeechProb = 0.2f },
            new() { Id = 1, Start = 5, End = 10, Text = "Speech two", NoSpeechProb = 0.3f }
        };

        var (result, _) = SegmentPostProcessor.Process(segments, null);

        result.Should().HaveCount(2);
    }

    [Fact]
    public void NoSpeechFilterSkipsWhenNotPopulated()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 5, Text = "Has value", NoSpeechProb = 0.1f },
            new() { Id = 1, Start = 5, End = 10, Text = "No value set" } // NoSpeechProb is null
        };

        var (result, _) = SegmentPostProcessor.Process(segments, null);

        result.Should().HaveCount(2, "null NoSpeechProb should not trigger filtering");
    }

    [Fact]
    public void ReindexesSegmentIds()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 3, Text = "First" },
            new() { Id = 1, Start = 3, End = 6, Text = "Duplicate" },
            new() { Id = 2, Start = 6, End = 9, Text = "Duplicate" }, // Will be removed
            new() { Id = 3, Start = 9, End = 12, Text = "Last" }
        };

        var (result, _) = SegmentPostProcessor.Process(segments, null);

        result.Should().HaveCount(3);
        result[0].Id.Should().Be(0);
        result[1].Id.Should().Be(1);
        result[2].Id.Should().Be(2);
    }

    [Fact]
    public void RebuildsTextFromRemainingSegments()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 3, Text = "Hello" },
            new() { Id = 1, Start = 3, End = 6, Text = "Hello" }, // Consecutive duplicate
            new() { Id = 2, Start = 6, End = 9, Text = "World" }
        };

        var (result, text) = SegmentPostProcessor.Process(segments, null);

        text.Should().Be("Hello World");
        result.Should().HaveCount(2);
    }

    [Fact]
    public void HandlesEmptySegmentList()
    {
        var (result, text) = SegmentPostProcessor.Process([], null);

        result.Should().BeEmpty();
        text.Should().BeEmpty();
    }

    [Fact]
    public void HandlesSingleSegment()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 5, Text = "Only segment" }
        };

        var (result, text) = SegmentPostProcessor.Process(segments, null);

        result.Should().HaveCount(1);
        result[0].Text.Should().Be("Only segment");
        text.Should().Be("Only segment");
    }

    [Fact]
    public void CustomThresholds()
    {
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 5, Text = "Normal text", NoSpeechProb = 0.4f },
            new() { Id = 1, Start = 5, End = 10, Text = "Borderline speech", NoSpeechProb = 0.5f }
        };

        // With custom threshold of 0.35, both should be filtered
        var options = new TranscribeOptions { NoSpeechThreshold = 0.35f };
        var (result, _) = SegmentPostProcessor.Process(segments, options);

        result.Should().BeEmpty();
    }

    [Fact]
    public void CompressionRatioUsesPrecomputedValueWhenAvailable()
    {
        // Segment has CompressionRatio already set (from decoder) — should use that, not recompute
        var segments = new List<TranscriptionSegment>
        {
            new() { Id = 0, Start = 0, End = 5, Text = "Normal text", CompressionRatio = 5.0f }
        };

        // Pre-computed ratio 5.0 > threshold 2.4 → should be filtered
        var (result, _) = SegmentPostProcessor.Process(segments, null);

        result.Should().BeEmpty();
    }

    [Fact]
    public void ComputeCompressionRatio_NormalText_BelowThreshold()
    {
        var ratio = SegmentPostProcessor.ComputeCompressionRatio(
            "The quick brown fox jumps over the lazy dog");

        ratio.Should().BeLessThan(2.4f);
    }

    [Fact]
    public void ComputeCompressionRatio_RepetitiveText_AboveThreshold()
    {
        var text = string.Join(" ", Enumerable.Repeat("hello", 50));
        var ratio = SegmentPostProcessor.ComputeCompressionRatio(text);

        ratio.Should().BeGreaterThan(2.4f);
    }

    [Fact]
    public void ComputeCompressionRatio_EmptyText_ReturnsOne()
    {
        var ratio = SegmentPostProcessor.ComputeCompressionRatio("");

        ratio.Should().Be(1.0f);
    }

    [Fact]
    public void PreservesSegmentMetadata()
    {
        var segments = new List<TranscriptionSegment>
        {
            new()
            {
                Id = 0, Start = 1.5, End = 3.0, Text = "Keep me",
                AvgLogProb = -0.3f, NoSpeechProb = 0.05f, CompressionRatio = 1.1f
            }
        };

        var (result, _) = SegmentPostProcessor.Process(segments, null);

        result.Should().HaveCount(1);
        result[0].Start.Should().Be(1.5);
        result[0].End.Should().Be(3.0);
        result[0].AvgLogProb.Should().Be(-0.3f);
        result[0].NoSpeechProb.Should().Be(0.05f);
        result[0].CompressionRatio.Should().Be(1.1f);
    }
}
