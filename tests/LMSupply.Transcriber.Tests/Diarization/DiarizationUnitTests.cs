using AwesomeAssertions;
using LMSupply.Transcriber.Diarization;

namespace LMSupply.Transcriber.Tests.Diarization;

/// <summary>Diarization pieces that need no model: clustering, frame labels to turns, speaker assignment, options.</summary>
public class DiarizationUnitTests
{
    private static float[] Rows(params float[][] rows) => rows.SelectMany(r => r).ToArray();

    [Fact]
    public void Clustering_SplitsAtTheThreshold()
    {
        // Two tight groups (cosine distance ~0 within, ~1 across).
        var embeddings = Rows([1, 0.01f], [1, 0], [0.02f, 1], [0, 1]);

        AgglomerativeClustering.Cluster(embeddings, 2, threshold: 0.5f).Should().Equal(0, 0, 1, 1);
        AgglomerativeClustering.Cluster(embeddings, 2, threshold: 1.5f).Should().OnlyContain(l => l == 0, "above every distance");
    }

    [Fact]
    public void Clustering_AFixedCountOverridesTheThreshold()
    {
        var embeddings = Rows([1, 0], [0.9f, 0.1f], [0, 1], [-1, 0]);

        AgglomerativeClustering.Cluster(embeddings, 2, threshold: 0.001f, numClusters: 1).Should().OnlyContain(l => l == 0);
        AgglomerativeClustering.Cluster(embeddings, 2, threshold: 5f, numClusters: 3).Distinct().Should().HaveCount(3);
    }

    [Fact]
    public void Clustering_IsCompleteLinkage()
    {
        // A chain a–b–c where neighbours are close but a and c are far: single linkage would join all three at 0.3,
        // complete linkage keeps {a,b} and c apart until the a–c distance.
        static float[] At(double deg) => [(float)Math.Cos(deg * Math.PI / 180), (float)Math.Sin(deg * Math.PI / 180)];
        var embeddings = Rows(At(0), At(40), At(80)); // 1 − cos 40° ≈ 0.234, 1 − cos 80° ≈ 0.826

        AgglomerativeClustering.Cluster(embeddings, 2, threshold: 0.5f).Distinct().Should().HaveCount(2);
    }

    [Fact]
    public void ToTurns_MergesShortGapsDropsShortTurnsAndNumbersBySpeakingOrder()
    {
        // 270 samples per frame at 16 kHz ≈ 16.9 ms. Speaker 2 speaks first, then speaker 0 with a 0.2 s pause.
        var labels = new int[400, 3];
        for (var f = 10; f < 100; f++) labels[f, 2] = 1;   // ~1.5 s
        for (var f = 120; f < 200; f++) labels[f, 0] = 1;
        for (var f = 212; f < 300; f++) labels[f, 0] = 1;  // gap of 12 frames ≈ 0.2 s → merged
        for (var f = 350; f < 360; f++) labels[f, 1] = 1;  // ~0.17 s → dropped

        var turns = SpeakerDiarizer.ToTurns(labels, 3);

        turns.Should().HaveCount(2);
        turns[0].Speaker.Should().Be(0, "the first voice heard is numbered first");
        turns[1].Speaker.Should().Be(1);
        turns[1].End.Should().BeGreaterThan(turns[1].Start + 2.5, "the two parts of the second voice merged");
    }

    private static TranscriptionSegment Segment(double start, double end) => new() { Start = start, End = end, Text = "x" };

    [Fact]
    public void Assign_TakesTheLargestOverlapThenTheNearestTurn()
    {
        var turns = new[] { new SpeakerTurn(0, 4, 0), new SpeakerTurn(4, 10, 1), new SpeakerTurn(20, 25, 0) };
        var segments = new[] { Segment(0.5, 3), Segment(3, 6), Segment(12, 13), Segment(18, 19.5) };

        var labelled = DiarizationStage.Assign(segments, turns);

        labelled.Select(s => s.Speaker).Should().Equal("S1", "S2", "S2", "S1");
        labelled[1].Text.Should().Be("x");
    }

    [Fact]
    public void Assign_WithNoTurnsLeavesSpeakersNull()
    {
        DiarizationStage.Assign([Segment(0, 1)], []).Single().Speaker.Should().BeNull();
    }

    [Fact]
    public void Options_AreValidated()
    {
        var zero = () => DiarizationStage.Validate(new TranscribeOptions { Diarize = true, NumSpeakers = 0 });
        var threshold = () => DiarizationStage.Validate(new TranscribeOptions { Diarize = true, SpeakerThreshold = 2.5f });
        var streaming = () => DiarizationStage.RejectOnStreaming(new TranscribeOptions { Diarize = true });

        zero.Should().Throw<ArgumentOutOfRangeException>();
        threshold.Should().Throw<ArgumentOutOfRangeException>();
        streaming.Should().Throw<NotSupportedException>();
        DiarizationStage.RejectOnStreaming(new TranscribeOptions());
    }
}
