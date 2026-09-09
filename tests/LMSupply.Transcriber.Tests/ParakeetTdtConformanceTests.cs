using System.Diagnostics;
using AwesomeAssertions;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Conformance against the real Parakeet TDT 0.6B v3 int8 export (downloaded on first run, ~670 MB; excluded from CI
/// via <c>Category=Integration</c>). The fixture is a 21.5 s English clip synthesized once, offline, with a local
/// text-to-speech voice from an original sentence and checked in. The Whisper comparison is informational — it prints
/// both transcripts and timings so the spike's A/B numbers live in the test output rather than in someone's memory.
/// </summary>
[Trait("Category", "Integration")]
public sealed class ParakeetTdtConformanceTests
{
    private static readonly string s_fixture = Path.Combine(AppContext.BaseDirectory, "Fixtures", "english-deploy-notice-21s.wav");

    [Fact]
    public async Task EnglishClip_IsTranscribed_WithSentenceSegments_AndNoLanguageClaim()
    {
        var ct = TestContext.Current.CancellationToken;
        var load = Stopwatch.StartNew();
        await using var model = await LocalTranscriber.LoadAsync("parakeet-tdt", cancellationToken: ct);
        load.Stop();

        var result = await model.TranscribeAsync(s_fixture, cancellationToken: ct);

        TestContext.Current.TestOutputHelper?.WriteLine(
            $"parakeet-tdt: load {load.Elapsed.TotalSeconds:F1}s, inference {result.InferenceTimeMs:F0} ms, RTF {result.RealTimeFactor:F3}, providers [{string.Join(",", model.ActiveProviders)}]");
        TestContext.Current.TestOutputHelper?.WriteLine($"text: {result.Text}");
        foreach (var s in result.Segments)
            TestContext.Current.TestOutputHelper?.WriteLine($"  [{s.Id}] {s.Start:F2}-{s.End:F2} ({s.AvgLogProb:F2}) {s.Text}");

        result.Text.Should().ContainEquivalentOf("deployment").And.ContainEquivalentOf("thursday").And.ContainEquivalentOf("release notes");
        result.Language.Should().Be("und", "the TDT export has no language-id output and the call gave no hint");
        result.Segments.Should().NotBeEmpty();
        result.Segments.Should().OnlyContain(s => s.End > s.Start && s.End <= result.DurationSeconds + 0.5);
        result.Segments.Select(s => s.Start).Should().BeInAscendingOrder();
        result.DurationSeconds.Should().BeApproximately(21.5, 0.3);
    }

    [Fact]
    public async Task EnglishClip_WithLanguageHint_EchoesTheHint()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await LocalTranscriber.LoadAsync("parakeet-tdt", cancellationToken: ct);

        var result = await model.TranscribeAsync(s_fixture, new TranscribeOptions { Language = "en" }, ct);

        result.Language.Should().Be("en");
        result.Text.Should().ContainEquivalentOf("deployment");
    }

    [Fact]
    public async Task Translate_IsRejectedUpFront()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await LocalTranscriber.LoadAsync("parakeet-tdt", cancellationToken: ct);

        var act = () => model.TranscribeAsync(s_fixture, new TranscribeOptions { Translate = true }, ct);

        await act.Should().ThrowAsync<NotSupportedException>();
    }

    [Fact]
    public async Task Streaming_YieldsTheSameSegmentsAsBatch()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await LocalTranscriber.LoadAsync("parakeet-tdt", cancellationToken: ct);

        var batch = await model.TranscribeAsync(s_fixture, cancellationToken: ct);
        var streamed = new List<TranscriptionSegment>();
        await foreach (var s in model.TranscribeStreamingAsync(s_fixture, cancellationToken: ct))
            streamed.Add(s);

        streamed.Select(s => s.Text).Should().Equal(batch.Segments.Select(s => s.Text));
    }

    [Fact]
    public async Task AgainstWhisperBaseEnglish_InformationalAB()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var whisper = await LocalTranscriber.LoadAsync("english", cancellationToken: ct);
        await using var parakeet = await LocalTranscriber.LoadAsync("parakeet-tdt", cancellationToken: ct);

        var w = await whisper.TranscribeAsync(s_fixture, cancellationToken: ct);
        var p = await parakeet.TranscribeAsync(s_fixture, cancellationToken: ct);

        var reference = "good morning everyone the deployment window for the reporting service opens at nine o'clock on thursday please merge your pending changes before wednesday evening and make sure the release notes mention the new export format if anything looks wrong after the rollout page the on-call engineer immediately thank you";
        TestContext.Current.TestOutputHelper?.WriteLine($"whisper-base.en : {w.InferenceTimeMs:F0} ms, RTF {w.RealTimeFactor:F3}, word overlap {WordOverlap(w.Text, reference):P0}\n  {w.Text}");
        TestContext.Current.TestOutputHelper?.WriteLine($"parakeet-tdt    : {p.InferenceTimeMs:F0} ms, RTF {p.RealTimeFactor:F3}, word overlap {WordOverlap(p.Text, reference):P0}\n  {p.Text}");

        w.Text.Should().NotBeNullOrWhiteSpace();
        p.Text.Should().NotBeNullOrWhiteSpace();
    }

    private static double WordOverlap(string hypothesis, string reference)
    {
        static HashSet<string> Words(string s) => new(s.ToLowerInvariant().Split([' ', ',', '.', '!', '?'], StringSplitOptions.RemoveEmptyEntries));
        var r = Words(reference);
        var h = Words(hypothesis);
        return r.Count == 0 ? 0 : (double)r.Intersect(h).Count() / r.Count;
    }
}
