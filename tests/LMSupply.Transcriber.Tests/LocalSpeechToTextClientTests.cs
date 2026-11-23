#pragma warning disable MEAI001 // Microsoft.Extensions.AI's speech-to-text contract is experimental; so is the client under test.

using AwesomeAssertions;
using LMSupply.Transcriber.Models;
using Microsoft.Extensions.AI;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// The client's mapping between the standard speech-to-text contract and the transcriber, against a transcriber that
/// answers with a fixed result and records what it was asked. What a real model does with a recording is in
/// <see cref="LocalSpeechToTextClientModelTests"/>.
/// </summary>
public class LocalSpeechToTextClientTests
{
    private static TranscriptionSegment Seg(int id, double start, double end, string text, string? speaker = null) =>
        new() { Id = id, Start = start, End = end, Text = text, Speaker = speaker };

    private static TranscriptionResult Result(params TranscriptionSegment[] segments) => new()
    {
        Text = string.Concat(segments.Select(s => s.Text)).Trim(),
        Language = "en",
        LanguageProbability = 0.97f,
        DurationSeconds = 12.5,
        Segments = segments,
    };

    private static LocalSpeechToTextClient ClientOver(FakeModel model) => model.AsSpeechToTextClient();

    private static MemoryStream Audio() => new([1, 2, 3]);

    [Fact]
    public async Task EachSegmentIsOneContent_InOrder_WithItsPlaceOnTheRecording_AndTheTextReadsAsTheTranscript()
    {
        var model = new FakeModel(Result(
            Seg(0, 0.0, 3.2, " Welcome to the meeting.", "S1"),
            Seg(1, 3.2, 7.9, " Thanks for coming.", "S2"),
            Seg(2, 7.9, 12.5, " Let us begin.", "S1")));

        var response = await ClientOver(model).GetTextAsync(Audio(), cancellationToken: TestContext.Current.CancellationToken);

        response.Contents.Should().HaveCount(3);
        response.Text.Should().Be(" Welcome to the meeting. Thanks for coming. Let us begin.");
        response.Text.Trim().Should().Be(model.Result.Text);
        var second = response.Contents[1].Should().BeOfType<TextContent>().Subject;
        second.AdditionalProperties![LocalSpeechToTextClient.SegmentStartSecondsKey].Should().Be(3.2);
        second.AdditionalProperties[LocalSpeechToTextClient.SegmentEndSecondsKey].Should().Be(7.9);
        second.AdditionalProperties[LocalSpeechToTextClient.SpeakerKey].Should().Be("S2");
        second.RawRepresentation.Should().BeSameAs(model.Result.Segments[1]);
        response.StartTime.Should().Be(TimeSpan.Zero);
        response.EndTime.Should().Be(TimeSpan.FromSeconds(12.5));
        response.ModelId.Should().Be("fake-model");
        response.RawRepresentation.Should().BeSameAs(model.Result);
        response.AdditionalProperties![LocalSpeechToTextClient.LanguageKey].Should().Be("en");
        response.AdditionalProperties[LocalSpeechToTextClient.LanguageProbabilityKey].Should().Be(0.97f);
        response.AdditionalProperties[LocalSpeechToTextClient.DurationSecondsKey].Should().Be(12.5);
    }

    [Fact]
    public async Task WithoutSpeakerLabelling_NoSpeakerKey_AndAGivenLanguageHasNoProbability()
    {
        var model = new FakeModel(new TranscriptionResult
        {
            Text = "hej",
            Language = "sv",
            Segments = [Seg(0, 0, 1, " hej")],
        });

        var response = await ClientOver(model).GetTextAsync(Audio(), new SpeechToTextOptions { SpeechLanguage = "sv" },
            TestContext.Current.CancellationToken);

        response.Contents.Single().AdditionalProperties!.Should().NotContainKey(LocalSpeechToTextClient.SpeakerKey);
        response.AdditionalProperties!.Should().NotContainKey(LocalSpeechToTextClient.LanguageProbabilityKey);
        response.AdditionalProperties.Should().NotContainKey(LocalSpeechToTextClient.DurationSecondsKey);
    }

    /// <summary>Parakeet's segments have no leading space and its transcript joins them with one.</summary>
    [Fact]
    public async Task SegmentsWithoutALeadingSpace_AreJoinedByOne()
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, "one two"), Seg(1, 1, 2, "three")));

        var response = await ClientOver(model).GetTextAsync(Audio(), cancellationToken: TestContext.Current.CancellationToken);

        response.Text.Should().Be("one two three");
    }

    [Fact]
    public async Task NoSegments_IsAnEmptyResponseWithNoTimes()
    {
        var model = new FakeModel(new TranscriptionResult { Text = "", Language = "en", Segments = [] });

        var response = await ClientOver(model).GetTextAsync(Audio(), cancellationToken: TestContext.Current.CancellationToken);

        response.Contents.Should().BeEmpty();
        response.Text.Should().BeEmpty();
        response.StartTime.Should().BeNull();
        response.EndTime.Should().BeNull();
    }

    [Fact]
    public async Task ByDefault_SegmentTimestampsAreAskedFor()
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a")));

        await ClientOver(model).GetTextAsync(Audio(), cancellationToken: TestContext.Current.CancellationToken);

        model.LastOptions.Should().NotBeNull();
        model.LastOptions!.WordTimestamps.Should().BeTrue("a long recording would otherwise be one segment per window");
        model.LastOptions.Language.Should().BeNull("no hint means the language is identified");
        model.LastOptions.Translate.Should().BeFalse();
    }

    [Fact]
    public async Task TheStreamTheCallerHolds_IsTheStreamTheModelReads()
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a")));
        using var audio = Audio();

        await ClientOver(model).GetTextAsync(audio, cancellationToken: TestContext.Current.CancellationToken);

        model.LastStream.Should().BeSameAs(audio);
    }

    [Theory]
    [InlineData("ko")]
    [InlineData("ko-KR")]
    [InlineData("korean")]
    public async Task TheSpeechLanguage_IsTheTranscribersHint(string hint)
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a")));

        await ClientOver(model).GetTextAsync(Audio(), new SpeechToTextOptions { SpeechLanguage = hint }, TestContext.Current.CancellationToken);

        model.LastOptions!.Language.Should().Be(hint, "the transcriber resolves the spelling of a hint");
    }

    [Theory]
    [InlineData("ko", "ko", false)]
    [InlineData("ko-KR", "ko", false)]
    [InlineData("en", "en", false)]
    [InlineData("en-US", "english", false)]
    [InlineData("ko", "en", true)]
    [InlineData("ko", "en-US", true)]
    [InlineData(null, "en", true)]
    public async Task TheTextLanguage_IsTheSpeechLanguage_OrEnglishByTranslation(string? speech, string text, bool translates)
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a")));

        await ClientOver(model).GetTextAsync(Audio(), new SpeechToTextOptions { SpeechLanguage = speech, TextLanguage = text },
            TestContext.Current.CancellationToken);

        model.LastOptions!.Translate.Should().Be(translates);
    }

    [Theory]
    [InlineData("ko", "ja")]
    [InlineData(null, "ko")]
    [InlineData("en", "fr")]
    public async Task AnyOtherTextLanguage_IsRefused_NotIgnored(string? speech, string text)
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a")));

        var act = () => ClientOver(model).GetTextAsync(Audio(), new SpeechToTextOptions { SpeechLanguage = speech, TextLanguage = text },
            TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<NotSupportedException>()).Which.Message.Should().Contain(text);
        model.Calls.Should().Be(0, "nothing was transcribed for an answer that could not be given");
    }

    [Fact]
    public async Task AModelIdThatIsNotThisModel_IsRefused_AndTheNameOfTheModelIsAccepted()
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a")));
        var client = ClientOver(model);

        var act = () => client.GetTextAsync(Audio(), new SpeechToTextOptions { ModelId = "another-model" }, TestContext.Current.CancellationToken);
        (await act.Should().ThrowAsync<ArgumentException>()).Which.Message.Should().Contain("another-model").And.Contain("fake-model");

        await client.GetTextAsync(Audio(), new SpeechToTextOptions { ModelId = "FAKE-MODEL" }, TestContext.Current.CancellationToken);
        model.Calls.Should().Be(1);
    }

    /// <summary>Speaker labelling has no standard option: it comes in as the library's own options, over which the standard ones apply.</summary>
    [Fact]
    public async Task TheLibrarysOwnOptions_AreUsedAsTheyAre_WithTheStandardOnesOverThem()
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a", "S1")));
        var options = new SpeechToTextOptions
        {
            SpeechLanguage = "ko",
            RawRepresentationFactory = _ => new TranscribeOptions { Diarize = true, MaxSpeakers = 4, Language = "en", InitialPrompt = "Attendees: Meena" },
        };

        await ClientOver(model).GetTextAsync(Audio(), options, TestContext.Current.CancellationToken);

        var used = model.LastOptions!;
        used.Diarize.Should().BeTrue();
        used.MaxSpeakers.Should().Be(4);
        used.InitialPrompt.Should().Be("Attendees: Meena");
        used.Language.Should().Be("ko", "the standard option is applied over the library's");
        used.WordTimestamps.Should().BeFalse("the library's options are used as they are");
    }

    [Fact]
    public async Task Streaming_YieldsTheSegmentsInOrder_EachWithItsOwnTimes()
    {
        var model = new FakeModel(Result(
            Seg(0, 0.0, 3.2, " Welcome.", "S1"),
            Seg(1, 3.2, 7.9, " Thanks.", "S2")));

        var updates = new List<SpeechToTextResponseUpdate>();
        await foreach (var update in ClientOver(model).GetStreamingTextAsync(Audio(), cancellationToken: TestContext.Current.CancellationToken))
            updates.Add(update);

        updates.Should().HaveCount(2);
        updates.Select(u => u.Kind).Should().OnlyContain(k => k == SpeechToTextResponseUpdateKind.TextUpdated);
        updates.Select(u => u.Text).Should().Equal(" Welcome.", " Thanks.");
        updates[1].StartTime.Should().Be(TimeSpan.FromSeconds(3.2));
        updates[1].EndTime.Should().Be(TimeSpan.FromSeconds(7.9));
        updates[1].Contents.Single().AdditionalProperties![LocalSpeechToTextClient.SpeakerKey].Should().Be("S2");
        updates.Select(u => u.ModelId).Should().OnlyContain(m => m == "fake-model");
    }

    [Fact]
    public async Task Streaming_RefusesWhatTheOtherFormRefuses_WhenItIsCalled_NotOnTheFirstMoveNext()
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a")));

        var act = () => ClientOver(model).GetStreamingTextAsync(Audio(), new SpeechToTextOptions { TextLanguage = "ja", SpeechLanguage = "ko" });

        act.Should().Throw<NotSupportedException>();
        await Task.CompletedTask;
    }

    [Fact]
    public async Task ACancelledToken_ReachesTheModel()
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a")));
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        var act = () => ClientOver(model).GetTextAsync(Audio(), cancellationToken: cts.Token);

        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public void Services_AreTheMetadata_TheClient_AndTheModel_AndNothingKeyed()
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a")));
        var client = ClientOver(model);

        client.GetService(typeof(SpeechToTextClientMetadata)).Should().BeOfType<SpeechToTextClientMetadata>()
            .Which.DefaultModelId.Should().Be("fake-model");
        client.Metadata.ProviderName.Should().Be("LMSupply");
        client.GetService(typeof(ISpeechToTextClient)).Should().BeSameAs(client);
        client.GetService(typeof(ITranscriberModel)).Should().BeSameAs(model);
        client.GetService(typeof(string)).Should().BeNull();
        client.GetService(typeof(ITranscriberModel), serviceKey: "k").Should().BeNull();
        client.Model.Should().BeSameAs(model);
    }

    [Fact]
    public async Task DisposingTheClient_LeavesTheModelLoaded()
    {
        var model = new FakeModel(Result(Seg(0, 0, 1, " a")));
        var client = ClientOver(model);

        client.Dispose();
        await client.GetTextAsync(Audio(), cancellationToken: TestContext.Current.CancellationToken);

        model.Disposed.Should().BeFalse();
    }

    private sealed class FakeModel(TranscriptionResult result) : ITranscriberModel
    {
        public TranscriptionResult Result { get; } = result;
        public TranscribeOptions? LastOptions { get; private set; }
        public Stream? LastStream { get; private set; }
        public int Calls { get; private set; }
        public bool Disposed { get; private set; }

        public string ModelId => "fake-model";
        public string? Language => null;
        public bool IsGpuActive => false;
        public IReadOnlyList<string> ActiveProviders => ["CPU"];
        public ExecutionProvider RequestedProvider => ExecutionProvider.Cpu;
        public long? EstimatedMemoryBytes => null;
        public Task WarmupAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public TranscriberModelInfo? GetModelInfo() => null;

        public Task<TranscriptionResult> TranscribeAsync(string audioPath, TranscribeOptions? options = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("The client reads a stream.");

        public Task<TranscriptionResult> TranscribeAsync(Stream audioStream, TranscribeOptions? options = null, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Calls++;
            LastOptions = options;
            LastStream = audioStream;
            return Task.FromResult(Result);
        }

        public Task<TranscriptionResult> TranscribeAsync(byte[] audioData, TranscribeOptions? options = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("The client reads a stream.");

        public IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(string audioPath, TranscribeOptions? options = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("The client reads a stream.");

        public ValueTask DisposeAsync()
        {
            Disposed = true;
            return ValueTask.CompletedTask;
        }
    }
}
