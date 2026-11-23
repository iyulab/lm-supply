using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

namespace LMSupply.Transcriber;

/// <summary>
/// A loaded <see cref="ITranscriberModel"/> as a Microsoft.Extensions.AI <see cref="ISpeechToTextClient"/>, so a library that
/// takes the standard contract uses a local Whisper or Parakeet model with no adapter of its own. Create it with
/// <see cref="TranscriberSpeechToTextClientExtensions.AsSpeechToTextClient"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Experimental.</b> Microsoft.Extensions.AI marks its speech-to-text contract experimental (<c>MEAI001</c>), and so is
/// this client: a caller suppresses the same diagnostic, and the shape here follows that contract if it changes.
/// </para>
/// <para>
/// The client does not own the model: disposing it leaves the model loaded, and the model is disposed by whoever loaded
/// it. <see cref="SpeechToTextOptions.ModelId"/>, when set, must name this model — there is no other model to route to.
/// </para>
/// <para>
/// <b>Input.</b> The stream holds a whole encoded recording (WAV, MP3, Opus and the other containers the transcriber
/// reads); its sample rate is read from the container, so <see cref="SpeechToTextOptions.SpeechSampleRate"/> is not used.
/// </para>
/// <para>
/// <b>Options.</b> <see cref="SpeechToTextOptions.SpeechLanguage"/> is the language hint of
/// <see cref="TranscribeOptions.Language"/> (none: the language is identified). <see cref="SpeechToTextOptions.TextLanguage"/>
/// may be the speech language, or English, which turns on Whisper's translation to English; any other text language is
/// refused with a <see cref="NotSupportedException"/> — a text in another language is never silently returned. Options with
/// no standard name — speaker labelling (<see cref="TranscribeOptions.Diarize"/>, <see cref="TranscribeOptions.NumSpeakers"/>,
/// <see cref="TranscribeOptions.MaxSpeakers"/>), an initial prompt, the decoding controls — come from
/// <see cref="SpeechToTextOptions.RawRepresentationFactory"/> returning a <see cref="TranscribeOptions"/>, which is then used
/// as it is, with the standard options applied over it. Without one the client asks for segment timestamps, because a
/// recording longer than one 30-second window would otherwise come back as one segment per window ending where the
/// window stops.
/// </para>
/// <para>
/// <b>Response.</b> One <see cref="TextContent"/> per transcribed segment, in order, so
/// <see cref="SpeechToTextResponse.Text"/> reads as the transcript. Each carries its place on the recording's own time axis
/// and, when speaker labelling was asked for, its speaker, in <see cref="AIContent.AdditionalProperties"/> under
/// <see cref="SegmentStartSecondsKey"/>, <see cref="SegmentEndSecondsKey"/> and <see cref="SpeakerKey"/> (a label stable
/// within the recording, <c>"S1"</c>, <c>"S2"</c>, … — it names a voice, not a person), and keeps the library's segment as its
/// raw representation. The response carries the span of the segments (<see cref="SpeechToTextResponse.StartTime"/>,
/// <see cref="SpeechToTextResponse.EndTime"/>), the model id, and under <see cref="LanguageKey"/>,
/// <see cref="LanguageProbabilityKey"/> and <see cref="DurationSecondsKey"/> the language of the transcript, how sure the
/// identification was (absent when a hint was given) and the recording's length; the library's own result is its raw
/// representation. Word-level times are not available (the transcriber does not align words).
/// </para>
/// <para>
/// <b>Streaming.</b> The recording is transcribed whole and the updates are its segments in order, one
/// <see cref="SpeechToTextResponseUpdateKind.TextUpdated"/> update each with the same properties as above: the stream form
/// is the standard shape for a caller that handles segments one by one, not a way to get text before the model has
/// finished.
/// </para>
/// <para>
/// <b>Services.</b> <see cref="GetService"/> hands out the <see cref="SpeechToTextClientMetadata"/> and the model under any
/// type it implements.
/// </para>
/// </remarks>
[Experimental("MEAI001")]
public sealed class LocalSpeechToTextClient : ISpeechToTextClient
{
    /// <summary>The key, in a segment's additional properties, of its start in seconds from the start of the recording (a <see cref="double"/>).</summary>
    public const string SegmentStartSecondsKey = "start_seconds";

    /// <summary>The key, in a segment's additional properties, of its end in seconds from the start of the recording (a <see cref="double"/>).</summary>
    public const string SegmentEndSecondsKey = "end_seconds";

    /// <summary>The key, in a segment's additional properties, of its speaker label (a <see cref="string"/>); absent without speaker labelling.</summary>
    public const string SpeakerKey = "speaker";

    /// <summary>The key, in a response's additional properties, of the language of the transcript (a <see cref="string"/> code).</summary>
    public const string LanguageKey = "language";

    /// <summary>The key, in a response's additional properties, of the probability of the identified language (a <see cref="float"/>); absent when the language was given.</summary>
    public const string LanguageProbabilityKey = "language_probability";

    /// <summary>The key, in a response's additional properties, of the recording's length in seconds (a <see cref="double"/>).</summary>
    public const string DurationSecondsKey = "duration_seconds";

    private readonly ITranscriberModel _model;

    internal LocalSpeechToTextClient(ITranscriberModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
        Metadata = new SpeechToTextClientMetadata(providerName: "LMSupply", providerUri: null, defaultModelId: model.ModelId);
    }

    /// <summary>The transcriber this client answers with.</summary>
    public ITranscriberModel Model => _model;

    /// <summary>What the client says about itself: the provider and the model it answers with.</summary>
    public SpeechToTextClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public async Task<SpeechToTextResponse> GetTextAsync(
        Stream audioSpeechStream,
        SpeechToTextOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audioSpeechStream);
        var transcribe = ToTranscribeOptions(options);
        var result = await _model.TranscribeAsync(audioSpeechStream, transcribe, cancellationToken).ConfigureAwait(false);
        return ToResponse(result);
    }

    /// <inheritdoc/>
    public IAsyncEnumerable<SpeechToTextResponseUpdate> GetStreamingTextAsync(
        Stream audioSpeechStream,
        SpeechToTextOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audioSpeechStream);
        var transcribe = ToTranscribeOptions(options); // refused now, not on the first MoveNext
        return StreamAsync(audioSpeechStream, transcribe, cancellationToken);
    }

    private async IAsyncEnumerable<SpeechToTextResponseUpdate> StreamAsync(
        Stream audioSpeechStream,
        TranscribeOptions transcribe,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var result = await _model.TranscribeAsync(audioSpeechStream, transcribe, cancellationToken).ConfigureAwait(false);
        var previous = default(TranscriptionSegment);
        foreach (var segment in result.Segments)
        {
            var update = new SpeechToTextResponseUpdate(new List<AIContent> { SegmentContent(segment, previous) })
            {
                Kind = SpeechToTextResponseUpdateKind.TextUpdated,
                ModelId = _model.ModelId,
                StartTime = TimeSpan.FromSeconds(segment.Start),
                EndTime = TimeSpan.FromSeconds(segment.End),
                RawRepresentation = segment,
            };
            previous = segment;
            yield return update;
        }
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
            return null;

        if (serviceType == typeof(SpeechToTextClientMetadata))
            return Metadata;
        if (serviceType.IsInstanceOfType(this))
            return this;
        if (serviceType.IsInstanceOfType(_model))
            return _model;
        return null;
    }

    /// <summary>Does nothing: the client does not own the model.</summary>
    public void Dispose()
    {
    }

    private TranscribeOptions ToTranscribeOptions(SpeechToTextOptions? options)
    {
        if (options?.ModelId is { Length: > 0 } requested &&
            !string.Equals(requested, _model.ModelId, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"This client answers with model '{_model.ModelId}'; the request names '{requested}'. " +
                "Load the other model and wrap it in a client of its own.",
                nameof(options));
        }

        var transcribe = options?.RawRepresentationFactory?.Invoke(this) as TranscribeOptions
            ?? new TranscribeOptions { WordTimestamps = true };

        var speech = string.IsNullOrWhiteSpace(options?.SpeechLanguage) ? null : options!.SpeechLanguage!.Trim();
        if (speech is not null)
            transcribe.Language = speech;

        var text = string.IsNullOrWhiteSpace(options?.TextLanguage) ? null : options!.TextLanguage!.Trim();
        if (text is not null)
        {
            if (IsEnglish(text))
            {
                // English text from English speech is a plain transcript; from anything else (or from speech whose
                // language is not given) it is Whisper's translation task.
                if (speech is null || !IsEnglish(speech))
                    transcribe.Translate = true;
            }
            else if (speech is null || !SameLanguage(speech, text))
            {
                throw new NotSupportedException(
                    $"The text language '{text}' cannot be produced: the transcriber returns the language that is spoken, " +
                    "or English by translation. Leave TextLanguage unset, or set it to the speech language or English.");
            }
        }

        return transcribe;
    }

    private SpeechToTextResponse ToResponse(TranscriptionResult result)
    {
        var contents = new List<AIContent>(result.Segments.Count);
        TranscriptionSegment? previous = null;
        foreach (var segment in result.Segments)
        {
            contents.Add(SegmentContent(segment, previous));
            previous = segment;
        }

        var response = new SpeechToTextResponse(contents)
        {
            ModelId = _model.ModelId,
            RawRepresentation = result,
            AdditionalProperties = new AdditionalPropertiesDictionary
            {
                [LanguageKey] = result.Language,
            },
        };
        if (result.LanguageProbability is { } probability)
            response.AdditionalProperties[LanguageProbabilityKey] = probability;
        if (result.DurationSeconds is { } duration)
            response.AdditionalProperties[DurationSecondsKey] = duration;
        if (result.Segments.Count > 0)
        {
            response.StartTime = TimeSpan.FromSeconds(result.Segments[0].Start);
            response.EndTime = TimeSpan.FromSeconds(result.Segments[^1].End);
        }

        return response;
    }

    // The text of a segment as it joins the one before it: segments of a multilingual Whisper model begin with a space and
    // those of a model that does not are joined by one, so SpeechToTextResponse.Text (the contents, run together) reads as
    // the transcript.
    private static TextContent SegmentContent(TranscriptionSegment segment, TranscriptionSegment? previous)
    {
        var text = segment.Text;
        if (previous is not null && text.Length > 0 && !char.IsWhiteSpace(text[0]) &&
            previous.Text.Length > 0 && !char.IsWhiteSpace(previous.Text[^1]))
        {
            text = " " + text;
        }

        var content = new TextContent(text)
        {
            RawRepresentation = segment,
            AdditionalProperties = new AdditionalPropertiesDictionary
            {
                [SegmentStartSecondsKey] = segment.Start,
                [SegmentEndSecondsKey] = segment.End,
            },
        };
        if (segment.Speaker is { } speaker)
            content.AdditionalProperties[SpeakerKey] = speaker;
        return content;
    }

    private static string PrimarySubtag(string language)
    {
        var end = language.IndexOfAny(['-', '_']);
        return (end < 0 ? language : language[..end]).Trim();
    }

    private static bool IsEnglish(string language) =>
        string.Equals(PrimarySubtag(language), "en", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(language, "english", StringComparison.OrdinalIgnoreCase);

    private static bool SameLanguage(string a, string b) =>
        string.Equals(PrimarySubtag(a), PrimarySubtag(b), StringComparison.OrdinalIgnoreCase);
}

/// <summary>Creates a Microsoft.Extensions.AI <see cref="ISpeechToTextClient"/> from a loaded transcriber.</summary>
[Experimental("MEAI001")]
public static class TranscriberSpeechToTextClientExtensions
{
    /// <summary>
    /// The transcriber as a Microsoft.Extensions.AI <see cref="ISpeechToTextClient"/>. The client does not own the model;
    /// dispose the model when it is no longer used.
    /// </summary>
    /// <param name="model">A loaded transcriber.</param>
    public static LocalSpeechToTextClient AsSpeechToTextClient(this ITranscriberModel model) => new(model);
}
