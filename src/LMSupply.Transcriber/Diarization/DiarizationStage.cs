namespace LMSupply.Transcriber.Diarization;

/// <summary>
/// Labels transcription segments with speakers. Shared by every transcriber backend: diarization runs on the same
/// samples after transcription, and each segment takes the speaker whose turns overlap it most.
/// </summary>
internal sealed class DiarizationStage : IAsyncDisposable
{
    private readonly TranscriberOptions _options;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private SpeakerDiarizer? _diarizer;

    public DiarizationStage(TranscriberOptions options) => _options = options;

    /// <summary>Throws when <paramref name="options"/> ask for diarization on a call that cannot run it.</summary>
    public static void RejectOnStreaming(TranscribeOptions? options)
    {
        if (options?.Diarize == true)
            throw new NotSupportedException(
                "TranscribeOptions.Diarize needs the whole recording and is not available on TranscribeStreamingAsync. " +
                "Use TranscribeAsync.");
    }

    public static void Validate(TranscribeOptions? options)
    {
        if (options?.NumSpeakers is <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), options.NumSpeakers,
                "TranscribeOptions.NumSpeakers must be positive when set.");
        if (options?.SpeakerThreshold is { } t && (t <= 0 || t >= 2 || float.IsNaN(t)))
            throw new ArgumentOutOfRangeException(nameof(options), t,
                "TranscribeOptions.SpeakerThreshold is a cosine distance and must be in (0, 2).");
    }

    public async Task<TranscriptionResult> ApplyAsync(
        TranscriptionResult result, float[] samples, TranscribeOptions? options, CancellationToken cancellationToken)
    {
        if (options?.Diarize != true)
            return result;

        var diarizer = await GetDiarizerAsync(cancellationToken);
        var turns = diarizer.Diarize(
            samples,
            options.NumSpeakers ?? 0,
            options.SpeakerThreshold ?? SpeakerDiarizer.DefaultThreshold,
            cancellationToken);

        return new TranscriptionResult
        {
            Text = result.Text,
            Language = result.Language,
            LanguageProbability = result.LanguageProbability,
            Segments = Assign(result.Segments, turns),
            DurationSeconds = result.DurationSeconds,
            InferenceTimeMs = result.InferenceTimeMs,
        };
    }

    /// <summary>
    /// Each segment takes the speaker with the largest total overlap; with no overlap, the turn nearest its midpoint.
    /// No turns at all leaves every speaker null.
    /// </summary>
    internal static IReadOnlyList<TranscriptionSegment> Assign(
        IReadOnlyList<TranscriptionSegment> segments, IReadOnlyList<SpeakerTurn> turns)
    {
        if (turns.Count == 0)
            return segments;

        var labelled = new List<TranscriptionSegment>(segments.Count);
        foreach (var segment in segments)
        {
            var overlap = new Dictionary<int, double>();
            foreach (var turn in turns)
            {
                var o = Math.Min(segment.End, turn.End) - Math.Max(segment.Start, turn.Start);
                if (o > 0)
                    overlap[turn.Speaker] = overlap.GetValueOrDefault(turn.Speaker) + o;
            }

            int speaker;
            if (overlap.Count > 0)
            {
                speaker = overlap.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).First().Key;
            }
            else
            {
                var mid = (segment.Start + segment.End) / 2;
                speaker = turns
                    .OrderBy(t => Math.Min(Math.Abs(mid - t.Start), Math.Abs(mid - t.End)))
                    .First().Speaker;
            }

            labelled.Add(new TranscriptionSegment
            {
                Id = segment.Id,
                Start = segment.Start,
                End = segment.End,
                Text = segment.Text,
                AvgLogProb = segment.AvgLogProb,
                NoSpeechProb = segment.NoSpeechProb,
                CompressionRatio = segment.CompressionRatio,
                Words = segment.Words,
                Speaker = $"S{speaker + 1}",
            });
        }
        return labelled;
    }

    private async Task<SpeakerDiarizer> GetDiarizerAsync(CancellationToken cancellationToken)
    {
        if (_diarizer is { } ready)
            return ready;
        await _gate.WaitAsync(cancellationToken);
        try
        {
            return _diarizer ??= await SpeakerDiarizer.LoadAsync(
                _options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory(),
                _options.DisableAutoDownload,
                _options.Provider,
                cancellationToken);
        }
        finally
        {
            _gate.Release();
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_diarizer is not null)
            await _diarizer.DisposeAsync();
        _gate.Dispose();
    }
}
