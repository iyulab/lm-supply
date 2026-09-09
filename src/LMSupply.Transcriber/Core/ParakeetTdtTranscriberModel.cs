using System.Diagnostics;
using System.Runtime.CompilerServices;
using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Inference;
using LMSupply.Transcriber.Audio;
using LMSupply.Transcriber.Decoding;
using LMSupply.Transcriber.Internal;
using LMSupply.Transcriber.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Transcriber.Core;

/// <summary>
/// NVIDIA Parakeet TDT(Token-and-Duration Transducer) 계열의 <see cref="ITranscriberModel"/> — onnx-asr이 내보낸
/// 세 ONNX(<c>nemo128.onnx</c> mel 전처리 · Conformer encoder · prediction-network+joint)와 SentencePiece 어휘로 돈다.
/// Whisper 경로(<see cref="OnnxTranscriberModel"/>)와 아무것도 공유하지 않는다: 창 길이, 특징 추출, 디코딩, 토크나이저가 전부 다르다.
/// </summary>
/// <remarks>
/// 시범 구현(spike)의 의도적 한계: 60 s 창을 겹침 없이 잇는다, 언어 식별 출력이 없어 <see cref="TranscriptionResult.Language"/>는
/// 힌트 또는 <c>"und"</c>다, 번역·단어 타임스탬프·빔 서치는 지원하지 않는다. 자동 선택(<c>"auto"</c>) 후보에는 들어가지 않는다 —
/// 별칭이나 저장소 id를 명시해야만 이 경로가 열린다.
/// </remarks>
internal sealed class ParakeetTdtTranscriberModel : ITranscriberModel
{
    private const string PreprocessorFile = "nemo128.onnx";
    private const string VocabFile = "vocab.txt";
    private const string ConfigFile = "config.json";

    /// <summary><see cref="AudioProcessor"/>가 모든 입력을 16 kHz mono로 맞춘다 — NeMo 전처리기가 기대하는 것과 같다.</summary>
    private const int SampleRate = 16000;

    /// <summary>NeMo 전처리 hop 10 ms; 인코더 프레임 1개 = hop × subsampling.</summary>
    private const double HopSeconds = 0.01;
    private const int DefaultSubsamplingFactor = 8;
    private const int LstmLayers = 2;
    private const int LstmHidden = 640;

    /// <summary>한 번에 인코더에 넣는 최대 길이. 겹침 없는 단순 분할 — 창 경계에서 단어가 잘릴 수 있다(spike 한계).</summary>
    private static readonly int s_windowSamples = AudioProcessor.SecondsToSamples(60);

    /// <summary>prediction network에는 항상 마지막 토큰 하나만 넣는다(이력은 LSTM 상태가 든다).</summary>
    private static readonly int[] s_singleTargetLength = [1];

    private readonly TranscriberOptions _options;
    private readonly TranscriberModelInfo _modelInfo;
    private readonly SemaphoreSlim _initLock = new(1, 1);
    private readonly ProviderBlacklist _providerBlacklist = new();

    private RecoverableOnnxSession? _preprocessor;
    private RecoverableOnnxSession? _encoder;
    private RecoverableOnnxSession? _decoderJoint;
    private SentencePieceVocabulary? _vocab;
    private int _hiddenSize;
    private int _numMelBins;
    private double _frameSeconds = HopSeconds * DefaultSubsamplingFactor;
    private bool _initialized;
    private bool _disposed;

    public ParakeetTdtTranscriberModel(TranscriberOptions options, TranscriberModelInfo modelInfo)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _modelInfo = modelInfo ?? throw new ArgumentNullException(nameof(modelInfo));
        _hiddenSize = modelInfo.HiddenSize;
        _numMelBins = modelInfo.NumMelBins;
    }

    public string ModelId => _modelInfo.Id;

    /// <summary>TDT 내보내기는 언어 식별 출력이 없다 — 전사마다 힌트 또는 <c>"und"</c>.</summary>
    public string? Language => null;

    public bool IsGpuActive => _encoder?.IsGpuActive ?? false;

    public IReadOnlyList<string> ActiveProviders => _encoder?.ActiveProviders ?? [];

    public ExecutionProvider RequestedProvider => _options.Provider;

    public long? EstimatedMemoryBytes => _modelInfo.SizeBytes > 0 ? _modelInfo.SizeBytes * 2 : null;

    public Task WarmupAsync(CancellationToken cancellationToken = default) => EnsureInitializedAsync(cancellationToken);

    public TranscriberModelInfo? GetModelInfo() => _modelInfo;

    public async Task<TranscriptionResult> TranscribeAsync(string audioPath, TranscribeOptions? options = null, CancellationToken cancellationToken = default)
        => await TranscribeSamplesAsync(await AudioProcessor.LoadAudioAsync(audioPath, cancellationToken), options, cancellationToken);

    public async Task<TranscriptionResult> TranscribeAsync(Stream audioStream, TranscribeOptions? options = null, CancellationToken cancellationToken = default)
        => await TranscribeSamplesAsync(await AudioProcessor.LoadAudioAsync(audioStream, cancellationToken), options, cancellationToken);

    public async Task<TranscriptionResult> TranscribeAsync(byte[] audioData, TranscribeOptions? options = null, CancellationToken cancellationToken = default)
        => await TranscribeSamplesAsync(await AudioProcessor.LoadAudioAsync(audioData, cancellationToken), options, cancellationToken);

    public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(
        string audioPath,
        TranscribeOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ValidateOptions(options);
        await EnsureInitializedAsync(cancellationToken);
        var samples = await AudioProcessor.LoadAudioAsync(audioPath, cancellationToken);

        var nextId = 0;
        foreach (var (chunk, offsetSeconds) in Windows(samples))
        {
            foreach (var segment in await TranscribeWindowAsync(chunk, offsetSeconds, cancellationToken))
            {
                yield return WithId(segment, nextId++);
            }
        }
    }

    private async Task<TranscriptionResult> TranscribeSamplesAsync(float[] samples, TranscribeOptions? options, CancellationToken cancellationToken)
    {
        ValidateOptions(options);
        await EnsureInitializedAsync(cancellationToken);

        var stopwatch = Stopwatch.StartNew();
        var segments = new List<TranscriptionSegment>();
        foreach (var (chunk, offsetSeconds) in Windows(samples))
        {
            foreach (var segment in await TranscribeWindowAsync(chunk, offsetSeconds, cancellationToken))
                segments.Add(WithId(segment, segments.Count));
        }
        stopwatch.Stop();

        return new TranscriptionResult
        {
            Text = string.Join(" ", segments.Select(s => s.Text)).Trim(),
            Language = options?.Language ?? "und",
            LanguageProbability = null,
            Segments = segments,
            DurationSeconds = AudioProcessor.GetDurationSeconds(samples),
            InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds
        };
    }

    private static void ValidateOptions(TranscribeOptions? options)
    {
        if (options?.Translate == true)
            throw new NotSupportedException("Parakeet TDT models transcribe only; translation to English is not supported by this model family.");
    }

    private static IEnumerable<(float[] Chunk, double OffsetSeconds)> Windows(float[] samples)
    {
        if (samples.Length == 0)
            yield break;

        for (var start = 0; start < samples.Length; start += s_windowSamples)
        {
            var length = Math.Min(s_windowSamples, samples.Length - start);
            var chunk = new float[length];
            Array.Copy(samples, start, chunk, 0, length);
            yield return (chunk, (double)start / SampleRate);
        }
    }

    private static TranscriptionSegment WithId(TranscriptionSegment s, int id) => new()
    {
        Id = id,
        Start = s.Start,
        End = s.End,
        Text = s.Text,
        AvgLogProb = s.AvgLogProb
    };

    // ── pipeline ────────────────────────────────────────────────────────────────

    private async Task<List<TranscriptionSegment>> TranscribeWindowAsync(float[] chunk, double offsetSeconds, CancellationToken cancellationToken)
    {
        var (features, featureFrames, featureLength) = await RunPreprocessorAsync(chunk, cancellationToken);
        var (encoded, encodedFrames, encodedLength) = await RunEncoderAsync(features, featureFrames, featureLength, cancellationToken);
        var tokens = await RunDecoderAsync(encoded, encodedFrames, encodedLength, cancellationToken);
        return BuildSegments(tokens, offsetSeconds);
    }

    private Task<(float[] Features, int Frames, long Length)> RunPreprocessorAsync(float[] chunk, CancellationToken cancellationToken)
    {
        var waveforms = new DenseTensor<float>(chunk, [1, chunk.Length]);
        var lens = new DenseTensor<long>(new long[] { chunk.Length }, [1]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("waveforms", waveforms),
            NamedOnnxValue.CreateFromTensor("waveforms_lens", lens)
        };

        return _preprocessor!.RunWithRecoveryAsync((session, runOptions) =>
        {
            using var results = session.Run(inputs, session.OutputNames, runOptions);
            var features = results.First(r => r.Name == "features").AsTensor<float>();
            var length = results.First(r => r.Name == "features_lens").AsTensor<long>()[0];
            return (features.ToArray(), features.Dimensions[2], length);
        }, cancellationToken: cancellationToken);
    }

    private Task<(float[] Encoded, int Frames, int Length)> RunEncoderAsync(float[] features, int frames, long featureLength, CancellationToken cancellationToken)
    {
        var signal = new DenseTensor<float>(features, [1, _numMelBins, frames]);
        var length = new DenseTensor<long>(new[] { featureLength }, [1]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio_signal", signal),
            NamedOnnxValue.CreateFromTensor("length", length)
        };

        return _encoder!.RunWithRecoveryAsync((session, runOptions) =>
        {
            using var results = session.Run(inputs, session.OutputNames, runOptions);
            var outputs = results.First(r => r.Name == "outputs").AsTensor<float>();   // [1, hidden, T'] channels-first
            var encodedLength = (int)results.First(r => r.Name == "encoded_lengths").AsTensor<long>()[0];
            return (outputs.ToArray(), outputs.Dimensions[2], encodedLength);
        }, cancellationToken: cancellationToken);
    }

    private Task<List<TdtToken>> RunDecoderAsync(float[] encoded, int encodedFrames, int encodedLength, CancellationToken cancellationToken)
    {
        var vocab = _vocab!;
        var hidden = _hiddenSize;
        var stateSize = LstmLayers * LstmHidden;

        // 전체 greedy 루프를 세션 Run 하나 안에서 돈다 — 복구(provider fallback)는 창 단위로 한 번만 적용된다.
        return _decoderJoint!.RunWithRecoveryAsync((session, runOptions) =>
            TdtGreedyDecoder.Decode(
                Math.Min(encodedLength, encodedFrames),
                vocab.Size,
                vocab.BlankId,
                stateSize,
                (frame, lastToken, stateH, stateC) =>
                {
                    var column = new float[hidden];
                    for (var c = 0; c < hidden; c++)
                        column[c] = encoded[c * encodedFrames + frame];

                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("encoder_outputs", new DenseTensor<float>(column, [1, hidden, 1])),
                        NamedOnnxValue.CreateFromTensor("targets", new DenseTensor<int>(new[] { lastToken }, [1, 1])),
                        NamedOnnxValue.CreateFromTensor("target_length", new DenseTensor<int>(s_singleTargetLength, [1])),
                        NamedOnnxValue.CreateFromTensor("input_states_1", new DenseTensor<float>(stateH, [LstmLayers, 1, LstmHidden])),
                        NamedOnnxValue.CreateFromTensor("input_states_2", new DenseTensor<float>(stateC, [LstmLayers, 1, LstmHidden]))
                    };

                    using var results = session.Run(inputs, session.OutputNames, runOptions);
                    var logits = results.First(r => r.Name == "outputs").AsTensor<float>().ToArray();
                    var h = results.First(r => r.Name == "output_states_1").AsTensor<float>().ToArray();
                    var c2 = results.First(r => r.Name == "output_states_2").AsTensor<float>().ToArray();
                    return new TdtJointOutput(logits, h, c2);
                }),
            cancellationToken: cancellationToken);
    }

    private List<TranscriptionSegment> BuildSegments(List<TdtToken> tokens, double offsetSeconds)
    {
        var segments = new List<TranscriptionSegment>();
        if (tokens.Count == 0)
            return segments;

        var vocab = _vocab!;
        var start = 0;
        for (var i = 0; i < tokens.Count; i++)
        {
            var piece = vocab.Piece(tokens[i].Id).TrimEnd();
            var sentenceEnd = piece.Length > 0 && piece[^1] is '.' or '?' or '!';
            if (sentenceEnd || i == tokens.Count - 1)
            {
                var slice = tokens.GetRange(start, i - start + 1);
                var text = vocab.Decode(slice.Select(t => t.Id));
                if (text.Length > 0)
                {
                    segments.Add(new TranscriptionSegment
                    {
                        Start = offsetSeconds + slice[0].Frame * _frameSeconds,
                        End = offsetSeconds + (slice[^1].Frame + 1) * _frameSeconds,
                        Text = text,
                        AvgLogProb = slice.Average(t => t.LogProb)
                    });
                }
                start = i + 1;
            }
        }

        return segments;
    }

    // ── initialization ───────────────────────────────────────────────────────────

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_initialized) return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_initialized) return;

            var modelDir = await ResolveModelDirectoryAsync(cancellationToken);

            var config = NemoConfigReader.ReadConfig(modelDir);
            if (config is { IsTdt: false })
                throw new InvalidDataException($"'{modelDir}/config.json' declares model_type '{config.ModelType}', not '{NemoModelConfig.ConformerTdt}'.");
            if (config?.FeaturesSize is { } mels) _numMelBins = mels;
            _frameSeconds = HopSeconds * (config?.SubsamplingFactor ?? DefaultSubsamplingFactor);

            var preprocessorPath = Require(Path.Combine(modelDir, PreprocessorFile));
            var encoderPath = Require(Path.Combine(modelDir, _modelInfo.EncoderFile));
            var decoderPath = Require(Path.Combine(modelDir, _modelInfo.DecoderFile));
            _vocab = await SentencePieceVocabulary.LoadAsync(Require(Path.Combine(modelDir, VocabFile)), cancellationToken);

            // 전처리기는 STFT 연산자 때문에 CPU 전용(onnx-asr도 CUDA/TensorRT에서 CPU로 돌린다).
            var pre = await OnnxSessionFactory.CreateWithInfoAsync(preprocessorPath, ExecutionProvider.Cpu, ConfigureSessionOptions, cancellationToken: cancellationToken);
            _preprocessor = RecoverableOnnxSession.FromResult(pre, preprocessorPath, ConfigureSessionOptions, logPrefix: "[ParakeetTdt:preprocessor]");

            var enc = await OnnxSessionFactory.CreateWithInfoAsync(encoderPath, _options.Provider, ConfigureSessionOptions, cancellationToken: cancellationToken);
            _encoder = RecoverableOnnxSession.FromResult(enc, encoderPath, ConfigureSessionOptions, logPrefix: "[ParakeetTdt:encoder]", blacklist: _providerBlacklist);

            var dec = await OnnxSessionFactory.CreateWithInfoAsync(decoderPath, _options.Provider, ConfigureSessionOptions, cancellationToken: cancellationToken);
            _decoderJoint = RecoverableOnnxSession.FromResult(dec, decoderPath, ConfigureSessionOptions, logPrefix: "[ParakeetTdt:decoder_joint]", blacklist: _providerBlacklist);

            Trace.TraceInformation($"[ParakeetTdt] Loaded {_modelInfo.Id} — encoder providers: {string.Join(",", _encoder.ActiveProviders)}, vocab {_vocab.Size}, frame {_frameSeconds:F3}s");
            _initialized = true;
        }
        finally
        {
            _initLock.Release();
        }
    }

    private async Task<string> ResolveModelDirectoryAsync(CancellationToken cancellationToken)
    {
        if (Directory.Exists(_modelInfo.Id))
            return _modelInfo.Id;

        var cacheDir = _options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir);
        // 파일을 명시한다 — 이 저장소는 저장소 루트에 두 양자화 변형을 나란히 두므로 discovery의 whisper용 «onnx 하위 폴더» 규칙이 맞지 않는다.
        return await downloader.DownloadModelAsync(
            _modelInfo.Id,
            files: [_modelInfo.EncoderFile, _modelInfo.DecoderFile, PreprocessorFile, VocabFile, ConfigFile],
            cancellationToken: cancellationToken);
    }

    private static string Require(string path)
        => File.Exists(path) ? path : throw new FileNotFoundException($"Parakeet TDT model file not found: {path}");

    private void ConfigureSessionOptions(SessionOptions options)
    {
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.EnableMemoryPattern = true;
        options.EnableCpuMemArena = true;
        if (_options.ThreadCount is { } threads && threads > 0)
        {
            options.IntraOpNumThreads = threads;
            options.InterOpNumThreads = threads;
        }
    }

    public ValueTask DisposeAsync()
    {
        if (_disposed) return ValueTask.CompletedTask;
        _disposed = true;
        _preprocessor?.Dispose();
        _encoder?.Dispose();
        _decoderJoint?.Dispose();
        _initLock.Dispose();
        return ValueTask.CompletedTask;
    }
}
