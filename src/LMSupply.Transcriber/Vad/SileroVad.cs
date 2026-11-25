using LMSupply.Exceptions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Transcriber.Vad;

/// <summary>
/// The Silero VAD model (v5, ONNX): the probability that each 32 ms window of 16 kHz audio holds speech, from a recurrent
/// state carried window to window. The weights are fetched on first use into the shared cache.
/// </summary>
internal sealed class SileroVad : IAsyncDisposable
{
    internal const string Repo = "onnx-community/silero-vad";
    internal const string ModelFile = "onnx/model.onnx";

    /// <summary>Bytes of the model file, as the repository lists it (2026-11-24).</summary>
    internal const long DownloadSizeBytes = 2_243_022;

    private const int ContextSamples = 64;
    private const int StateLayers = 2;
    private const int StateSize = 128;

    private readonly InferenceSession _session;

    private SileroVad(InferenceSession session) => _session = session;

    /// <summary>True when the file would be opened from the cache without a request. Never touches the network.</summary>
    public static async Task<bool> IsCachedAsync(string cacheDirectory, CancellationToken cancellationToken)
    {
        try
        {
            await FetchAsync(cacheDirectory, localFilesOnly: true, progress: null, cancellationToken);
            return true;
        }
        catch (ModelNotFoundException)
        {
            return false;
        }
    }

    /// <summary>
    /// Downloads the file into the cache (a no-op when cached); with <paramref name="localFilesOnly"/> only confirms it is
    /// cached and throws when it is not, making no request.
    /// </summary>
    public static Task DownloadAsync(
        string cacheDirectory, bool localFilesOnly, IProgress<DownloadProgress>? progress, CancellationToken cancellationToken)
        => FetchAsync(cacheDirectory, localFilesOnly, progress, cancellationToken);

    private static async Task<string> FetchAsync(
        string cacheDirectory, bool localFilesOnly, IProgress<DownloadProgress>? progress, CancellationToken cancellationToken)
    {
        using var downloader = new HuggingFaceDownloader(cacheDirectory, localFilesOnly: localFilesOnly);
        var directory = await downloader.DownloadModelAsync(Repo, [ModelFile], progress: progress, cancellationToken: cancellationToken);
        return Path.Combine(directory, ModelFile);
    }

    public static async Task<SileroVad> LoadAsync(
        string cacheDirectory, bool localFilesOnly, ExecutionProvider provider,
        IProgress<DownloadProgress>? progress, CancellationToken cancellationToken)
    {
        var path = await FetchAsync(cacheDirectory, localFilesOnly, progress, cancellationToken);

        // A 2 MB recurrent model run one 32 ms window at a time: a GPU only adds a transfer per window, so Auto is the CPU.
        var session = await OnnxSessionFactory.CreateAsync(
            path, provider == ExecutionProvider.Auto ? ExecutionProvider.Cpu : provider,
            configureOptions: null, progress, cancellationToken);
        return new SileroVad(session);
    }

    /// <summary>
    /// The speech probability of every 512-sample window of <paramref name="samples"/> (16 kHz mono); a last partial window
    /// is padded with silence, so speech at the very end is not lost.
    /// </summary>
    public float[] Probabilities(float[] samples, CancellationToken cancellationToken)
    {
        var windows = (samples.Length + SpeechSegmenter.WindowSamples - 1) / SpeechSegmenter.WindowSamples;
        var probabilities = new float[windows];
        var state = new float[StateLayers * 1 * StateSize];
        var context = new float[ContextSamples];
        var sampleRate = new DenseTensor<long>(new long[] { SpeechSegmenter.SampleRate }, Array.Empty<int>());

        for (var w = 0; w < windows; w++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var input = new float[ContextSamples + SpeechSegmenter.WindowSamples];
            Array.Copy(context, input, ContextSamples);
            var offset = w * SpeechSegmenter.WindowSamples;
            var take = Math.Min(SpeechSegmenter.WindowSamples, samples.Length - offset);
            Array.Copy(samples, offset, input, ContextSamples, take);
            Array.Copy(input, SpeechSegmenter.WindowSamples, context, 0, ContextSamples);

            using var outputs = _session.Run(
            [
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(input, [1, input.Length])),
                NamedOnnxValue.CreateFromTensor("state", new DenseTensor<float>(state, [StateLayers, 1, StateSize])),
                NamedOnnxValue.CreateFromTensor("sr", sampleRate),
            ]);

            foreach (var output in outputs)
            {
                if (output.Name == "output")
                    probabilities[w] = output.AsTensor<float>().GetValue(0);
                else if (output.Name == "stateN")
                    output.AsTensor<float>().ToArray().CopyTo(state, 0);
            }
        }

        return probabilities;
    }

    public ValueTask DisposeAsync()
    {
        _session.Dispose();
        return ValueTask.CompletedTask;
    }
}
