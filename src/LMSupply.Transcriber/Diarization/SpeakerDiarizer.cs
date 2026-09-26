using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Transcriber.Diarization;

/// <summary>A span of audio attributed to one speaker (0-based, stable within one recording).</summary>
internal readonly record struct SpeakerTurn(double Start, double End, int Speaker);

/// <summary>
/// Offline speaker diarization: pyannote segmentation-3.0 over 10 s windows (1 s step), one speaker embedding per
/// (window, local speaker), complete-linkage clustering, then per-frame aggregation. The same pipeline as sherpa-onnx's
/// offline pyannote diarization.
/// </summary>
internal sealed class SpeakerDiarizer : IAsyncDisposable
{
    internal const string SegmentationRepo = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0";
    internal const string SegmentationFile = "model.onnx";
    internal const string EmbeddingRepo = "csukuangfj/speaker-embedding-models";
    internal const string EmbeddingFile = "wespeaker_en_voxceleb_resnet34_LM.onnx";

    private const int SampleRate = 16000;
    private const int WindowSize = 160000;          // 10 s — segmentation model metadata
    private const int WindowShift = WindowSize / 10; // 1 s
    private const int ReceptiveFieldShift = 270;
    private const int ReceptiveFieldSize = 991;
    private const int LocalSpeakers = 3;
    private const int MinActiveFrames = 10;
    private const double MinDurationOn = 0.3;
    private const double MinDurationOff = 0.5;

    /// <summary>Default cosine-distance cut for the WeSpeaker ResNet34 embeddings.</summary>
    public const float DefaultThreshold = 0.5f;

    // Powerset classes of segmentation-3.0: silence, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}.
    private static readonly int[][] s_powerset = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]];

    private readonly InferenceSession _segmentation;
    private readonly InferenceSession _embedding;
    private readonly KaldiFbank _fbank = new();

    private SpeakerDiarizer(InferenceSession segmentation, InferenceSession embedding)
    {
        _segmentation = segmentation;
        _embedding = embedding;
    }

    public static async Task<SpeakerDiarizer> LoadAsync(
        string cacheDirectory, bool localFilesOnly, ExecutionProvider provider, CancellationToken cancellationToken)
    {
        using var downloader = new HuggingFaceDownloader(cacheDirectory, localFilesOnly: localFilesOnly);
        var segDir = await downloader.DownloadModelAsync(SegmentationRepo, [SegmentationFile], cancellationToken: cancellationToken);
        var embDir = await downloader.DownloadModelAsync(EmbeddingRepo, [EmbeddingFile], cancellationToken: cancellationToken);

        var segmentation = await OnnxSessionFactory.CreateAsync(Path.Combine(segDir, SegmentationFile), provider, cancellationToken: cancellationToken);
        try
        {
            var embedding = await OnnxSessionFactory.CreateAsync(Path.Combine(embDir, EmbeddingFile), provider, cancellationToken: cancellationToken);
            return new SpeakerDiarizer(segmentation, embedding);
        }
        catch
        {
            segmentation.Dispose();
            throw;
        }
    }

    /// <summary>Speaker turns of 16 kHz mono <paramref name="samples"/>, ordered by start.</summary>
    public IReadOnlyList<SpeakerTurn> Diarize(float[] samples, int numSpeakers = 0, float threshold = DefaultThreshold,
        CancellationToken cancellationToken = default)
    {
        if (samples.Length == 0)
            return [];

        // 1. Segmentation → per-window multi-label activity [frames, 3].
        var chunkStarts = ChunkStarts(samples.Length);
        var labels = new List<int[,]>(chunkStarts.Count);
        foreach (var start in chunkStarts)
        {
            cancellationToken.ThrowIfCancellationRequested();
            labels.Add(Segment(samples, start));
        }

        if (labels.Count == 1)
            return ToTurns(Trim(labels[0], samples.Length), speakerCount: LocalSpeakers);

        var speakersPerFrame = SpeakersPerFrame(labels);
        if (speakersPerFrame.All(c => c == 0))
            return [];

        // 2. One embedding per (window, local speaker) with enough non-overlapped activity.
        var pairs = new List<(int Chunk, int Speaker)>();
        var embeddings = new List<float[]>();
        for (var c = 0; c < labels.Count; c++)
        {
            var label = labels[c];
            var frames = label.GetLength(0);
            for (var s = 0; s < LocalSpeakers; s++)
            {
                var active = new bool[frames];
                var count = 0;
                for (var f = 0; f < frames; f++)
                {
                    var overlapped = label[f, 0] + label[f, 1] + label[f, 2] >= 2;
                    active[f] = label[f, s] == 1 && !overlapped;
                    if (active[f])
                        count++;
                }
                if (count < MinActiveFrames)
                    continue;

                cancellationToken.ThrowIfCancellationRequested();
                var audio = SpeakerAudio(samples, chunkStarts[c], active, frames);
                var embedding = Embed(audio);
                if (embedding.Any(float.IsNaN))
                    continue;
                pairs.Add((c, s));
                embeddings.Add(embedding);
            }
        }

        if (embeddings.Count == 0)
            return [];

        // 3. Cluster embeddings into global speakers.
        var dim = embeddings[0].Length;
        var flat = new float[embeddings.Count * dim];
        for (var i = 0; i < embeddings.Count; i++)
            embeddings[i].CopyTo(flat, i * dim);
        var clusters = AgglomerativeClustering.Cluster(flat, dim, threshold, numSpeakers);
        var globalSpeakers = clusters.Max() + 1;
        var map = new Dictionary<(int, int), int>();
        for (var i = 0; i < pairs.Count; i++)
            map[pairs[i]] = clusters[i];

        // 4. Aggregate relabelled windows into a global frame × speaker count; keep the top-k per frame.
        var totalFrames = (WindowSize + (labels.Count - 1) * WindowShift) / ReceptiveFieldShift + 1;
        var counts = new int[totalFrames, globalSpeakers];
        for (var c = 0; c < labels.Count; c++)
        {
            var offset = (int)((float)c * WindowShift / ReceptiveFieldShift + 0.5f);
            var label = labels[c];
            for (var s = 0; s < LocalSpeakers; s++)
            {
                if (!map.TryGetValue((c, s), out var g))
                    continue;
                for (var f = 0; f < label.GetLength(0) && offset + f < totalFrames; f++)
                    counts[offset + f, g] += label[f, s];
            }
        }

        var usable = totalFrames;
        if ((samples.Length - WindowSize) % WindowShift > 0)
            usable = Math.Min(totalFrames, samples.Length / ReceptiveFieldShift + 1);

        var final = new int[usable, globalSpeakers];
        for (var f = 0; f < usable; f++)
        {
            var k = speakersPerFrame[Math.Min(f, speakersPerFrame.Length - 1)];
            if (k == 0)
                continue;
            foreach (var g in TopK(counts, f, globalSpeakers, k))
                final[f, g] = 1;
        }

        return ToTurns(final, globalSpeakers);
    }

    private static List<int> ChunkStarts(int n)
    {
        var starts = new List<int>();
        if (n <= WindowSize)
        {
            starts.Add(0);
            return starts;
        }
        var full = (n - WindowSize) / WindowShift + 1;
        for (var i = 0; i < full; i++)
            starts.Add(i * WindowShift);
        if ((n - WindowSize) % WindowShift > 0)
            starts.Add(full * WindowShift);
        return starts;
    }

    private int[,] Segment(float[] samples, int start)
    {
        var buffer = new float[WindowSize];
        var length = Math.Min(WindowSize, samples.Length - start);
        Array.Copy(samples, start, buffer, 0, length);

        var input = new DenseTensor<float>(buffer, [1, 1, WindowSize]);
        using var results = _segmentation.Run([NamedOnnxValue.CreateFromTensor(_segmentation.InputMetadata.Keys.First(), input)]);
        var output = results[0].AsTensor<float>();
        var frames = output.Dimensions[1];
        var classes = output.Dimensions[2];

        var label = new int[frames, LocalSpeakers];
        for (var f = 0; f < frames; f++)
        {
            var best = 0;
            for (var k = 1; k < classes; k++)
                if (output[0, f, k] > output[0, f, best])
                    best = k;
            foreach (var s in s_powerset[best])
                label[f, s] = 1;
        }
        return label;
    }

    private static int[] SpeakersPerFrame(List<int[,]> labels)
    {
        var totalFrames = (WindowSize + (labels.Count - 1) * WindowShift) / ReceptiveFieldShift + 1;
        var count = new float[totalFrames];
        var weight = new float[totalFrames];
        for (var c = 0; c < labels.Count; c++)
        {
            var offset = (int)((float)c * WindowShift / ReceptiveFieldShift + 0.5f);
            var label = labels[c];
            for (var f = 0; f < label.GetLength(0) && offset + f < totalFrames; f++)
            {
                count[offset + f] += label[f, 0] + label[f, 1] + label[f, 2];
                weight[offset + f] += 1;
            }
        }
        var result = new int[totalFrames];
        for (var f = 0; f < totalFrames; f++)
            result[f] = (int)(count[f] / (weight[f] + 1e-12f) + 0.5f);
        return result;
    }

    private static float[] SpeakerAudio(float[] samples, int chunkStart, bool[] active, int frames)
    {
        var parts = new List<float>();
        var f = 0;
        while (f < frames)
        {
            if (!active[f])
            {
                f++;
                continue;
            }
            var begin = f;
            while (f < frames && active[f])
                f++;
            var end = f < frames ? f : frames - 1;
            var from = chunkStart + (int)((float)begin / frames * WindowSize);
            var to = Math.Min(samples.Length, chunkStart + (int)((float)end / frames * WindowSize));
            for (var i = from; i < to; i++)
                parts.Add(samples[i]);
        }
        return parts.ToArray();
    }

    private float[] Embed(float[] audio)
    {
        var features = _fbank.Compute(audio);
        var frames = features.Length / _fbank.NumBins;
        if (frames == 0)
            return [float.NaN];
        var input = new DenseTensor<float>(features, [1, frames, _fbank.NumBins]);
        using var results = _embedding.Run([NamedOnnxValue.CreateFromTensor(_embedding.InputMetadata.Keys.First(), input)]);
        return results[0].AsTensor<float>().ToArray();
    }

    private static IEnumerable<int> TopK(int[,] counts, int frame, int speakers, int k)
    {
        var order = Enumerable.Range(0, speakers).OrderByDescending(g => counts[frame, g]).ThenBy(g => g);
        return order.Take(k);
    }

    private static int[,] Trim(int[,] label, int numSamples)
    {
        if ((numSamples - WindowSize) % WindowShift <= 0 && numSamples >= WindowSize)
            return label;
        var keep = Math.Min(label.GetLength(0), numSamples / ReceptiveFieldShift);
        var trimmed = new int[keep, label.GetLength(1)];
        for (var f = 0; f < keep; f++)
            for (var s = 0; s < label.GetLength(1); s++)
                trimmed[f, s] = label[f, s];
        return trimmed;
    }

    /// <summary>Frame labels → turns: frame times, merging same-speaker gaps below 0.5 s, dropping turns under 0.3 s.</summary>
    internal static IReadOnlyList<SpeakerTurn> ToTurns(int[,] labels, int speakerCount)
    {
        const double scale = (double)ReceptiveFieldShift / SampleRate;
        const double offset = 0.5 * ReceptiveFieldSize / SampleRate;
        var frames = labels.GetLength(0);
        var turns = new List<SpeakerTurn>();

        for (var s = 0; s < Math.Min(speakerCount, labels.GetLength(1)); s++)
        {
            var own = new List<(double Start, double End)>();
            var activeStart = labels[0, s] > 0 ? 0 : -1;
            for (var f = 1; f < frames; f++)
            {
                if (activeStart >= 0 && labels[f, s] == 0)
                {
                    own.Add((activeStart * scale + offset, f * scale + offset));
                    activeStart = -1;
                }
                else if (activeStart < 0 && labels[f, s] == 1)
                {
                    activeStart = f;
                }
            }
            if (activeStart >= 0)
                own.Add((activeStart * scale + offset, (frames - 1) * scale + offset));

            for (var i = 0; i + 1 < own.Count;)
            {
                if (own[i + 1].Start - own[i].End < MinDurationOff)
                {
                    own[i] = (own[i].Start, own[i + 1].End);
                    own.RemoveAt(i + 1);
                }
                else
                {
                    i++;
                }
            }

            foreach (var (start, end) in own)
            {
                if (end - start <= MinDurationOn)
                    continue;
                turns.Add(new SpeakerTurn(start, end, s));
            }
        }

        // Number speakers by first appearance so labels read S1, S2, … in speaking order.
        turns.Sort((a, b) => a.Start.CompareTo(b.Start));
        var order = new Dictionary<int, int>();
        for (var i = 0; i < turns.Count; i++)
        {
            if (!order.TryGetValue(turns[i].Speaker, out var id))
                order[turns[i].Speaker] = id = order.Count;
            turns[i] = turns[i] with { Speaker = id };
        }
        return turns;
    }

    public ValueTask DisposeAsync()
    {
        _segmentation.Dispose();
        _embedding.Dispose();
        return ValueTask.CompletedTask;
    }
}
