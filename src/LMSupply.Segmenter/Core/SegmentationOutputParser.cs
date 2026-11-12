using LMSupply.Segmenter.Models;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Segmenter.Core;

/// <summary>
/// Turns a segmentation model's output tensors into a class map and a confidence map, according to
/// <see cref="SegmenterOutputKind"/>.
/// </summary>
internal static class SegmentationOutputParser
{
    // A query's class counts toward a pixel only when its probability is at least this; a trained query is almost always
    // decided, so what is dropped is a fraction of a percent of the score.
    private const float MinQueryClassProbability = 0.01f;

    public static (int[] ClassMap, float[] ConfidenceMap, int Width, int Height) Parse(
        SegmenterOutputKind kind, Tensor<float> first, Tensor<float>? second)
    {
        return kind switch
        {
            SegmenterOutputKind.ClassLogits => ParseClassLogits(first),
            SegmenterOutputKind.ForegroundAlpha => ParseForegroundAlpha(first),
            SegmenterOutputKind.QueryMasks => ParseQueryMasks(
                first, second ?? throw new InvalidOperationException("A query-based model has two outputs.")),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown output kind."),
        };
    }

    // [1, classes, height, width] of scores: argmax over the classes; the confidence is a sigmoid of the winning score.
    private static (int[], float[], int, int) ParseClassLogits(Tensor<float> output)
    {
        var dims = output.Dimensions.ToArray();
        var numClasses = dims[1];
        var height = dims[2];
        var width = dims[3];

        var classMap = new int[width * height];
        var confidenceMap = new float[width * height];

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var pixelIndex = y * width + x;
                var maxScore = float.MinValue;
                var bestClass = 0;

                for (var c = 0; c < numClasses; c++)
                {
                    var score = output[0, c, y, x];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        bestClass = c;
                    }
                }

                classMap[pixelIndex] = bestClass;
                confidenceMap[pixelIndex] = Sigmoid(maxScore);
            }
        }

        return (classMap, confidenceMap, width, height);
    }

    // One channel of foreground probability: above one half is class 1; the confidence is the probability of the class chosen.
    private static (int[], float[], int, int) ParseForegroundAlpha(Tensor<float> output)
    {
        var dims = output.Dimensions.ToArray();
        var channelsLast = dims.Length == 4 && dims[3] == 1 && dims[1] != 1;
        var height = channelsLast ? dims[1] : dims[^2];
        var width = channelsLast ? dims[2] : dims[^1];

        var classMap = new int[width * height];
        var confidenceMap = new float[width * height];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var alpha = channelsLast ? output[0, y, x, 0] : output[0, 0, y, x];
                var pixelIndex = y * width + x;
                classMap[pixelIndex] = alpha > 0.5f ? 1 : 0;
                confidenceMap[pixelIndex] = alpha > 0.5f ? alpha : 1f - alpha;
            }
        }

        return (classMap, confidenceMap, width, height);
    }

    // Queries and masks: the score of class c at a pixel is the sum over queries of (the query's probability of c) times
    // (the sigmoid of its mask at the pixel); the class with the highest score wins and "no object" is not a class.
    private static (int[], float[], int, int) ParseQueryMasks(Tensor<float> classQueries, Tensor<float> maskQueries)
    {
        var classDims = classQueries.Dimensions.ToArray();
        var maskDims = maskQueries.Dimensions.ToArray();
        var queries = classDims[1];
        var numClasses = classDims[2] - 1;
        var height = maskDims[2];
        var width = maskDims[3];
        var pixels = width * height;

        var logits = classQueries.ToArray();
        var masks = maskQueries.ToArray();

        // The (query, class, probability) triples that count, and the distinct classes among them.
        var entries = new List<(int Query, int Class, float Probability)>();
        var candidateIndex = new Dictionary<int, int>();
        var candidates = new List<int>();
        for (var q = 0; q < queries; q++)
        {
            var row = q * (numClasses + 1);
            var max = float.MinValue;
            for (var c = 0; c <= numClasses; c++)
            {
                max = Math.Max(max, logits[row + c]);
            }

            var sum = 0f;
            for (var c = 0; c <= numClasses; c++)
            {
                sum += MathF.Exp(logits[row + c] - max);
            }

            for (var c = 0; c < numClasses; c++)
            {
                var probability = MathF.Exp(logits[row + c] - max) / sum;
                if (probability < MinQueryClassProbability)
                {
                    continue;
                }

                entries.Add((q, c, probability));
                if (!candidateIndex.ContainsKey(c))
                {
                    candidateIndex[c] = candidates.Count;
                    candidates.Add(c);
                }
            }
        }

        var classMap = new int[pixels];
        var confidenceMap = new float[pixels];
        if (candidates.Count == 0)
        {
            return (classMap, confidenceMap, width, height);
        }

        var scores = new float[candidates.Count * pixels];
        var sigmoidOfQuery = new float[pixels];
        var lastQuery = -1;
        foreach (var (query, cls, probability) in entries)
        {
            if (query != lastQuery)
            {
                var offset = query * pixels;
                for (var i = 0; i < pixels; i++)
                {
                    sigmoidOfQuery[i] = Sigmoid(masks[offset + i]);
                }

                lastQuery = query;
            }

            var target = candidateIndex[cls] * pixels;
            for (var i = 0; i < pixels; i++)
            {
                scores[target + i] += probability * sigmoidOfQuery[i];
            }
        }

        for (var i = 0; i < pixels; i++)
        {
            var best = 0;
            var bestScore = float.MinValue;
            for (var k = 0; k < candidates.Count; k++)
            {
                var score = scores[k * pixels + i];
                if (score > bestScore)
                {
                    bestScore = score;
                    best = k;
                }
            }

            classMap[i] = candidates[best];
            confidenceMap[i] = Math.Clamp(bestScore, 0f, 1f);
        }

        return (classMap, confidenceMap, width, height);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
