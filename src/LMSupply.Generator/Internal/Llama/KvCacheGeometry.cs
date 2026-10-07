using LMSupply.Generator.Gguf;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// What llama.cpp allocates for the KV cache of a GGUF model, derived from the file's attention metadata
/// the way llama.cpp sizes it: only layers that keep a cache count (hybrid recurrent layers and layers that
/// share an earlier layer's cache do not), each with its own KV head count and per-head K/V dimensions
/// (grouped-query attention keeps far fewer KV heads than attention heads). Sliding-window layers hold a
/// window of cells instead of the whole context, so they cost a fixed amount rather than a per-token one.
/// </summary>
/// <param name="FullKElementsPerToken">K elements per context token across full-attention KV layers.</param>
/// <param name="FullVElementsPerToken">V elements per context token across full-attention KV layers.</param>
/// <param name="SwaKElementsPerCell">K elements per cell across sliding-window layers.</param>
/// <param name="SwaVElementsPerCell">V elements per cell across sliding-window layers.</param>
/// <param name="SlidingWindow">The sliding-window width in tokens, when sliding-window layers exist.</param>
internal sealed record KvCacheGeometry(
    long FullKElementsPerToken,
    long FullVElementsPerToken,
    long SwaKElementsPerCell,
    long SwaVElementsPerCell,
    int? SlidingWindow)
{
    /// <summary>llama.cpp pads the sliding-window cell count to this multiple.</summary>
    private const int CellPadding = 256;

    /// <summary>llama.cpp's default physical batch (<c>--ubatch-size</c>).</summary>
    internal const int DefaultUBatch = 512;

    /// <summary>
    /// The geometry from GGUF metadata, or null when the file does not carry enough of it (no layer count,
    /// or no way to derive a head dimension) — callers then fall back to a file-size heuristic.
    /// </summary>
    /// <remarks>
    /// Unknown layer kinds are counted as full-attention layers, so a model whose sliding-window pattern
    /// the file does not record is over-estimated, never under-estimated.
    /// </remarks>
    public static KvCacheGeometry? FromMetadata(GgufMetadata? metadata)
    {
        if (metadata?.LayerCount is not { } layers || layers <= 0)
            return null;

        int? derivedHeadDim = metadata.EmbeddingLength is > 0 && metadata.HeadCount is > 0
            ? metadata.EmbeddingLength.Value / metadata.HeadCount.Value
            : null;
        var keyLength = metadata.KeyLength ?? derivedHeadDim;
        if (keyLength is not > 0)
            return null;
        var valueLength = metadata.ValueLength ?? keyLength.Value;
        var keyLengthSwa = metadata.KeyLengthSwa ?? keyLength.Value;
        var valueLengthSwa = metadata.ValueLengthSwa ?? valueLength;

        var sharedTail = Math.Clamp(metadata.SharedKvLayers ?? 0, 0, layers);
        var interval = metadata.FullAttentionInterval ?? 0;
        var perLayerKv = metadata.HeadCountKvPerLayer;
        var pattern = metadata.SlidingWindowPattern;
        var hasWindow = metadata.SlidingWindow is > 0;

        long fullK = 0, fullV = 0, swaK = 0, swaV = 0;
        for (var layer = 0; layer < layers - sharedTail; layer++)
        {
            // Hybrid models: only every interval-th layer is attention; the rest keep recurrent state.
            if (interval > 1 && (layer + 1) % interval != 0)
                continue;

            var kvHeads = perLayerKv is not null && layer < perLayerKv.Count
                ? perLayerKv[layer]
                : metadata.HeadCountKv ?? metadata.HeadCount ?? 0;
            if (kvHeads <= 0)
                continue;

            var slidingLayer = hasWindow && pattern is not null && layer < pattern.Count && pattern[layer];
            if (slidingLayer)
            {
                swaK += (long)kvHeads * keyLengthSwa;
                swaV += (long)kvHeads * valueLengthSwa;
            }
            else
            {
                fullK += (long)kvHeads * keyLength.Value;
                fullV += (long)kvHeads * valueLength;
            }
        }

        return new KvCacheGeometry(fullK, fullV, swaK, swaV, swaK + swaV > 0 ? metadata.SlidingWindow : null);
    }

    /// <summary>Bytes the cache grows by per context token, for the given K/V cache types.</summary>
    public long BytesPerToken(string? cacheTypeK, string? cacheTypeV)
        => (long)Math.Ceiling(
            FullKElementsPerToken * BytesPerElement(cacheTypeK) + FullVElementsPerToken * BytesPerElement(cacheTypeV));

    /// <summary>
    /// Bytes the sliding-window layers hold for a context of <paramref name="contextLength"/> tokens —
    /// llama.cpp keeps min(context, pad(window × sequences + ubatch)) cells for them.
    /// </summary>
    public long SlidingWindowBytes(int contextLength, string? cacheTypeK, string? cacheTypeV, int sequences, int ubatch)
    {
        if (SlidingWindow is not > 0 || SwaKElementsPerCell + SwaVElementsPerCell == 0)
            return 0;

        var cells = (long)SlidingWindow.Value * Math.Max(1, sequences) + Math.Max(1, ubatch);
        cells = (cells + CellPadding - 1) / CellPadding * CellPadding;
        cells = Math.Min(cells, contextLength);
        return (long)Math.Ceiling(
            cells * (SwaKElementsPerCell * BytesPerElement(cacheTypeK) + SwaVElementsPerCell * BytesPerElement(cacheTypeV)));
    }

    /// <summary>Total KV cache bytes for a context of <paramref name="contextLength"/> tokens.</summary>
    public long TotalBytes(int contextLength, string? cacheTypeK, string? cacheTypeV, int sequences, int ubatch)
        => BytesPerToken(cacheTypeK, cacheTypeV) * contextLength
           + SlidingWindowBytes(contextLength, cacheTypeK, cacheTypeV, sequences, ubatch);

    /// <summary>
    /// Bytes per element of a llama.cpp cache type (<c>--cache-type-k/-v</c> value); f16 when unset or
    /// unknown. Block-quantized types include their per-block scale.
    /// </summary>
    internal static double BytesPerElement(string? cacheType) => cacheType?.ToLowerInvariant() switch
    {
        "f32" => 4.0,
        "q8_0" => 34.0 / 32,
        "q4_0" or "iq4_nl" => 18.0 / 32,
        "q4_1" => 20.0 / 32,
        "q5_0" => 22.0 / 32,
        "q5_1" => 24.0 / 32,
        _ => 2.0, // f16, bf16, unset
    };
}
