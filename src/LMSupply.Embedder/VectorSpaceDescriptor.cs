using System.Security.Cryptography;
using System.Text;

namespace LMSupply.Embedder;

/// <summary>
/// What the loader actually did on the way from text to a vector — every decision that changes the
/// numbers a model id produces — and the opaque revision string derived from it
/// (<see cref="IEmbeddingModel.VectorSpaceRevision"/>).
/// </summary>
/// <remarks>
/// <para>
/// The revision is a SHA-256 of an explicit canonical string (<see cref="Canonical"/>), never of a
/// runtime hash code, so it is the same across processes, machines and operating systems for the same
/// decisions. Paths are snapshot-relative with <c>/</c> separators for the same reason.
/// </para>
/// <para>
/// Deliberately <b>not</b> part of it: the execution provider and GPU (they change floating-point
/// noise, not the space), the llama-server binary version for a GGUF model (a binary update would
/// otherwise ask every consumer to re-embed), and the model files' own content (the model id names
/// them; a repository that rewrites its weights under the same id is outside what this library can see).
/// </para>
/// </remarks>
/// <param name="Backend"><c>onnx</c> or <c>gguf</c>.</param>
/// <param name="ModelFile">The model file, relative to the cached snapshot root, <c>/</c>-separated
/// (e.g. <c>onnx/model.onnx</c>, <c>onnx/model_fp16.onnx</c>). A quantization variant is a different space.</param>
/// <param name="Tokenizer">The tokenizer's <see cref="Text.ISequenceTokenizer.Signature"/>, or
/// <c>llama-server</c> when tokenization happens inside the server.</param>
/// <param name="Pooling">The pooling applied (ONNX) or sent to the server (GGUF), by name.</param>
/// <param name="Normalize">Whether vectors are L2-normalized.</param>
/// <param name="MaxSequenceLength">The sequence length in effect.</param>
/// <param name="Dimensions">The vector dimension the model produces.</param>
/// <param name="QueryPrefix">The prefix <see cref="IEmbeddingModel.EmbedQueryAsync(string, CancellationToken)"/> prepends, if any.</param>
/// <param name="PassagePrefix">The prefix <see cref="IEmbeddingModel.EmbedPassageAsync(string, CancellationToken)"/> prepends, if any.</param>
internal sealed record VectorSpaceDescriptor(
    string Backend,
    string ModelFile,
    string Tokenizer,
    string Pooling,
    bool Normalize,
    int MaxSequenceLength,
    int Dimensions,
    string? QueryPrefix,
    string? PassagePrefix)
{
    /// <summary>
    /// The epoch of this library's pooling and normalization code — raised when a release changes how
    /// token embeddings become a vector without any tokenizer change. Tokenizer epochs travel inside
    /// <see cref="Tokenizer"/>.
    /// </summary>
    internal const int EmbedderEpoch = 1;

    /// <summary>The format version of <see cref="Canonical"/> itself.</summary>
    internal const int Format = 1;

    /// <summary>
    /// The decisions as one ASCII line. Traced at load, so two revisions can be diffed by eye.
    /// </summary>
    public string Canonical =>
        $"vs/{Format};embedder/{EmbedderEpoch};backend={Backend};model={ModelFile};tokenizer={Tokenizer};" +
        $"pooling={Pooling};normalize={(Normalize ? '1' : '0')};maxseq={MaxSequenceLength};dims={Dimensions};" +
        $"query={Escape(QueryPrefix)};passage={Escape(PassagePrefix)}";

    /// <summary>The opaque revision: the first 16 hex characters of SHA-256 over <see cref="Canonical"/>.</summary>
    public string Revision
    {
        get
        {
            var hash = SHA256.HashData(Encoding.UTF8.GetBytes(Canonical));
            return Convert.ToHexStringLower(hash.AsSpan(0, 8));
        }
    }

    /// <summary>
    /// A model path relative to its snapshot root with <c>/</c> separators, so the same file yields the
    /// same string on every operating system. A path outside <paramref name="root"/> (a local model
    /// file) reduces to its file name.
    /// </summary>
    internal static string RelativeModelFile(string? root, string modelPath)
    {
        string relative;
        if (!string.IsNullOrEmpty(root))
        {
            var full = Path.GetFullPath(modelPath);
            var fullRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
            relative = full.StartsWith(fullRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)
                || full.StartsWith(fullRoot + Path.AltDirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)
                ? full[(fullRoot.Length + 1)..]
                : Path.GetFileName(full);
        }
        else
        {
            relative = Path.GetFileName(modelPath);
        }

        return relative.Replace('\\', '/');
    }

    private static string Escape(string? value) =>
        value is null ? "" : value.Replace("%", "%25").Replace(";", "%3B").Replace("=", "%3D");
}
