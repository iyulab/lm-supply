using Microsoft.Extensions.AI;

namespace LMSupply.Embedder;

/// <summary>Which of the model's input conventions an embedding generator applies to every text it embeds.</summary>
public enum EmbeddingTextKind
{
    /// <summary>
    /// Text with no retrieval role — similarity, clustering, features: the model's <see cref="Utils.ModelInfo.DefaultPrefix"/>
    /// is applied (<see cref="IEmbeddingModel.EmbedAsync(IReadOnlyList{string}, CancellationToken)"/>).
    /// </summary>
    Default = 0,

    /// <summary>A search query: the model's <see cref="Utils.ModelInfo.QueryPrefix"/> is applied (<see cref="IEmbeddingModel.EmbedQueryAsync(IReadOnlyList{string}, CancellationToken)"/>).</summary>
    Query = 1,

    /// <summary>A stored passage or document: the model's <see cref="Utils.ModelInfo.PassagePrefix"/> is applied (<see cref="IEmbeddingModel.EmbedPassageAsync(IReadOnlyList{string}, CancellationToken)"/>).</summary>
    Passage = 2,

    /// <summary>
    /// The text exactly as given, with no prefix (<see cref="IEmbeddingModel.EmbedRawAsync(IReadOnlyList{string}, CancellationToken)"/>),
    /// for texts that already carry the instruction the model expects.
    /// </summary>
    Raw = 3,
}

/// <summary>
/// A loaded <see cref="IEmbeddingModel"/> as a Microsoft.Extensions.AI <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/>,
/// so a library that takes the standard contract uses a local model with no adapter of its own. Create it with
/// <see cref="EmbeddingModelExtensions.AsEmbeddingGenerator"/>.
/// </summary>
/// <remarks>
/// <para>
/// The generator does not own the model: disposing it leaves the model loaded, and the model is disposed by whoever
/// loaded it. <see cref="EmbeddingGenerationOptions.Dimensions"/> is served as a Matryoshka truncation (the full
/// embedding, sliced and re-normalized) and must be between 1 and <see cref="IEmbeddingModel.Dimensions"/>.
/// <see cref="EmbeddingGenerationOptions.ModelId"/>, when set, must name this model: there is no other model to route to.
/// </para>
/// <para>
/// Models trained with prefixes (the E5 family, Nomic) embed best when each input uses its own convention. The default
/// (<see cref="EmbeddingTextKind.Default"/>) applies the model's convention for text with no retrieval role, which is
/// right for similarity and clustering. For retrieval, use one generator per side (<see cref="EmbeddingTextKind.Query"/>
/// for searches, <see cref="EmbeddingTextKind.Passage"/> for what is stored) when the consumer can tell them apart.
/// </para>
/// </remarks>
public sealed class LocalEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly IEmbeddingModel _model;

    internal LocalEmbeddingGenerator(IEmbeddingModel model, EmbeddingTextKind textKind)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (!Enum.IsDefined(textKind))
            throw new ArgumentOutOfRangeException(nameof(textKind), textKind, "Unknown embedding text kind.");

        _model = model;
        TextKind = textKind;
        Metadata = new EmbeddingGeneratorMetadata(
            providerName: "LMSupply",
            providerUri: null,
            defaultModelId: model.ModelId,
            defaultModelDimensions: model.Dimensions);
    }

    /// <summary>The input convention applied to every text.</summary>
    public EmbeddingTextKind TextKind { get; }

    /// <summary>The model this generator embeds with.</summary>
    public IEmbeddingModel Model => _model;

    /// <summary>Provider <c>"LMSupply"</c>, the model id and its native dimension count.</summary>
    public EmbeddingGeneratorMetadata Metadata { get; }

    /// <inheritdoc />
    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(values);

        if (options?.ModelId is { } requestedModel &&
            !string.Equals(requestedModel, _model.ModelId, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"This generator embeds with model '{_model.ModelId}'; the request names '{requestedModel}'. " +
                "Load that model and use its own generator.", nameof(options));
        }

        var texts = values as IReadOnlyList<string> ?? [.. values];
        if (texts.Count == 0)
            return [];

        var dimensions = options?.Dimensions;
        if (dimensions is { } requested && (requested < 1 || requested > _model.Dimensions))
        {
            throw new ArgumentOutOfRangeException(
                nameof(options), requested,
                $"Dimensions must be between 1 and {_model.Dimensions} (model '{_model.ModelId}'): a shorter vector is a " +
                "Matryoshka truncation of the model's own, a longer one cannot be produced.");
        }

        var vectors = (TextKind, dimensions) switch
        {
            (EmbeddingTextKind.Query, null) => await _model.EmbedQueryAsync(texts, cancellationToken).ConfigureAwait(false),
            (EmbeddingTextKind.Query, { } d) => await _model.EmbedQueryAsync(texts, d, cancellationToken).ConfigureAwait(false),
            (EmbeddingTextKind.Passage, null) => await _model.EmbedPassageAsync(texts, cancellationToken).ConfigureAwait(false),
            (EmbeddingTextKind.Passage, { } d) => await _model.EmbedPassageAsync(texts, d, cancellationToken).ConfigureAwait(false),
            (EmbeddingTextKind.Raw, null) => await _model.EmbedRawAsync(texts, cancellationToken).ConfigureAwait(false),
            (EmbeddingTextKind.Raw, { } d) => await _model.EmbedRawAsync(texts, d, cancellationToken).ConfigureAwait(false),
            (_, null) => await _model.EmbedAsync(texts, cancellationToken).ConfigureAwait(false),
            (_, { } d) => await _model.EmbedAsync(texts, d, cancellationToken).ConfigureAwait(false),
        };

        // Results are positional: result[i] belongs to input[i], so a short set cannot be aligned with its inputs.
        if (vectors.Length != texts.Count)
        {
            throw new InvalidOperationException(
                $"Model '{_model.ModelId}' returned {vectors.Length} embedding(s) for {texts.Count} input(s).");
        }

        var createdAt = DateTimeOffset.UtcNow;
        return new GeneratedEmbeddings<Embedding<float>>(
            vectors.Select(v => new Embedding<float>(v) { ModelId = _model.ModelId, CreatedAt = createdAt }));
    }

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
            return null;

        if (serviceType == typeof(EmbeddingGeneratorMetadata))
            return Metadata;
        if (serviceType.IsInstanceOfType(this))
            return this;
        if (serviceType.IsInstanceOfType(_model))
            return _model;
        return null;
    }

    /// <summary>Does nothing: the generator does not own the model (see the remarks on the type).</summary>
    public void Dispose()
    {
    }
}

/// <summary>Microsoft.Extensions.AI interop for <see cref="IEmbeddingModel"/>.</summary>
public static class EmbeddingModelExtensions
{
    /// <summary>
    /// The model as a Microsoft.Extensions.AI <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/>. The generator does
    /// not own the model; dispose the model when it is no longer used.
    /// </summary>
    /// <param name="model">A loaded embedding model.</param>
    /// <param name="textKind">
    /// The input convention applied to every text: the model's default prefix (default), its query prefix, its passage
    /// prefix, or none. Matters only for a model that declares prefixes (e.g. the E5 family); for others the four are
    /// the same.
    /// </param>
    public static LocalEmbeddingGenerator AsEmbeddingGenerator(
        this IEmbeddingModel model,
        EmbeddingTextKind textKind = EmbeddingTextKind.Default) =>
        new(model, textKind);
}
