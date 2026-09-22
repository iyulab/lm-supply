using System.Runtime.CompilerServices;
using System.Text.Json;
using LMSupply.Embedder;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// The teeth behind <see cref="IEmbeddingModel.VectorSpaceRevision"/>: for each cached model family a
/// golden vector is stored next to the revision that produced it, and a load today must either
/// reproduce both or move both. A release that changes the vectors without raising an epoch, and an
/// epoch raised without the vectors moving, are both red here — the revision is only worth storing if
/// it moves exactly when the numbers do.
/// </summary>
/// <remarks>
/// <para>
/// Goldens are generated on the CPU provider (GPU providers add floating-point noise that is not a
/// vector-space change) and stored under <c>Functional/Goldens/embedder-vector-space.json</c>. To
/// regenerate after a deliberate change, run with <c>LMSUPPLY_UPDATE_GOLDENS=1</c>; the fact then
/// rewrites the source fixture and passes.
/// </para>
/// <para>
/// This suite is <c>LocalOnly</c> (it loads real models), so it bites on a developer machine, not in
/// CI. The ecosystem harness runs the published package and is where a CI-side golden belongs.
/// </para>
/// </remarks>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Embedder")]
public sealed class EmbedderVectorSpaceGoldenTests
{
    // Relative L2 distance, not cosine: cosine is scale-invariant and would not see L2 normalization
    // being switched off, which is a vector-space change a consumer's stored norms depend on.
    private const double SameSpaceThreshold = 1e-3;
    private const string Text = "The quick brown fox jumps over the lazy dog. 東京タワーは美しい。";

    /// <summary>One model per tokenizer family and load path: SentencePiece by alias, SentencePiece
    /// with catalog prefixes, WordPiece by repository id (CLS pooling read from the model's files).</summary>
    public static TheoryData<string> Models => new()
    {
        "default",                  // BAAI/bge-m3 — SentencePiece (XLM-R), CLS, no prefix
        "fast",                     // intfloat/multilingual-e5-small — SentencePiece, mean, "query: "/"passage: "
        "BAAI/bge-small-en-v1.5",   // WordPiece by repository id — CLS from 1_Pooling/config.json
    };

    private sealed record Golden(string Model, string Revision, string Text, float[] Vector);

    [Theory]
    [MemberData(nameof(Models))]
    public async Task Revision_moves_exactly_when_the_vectors_move(string model)
    {
        var options = new EmbedderOptions { Provider = ExecutionProvider.Cpu };
        await using var embedder = await LocalEmbedder.LoadAsync(model, options, cancellationToken: TestContext.Current.CancellationToken);

        embedder.VectorSpaceRevision.Should().NotBeNullOrEmpty("a model this library loaded reports the space it embeds into");
        var vector = await embedder.EmbedPassageAsync(Text, TestContext.Current.CancellationToken);

        var goldens = ReadGoldens();
        if (Environment.GetEnvironmentVariable("LMSUPPLY_UPDATE_GOLDENS") == "1")
        {
            goldens[model] = new Golden(model, embedder.VectorSpaceRevision!, Text, vector);
            WriteGoldens(goldens);
            return;
        }

        goldens.Should().ContainKey(model, "no golden is stored for this model — generate one with LMSUPPLY_UPDATE_GOLDENS=1");
        var golden = goldens[model];
        golden.Text.Should().Be(Text, "the golden was embedded from a different text; regenerate it");

        var distance = RelativeDistance(vector, golden.Vector);
        var cosine = Cosine(vector, golden.Vector);
        var sameVectors = distance <= SameSpaceThreshold;
        var sameRevision = embedder.VectorSpaceRevision == golden.Revision;

        switch (sameVectors, sameRevision)
        {
            case (true, true):
                return;
            case (false, true):
                Assert.Fail(
                    $"{model}: the vectors moved (relative L2 distance to golden {distance:E2}, cosine {cosine:F6}) but VectorSpaceRevision did not " +
                    $"({golden.Revision}). A release changed what this model produces without raising the epoch " +
                    "of the component that changed (TokenizerEpochs / VectorSpaceDescriptor.EmbedderEpoch).");
                break;
            case (true, false):
                Assert.Fail(
                    $"{model}: VectorSpaceRevision moved ({golden.Revision} -> {embedder.VectorSpaceRevision}) but the " +
                    $"vectors did not (relative L2 distance {distance:E2}). An epoch was raised, or the descriptor now reads a value that " +
                    "does not affect the vectors — consumers would re-embed for nothing.");
                break;
            case (false, false):
                Assert.Fail(
                    $"{model}: both the vectors (relative L2 distance {distance:E2}, cosine {cosine:F6}) and the revision moved " +
                    $"({golden.Revision} -> {embedder.VectorSpaceRevision}), as a deliberate change should. " +
                    "Regenerate the golden with LMSUPPLY_UPDATE_GOLDENS=1 and name the change in the CHANGELOG.");
                break;
        }
    }

    [Fact]
    public async Task The_same_repository_by_alias_and_by_id_reports_the_same_revision_when_the_loader_makes_the_same_decisions()
    {
        // bge-m3 has no catalog prefix and declares CLS in its own files, so the alias path and the
        // repository-id path should agree. (The e5 family would not: the catalog gives it prefixes the
        // repository does not declare — a different vector space through EmbedQueryAsync, honestly.)
        var options = () => new EmbedderOptions { Provider = ExecutionProvider.Cpu };
        await using var byAlias = await LocalEmbedder.LoadAsync("default", options(), cancellationToken: TestContext.Current.CancellationToken);
        await using var byId = await LocalEmbedder.LoadAsync("BAAI/bge-m3", options(), cancellationToken: TestContext.Current.CancellationToken);

        byId.VectorSpaceRevision.Should().Be(byAlias.VectorSpaceRevision,
            "the two load paths opened the same file with the same tokenizer, pooling, length and prefixes");
    }

    private static double RelativeDistance(float[] a, float[] b)
    {
        a.Length.Should().Be(b.Length);
        double diff = 0, norm = 0;
        for (var i = 0; i < a.Length; i++)
        {
            diff += (a[i] - b[i]) * (double)(a[i] - b[i]);
            norm += b[i] * (double)b[i];
        }

        return Math.Sqrt(diff) / Math.Sqrt(norm);
    }

    private static float Cosine(float[] a, float[] b)
    {
        a.Length.Should().Be(b.Length);
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }

        return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb)));
    }

    private static string GoldensPath([CallerFilePath] string thisFile = "") =>
        Path.Combine(Path.GetDirectoryName(thisFile)!, "Goldens", "embedder-vector-space.json");

    private static Dictionary<string, Golden> ReadGoldens()
    {
        var path = GoldensPath();
        if (!File.Exists(path))
            return [];

        var list = JsonSerializer.Deserialize<List<Golden>>(File.ReadAllText(path)) ?? [];
        return list.ToDictionary(g => g.Model);
    }

    private static void WriteGoldens(Dictionary<string, Golden> goldens)
    {
        var path = GoldensPath();
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var json = JsonSerializer.Serialize(goldens.Values.OrderBy(g => g.Model).ToList());
        File.WriteAllText(path, json);
    }
}
