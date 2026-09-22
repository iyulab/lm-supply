using AwesomeAssertions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// <see cref="IEmbeddingModel.VectorSpaceRevision"/> is derived from what the loader did, and it must
/// move for every decision that moves the vectors and for nothing else. These facts pin the derivation
/// itself; the golden-vector facts in the integration suite pin that the decisions it reads are the
/// ones that actually run.
/// </summary>
public sealed class VectorSpaceDescriptorTests
{
    private static VectorSpaceDescriptor Baseline() => new(
        Backend: "onnx",
        ModelFile: "onnx/model.onnx",
        Tokenizer: "wordpiece/2;clean=1;chinese=1;accents=1;lower=1;punct=1",
        Pooling: "Mean",
        Normalize: true,
        MaxSequenceLength: 512,
        Dimensions: 384,
        QueryPrefix: null,
        PassagePrefix: null);

    [Fact]
    public void Revision_is_deterministic_and_short()
    {
        var a = Baseline().Revision;
        var b = Baseline().Revision;

        a.Should().Be(b);
        a.Should().MatchRegex("^[0-9a-f]{16}$", "the revision is the first 16 hex characters of SHA-256 over the canonical line, not a runtime hash");
    }

    [Fact]
    public void Revision_is_the_same_value_this_library_computed_when_the_fact_was_written()
    {
        // A process-independent derivation: the same canonical line hashes to the same string on every
        // machine, run and operating system. If this fact goes red without a deliberate format change,
        // the derivation has become process-dependent.
        // Independently computed: sha256("vs/1;embedder/1;backend=onnx;model=onnx/model.onnx;tokenizer=wordpiece/2;clean=1;chinese=1;accents=1;lower=1;punct=1;pooling=Mean;normalize=1;maxseq=512;dims=384;query=;passage=")[:16] in Python.
        Baseline().Revision.Should().Be("a62056feeca5361b");
    }

    public static TheoryData<string> Moves => new()
    {
        "model file (quantization variant)", "tokenizer signature (epoch)", "tokenizer signature (normalization)",
        "pooling", "normalization", "sequence length", "dimensions", "query prefix", "passage prefix", "backend",
    };

    private static VectorSpaceDescriptor Move(string what, VectorSpaceDescriptor d) => what switch
    {
        "model file (quantization variant)" => d with { ModelFile = "onnx/model_fp16.onnx" },
        "tokenizer signature (epoch)" => d with { Tokenizer = "wordpiece/3;clean=1;chinese=1;accents=1;lower=1;punct=1" },
        "tokenizer signature (normalization)" => d with { Tokenizer = "wordpiece/2;clean=1;chinese=1;accents=0;lower=0;punct=1" },
        "pooling" => d with { Pooling = "Cls" },
        "normalization" => d with { Normalize = false },
        "sequence length" => d with { MaxSequenceLength = 256 },
        "dimensions" => d with { Dimensions = 768 },
        "query prefix" => d with { QueryPrefix = "query: " },
        "passage prefix" => d with { PassagePrefix = "passage: " },
        "backend" => d with { Backend = "gguf" },
        _ => throw new ArgumentOutOfRangeException(nameof(what)),
    };

    [Theory]
    [MemberData(nameof(Moves))]
    public void Revision_moves_with_each_vector_affecting_decision(string what)
    {
        var baseline = Baseline();
        var moved = Move(what, baseline);

        moved.Canonical.Should().NotBe(baseline.Canonical, what);
        moved.Revision.Should().NotBe(baseline.Revision, $"a change to {what} produces different vectors");
    }

    [Fact]
    public void Prefixes_cannot_collide_with_the_separators()
    {
        // "a;b=c" as a prefix must not read as extra fields in the canonical line.
        var tricky = Baseline() with { QueryPrefix = "q;pooling=Cls" };
        var plain = Baseline() with { QueryPrefix = "q" , Pooling = "Cls" };

        tricky.Canonical.Should().Contain("query=q%3Bpooling%3DCls");
        tricky.Revision.Should().NotBe(plain.Revision);
    }

    [Theory]
    [InlineData("onnx/model.onnx")]
    [InlineData("onnx\\model.onnx")]
    public void Model_file_is_snapshot_relative_with_forward_slashes_on_every_platform(string relative)
    {
        var root = Path.Combine(Path.GetTempPath(), $"vs-{Guid.NewGuid():N}");
        var modelPath = Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar));

        VectorSpaceDescriptor.RelativeModelFile(root, modelPath).Should().Be("onnx/model.onnx");
        VectorSpaceDescriptor.RelativeModelFile(root + Path.DirectorySeparatorChar, modelPath).Should().Be("onnx/model.onnx", "a trailing separator on the root changes nothing");
    }

    [Fact]
    public void A_model_file_outside_the_root_reduces_to_its_file_name()
    {
        var root = Path.Combine(Path.GetTempPath(), $"vs-{Guid.NewGuid():N}");
        var elsewhere = Path.Combine(Path.GetTempPath(), $"vs-{Guid.NewGuid():N}", "my-model.onnx");

        VectorSpaceDescriptor.RelativeModelFile(root, elsewhere).Should().Be("my-model.onnx");
        VectorSpaceDescriptor.RelativeModelFile(null, elsewhere).Should().Be("my-model.onnx");
    }

    [Fact]
    public void The_interface_default_is_null_so_a_consumer_implementation_keeps_compiling()
    {
        IEmbeddingModel model = new NoRevisionModel();

        model.VectorSpaceRevision.Should().BeNull();
    }

    private sealed class NoRevisionModel : IEmbeddingModel
    {
        public string ModelId => "double";
        public int Dimensions => 1;
        public bool IsGpuActive => false;
        public IReadOnlyList<string> ActiveProviders => [];
        public ExecutionProvider RequestedProvider => ExecutionProvider.Cpu;
        public long? EstimatedMemoryBytes => null;
        public ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default) => new([1f]);
        public ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default) => new([[1f]]);
        public ValueTask<float[]> EmbedAsync(string text, int dimensions, CancellationToken cancellationToken = default) => new([1f]);
        public ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken cancellationToken = default) => new([[1f]]);
        public Task WarmupAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Utils.ModelInfo? GetModelInfo() => null;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
