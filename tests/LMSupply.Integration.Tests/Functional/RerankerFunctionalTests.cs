using LMSupply.Exceptions;
using LMSupply.Generator;
using LMSupply.Reranker;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Comprehensive functional tests for the Reranker domain.
/// Tests L (loading), I (inference), Q (quality), E (edge cases) axes.
/// Also validates known bugs BUG-001/002/003 from the test plan.
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Reranker")]
public class RerankerFunctionalTests
{
    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalReranker.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_FastAlias_LoadsSuccessfully()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_WarmupAsync_CompletesWithoutError()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.WarmupAsync();
        await act.Should().NotThrowAsync();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_InvalidModelId_ThrowsModelNotFoundException()
    {
        var act = () => LocalReranker.LoadAsync("completely-nonexistent-reranker-xyz-999");
        await act.Should().ThrowAsync<ModelNotFoundException>();
    }

    [Theory]
    [Trait("Axis", "Loading")]
    [InlineData("")]
    [InlineData("///invalid")]
    public async Task L_MalformedModelId_ThrowsAppropriateException(string modelId)
    {
        var act = () => LocalReranker.LoadAsync(modelId);
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_ExplicitCpuProvider_LoadsSuccessfully()
    {
        var options = new RerankerOptions { Provider = ExecutionProvider.Cpu };
        await using var model = await LocalReranker.LoadAsync("fast", options, cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetAvailableModels_ReturnsNonEmptyList()
    {
        var models = LocalReranker.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("default");
        models.Should().Contain("fast");
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_BasicReranking_ReturnsCorrectCount()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "hello";
        string[] docs = ["hello", "world", "xyz"];

        var results = await model.RerankAsync(query, docs, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(3);
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_BasicReranking_ExactMatchIsTop1()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "hello";
        string[] docs = ["hello", "world", "xyz"];

        var results = await model.RerankAsync(query, docs, cancellationToken: TestContext.Current.CancellationToken);

        results[0].Document.Should().Be("hello",
            "exact match should be ranked first");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_ScoreAsync_ReturnsScoresInOriginalOrder()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "machine learning";
        string[] docs = ["weather report", "deep learning neural networks", "cooking recipe"];

        var scores = await model.ScoreAsync(query, docs, TestContext.Current.CancellationToken);

        scores.Should().HaveCount(3);
        // ML-related doc should have highest score
        scores[1].Should().BeGreaterThan(scores[0]);
        scores[1].Should().BeGreaterThan(scores[2]);
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_RerankBatchAsync_ReturnsCorrectBatchCount()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        string[] queries = ["query1", "query2"];
        string[][] docSets = [["doc1a", "doc1b"], ["doc2a", "doc2b"]];

        var results = await model.RerankBatchAsync(queries, docSets, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(2);
        results[0].Should().HaveCount(2);
        results[1].Should().HaveCount(2);
    }

    // ── Q axis: Quality / Semantic Correctness ──────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ExactMatch_HasHighestScore()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "machine learning";
        string[] docs =
        [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
            "Deep learning uses neural networks.",
            "I like cooking pasta.",
        ];

        var results = await model.RerankAsync(query, docs, cancellationToken: TestContext.Current.CancellationToken);

        // ML-related docs should rank higher
        var topDoc = results[0].Document;
        topDoc.Should().ContainAny("machine", "Machine", "learning", "neural",
            "top result should be ML-related");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ScoresAreMeaningful_NotGarbage()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "What is artificial intelligence?";
        string[] docs =
        [
            "Artificial intelligence is the simulation of human intelligence.",
            "The price of gold fluctuated today.",
        ];

        var scores = await model.ScoreAsync(query, docs, TestContext.Current.CancellationToken);

        // The relevant doc should have a significantly higher score
        scores[0].Should().BeGreaterThan(scores[1],
            "relevant document should score higher");

        // At least the top score should not be near-zero
        MathF.Abs(scores.Max()).Should().BeGreaterThan(1e-4f,
            "max score should not be near-zero (indicates model output issue)");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_TopN_ReturnsExactCount()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "test";
        string[] docs = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];

        var results = await model.RerankAsync(query, docs, topK: 3, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(3);
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_OrderConsistency_SameInputSameOutput()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "machine learning";
        string[] docs = ["ML paper", "cooking recipe", "deep learning", "weather"];

        var results1 = await model.RerankAsync(query, docs, cancellationToken: TestContext.Current.CancellationToken);
        var results2 = await model.RerankAsync(query, docs, cancellationToken: TestContext.Current.CancellationToken);

        // Same input should produce same ordering
        for (var i = 0; i < results1.Count; i++)
        {
            results1[i].OriginalIndex.Should().Be(results2[i].OriginalIndex,
                $"rank {i} should be consistent across calls");
        }
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ResultsContainOriginalIndex()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "test query";
        string[] docs = ["alpha", "beta", "gamma"];

        var results = await model.RerankAsync(query, docs, cancellationToken: TestContext.Current.CancellationToken);

        // Every original index 0, 1, 2 should appear exactly once
        var indices = results.Select(r => r.OriginalIndex).OrderBy(i => i).ToList();
        indices.Should().BeEquivalentTo([0, 1, 2]);
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ResultsAreSortedByScoreDescending()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "machine learning";
        string[] docs = ["ML paper", "cooking recipe", "deep learning", "weather"];

        var results = await model.RerankAsync(query, docs, cancellationToken: TestContext.Current.CancellationToken);

        for (var i = 1; i < results.Count; i++)
        {
            results[i - 1].Score.Should().BeGreaterThanOrEqualTo(results[i].Score,
                $"results should be sorted descending by score (index {i - 1} vs {i})");
        }
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ConcurrentInference_ProducesConsistentResults()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "machine learning";
        string[] docs = ["ML paper", "cooking recipe", "deep learning"];

        var tasks = Enumerable.Range(0, 5)
            .Select(_ => model.RerankAsync(query, docs))
            .ToArray();

        var allResults = await Task.WhenAll(tasks);

        // All runs should produce the same top-1
        var firstTop = allResults[0][0].OriginalIndex;
        foreach (var results in allResults)
        {
            results[0].OriginalIndex.Should().Be(firstTop,
                "concurrent calls should produce consistent top-1");
        }
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SingleDocument_ReturnsThatDocument()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "test";
        string[] docs = ["only document"];

        var results = await model.RerankAsync(query, docs, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(1);
        results[0].Document.Should().Be("only document");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_EmptyDocuments_ThrowsArgumentException()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.RerankAsync("query", Array.Empty<string>());
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_EmptyQuery_ThrowsArgumentException()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.RerankAsync("", ["doc1"]);
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_WhitespaceQuery_ThrowsArgumentException()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.RerankAsync("   ", ["doc1"]);
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SpecialCharacters_DoNotCrash()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "🎉 emoji query";
        string[] docs = ["hello 🐱", "日本語テスト", "مرحبا"];

        var act = () => model.RerankAsync(query, docs);
        var result = await act.Should().NotThrowAsync();
        result.Subject.Should().HaveCount(3);
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_LongDocuments_DoNotOOM()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "test";
        var longDoc = new string('a', 5000); // Exceed typical context window
        string[] docs = [longDoc, "short"];

        var act = () => model.RerankAsync(query, docs);
        await act.Should().NotThrowAsync("long documents should truncate, not OOM");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_NullOptions_UsesDefaults()
    {
        await using var model = await LocalReranker.LoadAsync("fast", options: null, cancellationToken: TestContext.Current.CancellationToken);

        var results = await model.RerankAsync("test", ["doc1"], cancellationToken: TestContext.Current.CancellationToken);
        results.Should().HaveCount(1);
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_TopK_GreaterThanDocCount_ReturnsAll()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        string[] docs = ["a", "b", "c"];
        var results = await model.RerankAsync("test", docs, topK: 10, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(3, "topK > doc count should return all docs");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_TopK_One_ReturnsSingleResult()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        string[] docs = ["a", "b", "c"];
        var results = await model.RerankAsync("test", docs, topK: 1, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(1);
    }

    // ── BUG validation: Known defects from test plan ────────────────

    /// <summary>
    /// BUG-003: Score validation — verify near-zero score warning exists.
    /// The Reranker.cs and LlamaServerRerankerModel.cs now have
    /// Trace.TraceWarning for near-zero scores. This test validates that
    /// meaningful scores are produced by compatible ONNX models.
    /// </summary>
    [Fact]
    [Trait("Axis", "Quality")]
    [Trait("Bug", "BUG-003")]
    public async Task Bug003_OnnxModel_ProducesNonZeroScores()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "What is machine learning?";
        string[] docs =
        [
            "Machine learning is a branch of AI that uses data to learn.",
            "The sun rises in the east.",
        ];

        var scores = await model.ScoreAsync(query, docs, TestContext.Current.CancellationToken);

        // Compatible ONNX models should NOT produce all near-zero scores
        var maxScore = scores.Max(s => MathF.Abs(s));
        maxScore.Should().BeGreaterThan(1e-4f,
            "ONNX cross-encoder should produce meaningful scores (BUG-003 validation)");
    }

    // ── Gap Tests: Models Listing, Properties ──────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ReturnsModelInfo()
    {
        var models = LocalReranker.GetAllModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty($"Id should be set for {m.AliasName}");
            m.AliasName.Should().NotBeNullOrEmpty($"AliasName should be set for {m.Id}");
            m.DisplayName.Should().NotBeNullOrEmpty($"DisplayName should be set for {m.AliasName}");
            m.MaxSequenceLength.Should().BeGreaterThan(0, $"MaxSequenceLength should be positive for {m.AliasName}");
            m.OnnxFile.Should().NotBeNullOrEmpty($"OnnxFile should be set for {m.AliasName}");
            m.TokenizerFile.Should().NotBeNullOrEmpty($"TokenizerFile should be set for {m.AliasName}");
        });
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_ModelProperties_ArePopulated()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ActiveProviders.Should().NotBeNull();
        model.RequestedProvider.Should().BeDefined();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ManyDocuments_RanksCorrectly()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var query = "What is machine learning?";
        string[] docs =
        [
            "The recipe calls for two cups of flour.",
            "Machine learning is a type of artificial intelligence.",
            "The football team won the championship.",
            "Neural networks are inspired by biological neurons.",
            "The garden needs watering twice a week.",
            "Supervised learning uses labeled training data.",
            "The movie received excellent reviews.",
            "Deep learning uses multiple neural network layers.",
            "The traffic was heavy during rush hour.",
            "Reinforcement learning trains agents through rewards.",
        ];

        var results = await model.RerankAsync(query, docs, topK: 3, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(3);
        // All top 3 should be ML-related
        results.Should().AllSatisfy(r =>
            r.Document.Should().ContainAny("machine", "Machine", "learning", "neural", "Neural",
                "top results should be ML-related"));
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_DuplicateDocuments_DoesNotCrash()
    {
        await using var model = await LocalReranker.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        string[] docs = ["same doc", "same doc", "same doc"];
        var results = await model.RerankAsync("test", docs, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(3);
    }

    // ── Static Tests: Options Defaults & Model Registry (no model loading) ─

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_RerankerOptions_Defaults_AreReasonable()
    {
        var opts = new RerankerOptions();

        opts.ModelId.Should().Be("default", "default model ID should be 'default'");
        opts.MaxSequenceLength.Should().BeNull("default max sequence length should be null (uses model default)");
        opts.BatchSize.Should().Be(32, "default batch size should be 32");
        opts.DisableAutoDownload.Should().BeFalse("auto download should be enabled by default");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsAllStandardAliases()
    {
        var models = LocalReranker.GetAvailableModels().ToList();

        models.Should().Contain("default");
        models.Should().Contain("fast");
        models.Should().Contain("quality");
        models.Should().Contain("large");
        models.Should().Contain("multilingual");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_HasExpectedCount()
    {
        var models = LocalReranker.GetAllModels().ToList();

        models.Count.Should().BeGreaterThanOrEqualTo(6,
            "registry should have at least 6 reranker models");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_WellKnownModels_Reranker_HasValidConstants()
    {
        WellKnownModels.Reranker.Default.Should().Contain("ms-marco-MiniLM-L-6");
        WellKnownModels.Reranker.Fast.Should().Contain("TinyBERT");
        WellKnownModels.Reranker.Quality.Should().Contain("bge-reranker-base");
        WellKnownModels.Reranker.Large.Should().Contain("bge-reranker-large");
        WellKnownModels.Reranker.Multilingual.Should().Contain("bge-reranker-v2-m3");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_RerankerOptions_Clone_IsIndependentCopy()
    {
        var original = new RerankerOptions
        {
            ModelId = "quality",
            MaxSequenceLength = 256,
            BatchSize = 16,
            Provider = ExecutionProvider.Cpu
        };

        var clone = original.Clone();

        clone.ModelId.Should().Be("quality");
        clone.MaxSequenceLength.Should().Be(256);
        clone.BatchSize.Should().Be(16);

        clone.ModelId = "fast";
        clone.BatchSize = 64;
        original.ModelId.Should().Be("quality", "original unaffected by clone mutation");
        original.BatchSize.Should().Be(16, "original batch size unaffected");
    }

    // ── GGUF Model Tests (R-Q5/Q6) ───────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    [Trait("Category", "GGUF")]
    public async Task L_GgufModel_LoadsSuccessfully()
    {
        // R-Q5: Load a GGUF reranker via llama-server (cross-encoder only)
        await using var model = await LocalReranker.LoadAsync("BAAI/bge-reranker-v2-m3-GGUF", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    [Trait("Category", "GGUF")]
    public async Task Q_GgufReranker_ProducesValidScores()
    {
        // R-Q5: GGUF reranker should produce meaningful, non-zero scores
        await using var model = await LocalReranker.LoadAsync("BAAI/bge-reranker-v2-m3-GGUF", cancellationToken: TestContext.Current.CancellationToken);

        var query = "What is machine learning?";
        string[] docs =
        [
            "Machine learning is a branch of AI that learns from data.",
            "The weather forecast predicts rain tomorrow.",
            "Deep learning uses neural networks with many layers."
        ];

        var results = await model.RerankAsync(query, docs, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(3);
        // Verify scores are non-zero (BUG-003: near-zero means incompatible model)
        results.Should().Contain(r => Math.Abs(r.Score) > 0.001,
            "GGUF reranker should produce meaningful (non-near-zero) scores");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    [Trait("Category", "GGUF")]
    public async Task Q_GgufReranker_RanksRelevantHigher()
    {
        // R-Q6: GGUF reranker should rank relevant doc higher than irrelevant
        await using var model = await LocalReranker.LoadAsync("BAAI/bge-reranker-v2-m3-GGUF", cancellationToken: TestContext.Current.CancellationToken);

        var query = "How does photosynthesis work?";
        string[] docs =
        [
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "The stock market closed higher today.",
        ];

        var results = await model.RerankAsync(query, docs, topK: 1, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(1);
        results[0].Document.Should().Contain("Photosynthesis",
            "relevant document should be ranked first");
    }
}
