using System.Diagnostics;
using LMSupply.Embedder;
using LMSupply.Embedder.Utils;
using LMSupply.Exceptions;
using LMSupply.Generator;
using LMSupply.Integration.Tests.Helpers;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Comprehensive functional tests for the Embedder domain.
/// Tests L (loading), I (inference), Q (quality), E (edge cases) axes.
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Embedder")]
public class EmbedderFunctionalTests
{
    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalEmbedder.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
        model.Dimensions.Should().Be(384, "default model (BGE-small) has 384 dimensions");
    }

    [Theory]
    [Trait("Axis", "Loading")]
    [InlineData("fast", 384)]
    [InlineData("default", 384)]
    public async Task L_KnownAliases_LoadWithExpectedDimensions(string alias, int expectedDims)
    {
        await using var model = await LocalEmbedder.LoadAsync(alias, cancellationToken: TestContext.Current.CancellationToken);

        model.Dimensions.Should().Be(expectedDims);
        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_WarmupAsync_CompletesWithoutError()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.WarmupAsync();
        await act.Should().NotThrowAsync();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_InvalidModelId_ThrowsModelNotFoundException()
    {
        var act = () => LocalEmbedder.LoadAsync("completely-nonexistent-model-xyz-999");
        await act.Should().ThrowAsync<ModelNotFoundException>();
    }

    [Theory]
    [Trait("Axis", "Loading")]
    [InlineData("")]
    [InlineData("///invalid")]
    public async Task L_MalformedModelId_ThrowsAppropriateException(string modelId)
    {
        var act = () => LocalEmbedder.LoadAsync(modelId);
        // Should throw ArgumentException or ModelNotFoundException, not crash
        await act.Should().ThrowAsync<Exception>();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_ExplicitCpuProvider_LoadsSuccessfully()
    {
        var options = new EmbedderOptions { Provider = ExecutionProvider.Cpu };
        await using var model = await LocalEmbedder.LoadAsync("fast", options, cancellationToken: TestContext.Current.CancellationToken);

        model.Dimensions.Should().BeGreaterThan(0);
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetAvailableModels_ReturnsNonEmptyList()
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("all-MiniLM-L6-v2");
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_SingleText_ReturnsCorrectDimensionVector()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var embedding = await model.EmbedAsync("hello world", TestContext.Current.CancellationToken);

        embedding.Should().NotBeNull();
        embedding.Length.Should().Be(model.Dimensions);
        embedding.Should().Contain(v => v != 0f, "embedding should not be all zeros");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_BatchTexts_ReturnsCorrectCountAndOrder()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var texts = new[] { "cat", "dog", "airplane" };
        var embeddings = await model.EmbedAsync(texts, TestContext.Current.CancellationToken);

        embeddings.Should().HaveCount(3);
        foreach (var emb in embeddings)
        {
            emb.Length.Should().Be(model.Dimensions);
        }
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_BatchEmbeddings_PreserveOrder()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var texts = new[] { "alpha", "beta", "gamma" };
        var batchResult = await model.EmbedAsync(texts, TestContext.Current.CancellationToken);

        // Each individual embed should match the batch
        for (var i = 0; i < texts.Length; i++)
        {
            var singleResult = await model.EmbedAsync(texts[i], TestContext.Current.CancellationToken);
            var similarity = LocalEmbedder.CosineSimilarity(batchResult[i], singleResult);
            similarity.Should().BeGreaterThan(0.99f,
                $"batch[{i}] should match single embed for '{texts[i]}'");
        }
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_EmbeddingsAreNormalized_ByDefault()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var embedding = await model.EmbedAsync("test normalization", TestContext.Current.CancellationToken);

        // L2 norm should be approximately 1.0 for normalized vectors
        var l2Norm = MathF.Sqrt(embedding.Sum(v => v * v));
        l2Norm.Should().BeApproximately(1.0f, 0.01f, "embeddings should be L2-normalized by default");
    }

    // ── Q axis: Quality / Semantic Correctness ──────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_SimilarTexts_HaveHigherSimilarity()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var catEmb = await model.EmbedAsync("cat", TestContext.Current.CancellationToken);
        var kittenEmb = await model.EmbedAsync("kitten", TestContext.Current.CancellationToken);
        var airplaneEmb = await model.EmbedAsync("airplane", TestContext.Current.CancellationToken);

        var simCatKitten = LocalEmbedder.CosineSimilarity(catEmb, kittenEmb);
        var simCatAirplane = LocalEmbedder.CosineSimilarity(catEmb, airplaneEmb);

        simCatKitten.Should().BeGreaterThan(simCatAirplane,
            "cat-kitten similarity should be greater than cat-airplane");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_IdenticalTexts_HaveMaxSimilarity()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var emb1 = await model.EmbedAsync("The quick brown fox", TestContext.Current.CancellationToken);
        var emb2 = await model.EmbedAsync("The quick brown fox", TestContext.Current.CancellationToken);

        var similarity = LocalEmbedder.CosineSimilarity(emb1, emb2);
        similarity.Should().BeApproximately(1.0f, 0.001f,
            "identical texts should have cosine similarity ~1.0");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_DimensionMatchesModelSpec()
    {
        // BGE-small: 384, MiniLM-L6: 384
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var embedding = await model.EmbedAsync("dimension check", TestContext.Current.CancellationToken);
        embedding.Length.Should().Be(384, "all-MiniLM-L6-v2 should produce 384-dim vectors");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_RepeatInference_IsDeterministic()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var emb1 = await model.EmbedAsync("determinism check", TestContext.Current.CancellationToken);
        var emb2 = await model.EmbedAsync("determinism check", TestContext.Current.CancellationToken);

        var similarity = LocalEmbedder.CosineSimilarity(emb1, emb2);
        similarity.Should().BeApproximately(1.0f, 0.0001f,
            "repeated inference should produce identical results");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_SemanticClusters_AreDistinguishable()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // Cluster 1: Animals
        var dog = await model.EmbedAsync("dog", TestContext.Current.CancellationToken);
        var puppy = await model.EmbedAsync("puppy", TestContext.Current.CancellationToken);

        // Cluster 2: Technology
        var computer = await model.EmbedAsync("computer", TestContext.Current.CancellationToken);
        var laptop = await model.EmbedAsync("laptop", TestContext.Current.CancellationToken);

        var withinAnimals = LocalEmbedder.CosineSimilarity(dog, puppy);
        var withinTech = LocalEmbedder.CosineSimilarity(computer, laptop);
        var crossCluster = LocalEmbedder.CosineSimilarity(dog, computer);

        withinAnimals.Should().BeGreaterThan(crossCluster, "within-cluster > cross-cluster (animals vs tech)");
        withinTech.Should().BeGreaterThan(crossCluster, "within-cluster > cross-cluster (tech vs animals)");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ConcurrentInference_ProducesConsistentResults()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        const string text = "concurrent test";
        var tasks = Enumerable.Range(0, 5)
            .Select(_ => model.EmbedAsync(text).AsTask())
            .ToArray();

        var results = await Task.WhenAll(tasks);

        // All results should be essentially identical
        for (var i = 1; i < results.Length; i++)
        {
            var sim = LocalEmbedder.CosineSimilarity(results[0], results[i]);
            sim.Should().BeGreaterThan(0.999f,
                $"concurrent result[{i}] should match result[0]");
        }
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_EmojiText_DoesNotCrash()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = async () => await model.EmbedAsync(TestDataHelper.EmojiText);
        var result = await act.Should().NotThrowAsync();

        // Should return a valid vector
        result.Subject.Length.Should().Be(model.Dimensions);
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SpecialCharacters_DoesNotCrash()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var specialTexts = new[]
        {
            "\t\n\r", // whitespace characters
            "zero\u200Bwidth", // zero-width space
            "مرحبا", // Arabic (RTL)
            "日本語テスト", // Japanese
            "中文测试", // Chinese
        };

        foreach (var text in specialTexts)
        {
            var embedding = await model.EmbedAsync(text, TestContext.Current.CancellationToken);
            embedding.Length.Should().Be(model.Dimensions,
                $"special text '{text}' should produce valid embedding");
        }
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_VeryLongText_DoesNotOOM()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // ~10000 tokens worth of text — should truncate, not OOM
        var longText = TestDataHelper.GenerateLongText(10_000);

        var embedding = await model.EmbedAsync(longText, TestContext.Current.CancellationToken);
        embedding.Length.Should().Be(model.Dimensions);
        embedding.Should().Contain(v => v != 0f, "long text embedding should not be all zeros");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_EmptyString_HandlesGracefully()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // Empty string should either produce a valid embedding or throw ArgumentException — never crash
        try
        {
            var embedding = await model.EmbedAsync("", TestContext.Current.CancellationToken);
            embedding.Length.Should().Be(model.Dimensions,
                "if empty string is accepted, it should produce a valid-dimension vector");
        }
        catch (ArgumentException)
        {
            // Acceptable: empty input rejected with clear error
        }
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_EmptyBatch_ReturnsEmptyArray()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var embeddings = await model.EmbedAsync(Array.Empty<string>(), TestContext.Current.CancellationToken);

        embeddings.Should().BeEmpty("empty batch should return empty result, not throw");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SingleCharacter_ProducesValidEmbedding()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var embedding = await model.EmbedAsync("a", TestContext.Current.CancellationToken);
        embedding.Length.Should().Be(model.Dimensions);
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_NullOptions_UsesDefaults()
    {
        // options=null should not cause NullReferenceException
        await using var model = await LocalEmbedder.LoadAsync("fast", options: null, cancellationToken: TestContext.Current.CancellationToken);

        var embedding = await model.EmbedAsync("null options test", TestContext.Current.CancellationToken);
        embedding.Length.Should().BeGreaterThan(0);
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_BatchWithSingleItem_Works()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        string[] singleBatch = ["single item batch"];
        var embeddings = await model.EmbedAsync(singleBatch, TestContext.Current.CancellationToken);
        embeddings.Should().HaveCount(1);
        embeddings[0].Length.Should().Be(model.Dimensions);
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_MixedLengthBatch_Works()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var texts = new[]
        {
            "a",
            "a medium length sentence for testing",
            TestDataHelper.GenerateLongText(500),
        };

        var embeddings = await model.EmbedAsync(texts, TestContext.Current.CancellationToken);
        embeddings.Should().HaveCount(3);
        foreach (var emb in embeddings)
        {
            emb.Length.Should().Be(model.Dimensions);
        }
    }

    // ── Gap Tests: Models Listing, Properties, Quality Alias ───────

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsDefaultModel()
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();

        models.Should().Contain("bge-small-en-v1.5",
            "default alias maps to bge-small-en-v1.5");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_ModelProperties_ArePopulated()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ActiveProviders.Should().NotBeNull();
        model.RequestedProvider.Should().BeDefined();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_QualityAlias_LoadsWithHigherDimensions()
    {
        await using var model = await LocalEmbedder.LoadAsync("quality", cancellationToken: TestContext.Current.CancellationToken);

        model.Dimensions.Should().BeGreaterThanOrEqualTo(384,
            "quality model should have >= 384 dimensions");
        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_CosineSimilarity_IsSymmetric()
    {
        await using var model = await LocalEmbedder.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var emb1 = await model.EmbedAsync("hello world", TestContext.Current.CancellationToken);
        var emb2 = await model.EmbedAsync("goodbye world", TestContext.Current.CancellationToken);

        var sim12 = LocalEmbedder.CosineSimilarity(emb1, emb2);
        var sim21 = LocalEmbedder.CosineSimilarity(emb2, emb1);

        sim12.Should().BeApproximately(sim21, 0.0001f,
            "cosine similarity should be symmetric");
    }

    // ── Static Vector Operations (no model required) ─────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_CosineSimilarity_IdenticalVectors_ReturnsOne()
    {
        float[] vec = [1.0f, 2.0f, 3.0f, 4.0f];

        var sim = LocalEmbedder.CosineSimilarity(vec, vec);

        sim.Should().BeApproximately(1.0f, 0.0001f,
            "identical vectors should have cosine similarity 1.0");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_CosineSimilarity_OppositeVectors_ReturnsNegativeOne()
    {
        float[] vec = [1.0f, 2.0f, 3.0f];
        float[] neg = [-1.0f, -2.0f, -3.0f];

        var sim = LocalEmbedder.CosineSimilarity(vec, neg);

        sim.Should().BeApproximately(-1.0f, 0.0001f,
            "opposite vectors should have cosine similarity -1.0");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_CosineSimilarity_OrthogonalVectors_ReturnsZero()
    {
        float[] a = [1.0f, 0.0f, 0.0f];
        float[] b = [0.0f, 1.0f, 0.0f];

        var sim = LocalEmbedder.CosineSimilarity(a, b);

        sim.Should().BeApproximately(0.0f, 0.0001f,
            "orthogonal vectors should have cosine similarity ~0");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_EuclideanDistance_IdenticalVectors_ReturnsZero()
    {
        float[] vec = [1.0f, 2.0f, 3.0f];

        var dist = LocalEmbedder.EuclideanDistance(vec, vec);

        dist.Should().BeApproximately(0.0f, 0.0001f,
            "identical vectors should have distance 0");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_EuclideanDistance_KnownVectors_IsCorrect()
    {
        float[] a = [0.0f, 0.0f, 0.0f];
        float[] b = [3.0f, 4.0f, 0.0f];

        var dist = LocalEmbedder.EuclideanDistance(a, b);

        dist.Should().BeApproximately(5.0f, 0.0001f,
            "distance from origin to (3,4,0) should be 5 (Pythagorean)");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_DotProduct_KnownVectors_IsCorrect()
    {
        float[] a = [1.0f, 2.0f, 3.0f];
        float[] b = [4.0f, 5.0f, 6.0f];

        var dot = LocalEmbedder.DotProduct(a, b);

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        dot.Should().BeApproximately(32.0f, 0.0001f,
            "dot product of [1,2,3]·[4,5,6] = 32");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_DotProduct_OrthogonalVectors_ReturnsZero()
    {
        float[] a = [1.0f, 0.0f];
        float[] b = [0.0f, 1.0f];

        var dot = LocalEmbedder.DotProduct(a, b);

        dot.Should().BeApproximately(0.0f, 0.0001f,
            "orthogonal vectors should have dot product 0");
    }

    // ── Model Registry Validation (no model loading) ─────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsAllStandardAliases()
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();

        models.Should().Contain("default");
        models.Should().Contain("fast");
        models.Should().Contain("quality");
        models.Should().Contain("large");
        models.Should().Contain("multilingual");
        models.Should().Contain("auto");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsMultilingualModels()
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();

        models.Should().Contain("bge-m3", "multilingual BGE-M3 should be registered");
        models.Should().Contain("multilingual-e5-small", "multilingual E5 small should be registered");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_HasExpectedCount()
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();

        // 6 aliases (auto, default, fast, quality, large, multilingual)
        // + 14 model names = 20 total
        models.Count.Should().BeGreaterThanOrEqualTo(18,
            "registry should have at least 18 entries (aliases + models)");
    }

    // ── Model-Loading Tests: Quality/Large/Multilingual Aliases ──

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_LargeAlias_LoadsWithExpectedDimensions()
    {
        await using var model = await LocalEmbedder.LoadAsync("large", cancellationToken: TestContext.Current.CancellationToken);

        model.Dimensions.Should().Be(768,
            "large alias (nomic-embed-text-v1.5) should have 768 dimensions");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_MultilingualAlias_LoadsWithHighDimensions()
    {
        await using var model = await LocalEmbedder.LoadAsync("multilingual", cancellationToken: TestContext.Current.CancellationToken);

        model.Dimensions.Should().Be(1024,
            "multilingual alias (bge-m3) should have 1024 dimensions");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_AutoAlias_LoadsSuccessfully()
    {
        await using var model = await LocalEmbedder.LoadAsync("auto", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
        model.Dimensions.Should().BeOneOf([384, 768, 1024],
            "auto should select a model with standard dimensions");
    }

    // ── Multilingual Quality Tests ───────────────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_MultilingualModel_CrossLanguageSimilarity()
    {
        await using var model = await LocalEmbedder.LoadAsync("multilingual", cancellationToken: TestContext.Current.CancellationToken);

        var engEmb = await model.EmbedAsync("hello", TestContext.Current.CancellationToken);
        var korEmb = await model.EmbedAsync("안녕하세요", TestContext.Current.CancellationToken);

        var similarity = LocalEmbedder.CosineSimilarity(engEmb, korEmb);

        similarity.Should().BeGreaterThan(0.3f,
            "multilingual model should find cross-language greeting similarity > 0.3");
    }

    // ── Options Defaults (no model loading) ───────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_EmbedderOptions_Defaults_AreReasonable()
    {
        var opts = new EmbedderOptions();

        opts.MaxSequenceLength.Should().Be(512, "default max sequence length should be 512");
        opts.NormalizeEmbeddings.Should().BeTrue("embeddings should be normalized by default");
        opts.PoolingMode.Should().Be(PoolingMode.Mean, "default pooling should be Mean");
        opts.DoLowerCase.Should().BeTrue("default should lowercase for uncased models");
        opts.Provider.Should().Be(ExecutionProvider.Auto, "default provider should be Auto");
        opts.CacheDirectory.Should().BeNull("default cache directory should be null");
        opts.ThreadCount.Should().BeNull("default thread count should be null");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_WellKnownModels_Embedder_HasValidConstants()
    {
        WellKnownModels.Embedder.Default.Should().Contain("bge-small-en");
        WellKnownModels.Embedder.Fast.Should().Contain("MiniLM");
        WellKnownModels.Embedder.Quality.Should().Contain("gte-base");
        WellKnownModels.Embedder.Large.Should().Contain("nomic");
        WellKnownModels.Embedder.Multilingual.Should().Contain("multilingual");
        WellKnownModels.Embedder.MultilingualLarge.Should().Contain("bge-m3");
    }

    // ── GetAllModels API (no model loading) ───────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ReturnsDeduplicatedModelInfo()
    {
        var models = LocalEmbedder.GetAllModels().ToList();

        // 14 unique models by RepoId in the registry
        models.Count.Should().BeGreaterThanOrEqualTo(14,
            "registry should have at least 14 unique models");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ContainsBgeSmall()
    {
        var models = LocalEmbedder.GetAllModels().ToList();

        models.Should().Contain(m => m.RepoId == "BAAI/bge-small-en-v1.5",
            "default model BGE-small should be in GetAllModels");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ModelsHaveValidProperties()
    {
        var models = LocalEmbedder.GetAllModels().ToList();

        foreach (var model in models)
        {
            model.RepoId.Should().NotBeNullOrEmpty($"RepoId should be set for all models");
            model.Dimensions.Should().BeGreaterThan(0, $"Dimensions should be positive for {model.RepoId}");
            model.MaxSequenceLength.Should().BeGreaterThan(0, $"MaxSequenceLength should be positive for {model.RepoId}");
        }
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_NoDuplicateRepoIds()
    {
        var models = LocalEmbedder.GetAllModels().ToList();
        var repoIds = models.Select(m => m.RepoId).ToList();

        repoIds.Should().OnlyHaveUniqueItems("GetAllModels should return deduplicated models");
    }

    // ── Vector Operation Edge Cases (static, no model) ───────────

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_CosineSimilarity_EmptyVectors_ThrowsArgumentException()
    {
        float[] empty = [];
        float[] vec = [1.0f, 2.0f];

        var act = () => LocalEmbedder.CosineSimilarity(empty, vec);
        act.Should().Throw<ArgumentException>("empty vector should throw");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_CosineSimilarity_MismatchedLengths_ThrowsArgumentException()
    {
        float[] a = [1.0f, 2.0f];
        float[] b = [1.0f, 2.0f, 3.0f];

        var act = () => LocalEmbedder.CosineSimilarity(a, b);
        act.Should().Throw<ArgumentException>("mismatched lengths should throw");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_EuclideanDistance_EmptyVectors_ThrowsArgumentException()
    {
        float[] empty = [];
        float[] vec = [1.0f];

        var act = () => LocalEmbedder.EuclideanDistance(empty, vec);
        act.Should().Throw<ArgumentException>("empty vector should throw");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_DotProduct_EmptyVectors_ThrowsArgumentException()
    {
        float[] empty = [];
        float[] vec = [1.0f];

        var act = () => LocalEmbedder.DotProduct(empty, vec);
        act.Should().Throw<ArgumentException>("empty vector should throw");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_DotProduct_MismatchedLengths_ThrowsArgumentException()
    {
        float[] a = [1.0f];
        float[] b = [1.0f, 2.0f];

        var act = () => LocalEmbedder.DotProduct(a, b);
        act.Should().Throw<ArgumentException>("mismatched lengths should throw");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_CosineSimilarity_LargeVectors_Works()
    {
        // Simulate realistic embedding dimensions
        var a = Enumerable.Range(0, 384).Select(i => (float)i / 384).ToArray();
        var b = Enumerable.Range(0, 384).Select(i => (float)(383 - i) / 384).ToArray();

        var sim = LocalEmbedder.CosineSimilarity(a, b);

        sim.Should().BeInRange(-1.0f, 1.0f, "cosine similarity must be in [-1, 1]");
    }

    // ── GGUF Model Tests (E-Q4) ───────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    [Trait("Category", "GGUF")]
    public async Task L_GgufModel_LoadsSuccessfully()
    {
        // E-Q4: Load a GGUF embedding model via llama-server
        // Requires llama-server binary (auto-downloaded by LlamaServerUpdateService)
        await using var model = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
        model.Dimensions.Should().BeGreaterThan(0, "GGUF embedder should report dimensions");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    [Trait("Category", "GGUF")]
    public async Task Q_GgufVsOnnx_BothProduceMeaningfulEmbeddings()
    {
        // E-Q4: Both GGUF and ONNX should produce valid embeddings for the same text
        await using var onnxModel = await LocalEmbedder.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);
        await using var ggufModel = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF", cancellationToken: TestContext.Current.CancellationToken);

        var text = "Machine learning is a subset of artificial intelligence.";

        var onnxEmb = await onnxModel.EmbedAsync(text, TestContext.Current.CancellationToken);
        var ggufEmb = await ggufModel.EmbedAsync(text, TestContext.Current.CancellationToken);

        onnxEmb.Length.Should().BeGreaterThan(0, "ONNX embedding should have dimensions");
        ggufEmb.Length.Should().BeGreaterThan(0, "GGUF embedding should have dimensions");

        // Both should be normalized (L2 norm ≈ 1.0)
        var onnxNorm = MathF.Sqrt(onnxEmb.Sum(x => x * x));
        var ggufNorm = MathF.Sqrt(ggufEmb.Sum(x => x * x));
        onnxNorm.Should().BeApproximately(1.0f, 0.1f, "ONNX embedding should be roughly normalized");
        ggufNorm.Should().BeApproximately(1.0f, 0.1f, "GGUF embedding should be roughly normalized");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    [Trait("Category", "GGUF")]
    public async Task Q_GgufModel_SemanticSimilarity_IsCoherent()
    {
        // E-Q4: GGUF embeddings should preserve semantic similarity
        await using var model = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF", cancellationToken: TestContext.Current.CancellationToken);

        var embCat = await model.EmbedAsync("The cat sat on the mat.", TestContext.Current.CancellationToken);
        var embKitten = await model.EmbedAsync("A kitten rested on the rug.", TestContext.Current.CancellationToken);
        var embCar = await model.EmbedAsync("The car drove down the highway.", TestContext.Current.CancellationToken);

        var simRelated = LocalEmbedder.CosineSimilarity(embCat, embKitten);
        var simUnrelated = LocalEmbedder.CosineSimilarity(embCat, embCar);

        simRelated.Should().BeGreaterThan(simUnrelated,
            "semantically similar texts should have higher similarity");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    [Trait("Category", "GGUF")]
    public async Task Q_GgufVsOnnx_SameModelFamily_HighCosineSimilarity()
    {
        // E-Q4: Same model family (nomic-embed-text-v1.5) in GGUF and ONNX
        // should produce very similar embeddings (cosine similarity > 0.8)
        await using var onnxModel = await LocalEmbedder.LoadAsync("large", cancellationToken: TestContext.Current.CancellationToken); // nomic-embed-text-v1.5 ONNX
        await using var ggufModel = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF", cancellationToken: TestContext.Current.CancellationToken);

        var text = "Machine learning is a subset of artificial intelligence.";

        var onnxEmb = await onnxModel.EmbedAsync(text, TestContext.Current.CancellationToken);
        var ggufEmb = await ggufModel.EmbedAsync(text, TestContext.Current.CancellationToken);

        // Same model family should produce same-dimension embeddings
        onnxEmb.Length.Should().Be(ggufEmb.Length,
            "ONNX and GGUF of same model family should have same dimensions");

        // Embeddings from same model family should be highly similar
        var similarity = LocalEmbedder.CosineSimilarity(onnxEmb, ggufEmb);
        similarity.Should().BeGreaterThan(0.8f,
            "same model family in GGUF vs ONNX should produce similar embeddings");
    }
}
