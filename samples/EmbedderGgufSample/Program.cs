using System.Globalization;
using LMSupply;
using LMSupply.Embedder;

Console.WriteLine("=== LMSupply GGUF Embedder Sample ===\n");

// Progress reporter
var progress = new Progress<DownloadProgress>(p =>
{
    if (p.TotalBytes > 0)
    {
        var percent = (double)p.BytesDownloaded / p.TotalBytes;
        Console.Write($"\rDownloading {p.FileName}: {percent:P0}");
    }
});

// =============================================================================
// Load GGUF Embedding Model
// =============================================================================
// GGUF models are auto-detected by repo name containing "-GGUF" or ".gguf" extension
// Popular GGUF embedding models:
// - nomic-ai/nomic-embed-text-v1.5-GGUF (768 dimensions, 8K context)
// - BAAI/bge-small-en-v1.5-GGUF (384 dimensions)

Console.WriteLine("Loading GGUF embedding model...");
Console.WriteLine("Note: This downloads the model on first use (~300-500MB).\n");

await using var model = await LocalEmbedder.LoadAsync(
    "nomic-ai/nomic-embed-text-v1.5-GGUF",  // Auto-detected as GGUF by "-GGUF" suffix
    progress: progress
);
Console.WriteLine();

var info = model.GetModelInfo();
Console.WriteLine($"Model: {info?.RepoId ?? model.ModelId}");
Console.WriteLine($"Dimensions: {model.Dimensions}");
Console.WriteLine();

// =============================================================================
// 1. Basic Embedding
// =============================================================================
Console.WriteLine("1. Basic Embedding");
Console.WriteLine(new string('=', 60));

var text = "Artificial intelligence is transforming how we work and live.";
var embedding = await model.EmbedAsync(text);

Console.WriteLine($"Text: \"{text}\"");
Console.WriteLine($"Embedding dimensions: {embedding.Length}");
Console.WriteLine($"First 5 values: [{string.Join(", ", embedding.Take(5).Select(v => v.ToString("F4", CultureInfo.InvariantCulture)))}...]");
Console.WriteLine();

// =============================================================================
// 2. Semantic Similarity
// =============================================================================
Console.WriteLine("2. Semantic Similarity");
Console.WriteLine(new string('=', 60));

var sentences = new[]
{
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "The dog ran in the park.",
    "Machine learning models process data."
};

Console.WriteLine("Sentences:");
for (int i = 0; i < sentences.Length; i++)
{
    Console.WriteLine($"  [{i}] {sentences[i]}");
}
Console.WriteLine();

// Generate embeddings for all sentences
var embeddings = await model.EmbedAsync(sentences);

// Compute similarity matrix
Console.WriteLine("Similarity Matrix (cosine similarity):");
Console.Write("      ");
for (int i = 0; i < sentences.Length; i++)
{
    Console.Write($"  [{i}]  ");
}
Console.WriteLine();

for (int i = 0; i < sentences.Length; i++)
{
    Console.Write($"[{i}]  ");
    for (int j = 0; j < sentences.Length; j++)
    {
        var similarity = LocalEmbedder.CosineSimilarity(embeddings[i], embeddings[j]);
        Console.Write($" {similarity:F3} ");
    }
    Console.WriteLine();
}
Console.WriteLine();

Console.WriteLine("Analysis:");
Console.WriteLine($"  - Sentences 0 and 1 are semantically similar (cat/feline, mat/rug)");
Console.WriteLine($"    Similarity: {LocalEmbedder.CosineSimilarity(embeddings[0], embeddings[1]):F3}");
Console.WriteLine($"  - Sentences 0 and 3 are quite different (cat vs ML)");
Console.WriteLine($"    Similarity: {LocalEmbedder.CosineSimilarity(embeddings[0], embeddings[3]):F3}");
Console.WriteLine();

// =============================================================================
// 3. Semantic Search
// =============================================================================
Console.WriteLine("3. Semantic Search");
Console.WriteLine(new string('=', 60));

var documents = new[]
{
    "Python is a popular programming language for data science.",
    "The Eiffel Tower is located in Paris, France.",
    "Neural networks are a fundamental part of deep learning.",
    "The Great Wall of China is visible from space.",
    "TensorFlow and PyTorch are popular machine learning frameworks."
};

Console.WriteLine("Documents:");
foreach (var doc in documents)
{
    Console.WriteLine($"  - {doc}");
}
Console.WriteLine();

// Embed all documents (the passage side of retrieval)
var docEmbeddings = await model.EmbedPassageAsync(documents);

// Search query
var query = "What are good tools for machine learning?";
Console.WriteLine($"Query: \"{query}\"");
Console.WriteLine();

var queryEmbedding = await model.EmbedQueryAsync(query);

// Rank by similarity
var results = documents
    .Select((doc, idx) => new
    {
        Document = doc,
        Score = LocalEmbedder.CosineSimilarity(queryEmbedding, docEmbeddings[idx])
    })
    .OrderByDescending(r => r.Score)
    .ToList();

Console.WriteLine("Search Results (ranked by relevance):");
for (int i = 0; i < results.Count; i++)
{
    Console.WriteLine($"  {i + 1}. [{results[i].Score:F3}] {results[i].Document}");
}
Console.WriteLine();

// =============================================================================
// 4. Batch Processing Performance
// =============================================================================
Console.WriteLine("4. Batch Processing");
Console.WriteLine(new string('=', 60));

var batchTexts = Enumerable.Range(1, 10)
    .Select(i => $"This is sample text number {i} for batch processing.")
    .ToList();

Console.WriteLine($"Processing {batchTexts.Count} texts...");

var sw = System.Diagnostics.Stopwatch.StartNew();
var batchEmbeddings = await model.EmbedAsync(batchTexts);
sw.Stop();

Console.WriteLine($"Time: {sw.ElapsedMilliseconds}ms");
Console.WriteLine($"Average: {sw.ElapsedMilliseconds / (double)batchTexts.Count:F1}ms per text");
Console.WriteLine($"Generated {batchEmbeddings.Length} embeddings of {batchEmbeddings[0].Length} dimensions each");
Console.WriteLine();

// =============================================================================
// 5. Text Clustering (Simple K-Means-like)
// =============================================================================
Console.WriteLine("5. Text Clustering (by similarity)");
Console.WriteLine(new string('=', 60));

var clusterTexts = new[]
{
    // Cluster 1: Technology
    "Artificial intelligence is changing industries.",
    "Machine learning models improve with more data.",
    "Cloud computing enables scalable applications.",
    // Cluster 2: Nature
    "The forest is home to many species of birds.",
    "Rivers flow from mountains to the sea.",
    "Flowers bloom in spring after the winter frost.",
    // Cluster 3: Food
    "Italian pasta is often served with tomato sauce.",
    "Sushi is a traditional Japanese dish.",
    "French cuisine is known for its sophistication."
};

Console.WriteLine("Texts to cluster:");
foreach (var t in clusterTexts)
{
    Console.WriteLine($"  - {t}");
}
Console.WriteLine();

var clusterEmbeddings = await model.EmbedAsync(clusterTexts);

// Simple clustering: Find which texts are most similar to each other
Console.WriteLine("Similarity Groups (texts with similarity > 0.5):");

var grouped = new List<List<int>>();
var assigned = new HashSet<int>();

for (int i = 0; i < clusterTexts.Length; i++)
{
    if (assigned.Contains(i)) continue;

    var group = new List<int> { i };
    assigned.Add(i);

    for (int j = i + 1; j < clusterTexts.Length; j++)
    {
        if (assigned.Contains(j)) continue;

        var sim = LocalEmbedder.CosineSimilarity(clusterEmbeddings[i], clusterEmbeddings[j]);
        if (sim > 0.5)
        {
            group.Add(j);
            assigned.Add(j);
        }
    }

    grouped.Add(group);
}

for (int g = 0; g < grouped.Count; g++)
{
    Console.WriteLine($"  Group {g + 1}:");
    foreach (var idx in grouped[g])
    {
        Console.WriteLine($"    - {clusterTexts[idx]}");
    }
}
Console.WriteLine();

Console.WriteLine("=== All Samples Complete ===");
