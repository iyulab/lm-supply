namespace LMSupply.Transcriber.Diarization;

/// <summary>
/// Complete-linkage agglomerative clustering of L2-normalized embeddings on cosine distance (1 − cosine similarity),
/// cut either at a distance threshold or at a fixed cluster count. The nearest-neighbour-chain algorithm gives the
/// exact complete-linkage hierarchy in O(n²) time over a condensed distance matrix.
/// </summary>
internal static class AgglomerativeClustering
{
    /// <summary>
    /// Clusters <paramref name="embeddings"/> (row-major, <paramref name="dim"/> columns). Returns a label per row,
    /// numbered 0.. in order of first appearance. With <paramref name="numClusters"/> &gt; 0 the threshold is ignored.
    /// </summary>
    public static int[] Cluster(float[] embeddings, int dim, float threshold, int numClusters = 0)
    {
        var n = embeddings.Length / dim;
        if (n == 0)
            return [];
        if (n == 1)
            return [0];

        var normalized = (float[])embeddings.Clone();
        for (var i = 0; i < n; i++)
        {
            var row = normalized.AsSpan(i * dim, dim);
            var norm = 0f;
            foreach (var v in row)
                norm += v * v;
            norm = MathF.Sqrt(norm);
            if (norm > 0)
                for (var k = 0; k < dim; k++)
                    row[k] /= norm;
        }

        var distance = new float[(long)n * (n - 1) / 2];
        for (var i = 0; i < n; i++)
        {
            var a = normalized.AsSpan(i * dim, dim);
            for (var j = i + 1; j < n; j++)
            {
                var b = normalized.AsSpan(j * dim, dim);
                var dot = 0f;
                for (var k = 0; k < dim; k++)
                    dot += a[k] * b[k];
                distance[Index(i, j, n)] = Math.Max(0f, 1f - dot);
            }
        }

        var merges = NearestNeighborChain(distance, n);
        merges.Sort((x, y) => x.Height.CompareTo(y.Height));

        // Apply merges in height order: all below the threshold, or the first n − k.
        var parent = new int[n];
        for (var i = 0; i < n; i++)
            parent[i] = i;
        var toApply = numClusters > 0
            ? Math.Max(0, n - numClusters)
            : merges.Count(m => m.Height <= threshold);
        for (var m = 0; m < toApply && m < merges.Count; m++)
            Union(parent, merges[m].A, merges[m].B);

        var labels = new int[n];
        var ids = new Dictionary<int, int>();
        for (var i = 0; i < n; i++)
        {
            var root = Find(parent, i);
            if (!ids.TryGetValue(root, out var id))
                ids[root] = id = ids.Count;
            labels[i] = id;
        }
        return labels;
    }

    private static long Index(int i, int j, int n)
    {
        if (i > j)
            (i, j) = (j, i);
        return (long)i * n - (long)i * (i + 1) / 2 + (j - i - 1);
    }

    /// <summary>
    /// Nearest-neighbour chain for complete linkage (a reducible linkage, so the chain finds the exact hierarchy).
    /// Clusters are represented by one member index; merges carry a representative of each side.
    /// </summary>
    private static List<(int A, int B, float Height)> NearestNeighborChain(float[] distance, int n)
    {
        var active = new bool[n];
        Array.Fill(active, true);
        var merges = new List<(int, int, float)>(n - 1);
        var chain = new Stack<int>();
        var remaining = n;

        while (remaining > 1)
        {
            if (chain.Count == 0)
                for (var i = 0; i < n; i++)
                    if (active[i])
                    {
                        chain.Push(i);
                        break;
                    }

            while (true)
            {
                var top = chain.Peek();
                var previous = chain.Count > 1 ? chain.ElementAt(1) : -1;

                // Nearest active neighbour of top; prefer the previous chain element on ties so the chain terminates.
                var best = -1;
                var bestDistance = float.PositiveInfinity;
                if (previous >= 0)
                {
                    best = previous;
                    bestDistance = distance[Index(top, previous, n)];
                }
                for (var j = 0; j < n; j++)
                {
                    if (!active[j] || j == top)
                        continue;
                    var d = distance[Index(top, j, n)];
                    if (d < bestDistance)
                    {
                        bestDistance = d;
                        best = j;
                    }
                }

                if (best == previous)
                {
                    chain.Pop();
                    chain.Pop();
                    merges.Add((top, previous, bestDistance));

                    // Complete linkage: the merged cluster (kept at `previous`) is as far as the farther of the two.
                    active[top] = false;
                    for (var k = 0; k < n; k++)
                    {
                        if (!active[k] || k == previous)
                            continue;
                        var merged = Math.Max(distance[Index(previous, k, n)], distance[Index(top, k, n)]);
                        distance[Index(previous, k, n)] = merged;
                    }
                    remaining--;
                    break;
                }

                chain.Push(best);
            }
        }

        return merges;
    }

    private static int Find(int[] parent, int i)
    {
        while (parent[i] != i)
        {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }

    private static void Union(int[] parent, int a, int b)
    {
        var ra = Find(parent, a);
        var rb = Find(parent, b);
        if (ra != rb)
            parent[rb] = ra;
    }
}
