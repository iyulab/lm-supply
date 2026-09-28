namespace LMSupply.Transcriber.Diarization;

/// <summary>
/// Complete-linkage agglomerative clustering of L2-normalized embeddings on cosine distance (1 − cosine similarity),
/// cut at a distance threshold or at a speaker count, then pyannote's small-cluster rule: a cluster with fewer members
/// than <see cref="MinClusterSize"/> (scaled down for short recordings) is not a speaker of its own — it is reassigned
/// to the nearest large cluster by centroid, and a requested count counts large clusters only. Without that rule a
/// one-window outlier takes one of the requested slots and its turns vanish in aggregation, so a count of 3 yields 2.
/// The nearest-neighbour-chain algorithm gives the exact complete-linkage hierarchy in O(n²) time over a condensed
/// distance matrix.
/// </summary>
internal static class AgglomerativeClustering
{
    /// <summary>pyannote 3.1's <c>min_cluster_size</c>; the effective size is at most a tenth of the embeddings.</summary>
    internal const int MinClusterSize = 12;

    /// <summary>
    /// Clusters <paramref name="embeddings"/> (row-major, <paramref name="dim"/> columns). Returns a label per row,
    /// numbered 0.. in order of first appearance. <paramref name="numClusters"/> &gt; 0 asks for that many large
    /// clusters; otherwise the threshold decides, clamped to [<paramref name="minClusters"/>,
    /// <paramref name="maxClusters"/>] when those are &gt; 0. A requested count is met with the cut nearest the
    /// threshold that gives it, or the nearest count the hierarchy allows.
    /// </summary>
    public static int[] Cluster(
        float[] embeddings, int dim, float threshold, int numClusters = 0, int minClusters = 0, int maxClusters = 0)
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

        // large[m] = the number of large clusters after the first m merges.
        var minSize = Math.Min(MinClusterSize, Math.Max(1, (int)Math.Round(0.1 * n)));
        var large = LargeCounts(merges, n, minSize);
        var cut = merges.Count(m => m.Height <= threshold);

        var target = numClusters > 0 ? numClusters
            : minClusters > 0 && large[cut] < minClusters ? minClusters
            : maxClusters > 0 && large[cut] > maxClusters ? maxClusters
            : 0;
        if (target > 0)
        {
            var best = cut;
            for (var m = 0; m <= merges.Count; m++)
            {
                var gap = Math.Abs(large[m] - target);
                var bestGap = Math.Abs(large[best] - target);
                if (gap < bestGap || (gap == bestGap && HeightGap(merges, m, threshold) < HeightGap(merges, best, threshold)))
                    best = m;
            }
            cut = best;
        }

        var parent = new int[n];
        for (var i = 0; i < n; i++)
            parent[i] = i;
        for (var m = 0; m < cut; m++)
            Union(parent, merges[m].A, merges[m].B);

        var roots = new int[n];
        var members = new Dictionary<int, List<int>>();
        for (var i = 0; i < n; i++)
        {
            roots[i] = Find(parent, i);
            if (!members.TryGetValue(roots[i], out var list))
                members[roots[i]] = list = [];
            list.Add(i);
        }

        // Small clusters join the nearest large cluster by centroid; with no large cluster, everything is one speaker.
        var largeRoots = members.Where(kv => kv.Value.Count >= minSize).Select(kv => kv.Key).ToList();
        if (largeRoots.Count == 0)
            return new int[n];
        var centroids = members.ToDictionary(kv => kv.Key, kv => Centroid(normalized, dim, kv.Value));
        var assigned = new Dictionary<int, int>();
        foreach (var (root, list) in members)
        {
            if (list.Count >= minSize)
            {
                assigned[root] = root;
                continue;
            }
            var own = centroids[root];
            assigned[root] = largeRoots.MaxBy(r => Dot(own, centroids[r]));
        }

        var labels = new int[n];
        var ids = new Dictionary<int, int>();
        for (var i = 0; i < n; i++)
        {
            var root = assigned[roots[i]];
            if (!ids.TryGetValue(root, out var id))
                ids[root] = id = ids.Count;
            labels[i] = id;
        }
        return labels;
    }

    private static int[] LargeCounts(List<(int A, int B, float Height)> merges, int n, int minSize)
    {
        var parent = new int[n];
        var size = new int[n];
        for (var i = 0; i < n; i++)
        {
            parent[i] = i;
            size[i] = 1;
        }
        var large = new int[merges.Count + 1];
        large[0] = minSize <= 1 ? n : 0;
        for (var m = 0; m < merges.Count; m++)
        {
            var ra = Find(parent, merges[m].A);
            var rb = Find(parent, merges[m].B);
            var count = large[m];
            if (ra != rb)
            {
                count -= (size[ra] >= minSize ? 1 : 0) + (size[rb] >= minSize ? 1 : 0);
                parent[rb] = ra;
                size[ra] += size[rb];
                count += size[ra] >= minSize ? 1 : 0;
            }
            large[m + 1] = count;
        }
        return large;
    }

    private static float HeightGap(List<(int A, int B, float Height)> merges, int m, float threshold) =>
        Math.Abs((m == 0 ? 0f : merges[m - 1].Height) - threshold);

    private static float[] Centroid(float[] normalized, int dim, List<int> rows)
    {
        var c = new float[dim];
        foreach (var r in rows)
            for (var k = 0; k < dim; k++)
                c[k] += normalized[r * dim + k];
        var norm = MathF.Sqrt(c.Sum(v => v * v));
        if (norm > 0)
            for (var k = 0; k < dim; k++)
                c[k] /= norm;
        return c;
    }

    private static float Dot(float[] a, float[] b)
    {
        var d = 0f;
        for (var k = 0; k < a.Length; k++)
            d += a[k] * b[k];
        return d;
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
