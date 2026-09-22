namespace LMSupply.Download;

/// <summary>
/// A file in the model cache that <see cref="CacheManager.FindReclaimable"/> judged safe to delete: a
/// copy at a snapshot's root whose byte-identical twin lives in a subfolder of the same snapshot and is
/// what the manifest says the loader reads.
/// </summary>
/// <param name="RepoId">The repository whose snapshot holds the file.</param>
/// <param name="Path">The absolute path of the reclaimable copy (the one to delete).</param>
/// <param name="Size">Its length in bytes — what deleting it frees.</param>
/// <param name="TwinPath">The absolute path of the copy that stays (the one the manifest lists).</param>
/// <param name="Reason">A sentence a consumer can show as it is.</param>
public sealed record ReclaimableFile(string RepoId, string Path, long Size, string TwinPath, string Reason);
