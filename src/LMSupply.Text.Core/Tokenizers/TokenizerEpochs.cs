namespace LMSupply.Text;

/// <summary>
/// The implementation epoch of each tokenization algorithm in this library. An epoch is raised when a
/// release changes the ids an algorithm produces for the same model files — never for a performance or
/// API change. It is part of <see cref="ISequenceTokenizer.Signature"/>, so a consumer that stores what
/// depends on those ids (embedding vectors, for one) can see that a release moved them.
/// </summary>
/// <remarks>
/// History, so the numbers can be audited:
/// <list type="bullet">
/// <item><description><see cref="WordPiece"/> 1 → 2 in 0.70.0: BERT basic tokenization (cleaning, CJK
/// and punctuation splitting, lowercasing, accent stripping) runs before WordPiece; uncased models no
/// longer map capitalised or punctuated words to <c>[UNK]</c>.</description></item>
/// <item><description><see cref="SentencePiece"/> 1 → 2 (XLM-R id offset fix): raw SentencePiece ids are
/// mapped onto the model's vocabulary ids when the two disagree by a constant offset.</description></item>
/// </list>
/// Raising an epoch is a deliberate act: the golden-vector facts in the integration suite fail when the
/// vectors move and the epoch does not, and when the epoch moves and the vectors do not.
/// </remarks>
internal static class TokenizerEpochs
{
    /// <summary>WordPiece with BERT basic tokenization.</summary>
    public const int WordPiece = 2;

    /// <summary>SentencePiece (Unigram/BPE) with the vocabulary id map.</summary>
    public const int SentencePiece = 2;
}
