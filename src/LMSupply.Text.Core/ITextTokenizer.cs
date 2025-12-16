namespace LMSupply.Text;

/// <summary>
/// Common interface for text tokenizers.
/// Provides encode/decode operations with support for special tokens.
/// </summary>
public interface ITextTokenizer : IDisposable
{
    /// <summary>
    /// Gets the vocabulary size.
    /// </summary>
    int VocabSize { get; }

    /// <summary>
    /// Gets the PAD (padding) token ID.
    /// </summary>
    int PadTokenId { get; }

    /// <summary>
    /// Gets the UNK (unknown) token ID.
    /// </summary>
    int UnkTokenId { get; }

    /// <summary>
    /// Gets the BOS (beginning of sequence) token ID, if available.
    /// </summary>
    int? BosTokenId { get; }

    /// <summary>
    /// Gets the EOS (end of sequence) token ID, if available.
    /// </summary>
    int? EosTokenId { get; }

    /// <summary>
    /// Gets the CLS (classification) token ID, if available (BERT-style).
    /// </summary>
    int? ClsTokenId { get; }

    /// <summary>
    /// Gets the SEP (separator) token ID, if available (BERT-style).
    /// </summary>
    int? SepTokenId { get; }

    /// <summary>
    /// Encodes text into token IDs.
    /// </summary>
    /// <param name="text">Text to encode.</param>
    /// <param name="addSpecialTokens">Whether to add special tokens (BOS/EOS or CLS/SEP).</param>
    /// <returns>Array of token IDs.</returns>
    int[] Encode(string text, bool addSpecialTokens = true);

    /// <summary>
    /// Decodes token IDs back to text.
    /// </summary>
    /// <param name="tokenIds">Token IDs to decode.</param>
    /// <param name="skipSpecialTokens">Whether to skip special tokens in output.</param>
    /// <returns>Decoded text string.</returns>
    string Decode(ReadOnlySpan<int> tokenIds, bool skipSpecialTokens = true);

    /// <summary>
    /// Checks if a token ID is a special token.
    /// </summary>
    /// <param name="tokenId">Token ID to check.</param>
    /// <returns>True if the token is a special token.</returns>
    bool IsSpecialToken(int tokenId);
}

/// <summary>
/// Extended tokenizer interface for sequence encoding with attention masks.
/// Used for encoder models (BERT, embedders, rerankers).
/// </summary>
public interface ISequenceTokenizer : ITextTokenizer
{
    /// <summary>
    /// Gets the maximum sequence length supported.
    /// </summary>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Encodes text into padded sequences with attention mask.
    /// </summary>
    /// <param name="text">Text to encode.</param>
    /// <param name="maxLength">Maximum sequence length (uses default if null).</param>
    /// <returns>Encoded sequence with input IDs and attention mask.</returns>
    EncodedSequence EncodeSequence(string text, int? maxLength = null);

    /// <summary>
    /// Encodes multiple texts into a batch.
    /// </summary>
    /// <param name="texts">Texts to encode.</param>
    /// <param name="maxLength">Maximum sequence length (uses default if null).</param>
    /// <returns>Batch of encoded sequences.</returns>
    EncodedBatch EncodeBatch(IReadOnlyList<string> texts, int? maxLength = null);
}

/// <summary>
/// Tokenizer interface for pair encoding (cross-encoders, rerankers).
/// </summary>
public interface IPairTokenizer : ISequenceTokenizer
{
    /// <summary>
    /// Encodes a text pair (e.g., query and document).
    /// Format: [CLS] text1 [SEP] text2 [SEP] [PAD...]
    /// </summary>
    /// <param name="text1">First text (e.g., query).</param>
    /// <param name="text2">Second text (e.g., document).</param>
    /// <param name="maxLength">Maximum sequence length (uses default if null).</param>
    /// <returns>Encoded pair with token type IDs.</returns>
    EncodedPair EncodePair(string text1, string text2, int? maxLength = null);

    /// <summary>
    /// Encodes multiple pairs with the same first text.
    /// </summary>
    /// <param name="text1">First text (e.g., query).</param>
    /// <param name="texts2">Second texts (e.g., documents).</param>
    /// <param name="maxLength">Maximum sequence length (uses default if null).</param>
    /// <returns>Batch of encoded pairs.</returns>
    EncodedPairBatch EncodePairBatch(string text1, IReadOnlyList<string> texts2, int? maxLength = null);
}
