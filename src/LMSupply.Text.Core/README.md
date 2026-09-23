# LMSupply.Text.Core

Core text processing infrastructure for LMSupply packages.

## Overview

This package provides centralized tokenization and text processing utilities used by LMSupply packages that work with text data (Embedder, Reranker, Translator, etc.).

## Features

- **Tokenizer Factory**: Creates tokenizers from model directories with auto-detection
- **Multiple Tokenizer Types**: WordPiece, BPE, Unigram, SentencePiece support
- **Pair Encoding**: Cross-encoder tokenization for rerankers
- **Vocabulary Loading**: JSON and TXT format support
- **Batch Encoding**: Efficient batch processing with padding

## Tokenizer Interfaces

| Interface | Purpose | Use Case |
|-----------|---------|----------|
| `ITextTokenizer` | Basic encode/decode | General tokenization |
| `ISequenceTokenizer` | Single sequence with special tokens | Embeddings |
| `IPairTokenizer` | Sentence pair encoding | Rerankers, cross-encoders |

## TokenizerFactory Methods

### Single Sequence Tokenizers

```csharp
// Auto-detect and create appropriate tokenizer
var tokenizer = await TokenizerFactory.CreateAutoAsync(modelDir, maxSequenceLength: 512);

// Specific tokenizer types
var wordpiece = await TokenizerFactory.CreateWordPieceAsync(modelDir, maxLength);
var sentencepiece = await TokenizerFactory.CreateSentencePieceSequenceAsync(modelDir, maxLength);
```

### Pair Tokenizers (for Cross-Encoders)

```csharp
// Auto-detect tokenizer type and create pair tokenizer (recommended)
var pairTokenizer = await TokenizerFactory.CreateAutoPairAsync(modelDir, maxSequenceLength: 512);

// Specific pair tokenizer types
var wordpiecePair = await TokenizerFactory.CreateWordPiecePairAsync(modelDir, maxLength);
var sentencepiecePair = await TokenizerFactory.CreateSentencePiecePairAsync(modelDir, maxLength);
```

## Supported Tokenizer Types

| Type | Detection | Example Models |
|------|-----------|----------------|
| WordPiece | `vocab.txt` or `tokenizer.json` with `type: WordPiece` | BERT, MiniLM, BGE-v1 |
| Unigram | `tokenizer.json` with `type: Unigram` | bge-reranker-base, XLM-RoBERTa |
| BPE | `vocab.json` + `merges.txt` or `tokenizer.json` with `type: BPE` | GPT-2, RoBERTa |
| SentencePiece | `.spm` or `.model` files | mBART, translation models |

## Auto-Detection Logic

The `CreateAutoPairAsync` method automatically detects the tokenizer type:

1. **vocab.txt exists** → WordPiece tokenizer
   - BERT's basic tokenization runs before WordPiece — lowercasing and accent stripping for uncased
     models, punctuation and CJK ideographs split into words of their own — as the model declares it in
     `tokenizer.json` (`BertNormalizer` / `BertPreTokenizer`) or `tokenizer_config.json`
     (`do_lower_case`, `strip_accents`, `tokenize_chinese_chars`). Token ids match the HuggingFace
     `tokenizers` library.
2. **tokenizer.json exists** → Parse `model.type` field:
   - `WordPiece` → WordPiece tokenizer
   - `Unigram` or `BPE` → SentencePiece-compatible tokenizer
3. **.spm/.model exists** → SentencePiece tokenizer
4. **Fallback** → Attempt SentencePiece

## Usage Examples

### Basic Tokenization

```csharp
using LMSupply.Text;

// Create tokenizer
var tokenizer = await TokenizerFactory.CreateAutoAsync(modelPath);

// Encode text
var encoded = tokenizer.EncodeSequence("Hello, world!");
Console.WriteLine($"Tokens: {encoded.InputIds.Length}");

// Decode tokens
var decoded = tokenizer.Decode(encoded.InputIds, skipSpecialTokens: true);
```

### Pair Encoding for Rerankers

```csharp
using LMSupply.Text;

// Create pair tokenizer (auto-detects WordPiece/Unigram/BPE)
var pairTokenizer = await TokenizerFactory.CreateAutoPairAsync(modelPath, maxSequenceLength: 512);

// Encode query-document pair
var encoded = pairTokenizer.EncodePair(
    "What is machine learning?",
    "Machine learning is a branch of AI..."
);

// Batch encode for multiple documents
var batch = pairTokenizer.EncodePairBatch(
    "What is machine learning?",
    new[] { "Doc 1...", "Doc 2...", "Doc 3..." }
);
```

### Batch Processing

```csharp
var tokenizer = await TokenizerFactory.CreateAutoAsync(modelPath);

var texts = new[] { "First text", "Second text", "Third text" };
var batch = tokenizer.EncodeBatch(texts, maxLength: 256);

// Access batch tensors
long[,] inputIds = batch.InputIds;
long[,] attentionMask = batch.AttentionMask;
```

## Encoded Types

### EncodedSequence

Single encoded sequence with special tokens:

```csharp
public record EncodedSequence(
    long[] InputIds,        // Token IDs with [CLS], [SEP]
    long[] AttentionMask,   // 1 for real tokens, 0 for padding
    int ActualLength        // Length before padding
);
```

### EncodedPair

Encoded sentence pair for cross-encoders:

```csharp
public record EncodedPair(
    long[] InputIds,        // [CLS] text1 [SEP] text2 [SEP]
    long[] AttentionMask,   // Attention mask
    long[] TokenTypeIds,    // 0 for text1, 1 for text2
    int ActualLength        // Length before padding
);
```

### EncodedBatch / EncodedPairBatch

Batched versions for efficient inference:

```csharp
public class EncodedBatch
{
    public long[,] InputIds { get; }
    public long[,] AttentionMask { get; }
    public int BatchSize { get; }
    public int SequenceLength { get; }
}
```

## Special Tokens

| Token | WordPiece | SentencePiece |
|-------|-----------|---------------|
| Start | `[CLS]` | `<s>` / `<bos>` |
| Separator | `[SEP]` | `</s>` / `<eos>` |
| Padding | `[PAD]` | `<pad>` |
| Unknown | `[UNK]` | `<unk>` |

The tokenizer automatically handles special token differences between tokenizer types.

## Version History

### v0.8.7
- Added `CreateAutoPairAsync` for automatic tokenizer type detection
- Added `SentencePiecePairTokenizer` for Unigram/BPE pair encoding
- Added `CreateSentencePiecePairAsync` for explicit SentencePiece pair tokenizers
- Fixed tokenizer type mismatch for bge-reranker-base and similar Unigram models

### v0.8.6
- Fixed vocab parsing for Array vs Object format in tokenizer.json
