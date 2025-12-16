# LMSupply.Text.Core

Core text processing infrastructure for LMSupply packages.

## Overview

This package provides centralized tokenization and text processing utilities used by LMSupply packages that work with text data (Embedder, Reranker, Translator, etc.).

## Features

- **Tokenizer Factory**: Creates tokenizers from model directories
- **Multiple Tokenizer Types**: WordPiece, BPE, SentencePiece support
- **Vocabulary Loading**: JSON and TXT format support
- **Batch Encoding**: Efficient batch processing with padding

## Usage

This is an infrastructure package typically used internally by other LMSupply packages.

```csharp
using LMSupply.Text;

// Create a tokenizer from model directory
var tokenizer = TokenizerFactory.CreateFromModelDirectory(modelPath);

// Encode text
var encoded = tokenizer.Encode("Hello, world!");
Console.WriteLine($"Tokens: {encoded.InputIds.Length}");

// Decode tokens
var decoded = tokenizer.Decode(encoded.InputIds, skipSpecialTokens: true);
```

## Supported Tokenizers

| Type | Format | Models |
|------|--------|--------|
| WordPiece | tokenizer.json, vocab.txt | BERT, BGE |
| BPE | tokenizer.json, vocab.json | GPT-2, RoBERTa |
| SentencePiece | tokenizer.model | XLM-R, mBART |
