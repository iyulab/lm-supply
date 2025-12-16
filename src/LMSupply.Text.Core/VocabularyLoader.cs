namespace LMSupply.Text;

/// <summary>
/// Utility for loading vocabularies from different file formats.
/// </summary>
public static class VocabularyLoader
{
    /// <summary>
    /// Loads vocabulary from a vocab.txt file (BERT format, one token per line).
    /// </summary>
    /// <param name="path">Path to vocab.txt.</param>
    /// <returns>Dictionary mapping tokens to IDs.</returns>
    public static async Task<Dictionary<string, int>> LoadFromVocabTxtAsync(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vocabulary file not found: {path}", path);

        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        var lines = await File.ReadAllLinesAsync(path);

        for (int i = 0; i < lines.Length; i++)
        {
            var token = lines[i].TrimEnd();
            if (!string.IsNullOrEmpty(token) && !vocab.ContainsKey(token))
            {
                vocab[token] = i;
            }
        }

        return vocab;
    }

    /// <summary>
    /// Loads vocabulary from a vocab.json file.
    /// </summary>
    /// <param name="path">Path to vocab.json.</param>
    /// <returns>Dictionary mapping tokens to IDs.</returns>
    public static async Task<Dictionary<string, int>> LoadFromVocabJsonAsync(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vocabulary file not found: {path}", path);

        var json = await File.ReadAllTextAsync(path);
        return ParseVocabJson(json);
    }

    /// <summary>
    /// Loads vocabulary from tokenizer.json (HuggingFace format).
    /// </summary>
    /// <param name="path">Path to tokenizer.json.</param>
    /// <returns>Dictionary mapping tokens to IDs.</returns>
    public static async Task<Dictionary<string, int>> LoadFromTokenizerJsonAsync(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Tokenizer file not found: {path}", path);

        var json = await File.ReadAllTextAsync(path);
        return ExtractVocabFromTokenizerJson(json);
    }

    /// <summary>
    /// Auto-detects and loads vocabulary from a model directory.
    /// </summary>
    /// <param name="modelDir">Model directory path.</param>
    /// <returns>Dictionary mapping tokens to IDs.</returns>
    public static async Task<Dictionary<string, int>> LoadFromModelDirectoryAsync(string modelDir)
    {
        // Try vocab.txt first (BERT format)
        var vocabTxtPath = Path.Combine(modelDir, "vocab.txt");
        if (File.Exists(vocabTxtPath))
        {
            return await LoadFromVocabTxtAsync(vocabTxtPath);
        }

        // Try vocab.json
        var vocabJsonPath = Path.Combine(modelDir, "vocab.json");
        if (File.Exists(vocabJsonPath))
        {
            return await LoadFromVocabJsonAsync(vocabJsonPath);
        }

        // Try tokenizer.json
        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");
        if (File.Exists(tokenizerJsonPath))
        {
            return await LoadFromTokenizerJsonAsync(tokenizerJsonPath);
        }

        throw new FileNotFoundException(
            $"No vocabulary file found in: {modelDir}. " +
            "Expected one of: vocab.txt, vocab.json, or tokenizer.json");
    }

    /// <summary>
    /// Extracts special token IDs from tokenizer.json added_tokens section.
    /// </summary>
    /// <param name="tokenizerJsonPath">Path to tokenizer.json.</param>
    /// <returns>Special tokens configuration.</returns>
    public static SpecialTokens ExtractSpecialTokensFromJson(string tokenizerJsonPath)
    {
        try
        {
            var json = File.ReadAllText(tokenizerJsonPath);
            using var doc = JsonDocument.Parse(json);

            int? clsId = null, sepId = null, padId = null, unkId = null;
            int? bosId = null, eosId = null, maskId = null;

            // Extract from added_tokens section
            if (doc.RootElement.TryGetProperty("added_tokens", out var addedTokens))
            {
                foreach (var token in addedTokens.EnumerateArray())
                {
                    if (token.TryGetProperty("content", out var content) &&
                        token.TryGetProperty("id", out var id))
                    {
                        var contentStr = content.GetString();
                        var tokenId = id.GetInt32();

                        switch (contentStr)
                        {
                            case "[CLS]": clsId = tokenId; break;
                            case "[SEP]": sepId = tokenId; break;
                            case "[PAD]": padId = tokenId; break;
                            case "[UNK]": unkId = tokenId; break;
                            case "[MASK]": maskId = tokenId; break;
                            case "<s>": bosId = tokenId; break;
                            case "</s>": eosId = tokenId; break;
                            case "<pad>": padId ??= tokenId; break;
                            case "<unk>": unkId ??= tokenId; break;
                        }
                    }
                }
            }

            return new SpecialTokens
            {
                ClsTokenId = clsId,
                SepTokenId = sepId,
                PadTokenId = padId ?? 0,
                UnkTokenId = unkId ?? 0,
                BosTokenId = bosId,
                EosTokenId = eosId,
                MaskTokenId = maskId
            };
        }
        catch
        {
            // Return BERT defaults if parsing fails
            return SpecialTokens.Bert;
        }
    }

    private static Dictionary<string, int> ParseVocabJson(string json)
    {
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);

        try
        {
            using var doc = JsonDocument.Parse(json);

            foreach (var property in doc.RootElement.EnumerateObject())
            {
                if (property.Value.TryGetInt32(out var id))
                {
                    vocab[property.Name] = id;
                }
            }
        }
        catch
        {
            // Fallback to manual parsing for malformed JSON
            vocab = ParseVocabJsonManual(json);
        }

        return vocab;
    }

    private static Dictionary<string, int> ParseVocabJsonManual(string json)
    {
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        var cleanJson = json.Trim();

        if (!cleanJson.StartsWith("{") || !cleanJson.EndsWith("}"))
            return vocab;

        cleanJson = cleanJson[1..^1]; // Remove { and }

        foreach (var entry in SplitJsonEntries(cleanJson))
        {
            var parts = entry.Split(':', 2);
            if (parts.Length != 2) continue;

            var token = parts[0].Trim().Trim('"');
            if (int.TryParse(parts[1].Trim(), out var id))
            {
                vocab[token] = id;
            }
        }

        return vocab;
    }

    private static Dictionary<string, int> ExtractVocabFromTokenizerJson(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);

            if (doc.RootElement.TryGetProperty("model", out var model) &&
                model.TryGetProperty("vocab", out var vocab))
            {
                var result = new Dictionary<string, int>(StringComparer.Ordinal);
                foreach (var property in vocab.EnumerateObject())
                {
                    if (property.Value.TryGetInt32(out var id))
                    {
                        result[property.Name] = id;
                    }
                }
                return result;
            }
        }
        catch
        {
            // Fall through to empty vocab
        }

        return [];
    }

    private static IEnumerable<string> SplitJsonEntries(string json)
    {
        var entries = new List<string>();
        var current = new StringBuilder();
        var inString = false;
        var escape = false;

        foreach (var c in json)
        {
            if (escape)
            {
                current.Append(c);
                escape = false;
                continue;
            }

            if (c == '\\')
            {
                current.Append(c);
                escape = true;
                continue;
            }

            if (c == '"')
            {
                inString = !inString;
                current.Append(c);
                continue;
            }

            if (c == ',' && !inString)
            {
                if (current.Length > 0)
                {
                    entries.Add(current.ToString().Trim());
                    current.Clear();
                }
                continue;
            }

            current.Append(c);
        }

        if (current.Length > 0)
        {
            entries.Add(current.ToString().Trim());
        }

        return entries;
    }
}
