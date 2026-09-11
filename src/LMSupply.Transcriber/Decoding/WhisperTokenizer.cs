using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LMSupply.Transcriber.Decoding;

/// <summary>
/// Whisper-specific tokenizer: decodes generated token IDs to text and, when the model's BPE merges
/// are available, encodes text (an initial prompt) to token IDs.
/// Uses GPT-2 BPE tokenizer with Whisper's special tokens.
/// </summary>
internal sealed partial class WhisperTokenizer : IDisposable
{
    /// <summary>
    /// Supported Whisper language codes mapped to ISO 639-1 codes.
    /// </summary>
    public static readonly IReadOnlyDictionary<string, string> SupportedLanguages = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
    {
        ["en"] = "english", ["zh"] = "chinese", ["de"] = "german", ["es"] = "spanish",
        ["ru"] = "russian", ["ko"] = "korean", ["fr"] = "french", ["ja"] = "japanese",
        ["pt"] = "portuguese", ["tr"] = "turkish", ["pl"] = "polish", ["ca"] = "catalan",
        ["nl"] = "dutch", ["ar"] = "arabic", ["sv"] = "swedish", ["it"] = "italian",
        ["id"] = "indonesian", ["hi"] = "hindi", ["fi"] = "finnish", ["vi"] = "vietnamese",
        ["he"] = "hebrew", ["uk"] = "ukrainian", ["el"] = "greek", ["ms"] = "malay",
        ["cs"] = "czech", ["ro"] = "romanian", ["da"] = "danish", ["hu"] = "hungarian",
        ["ta"] = "tamil", ["no"] = "norwegian", ["th"] = "thai", ["ur"] = "urdu",
        ["hr"] = "croatian", ["bg"] = "bulgarian", ["lt"] = "lithuanian", ["la"] = "latin",
        ["mi"] = "maori", ["ml"] = "malayalam", ["cy"] = "welsh", ["sk"] = "slovak",
        ["te"] = "telugu", ["fa"] = "persian", ["lv"] = "latvian", ["bn"] = "bengali",
        ["sr"] = "serbian", ["az"] = "azerbaijani", ["sl"] = "slovenian", ["kn"] = "kannada",
        ["et"] = "estonian", ["mk"] = "macedonian", ["br"] = "breton", ["eu"] = "basque",
        ["is"] = "icelandic", ["hy"] = "armenian", ["ne"] = "nepali", ["mn"] = "mongolian",
        ["bs"] = "bosnian", ["kk"] = "kazakh", ["sq"] = "albanian", ["sw"] = "swahili",
        ["gl"] = "galician", ["mr"] = "marathi", ["pa"] = "punjabi", ["si"] = "sinhala",
        ["km"] = "khmer", ["sn"] = "shona", ["yo"] = "yoruba", ["so"] = "somali",
        ["af"] = "afrikaans", ["oc"] = "occitan", ["ka"] = "georgian", ["be"] = "belarusian",
        ["tg"] = "tajik", ["sd"] = "sindhi", ["gu"] = "gujarati", ["am"] = "amharic",
        ["yi"] = "yiddish", ["lo"] = "lao", ["uz"] = "uzbek", ["fo"] = "faroese",
        ["ht"] = "haitian creole", ["ps"] = "pashto", ["tk"] = "turkmen", ["nn"] = "nynorsk",
        ["mt"] = "maltese", ["sa"] = "sanskrit", ["lb"] = "luxembourgish", ["my"] = "myanmar",
        ["bo"] = "tibetan", ["tl"] = "tagalog", ["mg"] = "malagasy", ["as"] = "assamese",
        ["tt"] = "tatar", ["haw"] = "hawaiian", ["ln"] = "lingala", ["ha"] = "hausa",
        ["ba"] = "bashkir", ["jw"] = "javanese", ["su"] = "sundanese", ["yue"] = "cantonese"
    };
    private readonly Dictionary<int, string> _idToToken;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<string, string> _bytesToUnicode;

    // GPT-2 byte-level BPE, for encoding: each byte's stand-in character, and each merge's rank
    // (lower merges first). Null ranks: the model directory had no merges, so text cannot be encoded.
    private static readonly string[] s_byteToUnicode = CreateByteToUnicode();
    private readonly Dictionary<(string Left, string Right), int>? _bpeRanks;

    // Default Whisper special token IDs (v1/v2 values, overridden by tokenizer.json)
    private const int DefaultEndOfTextToken = 50257;
    private const int DefaultStartOfTranscriptToken = 50258;
    private const int DefaultLanguageTokenStart = 50259;

    // Instance special token IDs (resolved from tokenizer.json at load time)
    public int EndOfTextToken { get; private set; } = DefaultEndOfTextToken;
    public int StartOfTranscriptToken { get; private set; } = DefaultStartOfTranscriptToken;
    public int TranslateToken { get; private set; } = 50358;
    public int TranscribeToken { get; private set; } = 50359;
    public int StartOfLmToken { get; private set; } = 50360;
    public int StartOfPrevToken { get; private set; } = 50361;
    public int NoSpeechToken { get; private set; } = 50362;
    public int NoTimestampsToken { get; private set; } = 50363;
    public int TimestampBeginToken { get; private set; } = 50364;
    public int LanguageTokenStart { get; private set; } = DefaultLanguageTokenStart;
    public int LanguageTokenEnd { get; private set; } = 50357;

    public int VocabSize { get; }

    private WhisperTokenizer(
        Dictionary<int, string> idToToken,
        Dictionary<string, int> tokenToId,
        IEnumerable<(string Left, string Right)>? merges = null)
    {
        _idToToken = idToToken;
        _tokenToId = tokenToId;
        _bytesToUnicode = CreateBytesToUnicode();
        _bpeRanks = merges is null ? null : RankMerges(merges);
        VocabSize = idToToken.Count;
    }

    /// <summary>
    /// Creates a tokenizer with default (v2) special token IDs and empty vocabulary.
    /// For testing only.
    /// </summary>
    internal static WhisperTokenizer CreateDefault()
    {
        return new WhisperTokenizer([], []);
    }

    /// <summary>
    /// Creates a tokenizer with default (v2) special token IDs and the given vocabulary. For
    /// testing only — lets tests exercise <see cref="Decode"/>-dependent logic without loading a
    /// real model's vocab.json.
    /// </summary>
    internal static WhisperTokenizer CreateForTesting(
        Dictionary<int, string> idToToken,
        Dictionary<string, int> tokenToId,
        IEnumerable<(string Left, string Right)>? merges = null)
    {
        return new WhisperTokenizer(idToToken, tokenToId, merges);
    }

    /// <summary>
    /// Creates a tokenizer from a model directory containing vocab.json.
    /// </summary>
    public static async Task<WhisperTokenizer> LoadAsync(string modelDir, CancellationToken cancellationToken = default)
    {
        var vocabPath = Path.Combine(modelDir, "vocab.json");
        if (!File.Exists(vocabPath))
        {
            throw new FileNotFoundException($"Vocabulary file not found: {vocabPath}");
        }

        var json = await File.ReadAllTextAsync(vocabPath, cancellationToken);
        using var doc = JsonDocument.Parse(json);

        var tokenToId = new Dictionary<string, int>(StringComparer.Ordinal);
        var idToToken = new Dictionary<int, string>();

        foreach (var property in doc.RootElement.EnumerateObject())
        {
            var token = property.Name;
            var id = property.Value.GetInt32();
            tokenToId[token] = id;
            idToToken[id] = token;
        }

        var tokenizer = new WhisperTokenizer(idToToken, tokenToId, LoadMerges(modelDir));

        // Resolve special token IDs from tokenizer.json (handles v2 vs v3 differences)
        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");
        if (File.Exists(tokenizerJsonPath))
        {
            tokenizer.LoadSpecialTokenIds(tokenizerJsonPath);
        }

        return tokenizer;
    }

    private void LoadSpecialTokenIds(string tokenizerJsonPath)
    {
        try
        {
            using var stream = File.OpenRead(tokenizerJsonPath);
            using var doc = JsonDocument.Parse(stream);

            if (!doc.RootElement.TryGetProperty("added_tokens", out var addedTokens))
                return;

            var specialTokenMap = new Dictionary<string, int>();
            foreach (var token in addedTokens.EnumerateArray())
            {
                if (token.TryGetProperty("content", out var content) &&
                    token.TryGetProperty("id", out var id))
                {
                    specialTokenMap[content.GetString()!] = id.GetInt32();
                }
            }

            // Map special token names to instance properties
            if (specialTokenMap.TryGetValue("<|endoftext|>", out var eot))
                EndOfTextToken = eot;
            if (specialTokenMap.TryGetValue("<|startoftranscript|>", out var sot))
                StartOfTranscriptToken = sot;
            if (specialTokenMap.TryGetValue("<|translate|>", out var translate))
                TranslateToken = translate;
            if (specialTokenMap.TryGetValue("<|transcribe|>", out var transcribe))
                TranscribeToken = transcribe;
            if (specialTokenMap.TryGetValue("<|startoflm|>", out var startOfLm))
                StartOfLmToken = startOfLm;
            if (specialTokenMap.TryGetValue("<|startofprev|>", out var startOfPrev))
                StartOfPrevToken = startOfPrev;
            if (specialTokenMap.TryGetValue("<|nospeech|>", out var noSpeech))
                NoSpeechToken = noSpeech;
            if (specialTokenMap.TryGetValue("<|notimestamps|>", out var noTs))
                NoTimestampsToken = noTs;
            if (specialTokenMap.TryGetValue("<|0.00|>", out var tsBegin))
                TimestampBeginToken = tsBegin;

            // Determine language token range from actual tokens
            var langStart = int.MaxValue;
            var langEnd = int.MinValue;
            foreach (var lang in SupportedLanguages.Keys)
            {
                if (specialTokenMap.TryGetValue($"<|{lang}|>", out var langId))
                {
                    langStart = Math.Min(langStart, langId);
                    langEnd = Math.Max(langEnd, langId);
                }
            }
            if (langStart != int.MaxValue)
            {
                LanguageTokenStart = langStart;
                LanguageTokenEnd = langEnd;
            }

            Trace.TraceInformation($"[WhisperTokenizer] Loaded special tokens from tokenizer.json - " +
                $"EOT={EndOfTextToken}, SOT={StartOfTranscriptToken}, Transcribe={TranscribeToken}, " +
                $"NoTimestamps={NoTimestampsToken}, NoSpeech={NoSpeechToken}, " +
                $"TimestampBegin={TimestampBeginToken}, LangRange=[{LanguageTokenStart}-{LanguageTokenEnd}]");
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"[WhisperTokenizer] Failed to load special tokens from tokenizer.json, using defaults: {ex.Message}");
        }
    }

    /// <summary>
    /// Decodes a sequence of token IDs to text.
    /// </summary>
    public string Decode(ReadOnlySpan<int> tokenIds, bool skipSpecialTokens = true)
    {
        var sb = new StringBuilder();

        foreach (var tokenId in tokenIds)
        {
            if (skipSpecialTokens && IsSpecialToken(tokenId))
                continue;

            if (_idToToken.TryGetValue(tokenId, out var token))
            {
                sb.Append(token);
            }
        }

        // Convert GPT-2 BPE tokens back to text
        return DecodeBytes(sb.ToString());
    }

    /// <summary>
    /// Whether this tokenizer can <see cref="Encode"/> text: the model directory held BPE merges.
    /// </summary>
    public bool CanEncode => _bpeRanks is not null;

    /// <summary>
    /// Encodes text to token IDs with GPT-2 byte-level BPE — the inverse of <see cref="Decode"/> for
    /// ordinary text. Special-token markup such as <c>&lt;|en|&gt;</c> is not recognised; it is
    /// encoded as the characters it is made of.
    /// </summary>
    /// <exception cref="NotSupportedException">The tokenizer was loaded without BPE merges.</exception>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (_bpeRanks is null)
        {
            throw new NotSupportedException(
                "This Whisper model's tokenizer has no BPE merges (merges.txt, or the merges in tokenizer.json), " +
                "so it cannot encode text. TranscribeOptions.InitialPrompt needs them.");
        }

        var ids = new List<int>();
        foreach (Match piece in PreTokenizer().Matches(text))
        {
            var bytes = Encoding.UTF8.GetBytes(piece.Value);
            var symbols = new List<string>(bytes.Length);
            foreach (var b in bytes)
            {
                symbols.Add(s_byteToUnicode[b]);
            }

            foreach (var symbol in Merge(symbols))
            {
                if (_tokenToId.TryGetValue(symbol, out var id))
                {
                    ids.Add(id);
                    continue;
                }

                // A merge the vocabulary lacks: fall back to its bytes, which a byte-level vocabulary has.
                foreach (var c in symbol)
                {
                    if (_tokenToId.TryGetValue(c.ToString(), out var byteId))
                    {
                        ids.Add(byteId);
                    }
                }
            }
        }

        return [.. ids];
    }

    // Applies merges to one pre-tokenized piece, lowest rank first, until no adjacent pair has one.
    private List<string> Merge(List<string> symbols)
    {
        while (symbols.Count > 1)
        {
            var bestRank = int.MaxValue;
            var bestIndex = -1;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_bpeRanks!.TryGetValue((symbols[i], symbols[i + 1]), out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
            {
                break;
            }

            var left = symbols[bestIndex];
            var right = symbols[bestIndex + 1];
            var merged = new List<string>(symbols.Count - 1);
            for (int i = 0; i < symbols.Count; i++)
            {
                if (i < symbols.Count - 1 && symbols[i] == left && symbols[i + 1] == right)
                {
                    merged.Add(left + right);
                    i++;
                }
                else
                {
                    merged.Add(symbols[i]);
                }
            }

            symbols = merged;
        }

        return symbols;
    }

    // GPT-2's pre-tokenization (the ByteLevel pre-tokenizer's regex): contractions, runs of letters,
    // runs of digits and runs of other characters, each led by at most one space; then whitespace.
    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")]
    private static partial Regex PreTokenizer();

    /// <summary>The character GPT-2's byte-level BPE uses to stand for <paramref name="b"/>.</summary>
    internal static string ByteToUnicode(byte b) => s_byteToUnicode[b];

    /// <summary>
    /// Reads BPE merges from a model directory: <c>merges.txt</c> (one "left right" pair per line, after
    /// a <c>#version</c> header), else <c>tokenizer.json</c>'s <c>model.merges</c> (strings or pairs).
    /// Null when neither has any.
    /// </summary>
    internal static List<(string Left, string Right)>? LoadMerges(string modelDir)
    {
        var mergesPath = Path.Combine(modelDir, "merges.txt");
        if (File.Exists(mergesPath))
        {
            var merges = new List<(string Left, string Right)>();
            foreach (var line in File.ReadLines(mergesPath))
            {
                if (line.StartsWith("#version", StringComparison.Ordinal))
                    continue;
                if (SplitMerge(line) is { } pair)
                    merges.Add(pair);
            }

            return merges.Count > 0 ? merges : null;
        }

        var tokenizerJsonPath = Path.Combine(modelDir, "tokenizer.json");
        if (!File.Exists(tokenizerJsonPath))
        {
            return null;
        }

        try
        {
            using var stream = File.OpenRead(tokenizerJsonPath);
            using var doc = JsonDocument.Parse(stream);
            if (!doc.RootElement.TryGetProperty("model", out var model)
                || !model.TryGetProperty("merges", out var entries)
                || entries.ValueKind != JsonValueKind.Array)
            {
                return null;
            }

            var merges = new List<(string Left, string Right)>();
            foreach (var entry in entries.EnumerateArray())
            {
                if (entry.ValueKind == JsonValueKind.String && SplitMerge(entry.GetString()!) is { } pair)
                    merges.Add(pair);
                else if (entry.ValueKind == JsonValueKind.Array && entry.GetArrayLength() == 2)
                    merges.Add((entry[0].GetString()!, entry[1].GetString()!));
            }

            return merges.Count > 0 ? merges : null;
        }
        catch (JsonException ex)
        {
            Trace.TraceWarning($"[WhisperTokenizer] Failed to read BPE merges from tokenizer.json: {ex.Message}");
            return null;
        }
    }

    private static (string Left, string Right)? SplitMerge(string line)
    {
        var space = line.IndexOf(' ');
        return space > 0 && space < line.Length - 1 ? (line[..space], line[(space + 1)..]) : null;
    }

    private static Dictionary<(string Left, string Right), int> RankMerges(IEnumerable<(string Left, string Right)> merges)
    {
        var ranks = new Dictionary<(string Left, string Right), int>();
        foreach (var pair in merges)
        {
            ranks.TryAdd(pair, ranks.Count);
        }

        return ranks;
    }

    private static string[] CreateByteToUnicode()
    {
        var map = new string[256];
        foreach (var (unicode, raw) in CreateBytesToUnicode())
        {
            map[raw[0]] = unicode;
        }

        return map;
    }

    /// <summary>
    /// Gets the token ID for a given token string.
    /// </summary>
    public int? GetTokenId(string token)
    {
        return _tokenToId.TryGetValue(token, out var id) ? id : null;
    }

    /// <summary>
    /// Gets the token string for a given token ID.
    /// </summary>
    public string? GetToken(int tokenId)
    {
        return _idToToken.TryGetValue(tokenId, out var token) ? token : null;
    }

    /// <summary>
    /// Checks if a token ID is a special token.
    /// </summary>
    public bool IsSpecialToken(int tokenId)
    {
        return tokenId >= EndOfTextToken;
    }

    /// <summary>
    /// Checks if a token ID is a timestamp token.
    /// </summary>
    public bool IsTimestampToken(int tokenId)
    {
        return tokenId >= TimestampBeginToken;
    }

    /// <summary>
    /// Converts a timestamp token to seconds.
    /// </summary>
    public float TimestampTokenToSeconds(int tokenId)
    {
        if (tokenId < TimestampBeginToken)
            return 0f;

        return (tokenId - TimestampBeginToken) * 0.02f; // Each token = 20ms
    }

    /// <summary>
    /// Checks if a token ID is a language token.
    /// </summary>
    public bool IsLanguageToken(int tokenId)
    {
        return tokenId >= LanguageTokenStart && tokenId <= LanguageTokenEnd;
    }

    /// <summary>
    /// Gets the language code for a language token.
    /// </summary>
    public string? GetLanguageFromToken(int tokenId)
    {
        if (!IsLanguageToken(tokenId))
            return null;

        var index = tokenId - LanguageTokenStart;
        var languageCodes = SupportedLanguages.Keys.ToList();

        if (index >= 0 && index < languageCodes.Count)
        {
            var lang = languageCodes[index];
            return lang;
        }

        return null;
    }

    /// <summary>
    /// Gets the token ID for a language code.
    /// </summary>
    public int? GetLanguageToken(string languageCode)
    {
        var languageCodes = SupportedLanguages.Keys.ToList();
        var index = languageCodes.FindIndex(c => c.Equals(languageCode, StringComparison.OrdinalIgnoreCase));

        if (index >= 0)
        {
            var token = LanguageTokenStart + index;
            return token;
        }

        return null;
    }

    /// <summary>
    /// Gets the SOT (start of transcript) sequence for transcription or translation.
    /// </summary>
    /// <param name="language">ISO 639-1 language code (e.g., "en", "zh", "ja"). Null to auto-detect.</param>
    /// <param name="timestamps">Whether to include timestamps in output.</param>
    /// <param name="translate">
    /// When true, uses the translate task token instead of transcribe. Whisper's translate task
    /// converts speech in any supported language into English text. Only meaningful for multilingual
    /// models; English-only (.en) variants must use transcribe.
    /// </param>
    /// <returns>Array of token IDs for the SOT sequence.</returns>
    public int[] GetSotSequence(string? language = null, bool timestamps = false, bool translate = false)
    {
        var tokens = new List<int> { StartOfTranscriptToken };

        // Add language token
        if (language != null)
        {
            var normalizedLang = NormalizeLanguageCode(language);
            var langToken = GetLanguageToken(normalizedLang);

            if (langToken.HasValue)
            {
                tokens.Add(langToken.Value);
                Trace.TraceInformation($"[WhisperTokenizer] Using language: {normalizedLang} (token: {langToken.Value})");
            }
            else
            {
                Trace.TraceWarning($"[WhisperTokenizer] Warning: Language '{language}' not found in vocabulary. " +
                    $"Supported codes: {string.Join(", ", SupportedLanguages.Keys.Take(10))}...");
            }
        }
        else
        {
            Trace.TraceInformation("[WhisperTokenizer] No language specified, model will auto-detect language.");
        }

        // Add task token (transcribe or translate)
        tokens.Add(translate ? TranslateToken : TranscribeToken);

        // Add no timestamps token if not using timestamps
        if (!timestamps)
        {
            tokens.Add(NoTimestampsToken);
        }

        Trace.TraceInformation($"[WhisperTokenizer] SOT sequence: [{string.Join(", ", tokens)}]");
        return [.. tokens];
    }

    /// <summary>
    /// Normalizes a language code to the format expected by Whisper.
    /// </summary>
    private static string NormalizeLanguageCode(string language)
    {
        // Handle common aliases and normalize to lowercase
        var normalized = language.ToLowerInvariant().Trim();

        // Handle full language names -> ISO codes
        return normalized switch
        {
            "chinese" or "mandarin" => "zh",
            "english" => "en",
            "japanese" => "ja",
            "korean" => "ko",
            "spanish" => "es",
            "french" => "fr",
            "german" => "de",
            "italian" => "it",
            "portuguese" => "pt",
            "russian" => "ru",
            "arabic" => "ar",
            "hindi" => "hi",
            "turkish" => "tr",
            "vietnamese" => "vi",
            "thai" => "th",
            "indonesian" => "id",
            "dutch" => "nl",
            "polish" => "pl",
            "swedish" => "sv",
            "hebrew" => "he",
            "cantonese" => "yue",
            _ => normalized
        };
    }

    // GPT-2 uses a specific byte-to-unicode mapping for BPE tokens
    private static Dictionary<string, string> CreateBytesToUnicode()
    {
        var bs = new List<int>();
        // Visible ASCII (33-126)
        for (int i = 33; i <= 126; i++) bs.Add(i);
        // Latin supplement (161-172, 174-255)
        for (int i = 161; i <= 172; i++) bs.Add(i);
        for (int i = 174; i <= 255; i++) bs.Add(i);

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var result = new Dictionary<string, string>();
        for (int i = 0; i < bs.Count; i++)
        {
            result[((char)cs[i]).ToString()] = ((char)bs[i]).ToString();
        }

        return result;
    }

    private string DecodeBytes(string text)
    {
        var sb = new StringBuilder();
        foreach (var c in text)
        {
            var cs = c.ToString();
            if (_bytesToUnicode.TryGetValue(cs, out var decoded))
            {
                sb.Append(decoded);
            }
            else
            {
                sb.Append(c);
            }
        }

        // The result might have multi-byte UTF-8 sequences
        // We need to decode them properly
        try
        {
            var bytes = new byte[sb.Length];
            for (int i = 0; i < sb.Length; i++)
            {
                bytes[i] = (byte)sb[i];
            }
            return Encoding.UTF8.GetString(bytes);
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[WhisperTokenizer] UTF-8 decoding failed, using fallback: {ex.Message}");
            return sb.ToString();
        }
    }

    public void Dispose()
    {
        // No unmanaged resources
    }
}
