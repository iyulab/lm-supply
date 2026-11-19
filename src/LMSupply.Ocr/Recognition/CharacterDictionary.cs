namespace LMSupply.Ocr.Recognition;

/// <summary>
/// Character dictionary for mapping CTC output indices to characters.
/// </summary>
internal sealed class CharacterDictionary
{
    private readonly List<string> _characters;
    private readonly Dictionary<string, int> _charToIndex;
    private readonly int _blankIndex;

    /// <summary>
    /// Creates a new character dictionary from a dictionary file.
    /// </summary>
    /// <param name="dictPath">Path to the dictionary file: one character per line, or a PaddleOCR <c>inference.yml</c> whose <c>character_dict</c> list is read.</param>
    /// <param name="useSpace">Whether to include space character.</param>
    public CharacterDictionary(string dictPath, bool useSpace = true)
    {
        _characters = [];
        _charToIndex = [];

        // Index 0 is reserved for blank token (CTC)
        _characters.Add("<blank>");
        _blankIndex = 0;

        // Load characters from the dictionary file: one per line, or the character_dict list of a PaddleOCR inference.yml
        IReadOnlyList<string> lines = IsInferenceConfig(dictPath) ? PaddleInferenceConfig.ReadCharacterDictionary(dictPath) : File.ReadAllLines(dictPath);
        foreach (var line in lines)
        {
            if (!string.IsNullOrEmpty(line))
            {
                var index = _characters.Count;
                _characters.Add(line);
                _charToIndex[line] = index;
            }
        }

        // Add space character if not present and requested
        if (useSpace && !_charToIndex.ContainsKey(" "))
        {
            var index = _characters.Count;
            _characters.Add(" ");
            _charToIndex[" "] = index;
        }
    }

    private static bool IsInferenceConfig(string path) =>
        path.EndsWith(".yml", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".yaml", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Gets the number of characters in the dictionary (including blank).
    /// </summary>
    public int Count => _characters.Count;

    /// <summary>
    /// Gets the blank token index (for CTC).
    /// </summary>
    public int BlankIndex => _blankIndex;

    /// <summary>
    /// Gets the character at the specified index.
    /// </summary>
    public string GetCharacter(int index)
    {
        if (index < 0 || index >= _characters.Count)
            return string.Empty;
        return _characters[index];
    }

    /// <summary>
    /// Decodes a sequence of indices to a string.
    /// </summary>
    /// <param name="indices">Sequence of character indices.</param>
    /// <returns>Decoded string.</returns>
    public string Decode(IEnumerable<int> indices)
    {
        var chars = new List<string>();
        var prevIndex = -1;

        foreach (var index in indices)
        {
            // Skip blank tokens and repeated characters (CTC decoding)
            if (index != _blankIndex && index != prevIndex)
            {
                chars.Add(GetCharacter(index));
            }
            prevIndex = index;
        }

        return string.Join("", chars);
    }

    /// <summary>
    /// Checks if the dictionary contains a character.
    /// </summary>
    public bool Contains(string character) => _charToIndex.ContainsKey(character);

    /// <summary>
    /// Gets the index of a character.
    /// </summary>
    public int GetIndex(string character)
    {
        return _charToIndex.TryGetValue(character, out var index) ? index : _blankIndex;
    }
}
