namespace LMSupply.Synthesizer.Core.English;

/// <summary>
/// The phonemes of the CMU Pronouncing Dictionary (ARPAbet): fifteen vowels, then twenty-four consonants.
/// </summary>
internal enum Arpa : byte
{
    AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW,
    B, CH, D, DH, F, G, HH, JH, K, L, M, N, NG, P, R, S, SH, T, TH, V, W, Y, Z, ZH
}

/// <summary>
/// One phoneme of a pronunciation: the symbol and, for a vowel, its lexical stress as the dictionary marks it
/// (0 none, 1 primary, 2 secondary). A consonant's stress is 0.
/// </summary>
internal readonly record struct Phone(Arpa Symbol, byte Stress = 0)
{
    private const int StressShift = 6;
    private const int SymbolMask = (1 << StressShift) - 1;

    public bool IsVowel => Symbol <= Arpa.UW;

    public bool IsStressed => Stress != 0;

    /// <summary>The phoneme in one byte: the symbol in the low six bits, the stress above them.</summary>
    public byte Pack() => (byte)((byte)Symbol | (Stress << StressShift));

    public static Phone Unpack(byte packed) => new((Arpa)(packed & SymbolMask), (byte)(packed >> StressShift));
}

internal static class Arpabet
{
    private static readonly string[] Names = Enum.GetNames<Arpa>();

    /// <summary>
    /// Parses one ARPAbet token as the dictionary writes it: a symbol, and on a vowel a stress digit (<c>AH0</c>,
    /// <c>IY1</c>). A vowel without a digit is unstressed.
    /// </summary>
    public static bool TryParse(ReadOnlySpan<char> token, out Phone phone)
    {
        phone = default;
        if (token.IsEmpty)
            return false;

        byte stress = 0;
        var last = token[^1];
        if (last is >= '0' and <= '2')
        {
            stress = (byte)(last - '0');
            token = token[..^1];
        }

        for (var i = 0; i < Names.Length; i++)
        {
            if (!token.SequenceEqual(Names[i]))
                continue;

            phone = new Phone((Arpa)i, stress);
            return phone.IsVowel || stress == 0;
        }

        return false;
    }

    /// <summary>Parses a pronunciation written as ARPAbet tokens separated by spaces.</summary>
    public static Phone[] Parse(string pronunciation)
    {
        var tokens = pronunciation.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var phones = new Phone[tokens.Length];
        for (var i = 0; i < tokens.Length; i++)
        {
            if (!TryParse(tokens[i], out phones[i]))
                throw new FormatException($"'{tokens[i]}' in '{pronunciation}' is not an ARPAbet phoneme.");
        }

        return phones;
    }
}
