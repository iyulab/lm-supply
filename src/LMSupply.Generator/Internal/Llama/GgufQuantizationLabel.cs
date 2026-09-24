using System.Text.RegularExpressions;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Reads the quantization label that GGUF publishers put in a file name
/// (<c>gemma-4-E4B-it-Q4_0.gguf</c>, <c>Qwen3.6-35B-A3B-UD-IQ4_XS.gguf</c>, <c>model.Q4_K_M-00001-of-00002.gguf</c>).
/// </summary>
internal static partial class GgufQuantizationLabel
{
    // A token bounded by a non-alphanumeric character on each side, so "Qwen3" and "bf16" inside a
    // longer word are not read as quantizations. The last match wins: the label sits at the end of the stem.
    [GeneratedRegex(@"(?<![A-Za-z0-9])(I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?)(?![A-Za-z0-9])",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex Label();

    /// <summary>
    /// The quantization label in <paramref name="fileName"/>, upper-cased, or <see langword="null"/> when the
    /// name carries none.
    /// </summary>
    public static string? FromFileName(string fileName)
    {
        var stem = Path.GetFileNameWithoutExtension(fileName);
        var matches = Label().Matches(stem);
        return matches.Count == 0 ? null : matches[^1].Groups[1].Value.ToUpperInvariant();
    }
}
