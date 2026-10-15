using System.Reflection;
using System.Runtime.CompilerServices;
using Xunit;

namespace LMSupply.Integration.Tests;

/// <summary>
/// No two top-level public types of the LMSupply packages share a simple name. An application typically uses several
/// packages together (an embedder with a reranker and a generator; OCR with a detector), so two types of one name in two
/// imported namespaces make every use ambiguous — CS0104 — and make a guide that names one of them ambiguous to its reader.
/// A nested type is always written through its parent, so it cannot collide this way.
/// </summary>
public class PublicTypeNameUniquenessTests
{
    private static readonly string[] Packages =
    [
        "LMSupply.Core", "LMSupply.Text.Core", "LMSupply.Vision.Core", "LMSupply.Embedder", "LMSupply.Reranker",
        "LMSupply.Generator", "LMSupply.Generator.Onnx", "LMSupply.Llama", "LMSupply.Translator", "LMSupply.Transcriber",
        "LMSupply.Synthesizer", "LMSupply.Captioner", "LMSupply.Ocr", "LMSupply.Detector", "LMSupply.Segmenter",
        "LMSupply.ImageGenerator",
    ];

    private static IEnumerable<Type> TopLevelPublicTypes() =>
        Packages.SelectMany(p => Assembly.Load(p).GetExportedTypes()).Where(t => !t.IsNested && !IsExtensionHolder(t));

    [Fact]
    public void EveryTopLevelPublicTypeName_IsUsedOnceAcrossThePackages()
    {
        var duplicates = TopLevelPublicTypes()
            .GroupBy(t => t.Name, StringComparer.Ordinal) // Name keeps the generic arity
            .Where(g => g.Count() > 1)
            .Select(g => $"{g.Key}: {string.Join(", ", g.Select(t => t.FullName).Order(StringComparer.Ordinal))}")
            .Order(StringComparer.Ordinal)
            .ToList();

        Assert.True(duplicates.Count == 0, "public type names used more than once:\n" + string.Join("\n", duplicates));
    }

    [Fact]
    public void TheScan_SeesThePackagesTypes()
    {
        var count = TopLevelPublicTypes().Count();
        Assert.True(count >= 300, $"expected the packages' public types, found {count}");
    }

    // A static class that only declares extension methods (ServiceCollectionExtensions) is never named by a caller.
    private static bool IsExtensionHolder(Type type)
    {
        if (type is not { IsAbstract: true, IsSealed: true })
            return false;
        var members = type.GetMembers(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly);
        return members.Length > 0 && members.All(m => m is MethodInfo method && method.IsDefined(typeof(ExtensionAttribute), false));
    }
}
