using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using AwesomeAssertions;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Every option a document shows a caller setting must exist. <c>docs/generator.md</c> told readers to set
/// <c>GeneratorOptions.GgufFileName</c>, a property the type never had — a reader who copies the snippet
/// gets a compile error, and nothing in the build notices because documentation is not compiled. This is
/// the mirror image of <see cref="OptionsReachabilityRosterTests"/>: that one asks whether a declared option
/// is read; this one asks whether a documented option is declared.
/// </summary>
/// <remarks>
/// It scans <c>README.md</c> and <c>docs/*.md</c> for object initializers of the form
/// <c>new &lt;Type&gt;Options { Name = …, … }</c> (nested initializers included), resolves the type by its
/// simple name in the library assemblies, and requires each assigned name to be a public settable property
/// (or public field) of that type. It lives here because this project references every module.
/// </remarks>
public class DocsOptionsSnippetRosterTests
{
    [Fact]
    public void EveryOptionADocumentSets_ExistsOnTheType()
    {
        var root = RepositoryRoot();
        var files = new[] { Path.Combine(root, "README.md") }
            .Concat(Directory.EnumerateFiles(Path.Combine(root, "docs"), "*.md"))
            .Where(File.Exists)
            .Order(StringComparer.Ordinal)
            .ToList();

        var types = OptionTypes();
        var findings = new List<string>();
        var snippets = 0;
        foreach (var file in files)
        {
            foreach (var snippet in FindSnippets(File.ReadAllText(file)))
            {
                snippets++;
                findings.AddRange(Check(snippet, types).Select(f => $"{Path.GetRelativePath(root, file)}:{snippet.Line}  {f}"));
            }
        }

        snippets.Should().BeGreaterThan(20, "the scan must find the documentation's option snippets, or an empty finding list proves nothing");
        findings.Should().BeEmpty("a document names an option the type does not have — fix the document, or add the option:\n" + string.Join('\n', findings));
    }

    // Positive control: the scanner must flag a phantom property, or the assertion above is vacuous.
    [Fact]
    public void AnOptionThatDoesNotExist_IsReported()
    {
        const string markdown = """
            ```csharp
            var options = new EmbedderOptions
            {
                MaxSequenceLength = 256,     // real
                NoSuchOption = true,         // phantom
                ServerUpdateOptions = new LlamaServerUpdateOptions { PinnedVersion = "b1", AlsoNotReal = 1 }
            };
            ```
            """;

        var findings = FindSnippets(markdown).SelectMany(s => Check(s, OptionTypes())).ToList();

        findings.Should().BeEquivalentTo(
        [
            "EmbedderOptions.NoSuchOption",
            "LlamaServerUpdateOptions.AlsoNotReal",
        ]);
    }

    [Fact]
    public void AnUnknownOptionsType_IsReported()
    {
        var findings = FindSnippets("var o = new ImaginaryOptions { Anything = 1 };").SelectMany(s => Check(s, OptionTypes())).ToList();

        findings.Should().Equal("ImaginaryOptions (no such type in the library assemblies)");
    }

    // ── scanner ─────────────────────────────────────────────────────────────────────────────

    internal sealed record Snippet(string TypeName, int Line, IReadOnlyList<string> Properties);

    private static readonly Regex Opening = new(@"new\s+([A-Z]\w*Options)\s*\{", RegexOptions.Compiled);
    private static readonly Regex Assignment = new(@"(?<![\w.])([A-Z]\w*)\s*=(?!=)", RegexOptions.Compiled);

    internal static IEnumerable<Snippet> FindSnippets(string markdown)
    {
        foreach (Match m in Opening.Matches(markdown))
        {
            var open = m.Index + m.Length - 1;
            var body = TopLevelBody(markdown, open);
            if (body is null)
                continue;

            var line = markdown.AsSpan(0, m.Index).Count('\n') + 1;
            var properties = Assignment.Matches(body).Select(a => a.Groups[1].Value).Distinct(StringComparer.Ordinal).ToList();
            yield return new Snippet(m.Groups[1].Value, line, properties);
        }
    }

    /// <summary>
    /// The initializer body between the brace at <paramref name="open"/> and its match, with nested
    /// braces, string literals and line comments blanked out — so only depth-one assignments remain.
    /// Nested initializers are found by their own <c>new …Options {</c> match.
    /// </summary>
    private static string? TopLevelBody(string text, int open)
    {
        var sb = new StringBuilder();
        var depth = 0;
        for (var i = open; i < text.Length; i++)
        {
            var c = text[i];
            if (c == '"')
            {
                // Skip a string literal (with escapes); it contributes nothing at depth one.
                i++;
                while (i < text.Length && text[i] != '"')
                    i += text[i] == '\\' ? 2 : 1;
                sb.Append(' ');
                continue;
            }
            if (c == '/' && i + 1 < text.Length && text[i + 1] == '/')
            {
                while (i < text.Length && text[i] != '\n')
                    i++;
                sb.Append('\n');
                continue;
            }
            if (c == '{')
            {
                depth++;
                sb.Append(depth == 1 ? ' ' : '{');
                continue;
            }
            if (c == '}')
            {
                depth--;
                if (depth == 0)
                    return sb.ToString();
                sb.Append('}');
                continue;
            }
            sb.Append(depth == 1 ? c : ' ');
        }
        return null;
    }

    internal static IEnumerable<string> Check(Snippet snippet, IReadOnlyDictionary<string, List<Type>> types)
    {
        if (!types.TryGetValue(snippet.TypeName, out var candidates))
        {
            yield return $"{snippet.TypeName} (no such type in the library assemblies)";
            yield break;
        }

        foreach (var property in snippet.Properties)
        {
            var exists = candidates.Any(t =>
                t.GetProperty(property, BindingFlags.Public | BindingFlags.Instance) is { SetMethod: not null }
                || t.GetField(property, BindingFlags.Public | BindingFlags.Instance) is not null);
            if (!exists)
                yield return $"{snippet.TypeName}.{property}";
        }
    }

    private static Dictionary<string, List<Type>> OptionTypes()
        => Directory.EnumerateFiles(AppContext.BaseDirectory, "LMSupply*.dll")
            .Select(path => Assembly.Load(AssemblyName.GetAssemblyName(path)))
            .SelectMany(a =>
            {
                try { return (IEnumerable<Type>)a.GetExportedTypes(); }
                catch (ReflectionTypeLoadException ex) { return ex.Types.Where(t => t is { IsPublic: true }).Select(t => t!); }
            })
            .Where(t => t.Name.EndsWith("Options", StringComparison.Ordinal))
            .GroupBy(t => t.Name, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.ToList(), StringComparer.Ordinal);

    private static string RepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "lm-supply.slnx")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("lm-supply.slnx not found above the test output directory");
    }
}
