using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using AwesomeAssertions;
using Xunit;

namespace LMSupply.Integration.Tests;

/// <summary>
/// What the documentation shows a caller writing must exist. Documentation is not compiled, so a renamed
/// builder method or an option type that never shipped stays in a guide until a reader copies it and gets
/// a compile error. This is the mirror image of <see cref="OptionsReachabilityRosterTests"/>: that one asks
/// whether a declared option is read; this one asks whether a documented name is declared.
/// </summary>
/// <remarks>
/// Two checks over the C# blocks of <c>README.md</c>, <c>docs/*.md</c> and each package's <c>src/*/README.md</c>:
/// <list type="number">
/// <item>Object initializers <c>new &lt;Type&gt;{Options|Config|Configuration|Defaults} { Name = … }</c> — the type
/// must exist in the library assemblies and every assigned name must be a public settable property.</item>
/// <item>Method calls <c>.Name(</c> — a public method of that name must exist on some public type in the
/// assemblies next to this test (the LMSupply packages, the sibling packages they integrate and the
/// framework packages their registrations extend) or in the BCL. Names from packages this test does not
/// reference, and names a sample defines itself, are listed in <see cref="KnownExternal"/>.</item>
/// </list>
/// Documented names that are known to be wrong and not yet repaired are pinned in <see cref="KnownDrift"/>
/// with the file they live in, so the list can only shrink deliberately.
/// </remarks>
public class DocsSnippetRosterTests
{
    /// <summary>
    /// Method names the docs use that no scanned assembly declares because they belong to packages the
    /// test output does not contain. A method a document defines for itself (its own service class or
    /// extension) is excluded per document by the scanner, not listed here.
    /// </summary>
    private static readonly HashSet<string> KnownExternal = new(StringComparer.Ordinal)
    {
        // The caller's own vector index in the vector-space-revision example
        "ReembedAsync",
    };

    /// <summary>
    /// Documented names that do not exist, pinned until the document is repaired (file → names). Each of
    /// these is a snippet a reader cannot compile today.
    /// </summary>
    private static readonly Dictionary<string, string[]> KnownDrift = new(StringComparer.Ordinal)
    {
    };

    /// <summary>Option-shaped types from other SDKs that a document legitimately shows (not ours to declare).</summary>
    private static readonly HashSet<string> KnownExternalTypes = new(StringComparer.Ordinal)
    {
    };

    [Fact]
    public void EveryNameADocumentUses_Exists_ExceptTheKnownDrift()
    {
        var root = RepositoryRoot();
        var files = new[] { Path.Combine(root, "README.md") }
            .Concat(Directory.EnumerateFiles(Path.Combine(root, "docs"), "*.md"))
            // Each package ships its own README to nuget.org; a reader of one package reads that one.
            .Concat(Directory.EnumerateDirectories(Path.Combine(root, "src")).Select(d => Path.Combine(d, "README.md")))
            .Where(File.Exists)
            .Order(StringComparer.Ordinal)
            .ToList();

        var types = OptionTypes();
        var methods = PublicMethodNames();
        var findings = new Dictionary<string, SortedSet<string>>(StringComparer.Ordinal);
        var blocks = 0;
        foreach (var file in files)
        {
            var relative = Path.GetRelativePath(root, file).Replace('\\', '/');
            var documentBlocks = CSharpBlocks(File.ReadAllText(file)).ToList();
            // A guide that shows how to write your own provider defines methods in one block and calls
            // them in the next; those are the document's own, not the library's.
            var defined = DefinedMethods(documentBlocks);
            // The same holds for an options type a sample declares for its own pipeline or step.
            var definedTypes = DefinedTypes(documentBlocks);
            foreach (var block in documentBlocks)
            {
                blocks++;
                var names = FindSnippets(block).Where(s => !definedTypes.Contains(s.TypeName)).SelectMany(s => Check(s, types))
                    .Concat(RegistrationCalls(block).Where(n => !methods.Contains(n) && !KnownExternal.Contains(n) && !defined.Contains(n)));
                foreach (var name in names)
                {
                    if (!findings.TryGetValue(relative, out var set))
                        findings[relative] = set = new SortedSet<string>(StringComparer.Ordinal);
                    set.Add(name);
                }
            }
        }

        blocks.Should().BeGreaterThan(40, "the scan must find the documentation's C# blocks, or an empty finding list proves nothing");

        var found = findings.Select(kv => $"{kv.Key}: {string.Join(",", kv.Value)}").Order(StringComparer.Ordinal).ToList();
        var expected = KnownDrift.Select(kv => $"{kv.Key}: {string.Join(",", kv.Value.Order(StringComparer.Ordinal))}").Order(StringComparer.Ordinal).ToList();
        Assert.True(expected.SequenceEqual(found),
            "a document uses a name the library does not have — fix the document, or pin it here as a deliberate decision.\n" +
            "found:\n  " + string.Join("\n  ", found) + "\nexpected:\n  " + string.Join("\n  ", expected));
    }

    // Positive controls: the scanner must flag phantoms, or the assertion above is vacuous.

    [Fact]
    public void APhantomProperty_AndAPhantomType_AreReported()
    {
        const string block = """
            var options = new EmbedderOptions
            {
                NormalizeEmbeddings = true,  // real
                NoSuchOption = true,         // phantom
                Nested = new ImaginaryOptions { Anything = 1 }
            };
            """;

        var findings = FindSnippets(block).SelectMany(s => Check(s, OptionTypes())).ToList();

        findings.Should().BeEquivalentTo(["EmbedderOptions.NoSuchOption", "EmbedderOptions.Nested", "ImaginaryOptions (no such type in the library assemblies)"]);
    }

    [Fact]
    public void AMethodTheDocumentDefinesItself_IsNotDrift_ButAnUndefinedOneStillIs()
    {
        var blocks = new[]
        {
            "public static class MyExtensions\n{\n    public static IServiceCollection AddMyEmbedding(this IServiceCollection s, string key) => s;\n}",
            "services.AddMyEmbedding(\"k\");\nservices.AddSomebodyElses(\"k\");",
        };

        var defined = DefinedMethods(blocks);
        var calls = RegistrationCalls(blocks[1]).Where(n => !PublicMethodNames().Contains(n) && !defined.Contains(n)).ToList();

        defined.Should().Contain("AddMyEmbedding");
        calls.Should().BeEquivalentTo(["AddSomebodyElses"]);
    }

    [Fact]
    public void AnOptionsTypeTheDocumentDeclaresItself_IsNotDrift_ButAnUndeclaredOneStillIs()
    {
        var blocks = new[]
        {
            "public class MyStepOptions { public int BatchSize { get; set; } }",
            "var a = new MyStepOptions { BatchSize = 1 };\nvar b = new TheirStepOptions { BatchSize = 1 };",
        };

        var definedTypes = DefinedTypes(blocks);
        var findings = FindSnippets(blocks[1]).Where(s => !definedTypes.Contains(s.TypeName)).SelectMany(s => Check(s, OptionTypes())).ToList();

        findings.Should().BeEquivalentTo(["TheirStepOptions (no such type in the library assemblies)"]);
    }

    [Fact]
    public void APhantomRegistrationCall_IsReported_AndARealOneIsNot()
    {
        var methods = PublicMethodNames();
        var calls = RegistrationCalls(
            "LocalEmbedder.LoadAsync(\"default\").ToString().UseImaginaryThing();\n" +
            "var r = await processor.NoSuchProcessAsync(url); // .NotACall( in a comment\n" +
            "var s = \"text with .NotACallEither( inside\".ToUpperInvariant();\n" +
            "var o = new TextChunkingPipeline.Options(ChunkSize: 1);").ToList();

        calls.Should().BeEquivalentTo(["LoadAsync", "ToString", "UseImaginaryThing", "NoSuchProcessAsync", "ToUpperInvariant"]);
        calls.Where(n => !methods.Contains(n)).Should().BeEquivalentTo(["UseImaginaryThing", "NoSuchProcessAsync"]);
    }

    /// <summary>
    /// Logging extension methods come from an assembly a shared framework also ships; whether it is copied to the test
    /// output depends on the machine's runtime version. The scan must see it either way.
    /// </summary>
    [Fact]
    public void AFrameworkSuppliedExtensionMethod_Resolves_WhetherOrNotItWasCopiedToTheOutput() =>
        Assert.Contains("LogError", PublicMethodNames());

    // ── scanner ─────────────────────────────────────────────────────────────────────────────

    private static readonly Regex Fence = new(@"```(?:csharp|cs|c#)\s*\n(.*?)```", RegexOptions.Compiled | RegexOptions.Singleline | RegexOptions.IgnoreCase);

    internal static IEnumerable<string> CSharpBlocks(string markdown) =>
        Fence.Matches(markdown).Select(m => m.Groups[1].Value);

    internal sealed record Snippet(string TypeName, IReadOnlyList<string> Properties);

    private static readonly Regex Opening = new(@"new\s+([A-Z]\w*(?:Options|Config|Configuration|Defaults))\s*\{", RegexOptions.Compiled);
    private static readonly Regex Assignment = new(@"(?<![\w.])([A-Z]\w*)\s*=(?!=)", RegexOptions.Compiled);
    private static readonly Regex NestedInitializer = new(@"\G\s*\{", RegexOptions.Compiled);
    // Every member call: `.Name(` after an identifier, a closing paren/bracket or a string literal — not a
    // decimal literal (`0.5f(` cannot occur) and not `new Type(` (no dot).
    private static readonly Regex Registration = new(@"(?<=[\w)\]""])\s*\.\s*([A-Z]\w*)\s*\(", RegexOptions.Compiled);

    // A method (or constructor/record) declaration: an access modifier, then a return type and a name,
    // then "(" — without crossing a statement or a body on the way.
    private static readonly Regex Definition = new(@"\b(?:public|private|internal|protected)\b[^;{}=()]*?\b([A-Z]\w*)\s*(?:<[^>()]*>)?\s*\(", RegexOptions.Compiled);

    /// <summary>Names a document's own C# blocks declare, so calls to them are not read as library API.</summary>
    internal static HashSet<string> DefinedMethods(IEnumerable<string> blocks) =>
        blocks.SelectMany(b => Definition.Matches(StripCommentsAndStrings(b)).Select(m => m.Groups[1].Value))
            .ToHashSet(StringComparer.Ordinal);

    private static readonly Regex TypeDefinition = new(@"\b(?:class|record|struct|interface)\s+([A-Z]\w*)", RegexOptions.Compiled);

    /// <summary>Types a document's own C# blocks declare, so initializers of them are not read as library options.</summary>
    internal static HashSet<string> DefinedTypes(IEnumerable<string> blocks) =>
        blocks.SelectMany(b => TypeDefinition.Matches(StripCommentsAndStrings(b)).Select(m => m.Groups[1].Value))
            .ToHashSet(StringComparer.Ordinal);

    // `new Outer.Inner(` constructs a nested type; collapse the qualified name so `.Inner(` is not read as a call.
    private static readonly Regex QualifiedConstruction = new(@"\bnew\s+[A-Za-z_][\w.]*", RegexOptions.Compiled);

    internal static IEnumerable<string> RegistrationCalls(string code) =>
        Registration.Matches(QualifiedConstruction.Replace(StripCommentsAndStrings(code), "new T"))
            .Select(m => m.Groups[1].Value).Distinct(StringComparer.Ordinal);

    /// <summary>Blank out string literals and comments so a `.Name(` inside them is not read as a call.</summary>
    private static string StripCommentsAndStrings(string code)
    {
        var sb = new StringBuilder(code.Length);
        for (var i = 0; i < code.Length; i++)
        {
            var c = code[i];
            if (c == '"')
            {
                sb.Append('"');
                i++;
                while (i < code.Length && code[i] != '"')
                {
                    if (code[i] == '\\') i++;
                    i++;
                }
                sb.Append('"');
                continue;
            }
            if (c == '/' && i + 1 < code.Length && code[i + 1] == '/')
            {
                while (i < code.Length && code[i] != '\n') i++;
                sb.Append('\n');
                continue;
            }
            sb.Append(c);
        }
        return sb.ToString();
    }

    internal static IEnumerable<Snippet> FindSnippets(string code)
    {
        foreach (Match m in Opening.Matches(code))
        {
            var body = TopLevelBody(code, m.Index + m.Length - 1);
            if (body is null)
                continue;
            // `Name = { ... }` is a nested collection/object initializer: legal on a get-only property, so it is
            // marked and checked for a readable property instead of a settable one.
            var properties = Assignment.Matches(body)
                .Select(a => NestedInitializer.IsMatch(body, a.Index + a.Length) ? a.Groups[1].Value + "{" : a.Groups[1].Value)
                .Distinct(StringComparer.Ordinal).ToList();
            yield return new Snippet(m.Groups[1].Value, properties);
        }
    }

    /// <summary>
    /// The initializer body between the brace at <paramref name="open"/> and its match, with nested braces,
    /// string literals and line comments blanked out — so only depth-one assignments remain.
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
        if (KnownExternalTypes.Contains(snippet.TypeName))
            yield break;
        if (!types.TryGetValue(snippet.TypeName, out var candidates))
        {
            yield return $"{snippet.TypeName} (no such type in the library assemblies)";
            yield break;
        }

        foreach (var entry in snippet.Properties)
        {
            var nested = entry.EndsWith('{');
            var property = nested ? entry[..^1] : entry;
            var exists = candidates.Any(t =>
                t.GetProperty(property, BindingFlags.Public | BindingFlags.Instance) is { } p && (nested || p.SetMethod is not null)
                || t.GetField(property, BindingFlags.Public | BindingFlags.Instance) is not null);
            if (!exists)
                yield return $"{snippet.TypeName}.{property}";
        }
    }

    /// <summary>
    /// Assemblies whose file name matches <paramref name="pattern"/>, from the test output and from the runtime's trusted
    /// platform assemblies. The output alone is not enough: an assembly a shared framework also ships (ASP.NET Core carries
    /// Microsoft.Extensions.*) is copied only when the package's version beats the installed framework's, so which names
    /// resolve would depend on the machine's runtime - green on one machine, red on a newer CI image.
    /// </summary>
    private static IEnumerable<Assembly> LoadedAssemblies(string pattern)
    {
        var glob = new Regex("^" + Regex.Escape(pattern).Replace(@"\*", ".*", StringComparison.Ordinal) + "$", RegexOptions.IgnoreCase);
        var platform = (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") as string ?? string.Empty)
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries);
        return Directory.EnumerateFiles(AppContext.BaseDirectory, pattern)
            .Concat(platform.Where(path => glob.IsMatch(Path.GetFileName(path))))
            .DistinctBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase)
            .Select(path =>
            {
                try { return Assembly.Load(AssemblyName.GetAssemblyName(path)); }
                catch (BadImageFormatException) { return null; }
                catch (FileLoadException) { return null; }
                catch (FileNotFoundException) { return null; }
            })
            .Where(a => a is not null)!;
    }

    private static List<Type> ExportedTypes(Assembly assembly)
    {
        try { return assembly.GetExportedTypes().ToList(); }
        catch (ReflectionTypeLoadException ex)
        {
            var loaded = new List<Type>();
            foreach (var t in ex.Types)
            {
                try { if (t is { IsPublic: true }) loaded.Add(t); }
                catch (FileNotFoundException) { }
                catch (TypeLoadException) { }
            }
            return loaded;
        }
        catch (FileNotFoundException) { return []; }
        catch (FileLoadException) { return []; }
        catch (TypeLoadException) { return []; }
    }

    /// <summary>
    /// The assemblies whose public method names count as "exists": the LMSupply packages, the sibling
    /// packages they integrate, and the framework packages their registrations extend. Test-runner and
    /// analyzer assemblies next to the output are deliberately not scanned.
    /// </summary>
    private static readonly string[] MethodNamePatterns =
    [
        "LMSupply*.dll", "Microsoft.Extensions.*.dll", "Microsoft.ML.OnnxRuntime*.dll", "Microsoft.AspNetCore.*.dll",
    ];

    /// <summary>BCL assemblies whose public method names the samples also call (LINQ, tasks, IO, JSON…).</summary>
    private static readonly Assembly[] FrameworkAssemblies =
    [
        typeof(object).Assembly, typeof(Enumerable).Assembly, typeof(Task).Assembly, typeof(System.Console).Assembly,
        typeof(File).Assembly, typeof(System.Text.Json.JsonSerializer).Assembly, typeof(Regex).Assembly,
        typeof(System.Collections.Concurrent.ConcurrentDictionary<,>).Assembly, typeof(HttpClient).Assembly,
        typeof(System.Diagnostics.Stopwatch).Assembly, typeof(Parallel).Assembly,
    ];

    private static Dictionary<string, List<Type>> OptionTypes() =>
        LoadedAssemblies("LMSupply*.dll")
            .Where(a => !a.GetName().Name!.EndsWith(".Tests", StringComparison.Ordinal))
            .SelectMany(ExportedTypes)
            .Where(t => t.Name.EndsWith("Options", StringComparison.Ordinal)
                     || t.Name.EndsWith("Config", StringComparison.Ordinal)
                     || t.Name.EndsWith("Configuration", StringComparison.Ordinal)
                     || t.Name.EndsWith("Defaults", StringComparison.Ordinal))
            .GroupBy(t => t.Name, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.ToList(), StringComparer.Ordinal);

    /// <summary>Every public method name on a public type in the assemblies next to this test.</summary>
    private static HashSet<string> PublicMethodNames() =>
        MethodNamePatterns.SelectMany(LoadedAssemblies)
            .Where(a => !a.GetName().Name!.EndsWith(".Tests", StringComparison.Ordinal))
            .Concat(FrameworkAssemblies)
            .Distinct()
            .SelectMany(ExportedTypes)
            .SelectMany(t =>
            {
                try { return t.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static).Select(m => m.Name).ToList(); }
                catch (TypeLoadException) { return []; }
                catch (FileNotFoundException) { return []; }
                catch (FileLoadException) { return []; }
            })
            .ToHashSet(StringComparer.Ordinal);

    private static string RepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "lm-supply.slnx")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("lm-supply.slnx not found above the test output directory");
    }
}
