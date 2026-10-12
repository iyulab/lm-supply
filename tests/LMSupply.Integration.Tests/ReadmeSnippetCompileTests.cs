using System.Reflection;
using System.Text.RegularExpressions;
using Iyu.Conventions.Testing.Snippets;
using Xunit;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Compiles every C# block in the README(s) against the current assemblies. <see cref="DocsSnippetRosterTests"/>
/// checks that a documented method name exists on some type and that an initializer's properties exist; a compiler also
/// checks the receiver, the arguments, the return types a block goes on to use, and the namespaces it needs.
/// </summary>
/// <remarks>
/// Each block is compiled as a top-level program by <see cref="SnippetCompiler"/>: its own usings, the document's usings
/// below, and the stand-ins below when the block uses the name without declaring it — values a reader already has from the
/// surrounding text (a loaded model, an input file), not part of what the block shows.
/// </remarks>
public partial class ReadmeSnippetCompileTests
{
    // A block that is deliberately not a program (a signature sketch, pseudocode), by the heading it sits under, with the
    // reason. Shrink this, never grow it silently.
    private static readonly SnippetRoster Fragments = new(new Dictionary<string, string>(StringComparer.Ordinal));

    private static readonly string[] CommonUsings =
        ["System", "System.Collections.Generic", "System.IO", "System.Linq", "System.Net.Http", "System.Threading", "System.Threading.Tasks"];

    // The repository README reads as one guide across packages; a package README is read with its own package's
    // namespaces only (see PackageNamespaces), so a block that needs another package's type has to say so.
    private static readonly string[] RootUsings =
    [
        "LMSupply", "LMSupply.Download", "LMSupply.Embedder", "LMSupply.Reranker", "LMSupply.Generator",
        "LMSupply.Generator.Abstractions", "LMSupply.Generator.Models", "LMSupply.Translator", "LMSupply.Transcriber",
        "LMSupply.Transcriber.Models", "LMSupply.Synthesizer", "LMSupply.Llama",
    ];

    private static readonly Dictionary<string, string> StandIns = new(StringComparer.Ordinal)
    {
        ["documents"] = "IReadOnlyList<string> documents = [];",
        ["loggerFactory"] = "Microsoft.Extensions.Logging.ILoggerFactory loggerFactory = null!;",
        ["AskUserToDownload"] = "static bool AskUserToDownload(long bytes) => true;",
        ["modelDir"] = "string modelDir = \"\";",
        ["modelPath"] = "string modelPath = \"\";",
        ["maxLength"] = "int maxLength = 512;",
    };

    // A package README's blocks read on from the package's own Quick Start: the model it loaded there is in scope.
    private static readonly Dictionary<string, Dictionary<string, string>> DocumentStandIns = new(StringComparer.Ordinal)
    {
        ["src/LMSupply.Generator/README.md"] = new() { ["generator"] = "LMSupply.Generator.Abstractions.IGeneratorModel generator = null!;" },
        ["src/LMSupply.ImageGenerator/README.md"] = new() { ["generator"] = "LMSupply.ImageGenerator.IImageGeneratorModel generator = null!;" },
        ["src/LMSupply.Ocr/README.md"] = new() { ["ocr"] = "LMSupply.Ocr.IOcr ocr = null!;" },
    };

    private static readonly IReadOnlyList<Microsoft.CodeAnalysis.MetadataReference> References = SnippetCompiler.ReferencesFor(
        [.. new[]
        {
            "LMSupply.Core", "LMSupply.Embedder", "LMSupply.Reranker", "LMSupply.Generator", "LMSupply.Generator.Onnx",
            "LMSupply.Llama", "LMSupply.Translator", "LMSupply.Transcriber", "LMSupply.Synthesizer", "LMSupply.Captioner",
            "LMSupply.Ocr", "LMSupply.Detector", "LMSupply.Segmenter", "LMSupply.ImageGenerator",
            "LMSupply.Text.Core", "LMSupply.Vision.Core",
        }.Select(Assembly.Load)]);

    private static readonly IReadOnlyList<DocSnippet> Snippets = [.. Documents().SelectMany(d => DocSnippets.Read(d.Path, d.Name))];

    public static TheoryData<string> Blocks() => new(Snippets.Select(s => s.Key));

    [Theory]
    [MemberData(nameof(Blocks))]
    public void ReadmeBlock_Compiles(string key)
    {
        var snippet = Snippets.Single(s => s.Key == key);
        if (!Fragments.IsFragment(snippet))
            CompilerFor(snippet.File).ShouldCompile(snippet);
    }

    [Fact]
    public void EveryReadmeBlock_IsFoundAndFragmentsNameRealHeadings() => Fragments.ShouldMatch([.. Snippets], atLeast: 60);

    /// <summary>Positive control: the compiler rejects a call the library does not have.</summary>
    [Fact]
    public void Compile_RejectsAMethodTheLibraryDoesNotHave()
    {
        var errors = CompilerFor("README.md").Compile("""
            await using var embedder = await LocalEmbedder.LoadAsync("default");
            float[] vector = await embedder.EmbedTextAsync("hello");
            """);

        Assert.NotEmpty(errors);
    }

    private static SnippetCompiler CompilerFor(string document) => new(References)
    {
        Usings = [.. CommonUsings, .. document == "README.md" ? RootUsings : [], .. PackageNamespaces(document)],
        StandIns = DocumentStandIns.TryGetValue(document, out var own)
            ? StandIns.Concat(own).ToDictionary(StringComparer.Ordinal)
            : StandIns,
    };

    // The repository README and every package README: the package READMEs are what nuget.org shows each package's readers.
    private static IEnumerable<(string Path, string Name)> Documents()
    {
        var root = RepoRoot();
        yield return (Path.Combine(root, "README.md"), "README.md");
        foreach (var readme in Directory.GetDirectories(Path.Combine(root, "src")).Order(StringComparer.Ordinal)
                     .Select(d => Path.Combine(d, "README.md")).Where(File.Exists))
            yield return (readme, Path.GetRelativePath(root, readme).Replace('\\', '/'));
    }

    // A package README is read with that package's root namespace in scope (LMSupply.Text for LMSupply.Text.Core):
    // the namespaces of its public types with the fewest segments.
    private static IEnumerable<string> PackageNamespaces(string document)
    {
        var match = PackageReadme().Match(document);
        if (!match.Success)
            return [];

        var namespaces = Assembly.Load(match.Groups["package"].Value).GetExportedTypes()
            .Select(t => t.Namespace).OfType<string>().Distinct().ToList();
        var fewest = namespaces.Min(n => n.Count(c => c == '.'));
        return namespaces.Where(n => n.Count(c => c == '.') == fewest);
    }

    [GeneratedRegex(@"^src/(?<package>[^/]+)/README\.md$")]
    private static partial Regex PackageReadme();

    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "lm-supply.slnx")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("lm-supply.slnx not found above the test output directory");
    }
}
