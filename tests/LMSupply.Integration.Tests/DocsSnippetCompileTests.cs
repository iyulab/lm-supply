using System.Reflection;
using Iyu.Conventions.Testing.Snippets;
using Microsoft.CodeAnalysis;
using Xunit;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Compiles the C# blocks of the guides under <c>docs/</c> against the current assemblies. A reader copies these; a block
/// that no longer matches the shipped API fails here instead of in their build. The READMEs are compiled by
/// <see cref="ReadmeSnippetCompileTests"/>.
/// </summary>
/// <remarks>
/// Each guide compiles as a whole (<see cref="SnippetCompiler.CompileDocument"/>): a type one block declares is visible to
/// the next, each block's statements run in a method of their own, and a block that restates a library type is checked
/// against the real type (<see cref="SnippetCompiler.CheckListings"/>). A guide's blocks mostly show no usings, so every
/// package's namespaces are in scope — no two packages share a public type name (PublicTypeNameUniquenessTests) — with the stand-ins below for
/// values the surrounding text gives the reader. A section that is deliberately not a program is in <see cref="Fragments"/> with the reason. A document with a C#
/// block is in <see cref="Checked"/>, in <see cref="Historical"/> or in <see cref="NotYetChecked"/> with the reason — never
/// silently in none.
/// </remarks>
public class DocsSnippetCompileTests
{
    // Guides whose every C# block compiles (or is a listed fragment).
    private static readonly string[] Checked =
    [
        "docs/API_DESIGN_GUIDELINES.md",
        "docs/GPU_PROVIDERS.md",
        "docs/MEMORY_REQUIREMENTS.md",
        "docs/MODEL_LIFECYCLE.md",
        "docs/TROUBLESHOOTING.md",
        "docs/captioner.md",
        "docs/decider.md",
        "docs/detector.md",
        "docs/embedder.md",
        "docs/generator.md",
        "docs/llama.md",
        "docs/ocr.md",
        "docs/reranker.md",
        "docs/segmenter.md",
        "docs/synthesizer.md",
        "docs/transcriber.md",
        "docs/translator.md",
    ];

    // Documents that show an API as it was, on purpose — a migration guide's «before», a decision record. They are not
    // compiled; their «after» code is what the guides above show.
    private static readonly Dictionary<string, string> Historical = new(StringComparer.Ordinal);

    // Guides with C# blocks that are not compiled yet, with the reason. Shrink this.
    private static readonly Dictionary<string, string> NotYetChecked = new(StringComparer.Ordinal);

    // A block that is deliberately not a program (a signature sketch, pseudocode), by the heading it sits under, with the
    // reason. Shrink this, never grow it silently.
    private static readonly SnippetRoster Fragments = new(new Dictionary<string, string>(StringComparer.Ordinal)
    {
        // docs/API_DESIGN_GUIDELINES.md states the shape every package follows, with {Domain} placeholders.
        ["2.1 Static Factory Class"] = "the static factory every package provides, as a template with {Domain} placeholders and bodiless members",
        ["2.2 Builder Pattern (Optional, for Complex Configuration)"] = "the builder a package may provide, as a template with {Domain} placeholders and bodiless members",
        ["3.1 Interface Naming"] = "the model interface's declaration line with its {Domain} placeholder, no body",
        ["3.2 Required Members"] = "the members every model interface declares, as a template with a {Domain} placeholder",
        ["3.3 Domain-Specific Methods"] = "method shapes with {Verb}, {Input}, {Output} and {Token} placeholders",
        ["4.1 Naming"] = "the options class's declaration line with its {Domain} placeholder, no body",
        ["4.2 Required Properties"] = "the properties every options class declares, as a template with a {Domain} placeholder",
        ["4.3 Common Optional Properties"] = "properties shown on their own, outside the options class each package declares them on",
        ["6.1 ModelInfo Record"] = "the model-information record as a template with a {Domain} placeholder and an elided parameter list",
        ["9.1 Disposable Pattern"] = "the model interface's declaration with a {Domain} placeholder",
    });

    private static readonly string[] LibraryAssemblies =
    [
        "LMSupply.Core", "LMSupply.Embedder", "LMSupply.Reranker", "LMSupply.Decider", "LMSupply.Generator", "LMSupply.Generator.Onnx",
        "LMSupply.Llama", "LMSupply.Translator", "LMSupply.Transcriber", "LMSupply.Synthesizer", "LMSupply.Captioner",
        "LMSupply.Ocr", "LMSupply.Detector", "LMSupply.Segmenter", "LMSupply.ImageGenerator",
        "LMSupply.Text.Core", "LMSupply.Vision.Core",
    ];

    private static readonly string[] AssembliesToLoad =
    [
        .. LibraryAssemblies,
        "Microsoft.Extensions.AI.Abstractions",
        // The transcriber guide's ASP.NET Core start-up block.
        "Microsoft.AspNetCore",
    ];

    private static readonly string[] CommonUsings =
    [
        "System", "System.Collections.Generic", "System.IO", "System.Linq", "System.Net.Http", "System.Threading",
        "System.Threading.Tasks",
    ];

    // A package guide is read with every namespace its package exports, as a reader who referenced that package has them;
    // the cross-package guides with the packages' root namespaces only, since two packages can export a type of the same
    // name (ModelInfo, GenerationOptions).
    // Values the guides' running text gives the reader, typed as the API takes them.
    private static readonly Dictionary<string, string> StandIns = new(StringComparer.Ordinal)
    {
        ["cancellationToken"] = "CancellationToken cancellationToken = default;",
        // The vision guides' «bytes from an API» examples.
        ["httpClient"] = "HttpClient httpClient = new();",
        ["imageUrl"] = "string imageUrl = \"\";",
    };

    // A guide's blocks read on from its own Quick Start: the model it loaded there, and the inputs its text names.
    private static readonly Dictionary<string, Dictionary<string, string>> DocumentStandIns = new(StringComparer.Ordinal)
    {
        ["docs/captioner.md"] = new() { ["captioner"] = "ICaptionerModel captioner = null!;" },
        ["docs/detector.md"] = new()
        {
            ["detector"] = "IDetectorModel detector = null!;",
            ["imageWidth"] = "int imageWidth = 640;",
            ["imageHeight"] = "int imageHeight = 480;",
            // Qualified: LMSupply.Vision exports a BoundingBox of another shape.
            ["otherBox"] = "LMSupply.Vision.BoundingBox otherBox = default;",
        },
        ["docs/embedder.md"] = new()
        {
            ["model"] = "IEmbeddingModel model = null!;",
            ["documents"] = "string[] documents = [];",
            ["embedding1"] = "float[] embedding1 = [];",
            ["embedding2"] = "float[] embedding2 = [];",
            // The reader's own vector store, which the vector-space-revision example compares and re-embeds.
            ["index"] = "VectorIndex index = new();",
            ["VectorIndex"] = "public sealed class VectorIndex { public string? EmbeddingRevision { get; set; } public Task ReembedAsync(IEmbeddingModel model) => Task.CompletedTask; }",
        },
        ["docs/generator.md"] = new()
        {
            ["generator"] = "IGeneratorModel generator = null!;",
            ["model"] = "IGeneratorModel model = null!;",
            ["oldModel"] = "IGeneratorModel oldModel = null!;",
            ["prompt"] = "string prompt = \"\";",
            ["messages"] = "IReadOnlyList<ChatMessage> messages = [];",
            ["options"] = "GeneratorOptions options = new();",
            ["progress"] = "IProgress<DownloadProgress> progress = null!;",
        },
        ["docs/llama.md"] = new()
        {
            ["model"] = "IGeneratorModel model = null!;",
            ["serverVersion"] = "string? serverVersion = \"b8672\";",
        },
        ["docs/ocr.md"] = new() { ["ocr"] = "IOcr ocr = null!;" },
        ["docs/reranker.md"] = new()
        {
            ["reranker"] = "IRerankerModel reranker = null!;",
            ["query"] = "string query = \"\";",
            ["documents"] = "string[] documents = [];",
            ["docs1"] = "string[] docs1 = [];",
            ["docs2"] = "string[] docs2 = [];",
            ["longDocument"] = "string longDocument = \"\";",
            ["progress"] = "IProgress<DownloadProgress> progress = null!;",
        },
        ["docs/segmenter.md"] = new() { ["segmenter"] = "ISegmenterModel segmenter = null!;" },
        ["docs/synthesizer.md"] = new()
        {
            ["synthesizer"] = "ISynthesizerModel synthesizer = null!;",
            ["result"] = "SynthesisResult result = null!;",
            ["sentences"] = "IReadOnlyList<string> sentences = [];",
            // The reader's own audio sink.
            ["ProcessAudioChunk"] = "static void ProcessAudioChunk(float[] samples, int sampleRate) { }",
        },
        ["docs/transcriber.md"] = new()
        {
            ["transcriber"] = "ITranscriberModel transcriber = null!;",
            ["result"] = "TranscriptionResult result = null!;",
            ["args"] = "string[] args = [];",
        },
        ["docs/translator.md"] = new() { ["translator"] = "ITranslatorModel translator = null!;" },
        ["docs/MODEL_LIFECYCLE.md"] = new()
        {
            ["model"] = "IEmbeddingModel model = null!;",
            ["text"] = "string text = \"\";",
            ["texts"] = "string[] texts = [];",
            ["query"] = "string query = \"\";",
            ["documents"] = "string[] documents = [];",
        },
        ["docs/MEMORY_REQUIREMENTS.md"] = new()
        {
            ["embedder"] = "IEmbeddingModel embedder = null!;",
            ["texts"] = "string[] texts = [];",
            ["thousandDocuments"] = "string[] thousandDocuments = [];",
        },
    };

    private static readonly IReadOnlyList<MetadataReference> References =
        SnippetCompiler.ReferencesFor([.. AssembliesToLoad.Select(Assembly.Load)]);

    private static readonly IReadOnlyList<DocSnippet> Snippets =
        [.. Checked.SelectMany(d => DocSnippets.Read(Path.Combine(RepositoryRoot(), d), d))];

    private static readonly Dictionary<string, Lazy<DocumentCompilation>> Documents = Checked.ToDictionary(
        d => d,
        d => new Lazy<DocumentCompilation>(() => CompilerFor(d).CompileDocument(
            [.. Snippets.Where(s => s.File == d && !Fragments.IsFragment(s))])),
        StringComparer.Ordinal);

    private static SnippetCompiler CompilerFor(string document) => new(References)
    {
        Usings =
        [
            .. CommonUsings,
            // The packages' own namespaces; a namespace they extend (Microsoft.Extensions.AI) is the reader's import.
            .. ExportedNamespaces(AssembliesToLoad).Where(n => n.StartsWith("LMSupply", StringComparison.Ordinal)),
        ],
        StandIns = DocumentStandIns.TryGetValue(document, out var own)
            ? StandIns.Concat(own).ToDictionary(StringComparer.Ordinal)
            : StandIns,
        CheckListings = true,
    };

    private static IEnumerable<string> ExportedNamespaces(IEnumerable<string> assemblies) =>
        assemblies.SelectMany(n => Assembly.Load(n).GetExportedTypes())
            .Select(t => t.Namespace)
            .OfType<string>()
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal);

    public static TheoryData<string> Blocks() => new(Snippets.Select(s => s.Key));

    public static TheoryData<string> CheckedDocuments() => new(Checked);

    [Theory]
    [MemberData(nameof(Blocks))]
    public void DocsBlock_Compiles(string key)
    {
        var snippet = Snippets.Single(s => s.Key == key);
        if (!Fragments.IsFragment(snippet))
            Documents[snippet.File].Value.ShouldCompile(snippet);
    }

    /// <summary>A document's type stand-ins compile against the current API (they belong to no single example).</summary>
    [Theory]
    [MemberData(nameof(CheckedDocuments))]
    public void Document_HasNoErrorsOutsideItsExamples(string document) => Documents[document].Value.ShouldHaveNoSharedErrors();

    [Fact]
    public void EveryCheckedBlock_IsFoundAndFragmentsNameRealHeadings() => Fragments.ShouldMatch([.. Snippets], atLeast: 229);

    /// <summary>
    /// The guides' type listings (an options class or an interface shown with its members) are checked against the real
    /// types, not compiled as a second type of the same name. A floor, so a listing the kit stops recognising shows here
    /// instead of passing unchecked.
    /// </summary>
    [Fact]
    public void TypeListings_AreCheckedAgainstTheRealTypes()
    {
        var listed = Checked.Sum(d => Documents[d].Value.ListedTypes);
        Assert.True(listed >= 12, $"expected the guides' type listings to be checked, found {listed}");
    }

    /// <summary>
    /// Positive control: the compiler rejects the local-path load the lifecycle guide showed before it was compiled — a
    /// <c>LocalEmbedder.LoadFromPathAsync</c> the embedder has never had (a path goes to <c>LoadAsync</c>). The name roster
    /// these tests replace passed it, because it only asked whether some public type declares a method of that name, and
    /// <c>LocalGenerator.LoadFromPathAsync</c> does.
    /// </summary>
    [Fact]
    public void Compile_RejectsTheLocalPathLoadTheLifecycleGuideShowedBeforeItWasCompiled()
    {
        var compiler = CompilerFor("docs/MODEL_LIFECYCLE.md");

        Assert.Empty(compiler.Compile("""await using var local = await LocalEmbedder.LoadAsync("/path/to/model.onnx");"""));
        Assert.NotEmpty(compiler.Compile("""await using var local = await LocalEmbedder.LoadFromPathAsync("/path/to/model");"""));
    }

    /// <summary>
    /// A guide with a C# block is checked, historical or listed as not yet checked with the reason, and the lists name only
    /// documents that exist and have C# blocks — so a new guide cannot be skipped silently and a repaired one cannot stay
    /// on the waiting list. The READMEs are <see cref="ReadmeSnippetCompileTests"/>'.
    /// </summary>
    [Fact]
    public void EveryGuideWithCSharp_IsCheckedOrListed()
    {
        var withCSharp = GuidesWithCSharp().ToList();
        var listed = Checked.Concat(Historical.Keys).Concat(NotYetChecked.Keys).ToList();

        Assert.All(withCSharp, d => Assert.True(listed.Contains(d), $"{d} has C# blocks but is in no list"));
        Assert.Equal(listed.Count, listed.Distinct(StringComparer.Ordinal).Count());
        Assert.All(listed, d => Assert.Contains(d, withCSharp));
    }

    /// <summary>
    /// Every Markdown document of the repository with a C# block, except the READMEs compiled elsewhere, changelogs, the
    /// tests' own, build output, dependencies (<c>node_modules</c>) and dot-folders. «Of the repository» is what git does not
    /// ignore: a folder a contributor keeps locally and ignores is not a document of the repository.
    /// </summary>
    private static IEnumerable<string> GuidesWithCSharp()
    {
        var root = RepositoryRoot();
        return MarkdownFilesGitDoesNotIgnore(root)
            .Where(p => !p.StartsWith("tests/", StringComparison.Ordinal)
                        && !p.Contains("/bin/", StringComparison.Ordinal) && !p.Contains("/obj/", StringComparison.Ordinal)
                        && !p.Contains("node_modules/", StringComparison.Ordinal)
                        && !p.StartsWith('.') && !p.Contains("/.", StringComparison.Ordinal)
                        && !IsReadmeTestsDocument(p)
                        && !Path.GetFileName(p).Equals("CHANGELOG.md", StringComparison.OrdinalIgnoreCase))
            .Where(p => DocSnippets.Read(Path.Combine(root, p), p).Count > 0)
            .Order(StringComparer.Ordinal);
    }

    // Tracked and untracked-but-not-ignored Markdown, relative and with forward slashes; every Markdown file when git is absent.
    private static List<string> MarkdownFilesGitDoesNotIgnore(string root)
    {
        try
        {
            using var git = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(
                "git", "ls-files -z --cached --others --exclude-standard -- *.md")
            {
                WorkingDirectory = root,
                RedirectStandardOutput = true,
                UseShellExecute = false,
            })!;
            var output = git.StandardOutput.ReadToEnd();
            git.WaitForExit();
            if (git.ExitCode == 0)
                return output.Split('\0', StringSplitOptions.RemoveEmptyEntries).Distinct(StringComparer.Ordinal).ToList();
        }
        catch (System.ComponentModel.Win32Exception)
        {
            // git is not installed: fall back to the file system below
        }

        return Directory.EnumerateFiles(root, "*.md", SearchOption.AllDirectories)
            .Select(p => Path.GetRelativePath(root, p).Replace('\\', '/'))
            .ToList();
    }

    // The repository README and the package READMEs (src/<package>/README.md) are ReadmeSnippetCompileTests'.
    private static bool IsReadmeTestsDocument(string path) =>
        path == "README.md" || (path.StartsWith("src/", StringComparison.Ordinal) && path.EndsWith("/README.md", StringComparison.Ordinal)
                                && path.Count(c => c == '/') == 2);

    private static string RepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "lm-supply.slnx")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("lm-supply.slnx not found above the test output directory");
    }
}
