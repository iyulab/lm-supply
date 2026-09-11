using System.Reflection;
using System.Runtime.Loader;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Every public option a caller can set must be read by the library. An option nothing reads is a
/// promise the library does not keep: setting it changes nothing and says nothing. In 0.63.0 three of
/// the transcriber's nine <c>TranscribeOptions</c> were never read, and <c>DisableAutoDownload</c> —
/// documented to throw when a model is not cached — was read by one module of the six that declare it.
/// </summary>
/// <remarks>
/// <para>
/// This scans the IL of every library assembly for a call to each option property's getter, outside
/// the options type itself, and pins the properties that have none. Wiring one up, or adding a new
/// option that nothing reads, then shows up here as a deliberate change to the roster.
/// </para>
/// <para>
/// It lives in this project because it is the one that references every module. It carries no
/// category trait, so it runs in CI.
/// </para>
/// <para>
/// Two limits, both deliberate. Reading is necessary, not sufficient — an option can be read and still
/// have no effect (<c>TranscribeOptions.Temperature</c> on a greedy decoder); that has no static signal.
/// And only properties an options type declares itself are checked: a property inherited from
/// <c>LMSupplyOptionsBase</c> may be honoured through a Core helper the module hands its options to,
/// which a direct-call scan cannot attribute to one module.
/// </para>
/// </remarks>
public class OptionsReachabilityRosterTests
{
    // Each entry has an open issue draft: wire the option, or remove it as a deliberate decision.
    private static readonly Dictionary<string, string[]> KnownUnread = new(StringComparer.Ordinal)
    {
        // No beam search and no prompt encoder (issue draft "TranscribeOptions knobs that do nothing").
        ["LMSupply.Transcriber.TranscribeOptions"] = ["BeamWidth", "InitialPrompt"],

        // Found by this roster's first run (issue draft "options nothing reads, across seven types").
        ["LMSupply.Runtime.RuntimeManagerOptions"] = ["MaxRetries", "ProxyPassword", "ProxyUrl", "ProxyUsername"],
        ["LMSupply.Captioner.CaptionerOptions"] = ["Prompt"],
        ["LMSupply.Ocr.OcrOptions"] = ["UsePolygon"],
        ["LMSupply.Synthesizer.SynthesizeOptions"] = ["Pitch"],
    };

    private static readonly Lazy<Scan> Result = new(Run);

    [Fact]
    public void EveryPublicOption_IsReadByTheLibrary_ExceptTheKnownRoster()
    {
        // One line per type, so a failure prints the whole roster it found — not just which keys differ.
        var unread = Result.Value.Unread
            .Where(kv => kv.Value.Length > 0)
            .Select(kv => $"{kv.Key}: {string.Join(",", kv.Value)}")
            .Order(StringComparer.Ordinal)
            .ToList();
        var expected = KnownUnread
            .Select(kv => $"{kv.Key}: {string.Join(",", kv.Value.Order(StringComparer.Ordinal))}")
            .Order(StringComparer.Ordinal)
            .ToList();

        unread.Should().Equal(
            expected,
            "a public option nothing in the library reads is a promise it does not keep. Wire it, or change "
            + "this roster as a deliberate decision and keep the option's documentation honest about it.");
    }

    // Positive controls: the scan must see reads it is known to have — same-assembly reads, reads that
    // live in async state machines, and a read from another assembly — or an empty roster above would
    // pass because the detector sees nothing.
    [Fact]
    public void Scan_SeesKnownReads()
    {
        var scan = Result.Value;

        scan.OptionTypes.Should().HaveCountGreaterThan(20, "the scan must find the library's options types");
        scan.Read.Should().Contain(
        [
            "LMSupply.Transcriber.TranscribeOptions.Language",
            "LMSupply.Transcriber.TranscribeOptions.MaxTokens",
            "LMSupply.Transcriber.TranscribeOptions.WordTimestamps",
            "LMSupply.Reranker.RerankerOptions.DisableAutoDownload",
        ]);
        scan.CrossAssemblyReads.Should().NotBeEmpty("some options are declared in one assembly and read in another");
    }

    private sealed record Scan(
        IReadOnlyList<Type> OptionTypes,
        IReadOnlyDictionary<string, string[]> Unread,
        IReadOnlySet<string> Read,
        IReadOnlySet<string> CrossAssemblyReads);

    private static Scan Run()
    {
        var assemblies = LibraryAssemblies();

        var optionTypes = assemblies
            .SelectMany(a => a.GetTypes())
            .Where(t => t is { IsPublic: true, IsClass: true, IsAbstract: false } || t is { IsNestedPublic: true, IsClass: true, IsAbstract: false })
            .Where(t => t.Name.EndsWith("Options", StringComparison.Ordinal))
            .OrderBy(t => t.FullName, StringComparer.Ordinal)
            .ToList();

        // (module, getter token) -> "Type.Property", for properties each type declares itself.
        var getters = new Dictionary<(Module, int), (Type Type, string Name)>();
        foreach (var type in optionTypes)
        {
            foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
            {
                if (property.GetMethod is { IsPublic: true } getter)
                {
                    getters[(getter.Module, getter.MetadataToken)] = (type, property.Name);
                }
            }
        }

        var read = new HashSet<string>(StringComparer.Ordinal);
        var crossAssembly = new HashSet<string>(StringComparer.Ordinal);
        const BindingFlags all = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance |
                                 BindingFlags.Static | BindingFlags.DeclaredOnly;

        // Reads inside an options type count only through a member the library calls from outside it:
        // LlamaOptions reads GpuOffloadRatio in the method that computes the effective layer count, and
        // that is how the option is honoured. Clone and constructors are the exception — every module
        // clones its options, and a copy is not a use.
        var readsInside = new Dictionary<(Module, int), List<string>>();
        var calledFromOutside = new HashSet<(Module, int)>();

        foreach (var assembly in assemblies)
        {
            foreach (var type in assembly.GetTypes())
            {
                var owner = optionTypes.FirstOrDefault(o => IsWithin(type, o));
                IEnumerable<MethodBase> bodies = type.GetMethods(all).Cast<MethodBase>().Concat(type.GetConstructors(all));
                foreach (var method in bodies)
                {
                    foreach (var target in Calls(method, type.Module))
                    {
                        var targetKey = (target.Module, target.MetadataToken);
                        if (getters.TryGetValue(targetKey, out var option))
                        {
                            var key = $"{option.Type.FullName}.{option.Name}";
                            if (owner == option.Type)
                            {
                                if (method is MethodInfo && method.Name != "Clone" && owner == method.DeclaringType)
                                {
                                    var methodKey = (method.Module, method.MetadataToken);
                                    if (!readsInside.TryGetValue(methodKey, out var list))
                                        readsInside[methodKey] = list = [];
                                    list.Add(key);
                                }

                                continue;
                            }

                            read.Add(key);
                            if (type.Assembly != option.Type.Assembly)
                                crossAssembly.Add(key);
                        }
                        else if (target.DeclaringType is { } declaring
                                 && optionTypes.Contains(declaring)
                                 && !IsWithin(type, declaring))
                        {
                            calledFromOutside.Add(targetKey);
                        }
                    }
                }
            }
        }

        foreach (var (method, keys) in readsInside)
        {
            if (calledFromOutside.Contains(method))
                read.UnionWith(keys);
        }

        var unread = optionTypes.ToDictionary(
            t => t.FullName!,
            t => t.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                .Where(p => p.GetMethod is { IsPublic: true })
                .Select(p => p.Name)
                .Where(name => !read.Contains($"{t.FullName}.{name}"))
                .Order(StringComparer.Ordinal)
                .ToArray(),
            StringComparer.Ordinal);

        return new Scan(optionTypes, unread, read, crossAssembly);
    }

    // Every method a body calls: call (0x28) / callvirt (0x6F) followed by a MethodDef (0x06) or
    // MemberRef (0x0A) token. A byte that merely looks like the opcode inside another operand yields a
    // token that resolves to something else, or to nothing; callers match exact methods only.
    private static IEnumerable<MethodBase> Calls(MethodBase method, Module module)
    {
        byte[]? il;
        try { il = method.GetMethodBody()?.GetILAsByteArray(); }
        catch (Exception) { yield break; }
        if (il is null) yield break;

        for (var i = 0; i + 4 < il.Length; i++)
        {
            if (il[i] is not (0x28 or 0x6F)) continue;
            var token = BitConverter.ToInt32(il, i + 1);
            if ((token >> 24) is not (0x06 or 0x0A)) continue;

            MethodBase? target;
            try { target = module.ResolveMethod(token); }
            catch (Exception) { continue; }

            if (target is not null)
                yield return target;
        }
    }

    private static bool IsWithin(Type type, Type container)
    {
        for (var t = type; t is not null; t = t.DeclaringType)
        {
            if (t == container) return true;
        }

        return false;
    }

    // Every library assembly copied next to the tests — not the tests themselves, and not the console
    // host, which is a consumer: its reading an option does not make the library honour it. Loaded by
    // name because the compiler drops a project reference the test code never names.
    private static List<Assembly> LibraryAssemblies()
        => [.. Directory.EnumerateFiles(AppContext.BaseDirectory, "LMSupply.*.dll")
            .Select(Path.GetFileNameWithoutExtension)
            .Where(name => name is not null
                && !name.EndsWith(".Tests", StringComparison.Ordinal)
                && !name.Contains(".Console", StringComparison.Ordinal))
            .Order(StringComparer.Ordinal)
            .Select(name => AssemblyLoadContext.Default.LoadFromAssemblyName(new AssemblyName(name!)))];
}
