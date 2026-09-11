using System.Reflection;
using AwesomeAssertions;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Every public option a caller can set must be read somewhere. In 0.63.0, five of the nine
/// <see cref="TranscribeOptions"/> knobs did nothing their documentation promised — three of them were
/// never read at all (<c>MaxTokens</c>, <c>InitialPrompt</c>, <c>BeamWidth</c>), so setting them changed
/// nothing and said nothing. This scans the transcriber assembly's IL for a call to each property's
/// getter outside the options type itself and pins the properties that still have none, so that
/// wiring one up, or adding a new one that nothing reads, is a visible decision.
/// </summary>
/// <remarks>
/// Reading is necessary, not sufficient: <c>Temperature</c> is read and still has no effect on a greedy
/// decoder. That kind of dead knob has no static signal; it is tracked in the issue draft instead.
/// </remarks>
public class TranscribeOptionsReachabilityTests
{
    // Known unread, each with an open decision (implement or remove) — see the TranscribeOptions issue draft.
    private static readonly string[] KnownUnread = ["BeamWidth", "InitialPrompt"];

    [Fact]
    public void EveryTranscribeOption_IsReadByTheTranscriber_ExceptTheKnownList()
    {
        Unread(typeof(TranscribeOptions)).Should().Equal(
            KnownUnread,
            "a public option nothing reads is a promise the library does not keep. Wire it, or change this list " +
            "as a deliberate decision and keep the option's documentation honest about it.");
    }

    [Fact]
    public void EveryTranscriberOption_IsReadByTheTranscriber_ExceptTheKnownList()
    {
        // DisableAutoDownload: documented to throw when the model is not cached, but nothing reads it —
        // the transcriber downloads regardless. The downloader it calls has no offline mode to pass it
        // to; see the DisableAutoDownload issue draft (the same gap in four more modules).
        Unread(typeof(TranscriberOptions)).Should().Equal(["DisableAutoDownload"]);
    }

    // Positive control: the scan must see calls it is known to have, or an empty result above would
    // pass because the detector sees nothing — including reads that live in async state machines.
    [Fact]
    public void Detector_SeesKnownReads()
    {
        var read = Read(typeof(TranscribeOptions));
        read.Should().Contain(["Language", "MaxTokens", "WordTimestamps", "NoSpeechThreshold"]);
    }

    private static List<string> Unread(Type optionsType)
    {
        var read = Read(optionsType);
        return [.. optionsType.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Where(p => p.GetMethod is { IsPublic: true })
            .Select(p => p.Name)
            .Where(name => !read.Contains(name))
            .OrderBy(name => name, StringComparer.Ordinal)];
    }

    private static HashSet<string> Read(Type optionsType)
    {
        var getters = optionsType.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Where(p => p.GetMethod is not null)
            .ToDictionary(p => p.GetMethod!.MetadataToken, p => p.Name);

        var assembly = optionsType.Assembly;
        var module = optionsType.Module;
        var read = new HashSet<string>(StringComparer.Ordinal);
        const BindingFlags all = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance |
                                 BindingFlags.Static | BindingFlags.DeclaredOnly;

        foreach (var type in assembly.GetTypes().Where(t => t != optionsType))
        {
            IEnumerable<MethodBase> bodies = type.GetMethods(all).Cast<MethodBase>().Concat(type.GetConstructors(all));
            foreach (var method in bodies)
            {
                byte[]? il;
                try { il = method.GetMethodBody()?.GetILAsByteArray(); }
                catch (Exception) { continue; }
                if (il is null) continue;

                for (var i = 0; i + 4 < il.Length; i++)
                {
                    // call (0x28) / callvirt (0x6F) followed by a method token. A byte that merely looks
                    // like the opcode inside another operand yields a token that resolves to something
                    // else, or to nothing; only an exact getter match counts.
                    if (il[i] is not (0x28 or 0x6F)) continue;
                    var token = BitConverter.ToInt32(il, i + 1);
                    if (!getters.ContainsKey(token) && (token >> 24) != 0x0A) continue;

                    try
                    {
                        var target = module.ResolveMethod(token);
                        if (target is not null && target.DeclaringType == optionsType &&
                            getters.TryGetValue(target.MetadataToken, out var name))
                        {
                            read.Add(name);
                        }
                    }
                    catch (Exception)
                    {
                        // Not a method token in this module — an operand byte, not a call.
                    }
                }
            }
        }

        return read;
    }
}
