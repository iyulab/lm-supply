using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// The server listens on <see cref="LlamaServerProcess.LoopbackHost"/> and every client connects to that same literal.
/// A <c>localhost</c> URL resolves to <c>::1</c> first; the server listens on IPv4 only, and on Windows a refused IPv6
/// connect takes about two seconds — measured 2,042 ms per new connection against 2 ms for the literal. So the startup
/// health poll and each client's first request paid it on every model load.
/// </summary>
public sealed class LoopbackAddressConventionTests
{
    [Fact]
    public void BaseUrl_UsesTheAddressTheServerListensOn()
    {
        var info = new LlamaServerInfo { ProcessId = 1, Port = 8081, ModelPath = "m.gguf" };

        info.BaseUrl.Should().Be("http://127.0.0.1:8081");
        LlamaServerProcess.LoopbackHost.Should().Be("127.0.0.1");
    }

    [Fact]
    public void LlamaSources_BuildNoLocalhostUrl()
    {
        var root = new DirectoryInfo(AppContext.BaseDirectory);
        while (root is not null && !File.Exists(Path.Combine(root.FullName, "lm-supply.slnx")))
            root = root.Parent;
        root.Should().NotBeNull("the test runs under the repository");

        var offenders = Directory.EnumerateFiles(Path.Combine(root!.FullName, "src", "LMSupply.Llama"), "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}"))
            .Where(f => File.ReadLines(f).Any(line => line.Contains("http://localhost", StringComparison.OrdinalIgnoreCase)
                && !line.TrimStart().StartsWith("///", StringComparison.Ordinal)))
            .Select(Path.GetFileName)
            .ToList();

        offenders.Should().BeEmpty("clients connect to LlamaServerProcess.LoopbackHost, the address the server listens on");
    }
}
