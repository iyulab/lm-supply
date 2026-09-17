using System.Text.RegularExpressions;
using AwesomeAssertions;

namespace LMSupply.Integration.Tests;

/// <summary>
/// A network behind a proxy is served by the .NET default proxy (<c>HTTPS_PROXY</c>, the system proxy, or
/// <c>HttpClient.DefaultProxy</c>) — the one mechanism every LMSupply download goes through. It only covers all of them
/// while no handler in the library turns it off or replaces it; this scan keeps it that way. (The library once declared
/// proxy options on a runtime-manager type no caller could reach; they were removed in favour of this.)
/// </summary>
public class DefaultProxyHonoredTests
{
    private static readonly Regex ProxyOverride = new(
        @"\bUseProxy\s*=\s*false\b|\bProxy\s*=\s*(?!null\b)[^=;]",
        RegexOptions.Compiled);

    [Theory]
    [InlineData("var handler = new HttpClientHandler { UseProxy = false };")]
    [InlineData("handler.Proxy = new WebProxy(url);")]
    [InlineData("new SocketsHttpHandler { Proxy = customProxy }")]
    public void Scanner_FlagsAHandlerThatBypassesOrReplacesTheDefaultProxy(string line)
        => ProxyOverride.IsMatch(line).Should().BeTrue();

    [Theory]
    [InlineData("var handler = new HttpClientHandler { AutomaticDecompression = DecompressionMethods.All };")]
    [InlineData("// UseProxy stays true so HTTPS_PROXY applies")]
    [InlineData("if (handler.Proxy == null) return;")]
    public void Scanner_AcceptsCodeThatLeavesTheDefaultProxyAlone(string line)
        => ProxyOverride.IsMatch(line.TrimStart().StartsWith("//", StringComparison.Ordinal) ? string.Empty : line).Should().BeFalse();

    [Fact]
    public void NoLibrarySource_BypassesOrReplacesTheDefaultProxy()
    {
        var src = Path.Combine(RepositoryRoot(), "src");
        var offenders = Directory.EnumerateFiles(src, "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                     && !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .SelectMany(f => File.ReadLines(f).Select((line, i) => (File: f, Line: i + 1, Text: line)))
            .Where(l => !l.Text.TrimStart().StartsWith("//", StringComparison.Ordinal) && ProxyOverride.IsMatch(l.Text))
            .Select(l => $"{Path.GetRelativePath(src, l.File)}:{l.Line}  {l.Text.Trim()}")
            .ToList();

        offenders.Should().BeEmpty("every download must honour the .NET default proxy — documented as the way to run behind one");
    }

    private static string RepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "lm-supply.slnx")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("lm-supply.slnx not found above the test output directory");
    }
}
