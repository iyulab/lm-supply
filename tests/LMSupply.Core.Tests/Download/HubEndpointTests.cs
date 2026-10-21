using System.Net;
using System.Net.Sockets;
using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Download;

public class HubEndpointResolveTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Unset_IsTheHuggingFaceHub(string? configured)
    {
        HubEndpoint.Resolve(configured).Should().Be("https://huggingface.co");
    }

    [Theory]
    [InlineData("https://hub.example.invalid", "https://hub.example.invalid")]
    [InlineData(" https://hub.example.invalid/ ", "https://hub.example.invalid")]
    [InlineData("http://hub.example.invalid:8080/mirror/", "http://hub.example.invalid:8080/mirror")]
    public void Set_IsThatBaseUrlWithoutTrailingSlash(string configured, string expected)
    {
        HubEndpoint.Resolve(configured).Should().Be(expected);
    }

    [Theory]
    [InlineData("hub.example.invalid")]
    [InlineData("ftp://hub.example.invalid")]
    [InlineData("not a url")]
    public void NotAnAbsoluteHttpUrl_IsRefusedWithTheVariableName(string configured)
    {
        var act = () => HubEndpoint.Resolve(configured);

        act.Should().Throw<InvalidOperationException>().WithMessage("*HF_ENDPOINT*");
    }
}

/// <summary>Sets the process-wide <c>HF_ENDPOINT</c>, so it runs alone.</summary>
[CollectionDefinition(nameof(HubEndpointSerialGroup), DisableParallelization = true)]
public sealed class HubEndpointSerialGroup;

[Collection(nameof(HubEndpointSerialGroup))]
public sealed class HubEndpointDownloadTests : IDisposable
{
    private readonly string? _original = Environment.GetEnvironmentVariable(HubEndpoint.EnvironmentVariable);
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-hub-endpoint-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(HubEndpoint.EnvironmentVariable, _original);
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    [Fact]
    public async Task Download_GoesToHfEndpoint_AndAnUnreachableHostIsTyped()
    {
        // A port nothing listens on: the connection is refused, the host never answers.
        var port = FreeLoopbackPort();
        Environment.SetEnvironmentVariable(HubEndpoint.EnvironmentVariable, $"http://127.0.0.1:{port}");
        using var downloader = new HuggingFaceDownloader(_cache);

        var act = () => downloader.DownloadModelAsync("example-org/example-model", ["model.onnx"], cancellationToken: TestContext.Current.CancellationToken);

        var failure = (await act.Should().ThrowAsync<ModelHostUnreachableException>()).Which;
        failure.Host.Should().Be("127.0.0.1");
        failure.InnerException!.Message.Should().Contain($":{port}", "the request went to HF_ENDPOINT, not the default hub");
        failure.ModelId.Should().Be("example-org/example-model");
        failure.Message.Should().Contain("127.0.0.1").And.Contain("HF_ENDPOINT");
        failure.Should().BeAssignableTo<ModelDownloadException>("a caller catching download failures keeps catching this one");
    }

    [Fact]
    public async Task Download_WithAMalformedHfEndpoint_SaysSoInsteadOfGoingToTheDefaultHub()
    {
        Environment.SetEnvironmentVariable(HubEndpoint.EnvironmentVariable, "hub.example.invalid");
        using var downloader = new HuggingFaceDownloader(_cache);

        var act = () => downloader.DownloadModelAsync("example-org/example-model", ["model.onnx"], cancellationToken: TestContext.Current.CancellationToken);

        await act.Should().ThrowAsync<InvalidOperationException>().WithMessage("*HF_ENDPOINT*hub.example.invalid*");
    }

    private static int FreeLoopbackPort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        return ((IPEndPoint)listener.LocalEndpoint).Port;
    }
}
