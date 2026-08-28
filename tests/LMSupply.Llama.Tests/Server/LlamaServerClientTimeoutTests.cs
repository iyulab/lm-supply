using System.Reflection;
using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Verifies LlamaServerClient's internally-created HttpClient uses a timeout suited to
/// local/CPU-bound inference (5 minutes) instead of HttpClient's 100-second default, which
/// multi-step tool-calling loops routinely exceed as each round's prompt grows with prior tool
/// results. Regression: ecosystem-e2e.yml ChatToolFlowTests hitting "HttpClient.Timeout of 100
/// seconds elapsing" once earlier blockers (Gemma4 ctx_other, ONNX provisioning order) stopped
/// masking it.
/// </summary>
public class LlamaServerClientTimeoutTests
{
    private static TimeSpan GetInternalHttpClientTimeout(LlamaServerClient client)
    {
        var field = typeof(LlamaServerClient).GetField("_httpClient", BindingFlags.NonPublic | BindingFlags.Instance);
        var httpClient = (HttpClient)field!.GetValue(client)!;
        return httpClient.Timeout;
    }

    [Fact]
    public void Constructor_NoHttpClientNoRequestTimeout_DefaultsToFiveMinutes()
    {
        using var client = new LlamaServerClient("http://localhost:9999");

        GetInternalHttpClientTimeout(client).Should().Be(TimeSpan.FromMinutes(5),
            "local/CPU-bound inference routinely exceeds HttpClient's 100-second default");
    }

    [Fact]
    public void Constructor_NoHttpClientWithRequestTimeout_UsesSuppliedValue()
    {
        var requested = TimeSpan.FromMinutes(10);
        using var client = new LlamaServerClient("http://localhost:9999", requestTimeout: requested);

        GetInternalHttpClientTimeout(client).Should().Be(requested);
    }

    [Fact]
    public void Constructor_ExternalHttpClientSupplied_RequestTimeoutIgnored()
    {
        using var externalClient = new HttpClient { Timeout = TimeSpan.FromSeconds(42) };
        using var client = new LlamaServerClient(
            "http://localhost:9999", externalClient, requestTimeout: TimeSpan.FromMinutes(10));

        GetInternalHttpClientTimeout(client).Should().Be(TimeSpan.FromSeconds(42),
            "a caller-supplied HttpClient owns its own timeout");
    }

    [Fact]
    public void LlamaServerConfig_RequestTimeout_DefaultsToFiveMinutes()
    {
        var config = new LlamaServerConfig { ModelPath = "model.gguf" };

        config.RequestTimeout.Should().Be(TimeSpan.FromMinutes(5));
    }
}
