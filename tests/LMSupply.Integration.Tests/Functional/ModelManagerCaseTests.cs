using System.Net;
using System.Net.Http.Json;
using AwesomeAssertions;
using Microsoft.AspNetCore.Mvc.Testing;

namespace LMSupply.Integration.Tests.Functional;

[Trait("Category", "Functional")]
public class ModelManagerCaseTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public ModelManagerCaseTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Theory]
    [InlineData("embedder")]
    [InlineData("Embedder")]
    [InlineData("EMBEDDER")]
    public async Task GET_RegistryByType_CaseInsensitive_Returns200(string type)
    {
        var response = await _client.GetAsync($"/api/registry/models/{type}", TestContext.Current.CancellationToken);
        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }

    [Theory]
    [InlineData("embedder")]
    [InlineData("Embedder")]
    [InlineData("EMBEDDER")]
    public async Task GET_CacheModelsByType_CaseInsensitive_Returns200(string type)
    {
        var response = await _client.GetAsync($"/api/cache/models/type/{type}", TestContext.Current.CancellationToken);
        response.StatusCode.Should().Be(HttpStatusCode.OK);
    }
}
