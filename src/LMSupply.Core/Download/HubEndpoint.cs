namespace LMSupply.Download;

/// <summary>
/// The Hugging Face Hub base URL every download and Hub API request of this library goes to: the <c>HF_ENDPOINT</c>
/// environment variable when it is set — the variable huggingface_hub reads, for a mirror or an internal hub — and
/// <c>https://huggingface.co</c> otherwise. Read on every request, so a host that sets the variable before its first
/// model load needs nothing else.
/// </summary>
internal static class HubEndpoint
{
    /// <summary>The environment variable huggingface_hub reads for the Hub base URL.</summary>
    public const string EnvironmentVariable = "HF_ENDPOINT";

    /// <summary>The Hub base URL when <see cref="EnvironmentVariable"/> is not set.</summary>
    public const string DefaultBaseUrl = "https://huggingface.co";

    /// <summary>The base URL without a trailing slash, e.g. <c>https://huggingface.co</c>.</summary>
    /// <exception cref="InvalidOperationException"><c>HF_ENDPOINT</c> is set but is not an absolute http(s) URL.</exception>
    public static string BaseUrl => Resolve(Environment.GetEnvironmentVariable(EnvironmentVariable));

    /// <summary>The Hub's model API root, <c>{base}/api/models</c>.</summary>
    public static string ModelsApiUrl => BaseUrl + "/api/models";

    /// <summary>The base URL a configured value names: <paramref name="configured"/> trimmed of a trailing slash, or the default.</summary>
    internal static string Resolve(string? configured)
    {
        if (string.IsNullOrWhiteSpace(configured))
            return DefaultBaseUrl;

        var value = configured.Trim();
        if (!Uri.TryCreate(value, UriKind.Absolute, out var uri) || (uri.Scheme != Uri.UriSchemeHttps && uri.Scheme != Uri.UriSchemeHttp))
        {
            throw new InvalidOperationException(
                $"{EnvironmentVariable} is set to '{value}', which is not an absolute http(s) URL. Set it to the Hub base URL " +
                $"(for example {DefaultBaseUrl}) or unset it.");
        }

        return value.TrimEnd('/');
    }

    /// <summary>The host name of the base URL, for a failure that names where the model was to come from.</summary>
    public static string Host => new Uri(BaseUrl).Host;
}
