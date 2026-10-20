using System.Text;
using System.Text.Json;
using LMSupply.Json;

namespace LMSupply.Llama.Server;

public sealed partial class LlamaServerClient
{
    /// <summary>
    /// Counts the number of tokens in the given text using the server's tokenizer.
    /// </summary>
    private Task<int> CountTokensCoreAsync(
        string text,
        CancellationToken cancellationToken = default)
        => TokenizeCountAsync(text, addSpecial: false, cancellationToken);

    /// <summary>
    /// Counts the tokens of <paramref name="text"/> on the server. With <paramref name="addSpecial"/>, the tokens the
    /// model adds around a prompt (a BOS token, for models that use one) are counted too — as generation counts a prompt.
    /// </summary>
    private async Task<int> TokenizeCountAsync(string text, bool addSpecial, CancellationToken cancellationToken)
        => (await TokenizeCoreAsync(text, addSpecial, cancellationToken)).Count;

    /// <summary>
    /// The token ids the model's tokenizer gives <paramref name="text"/>, using the server's <c>/tokenize</c>. Text that
    /// spells a special token of the model (such as <c>&lt;think&gt;</c>) becomes that token, as in a prompt. With
    /// <paramref name="addSpecial"/>, the tokens the model adds around a whole prompt (a BOS token, for models that use
    /// one) are included; without it — the default — only the text's own tokens are, which is what a single token's id
    /// needs: the same text can be a different token with a leading space (<c>"B"</c> and <c>" B"</c>), and the ids differ
    /// between models.
    /// </summary>
    private async Task<IReadOnlyList<int>> TokenizeCoreAsync(string text, bool addSpecial, CancellationToken cancellationToken)
    {
        var request = new TokenizeRequest { Content = text, AddSpecial = addSpecial ? true : null };
        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await PostWithTimeoutAsync(
            $"{_baseUrl}/tokenize",
            content,
            cancellationToken);

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var result = JsonSerializer.Deserialize(responseJson, JsonOptions.TypeInfo<TokenizeResponse>());

        return result?.Tokens ?? [];
    }

    /// <summary>
    /// Counts the tokens of the prompt a chat request renders to: the messages through the model's chat template,
    /// with the request's tools, tool choice and thinking setting — what <c>/v1/chat/completions</c> would send the
    /// model. Uses the server's <c>/apply-template</c>, then <c>/tokenize</c>. Null when the server has no
    /// <c>/apply-template</c>.
    /// </summary>
    private async Task<int?> CountChatPromptTokensCoreAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options,
        CancellationToken cancellationToken = default)
    {
        var request = BuildChatRequest(messages, options ?? new ChatCompletionOptions(), stream: false);
        var json = JsonSerializer.Serialize(request, JsonOptions.TypeInfoOf(request));
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var response = await PostWithTimeoutAsync(
            $"{_baseUrl}/apply-template",
            content,
            cancellationToken);

        if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            return null;

        await EnsureSuccessOrThrowContextExceptionAsync(response, _maxContextLength, cancellationToken);

        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        var rendered = JsonSerializer.Deserialize(responseJson, JsonOptions.TypeInfo<ApplyTemplateResponse>());
        if (rendered?.Prompt is not { } prompt)
            return null;

        // Generation tokenizes the rendered prompt with the model's special tokens (a BOS, where the model uses one).
        return await TokenizeCountAsync(prompt, addSpecial: true, cancellationToken);
    }
}
