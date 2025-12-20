using LMSupply.Generator;
using LMSupply.Generator.Models;
using LMSupply.Console.Host.Infrastructure;
using LMSupply.Console.Host.Models.OpenAI;
using LMSupply.Console.Host.Services;

namespace LMSupply.Console.Host.Endpoints;

public static class ChatEndpoints
{
    public static void MapChatEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/v1/chat")
            .WithTags("Chat")
            .WithOpenApi();

        // POST /v1/chat/completions - OpenAI compatible
        group.MapPost("/completions", async (ChatCompletionRequest request, HttpContext context, ModelManagerService manager) =>
        {
            var ct = context.RequestAborted;

            // Handle streaming separately to avoid IResult after response started
            if (request.Stream)
            {
                try
                {
                    var generator = await manager.GetGeneratorAsync(request.Model, ct);
                    var messages = request.Messages.Select(m =>
                        new ChatMessage(Enum.Parse<ChatRole>(m.Role, ignoreCase: true), m.Content));
                    var options = new GenerationOptions
                    {
                        MaxTokens = request.MaxTokens ?? 2048,
                        Temperature = request.Temperature ?? 0.7f,
                        TopP = request.TopP ?? 0.9f,
                        StopSequences = request.Stop?.ToList()
                    };

                    var tokens = generator.GenerateChatAsync(messages, options, ct);
                    await SseHelper.StreamChatCompletionAsync(context, generator.ModelId, tokens, ct);
                }
                catch (Exception ex)
                {
                    // If response hasn't started, we can still write error
                    if (!context.Response.HasStarted)
                    {
                        context.Response.StatusCode = 500;
                        await context.Response.WriteAsJsonAsync(new { error = ex.Message }, ct);
                    }
                }
                return;
            }

            // Non-streaming response
            try
            {
                var generator = await manager.GetGeneratorAsync(request.Model, ct);
                var messages = request.Messages.Select(m =>
                    new ChatMessage(Enum.Parse<ChatRole>(m.Role, ignoreCase: true), m.Content));
                var options = new GenerationOptions
                {
                    MaxTokens = request.MaxTokens ?? 2048,
                    Temperature = request.Temperature ?? 0.7f,
                    TopP = request.TopP ?? 0.9f,
                    StopSequences = request.Stop?.ToList()
                };

                var result = await generator.GenerateChatWithUsageAsync(messages, options, ct);
                var id = ApiHelper.GenerateId("chatcmpl");

                await context.Response.WriteAsJsonAsync(new ChatCompletionResponse
                {
                    Id = id,
                    Model = generator.ModelId,
                    Choices =
                    [
                        new ChatCompletionChoice
                        {
                            Index = 0,
                            Message = new ChatCompletionMessage
                            {
                                Role = "assistant",
                                Content = result.Content
                            },
                            FinishReason = "stop"
                        }
                    ],
                    Usage = new Usage
                    {
                        PromptTokens = result.Usage.PromptTokens,
                        CompletionTokens = result.Usage.CompletionTokens,
                        TotalTokens = result.Usage.TotalTokens
                    }
                }, ct);
            }
            catch (Exception ex)
            {
                context.Response.StatusCode = 500;
                await context.Response.WriteAsJsonAsync(new { error = ex.Message }, ct);
            }
        })
        .WithName("CreateChatCompletion")
        .WithSummary("Create a chat completion (OpenAI compatible)")
        .WithDescription("Creates a model response for the given chat conversation. Compatible with OpenAI's chat completions API.")
        .Produces<ChatCompletionResponse>()
        .Produces<ErrorResponse>(400)
        .Produces<ErrorResponse>(500);
    }
}
