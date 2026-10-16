using AwesomeAssertions;
using LMSupply.Console.Host.Endpoints;
using LMSupply.Console.Host.Models.OpenAI;
using Xunit;

namespace LMSupply.Integration.Tests;

/// <summary>
/// The console host's <c>POST /v1/audio/speech</c> picks the synthesizer model from the request. An OpenAI client
/// always sends a model of its own (<c>tts-1</c>) and one of OpenAI's voice names, so a voice name it knows picks the
/// model; a client that names an LMSupply synthesizer model sends <c>model</c> alone, and that model is used —
/// not the default voice's, as when <c>voice</c> silently defaulted to <c>alloy</c>.
/// </summary>
public class SpeechModelResolutionTests
{
    [Theory]
    [InlineData("lessac", null, "lessac")]
    [InlineData("british", null, "british")]
    [InlineData("tts-1", "alloy", "default")]
    [InlineData("tts-1-hd", "NOVA", "default")]
    [InlineData("lessac", "lessac", "lessac")]
    [InlineData("default", null, "default")]
    [InlineData("", null, "default")]
    public void TheModel_IsTheVoicesWhenAnOpenAIVoiceIsSent_AndTheRequestsOwnOtherwise(string model, string? voice, string expected)
    {
        var request = new SpeechRequest { Model = model, Voice = voice, Input = "Hello." };

        SynthesizeEndpoints.ResolveModelId(request).Should().Be(expected);
    }

    [Fact]
    public void ARequestWithoutAVoice_HasNone()
    {
        new SpeechRequest { Input = "Hello." }.Voice.Should().BeNull("a default voice would override the model the request names");
    }
}
