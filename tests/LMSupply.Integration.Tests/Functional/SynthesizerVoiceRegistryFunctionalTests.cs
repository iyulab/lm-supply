using AwesomeAssertions;
using LMSupply.Synthesizer;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// The voices the registry changed in 0.88.0 load and produce audio: the new default (LJ Speech) and the Korean voice, which
/// used to name an x_low build that was never published. Loading only — whether the audio is speech is
/// <see cref="SynthesizerIntelligibilityFacts"/>'s question.
/// </summary>
[Trait("Category", "Integration")]
public sealed class SynthesizerVoiceRegistryFunctionalTests
{
    [Theory]
    [InlineData("default", "Hello, this is the default voice.")]
    [InlineData("korean", "안녕하세요. 반갑습니다.")]
    public async Task Voice_LoadsAndProducesAudio(string alias, string text)
    {
        await using var model = await LocalSynthesizer.LoadAsync(alias, cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.SynthesizeAsync(text, cancellationToken: TestContext.Current.CancellationToken);

        result.AudioSamples.Should().NotBeEmpty();
        result.ToWavBytes().Length.Should().BeGreaterThan(44, "a WAV longer than its header");
    }
}
