using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// <see cref="TranscribeOptions.Temperature"/> used to divide the logits and then take the argmax —
/// a positive scale never changes which logit is largest, so every value decoded the same thing. It now
/// does what Whisper's reference decoder does with it: sampling above 0, and a fallback that decodes a
/// window again at rising temperature when its result looks like a failure (too repetitive, or too
/// unsure), which is also what <see cref="TranscribeOptions.CompressionRatioThreshold"/> was documented
/// to trigger.
/// </summary>
public class WhisperDecoderTemperatureFallbackTests
{
    private const int StartOfTranscript = 50258;
    private const int Transcribe = 50359;
    private const int NoTimestamps = 50363;
    private const int VocabSize = 51865;
    private const int TextA = 440;
    private const int TextB = 264;

    private static readonly int[] Prompt = [StartOfTranscript, Transcribe, NoTimestamps];

    private static readonly DecodingResult Good = Result(compression: 1.4f, avgLogProb: -0.3f);
    private static readonly DecodingResult Repetitive = Result(compression: 3.5f, avgLogProb: -0.2f);
    private static readonly DecodingResult Unsure = Result(compression: 1.4f, avgLogProb: -1.6f);

    [Fact]
    public void Defaults_AreWhispers()
    {
        var options = new TranscribeOptions();

        options.Temperature.Should().Be(0f);
        options.TemperatureIncrementOnFallback.Should().Be(0.2f);
        options.LogProbThreshold.Should().Be(-1.0f);
        options.CompressionRatioThreshold.Should().Be(2.4f);
    }

    [Fact]
    public async Task AWindowThatPasses_IsDecodedOnce_Greedily()
    {
        var (result, temperatures) = await RunAsync(new TranscribeOptions(), Good);

        temperatures.Should().Equal(0f);
        result.Temperature.Should().Be(0f);
    }

    [Fact]
    public async Task ARepetitiveWindow_IsDecodedAgainAtRisingTemperature_UntilOnePasses()
    {
        var (result, temperatures) = await RunAsync(new TranscribeOptions(), Repetitive, Repetitive, Good);

        temperatures.Should().Equal([0f, 0.2f, 0.4f], (a, b) => Math.Abs(a - b) < 1e-5f);
        result.Temperature.Should().BeApproximately(0.4f, 1e-5f);
    }

    [Fact]
    public async Task AnUnsureWindow_IsDecodedAgain()
    {
        var (result, temperatures) = await RunAsync(new TranscribeOptions(), Unsure, Good);

        temperatures.Should().HaveCount(2);
        result.AvgLogProb.Should().Be(Good.AvgLogProb, "the second attempt passed and is the one kept");
        result.Temperature.Should().BeApproximately(0.2f, 1e-5f);
    }

    [Fact]
    public async Task WhenEveryAttemptFails_TheLastIsKept_AtTemperatureOne()
    {
        var (result, temperatures) = await RunAsync(new TranscribeOptions(), Repetitive);

        temperatures.Should().HaveCount(6, "0.0, 0.2, 0.4, 0.6, 0.8 and 1.0");
        temperatures[^1].Should().BeApproximately(1f, 1e-5f);
        result.Temperature.Should().BeApproximately(1f, 1e-5f);
    }

    [Fact]
    public async Task AStartingTemperature_IsTheFirstAttempt()
    {
        var (_, temperatures) = await RunAsync(new TranscribeOptions { Temperature = 0.5f }, Repetitive);

        temperatures.Should().Equal([0.5f, 0.7f, 0.9f], (a, b) => Math.Abs(a - b) < 1e-5f);
    }

    [Fact]
    public async Task AnIncrementOfZero_DecodesEveryWindowOnce()
    {
        var (_, temperatures) = await RunAsync(new TranscribeOptions { TemperatureIncrementOnFallback = 0f }, Repetitive);

        temperatures.Should().Equal(0f);
    }

    [Fact]
    public async Task Silence_IsNotDecodedAgain()
    {
        var silence = Result(compression: 1.0f, avgLogProb: -2.0f, noSpeech: 0.9f);

        var (_, temperatures) = await RunAsync(new TranscribeOptions(), silence);

        temperatures.Should().Equal(0f);
    }

    [Fact]
    public async Task WithoutALogProbThreshold_OnlyCompressionDecides()
    {
        var (_, temperatures) = await RunAsync(new TranscribeOptions { LogProbThreshold = null }, Unsure);

        temperatures.Should().Equal(0f);
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.5f)]
    [InlineData(float.NaN)]
    public void ATemperatureOutsideZeroToOne_IsRefused(float temperature)
    {
        var act = () => WhisperDecoder.ValidateSamplingOptions(new TranscribeOptions { Temperature = temperature });

        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*Temperature*");
    }

    [Fact]
    public void ANegativeIncrement_IsRefused()
    {
        var act = () => WhisperDecoder.ValidateSamplingOptions(new TranscribeOptions { TemperatureIncrementOnFallback = -0.2f });

        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*TemperatureIncrementOnFallback*");
    }

    [Fact]
    public void AtTemperatureZero_TheMostLikelyTokenIsAlwaysTaken()
    {
        var decoder = WhisperDecoder.CreateForTesting(WhisperTokenizer.CreateDefault(), new Random(7));

        for (var i = 0; i < 50; i++)
        {
            decoder.SelectNextToken(TwoCandidates(), [.. Prompt], Prompt, options: null, temperature: 0f)
                .Should().Be(TextA);
        }
    }

    [Fact]
    public void AboveZero_TheRunnerUpIsSometimesDrawn_AndLowerTemperatureConcentrates()
    {
        var decoder = WhisperDecoder.CreateForTesting(WhisperTokenizer.CreateDefault(), new Random(7));

        var hot = Draws(decoder, temperature: 1.0f);
        var cold = Draws(decoder, temperature: 0.1f);

        // TextA 1.0 vs TextB 0.5: at T=1 TextB has probability ~0.38, at T=0.1 ~0.007.
        hot.Should().OnlyContain(t => t == TextA || t == TextB);
        hot.Count(t => t == TextB).Should().BeInRange(120, 260);
        cold.Count(t => t == TextB).Should().BeLessThan(20);
    }

    [Fact]
    public void ASuppressedToken_IsNeverDrawn()
    {
        var decoder = WhisperDecoder.CreateForTesting(WhisperTokenizer.CreateDefault(), new Random(7));

        // Three identical tokens in a row trip the hallucination guard, which removes that token.
        for (var i = 0; i < 200; i++)
        {
            var logits = TwoCandidates();
            decoder.SelectNextToken(logits, [.. Prompt, TextB, TextA, TextA, TextA], Prompt, options: null, temperature: 1.0f)
                .Should().Be(TextB);
        }
    }

    private static List<int> Draws(WhisperDecoder decoder, float temperature)
    {
        var draws = new List<int>();
        for (var i = 0; i < 500; i++)
        {
            draws.Add(decoder.SelectNextToken(TwoCandidates(), [.. Prompt], Prompt, options: null, temperature));
        }

        return draws;
    }

    private static float[] TwoCandidates()
    {
        var logits = new float[VocabSize];
        Array.Fill(logits, -1000f);
        logits[TextA] = 1.0f;
        logits[TextB] = 0.5f;
        return logits;
    }

    private static DecodingResult Result(float compression, float avgLogProb, float noSpeech = 0.05f) => new()
    {
        Text = "text",
        Language = "en",
        Segments = [],
        CompressionRatio = compression,
        AvgLogProb = avgLogProb,
        NoSpeechProb = noSpeech,
    };

    // Runs the fallback loop with a decode that returns the scripted results in order (repeating the last).
    private static async Task<(DecodingResult Result, List<float> Temperatures)> RunAsync(
        TranscribeOptions options, params DecodingResult[] script)
    {
        var temperatures = new List<float>();
        var result = await WhisperDecoder.DecodeWithFallbackAsync(
            (temperature, _) =>
            {
                var scripted = script[Math.Min(temperatures.Count, script.Length - 1)];
                temperatures.Add(temperature);
                return Task.FromResult(new DecodingResult
                {
                    Text = scripted.Text,
                    Language = scripted.Language,
                    Segments = scripted.Segments,
                    CompressionRatio = scripted.CompressionRatio,
                    AvgLogProb = scripted.AvgLogProb,
                    NoSpeechProb = scripted.NoSpeechProb,
                    Temperature = temperature,
                });
            },
            options,
            TestContext.Current.CancellationToken);

        return (result, temperatures);
    }
}
