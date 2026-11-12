using LMSupply.Captioner;
using LMSupply.Detector;
using LMSupply.ImageGenerator;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Three scenes made once by the image generator, from prompts whose answer is known.
/// </summary>
public sealed class GeneratedScenes : IAsyncLifetime
{
    public byte[] Apple { get; private set; } = [];

    public byte[] Bus { get; private set; } = [];

    public byte[] Portrait { get; private set; } = [];

    public async ValueTask InitializeAsync()
    {
        await using var generator = await LocalImageGenerator.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);
        var options = new ImageGenerationOptions { Steps = 4, Seed = 42, GuidanceScale = 8.5f };

        Apple = (await generator.GenerateAsync("a red apple on a wooden table", options, TestContext.Current.CancellationToken)).ImageData;
        Bus = (await generator.GenerateAsync("a yellow school bus on a city street", options, TestContext.Current.CancellationToken)).ImageData;
        Portrait = (await generator.GenerateAsync("a portrait photo of a woman standing in front of a plain wall", options, TestContext.Current.CancellationToken)).ImageData;
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

/// <summary>
/// Known-answer checks across the vision domains. A test that only asks for a non-empty, well-formed output cannot tell a
/// model that works from one that returns noise (an image generator and two segmentation models shipped that way): these
/// give each domain a picture whose content is known and ask it what it sees. Every fact needs the models in the local
/// cache, so they run at the release gate with the other local-only facts.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Axis", "KnownAnswer")]
public class KnownAnswerScenarioTests(GeneratedScenes scenes) : IClassFixture<GeneratedScenes>
{
    [Fact]
    public async Task Detector_FindsTheBusAndTheApple()
    {
        await using var detector = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var onBus = await detector.DetectAsync(scenes.Bus, TestContext.Current.CancellationToken);
        var onApple = await detector.DetectAsync(scenes.Apple, TestContext.Current.CancellationToken);

        // Observed: a bus at 97 % and an apple at 97 %.
        onBus.Should().Contain(d => d.Label == "bus" && d.Confidence > 0.5f);
        onApple.Should().Contain(d => d.Label == "apple" && d.Confidence > 0.5f);
    }

    [Fact]
    public async Task FaceDetector_FindsTheFaceInAPortrait()
    {
        await using var detector = await LocalDetector.LoadAsync("face", cancellationToken: TestContext.Current.CancellationToken);

        var found = await detector.DetectAsync(scenes.Portrait, TestContext.Current.CancellationToken);

        // Observed: one face at 93 %.
        found.Should().Contain(d => d.Confidence > 0.5f);
    }

    [Fact]
    public async Task Captioner_NamesWhatThePictureShows()
    {
        await using var captioner = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var onBus = await captioner.CaptionAsync(scenes.Bus, TestContext.Current.CancellationToken);
        var onApple = await captioner.CaptionAsync(scenes.Apple, TestContext.Current.CancellationToken);

        // Observed: "a yellow bus is parked on the side of the road" and "a apple with a red apple on it".
        onBus.Caption.Should().Contain("bus");
        onApple.Caption.Should().Contain("apple");
    }
}
