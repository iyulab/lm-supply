namespace LMSupply.Captioner.Tests;

/// <summary>
/// Manual evaluation: captions every image in <c>LMSUPPLY_CAPTION_EVAL_DIR</c> with <c>default</c> and <c>quality</c>
/// side by side into <c>captions.txt</c> there. Skipped when the variable is unset.
/// </summary>
[Trait("Category", "LocalOnly")]
public sealed class Florence2EvalProbe
{
    [Fact]
    public async Task CaptionEvalDirectory()
    {
        var dir = Environment.GetEnvironmentVariable("LMSUPPLY_CAPTION_EVAL_DIR");
        Assert.SkipWhen(string.IsNullOrEmpty(dir), "LMSUPPLY_CAPTION_EVAL_DIR is not set.");

        var ct = TestContext.Current.CancellationToken;
        await using var vit = await LocalCaptioner.LoadAsync("fast", cancellationToken: ct);
        await using var florence = await LocalCaptioner.LoadAsync("quality", cancellationToken: ct);

        var lines = new List<string>();
        foreach (var image in Directory.GetFiles(dir!, "*.jpg").Order())
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var a = await vit.CaptionAsync(image, ct);
            var vitMs = sw.ElapsedMilliseconds;
            sw.Restart();
            var b = await florence.CaptionAsync(image, ct);
            lines.Add($"{Path.GetFileName(image)} | vit {vitMs} ms: {a.Caption} | florence {sw.ElapsedMilliseconds} ms: {b.Caption}");
        }

        await File.WriteAllLinesAsync(Path.Combine(dir!, "captions.txt"), lines, ct);
    }
}
