using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using AwesomeAssertions;

namespace LMSupply.Decider.Tests;

/// <summary>
/// Reads a multiple-choice knowledge set (four options, one right) through the decider and reports the share answered right,
/// per category and in all, with the time. The set is a JSON Lines file the caller supplies in
/// <c>LMSUPPLY_DECIDER_KNOWLEDGE_SET</c> — one object per line with <c>question</c>, <c>A</c>–<c>D</c>, <c>answer</c> (1–4) and
/// <c>Category</c>; the file is not part of this repository (a published benchmark keeps its own licence). It says which of two
/// models knows more of what the set asks; it is not a measure of how well a model writes. The model comes from
/// <c>LMSUPPLY_DECIDER_QWEN_MODEL</c>, the variable the instruct-model tests read, and the test never downloads.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
[Trait("Category", "GGUF")]
public sealed class LiveKnowledgeSetTests
{
    private static readonly string[] OptionKeys = ["A", "B", "C", "D"];

    private sealed record Item(string Question, string[] Options, int Answer, string Category);

    private static void Report(string line)
    {
        TestContext.Current.TestOutputHelper?.WriteLine(line);
        if (Environment.GetEnvironmentVariable("LMSUPPLY_DECIDER_REPORT") is { Length: > 0 } file)
            File.AppendAllText(file, line + Environment.NewLine, Encoding.UTF8);
    }

    [Fact]
    public async Task AnInstructModel_ReadsAMultipleChoiceKnowledgeSet_ReportedNotGated()
    {
        var setPath = Environment.GetEnvironmentVariable("LMSUPPLY_DECIDER_KNOWLEDGE_SET");
        var modelPath = Environment.GetEnvironmentVariable("LMSUPPLY_DECIDER_QWEN_MODEL");
        if (string.IsNullOrWhiteSpace(setPath) || !File.Exists(setPath) || string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            Assert.Skip("LMSUPPLY_DECIDER_KNOWLEDGE_SET or LMSUPPLY_DECIDER_QWEN_MODEL is unset or does not name an existing file; this test never downloads.");

        var items = File.ReadLines(setPath).Where(l => l.Length > 0).Select(l =>
        {
            using var doc = JsonDocument.Parse(l);
            var r = doc.RootElement;
            return new Item(r.GetProperty("question").GetString()!,
                OptionKeys.Select(k => r.GetProperty(k).GetString()!).ToArray(),
                r.GetProperty("answer").GetInt32() - 1, r.GetProperty("Category").GetString()!);
        }).ToList();

        var ct = TestContext.Current.CancellationToken;
        var label = "knowledge " + Path.GetFileName(modelPath);
        await using var decider = await LocalDecider.LoadAsync(modelPath, new DeciderOptions { DisableAutoDownload = true }, cancellationToken: ct);

        var right = new Dictionary<string, (int Right, int Total)>();
        int refused = 0, correct = 0;
        var clock = Stopwatch.StartNew();
        foreach (var item in items)
        {
            var (r, t) = right.GetValueOrDefault(item.Category);
            try
            {
                var decision = await decider.DecideAsync(item.Question, "q", DecisionQuestion.Choice("다음 중 올바른 답은?", item.Options), ct);
                var ok = decision.Choice == item.Options[item.Answer];
                correct += ok ? 1 : 0;
                right[item.Category] = (r + (ok ? 1 : 0), t + 1);
            }
            catch (Exception e) when (e is ArgumentException or LMSupply.Exceptions.ContextLengthExceededException)
            {
                // Duplicate options or a question longer than the model was loaded to take: counted apart, not as wrong.
                refused++;
            }
        }

        foreach (var (category, (r, t)) in right.OrderBy(k => k.Key))
            Report($"{label} {category,-18} {r}/{t}");
        var answered = items.Count - refused;
        Report($"{label} total {correct}/{answered} ({(100.0 * correct / Math.Max(1, answered)).ToString("0.0", CultureInfo.InvariantCulture)}%) refused={refused} time {clock.ElapsedMilliseconds} ms");
        answered.Should().BeGreaterThan(0);
    }
}
