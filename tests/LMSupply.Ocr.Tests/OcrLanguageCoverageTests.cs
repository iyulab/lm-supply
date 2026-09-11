using System.Diagnostics;
using AwesomeAssertions;
using LMSupply.Ocr.Models;

namespace LMSupply.Ocr.Tests;

/// <summary>
/// Which recognizer a language code reaches, and what happens when it reaches none.
///
/// <para>
/// The language lists here come from the recognizers' own character dictionaries, not from a guess:
/// the Chinese model's dictionary holds every hiragana and katakana, and the Latin model's holds the
/// letters that set Turkish, Hungarian, Slovak and the rest apart from English. Before they were
/// listed, those codes fell through to the English recognizer — whose dictionary has none of those
/// letters — without saying so. A caller following the OS locale got the wrong model and no signal.
/// </para>
/// </summary>
public class OcrLanguageCoverageTests
{
    [Theory]
    [InlineData("ja")]
    [InlineData("ja-JP")]
    public void Japanese_ReachesTheChineseRecognizer_WhoseDictionaryHoldsKana(string code)
    {
        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code).Subfolder.Should().Be("languages/chinese");
    }

    [Theory]
    [InlineData("tr")]
    [InlineData("tr-TR")]
    [InlineData("hu")]
    [InlineData("sk")]
    [InlineData("hr")]
    [InlineData("bs")]
    [InlineData("sl")]
    [InlineData("is")]
    [InlineData("et")]
    [InlineData("lt")]
    [InlineData("sq")]
    [InlineData("ga")]
    [InlineData("id")]
    [InlineData("ms")]
    [InlineData("sw")]
    [InlineData("tl")]
    [InlineData("fil-PH")]
    [InlineData("nb-NO")]
    [InlineData("nn")]
    public void LatinScriptLanguages_ReachTheLatinRecognizer(string code)
    {
        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code).Subfolder.Should().Be("languages/latin");
    }

    [Fact]
    public void AnUnsupportedLanguage_FallsBackToEnglish_AndSaysSo()
    {
        using var capture = new TraceCapture();

        var model = OcrRecognitionModelRegistry.Default.ResolveForLanguage("xx-unsupported-test");

        model.Subfolder.Should().Be("languages/english");
        capture.Messages.Should().Contain(m => m.Contains("xx-unsupported-test") && m.Contains("English"),
            "a recognizer that cannot read the script produces confident garbage, so the substitution " +
            "must be visible to whoever is looking at the output");
    }

    [Theory]
    [InlineData("en")]
    [InlineData("ko-KR")]
    [InlineData("tr")]
    public void ASupportedLanguage_ResolvesWithoutAWarning(string code)
    {
        using var capture = new TraceCapture();

        OcrRecognitionModelRegistry.Default.ResolveForLanguage(code);

        capture.Messages.Should().NotContain(m => m.Contains($"'{code}'"));
    }

    [Fact]
    public void SupportedLanguages_ListTheNewlyMappedCodes()
    {
        OcrRecognitionModelRegistry.Default.GetSupportedLanguages()
            .Should().Contain(["ja", "tr", "hu", "nb"]);
    }

    /// <summary>Collects trace warnings raised on this thread's watch; other tests' traces are filtered by content.</summary>
    private sealed class TraceCapture : TraceListener
    {
        private readonly List<string> _messages = [];

        public TraceCapture() => Trace.Listeners.Add(this);

        public IReadOnlyList<string> Messages
        {
            get
            {
                lock (_messages)
                {
                    return [.. _messages];
                }
            }
        }

        public override void Write(string? message) => Record(message);

        public override void WriteLine(string? message) => Record(message);

        public override void TraceEvent(TraceEventCache? eventCache, string source, TraceEventType eventType, int id, string? message)
            => Record(message);

        public override void TraceEvent(TraceEventCache? eventCache, string source, TraceEventType eventType, int id, string? format, params object?[]? args)
            => Record(args is { Length: > 0 } && format is not null ? string.Format(System.Globalization.CultureInfo.InvariantCulture, format, args) : format);

        private void Record(string? message)
        {
            if (message is null)
            {
                return;
            }

            lock (_messages)
            {
                _messages.Add(message);
            }
        }

        protected override void Dispose(bool disposing)
        {
            Trace.Listeners.Remove(this);
            base.Dispose(disposing);
        }
    }
}
