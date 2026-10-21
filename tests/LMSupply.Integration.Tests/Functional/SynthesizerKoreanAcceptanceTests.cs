using LMSupply.Exceptions;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// The built-in Korean phonemizer, judged by ear: plain text goes in, the catalog's Korean voice speaks it with no
/// phonemizer set, Whisper transcribes the audio, and the character error rate against the text is the measure (Korean
/// spacing varies too much between writers for a word error rate). A voice or the transcriber that is not in the local
/// model cache is skipped, not downloaded.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Synthesizer")]
public sealed class SynthesizerKoreanAcceptanceTests(ITestOutputHelper output)
{
    // Measured (Whisper "quality", CPU), character error rate over all 165 characters: 0.176 in each of 3 runs (prose
    // 0.171-0.178, figures 0.167-0.194). Runs differ because the voice samples noise. The errors that recur are a vowel
    // after a vowel heard as one (서울은 → 서른), ㅊ after a vowel heard as ㅈ (같이 → 갈지, 이천 → 이전) and 회의 heard
    // as other words. For comparison, the same symbols with ㅏ written a (the plain IPA letter) measured about 0.5 on the
    // first 12 sentences. The bound leaves room for about 1.7 times the measured rate.
    private const double MaxCharacterErrorRate = 0.30;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static readonly (string Set, string Text)[] Sentences =
    [
        ("prose", "오늘 날씨가 정말 좋습니다."),
        ("prose", "안녕하세요, 만나서 반갑습니다."),
        ("prose", "저는 한국어를 공부하고 있어요."),
        ("prose", "내일 아침에 회의가 있습니다."),
        ("prose", "이 책은 도서관에서 빌렸어요."),
        ("prose", "저녁에 같이 밥 먹을까요?"),
        ("prose", "서울은 사람이 많아요."),
        ("prose", "물 한 잔 주세요."),
        ("prose", "여기서 기차역까지 멀어요?"),
        ("prose", "주말에는 집에서 쉬었어요."),
        ("prose", "우리 내일 다시 만나요."),
        ("prose", "그 영화는 정말 재미있었어요."),
        ("numbers", "올해는 2026년입니다."),
        ("numbers", "이 노트북은 150만 원이에요."),
        ("numbers", "회의실은 3층에 있습니다."),
    ];

    [Fact]
    public async Task TextSpokenByTheKoreanVoice_WhisperHears()
    {
        await using var synthesizer = await LoadCachedAsync(
            () => LocalSynthesizer.LoadAsync("korean", new SynthesizerOptions { DisableAutoDownload = true }, cancellationToken: Ct));
        await using var transcriber = await LoadCachedAsync(
            () => LocalTranscriber.LoadAsync("quality", new TranscriberOptions { DisableAutoDownload = true }, cancellationToken: Ct));

        var totals = new Dictionary<string, (int Errors, int Characters)>();
        foreach (var (set, text) in Sentences)
        {
            var audio = await synthesizer.SynthesizeAsync(text, cancellationToken: Ct);
            audio.UnknownPhonemes.Should().BeEmpty("the built-in phonemizer writes only symbols the voice has");

            var heard = await transcriber.TranscribeAsync(audio.ToWavBytes(), new TranscribeOptions { Language = "ko" }, Ct);
            var said = Characters(text);
            var errors = EditDistance(said, Characters(heard.Text));
            output.WriteLine($"[{set}] {errors}/{said.Length} said «{text}» heard «{heard.Text.Trim()}» phonemes «{string.Join(" | ", audio.Phonemes)}»");

            var (e, c) = totals.GetValueOrDefault(set);
            totals[set] = (e + errors, c + said.Length);
        }

        foreach (var (set, (errors, characters)) in totals)
            output.WriteLine($"[{set}] character error rate {(double)errors / characters:F3} ({errors}/{characters})");

        var cer = (double)totals.Values.Sum(t => t.Errors) / totals.Values.Sum(t => t.Characters);
        output.WriteLine($"[all] character error rate {cer:F3}");
        Assert.True(cer <= MaxCharacterErrorRate, $"character error rate {cer:F3}, bound {MaxCharacterErrorRate}");
    }

    private static async Task<T> LoadCachedAsync<T>(Func<Task<T>> load)
    {
        try
        {
            return await load();
        }
        catch (ModelNotFoundException ex)
        {
            Assert.Skip($"Not in the local model cache: {ex.Message}");
            throw;
        }
    }

    // The letters of a text, with figures read out by a converter of the test's own (not the phonemizer's), so that «3»
    // and «삼» are the same on both sides; spaces and punctuation are dropped.
    private static string Characters(string text)
    {
        var result = new System.Text.StringBuilder();
        for (var i = 0; i < text.Length; i++)
        {
            if (char.IsAsciiDigit(text[i]))
            {
                var start = i;
                while (i + 1 < text.Length && (char.IsAsciiDigit(text[i + 1]) || text[i + 1] == ','))
                    i++;
                result.Append(SinoKorean(long.Parse(text[start..(i + 1)].Replace(",", ""), System.Globalization.CultureInfo.InvariantCulture)));
            }
            else if (char.IsLetter(text[i]))
            {
                result.Append(text[i]);
            }
        }

        return result.ToString();
    }

    private static string SinoKorean(long value)
    {
        const string digits = "영일이삼사오육칠팔구";
        if (value == 0)
            return "영";

        var result = new System.Text.StringBuilder();
        string[] myriads = ["", "만", "억", "조"];
        string[] places = ["", "십", "백", "천"];
        var groups = new List<int>();
        for (; value > 0; value /= 10000)
            groups.Add((int)(value % 10000));

        for (var g = groups.Count - 1; g >= 0; g--)
        {
            if (groups[g] == 0)
                continue;
            if (groups[g] != 1 || g != 1)
            {
                for (var p = 3; p >= 0; p--)
                {
                    var d = groups[g] / (int)Math.Pow(10, p) % 10;
                    if (d == 0) continue;
                    if (d > 1 || p == 0) result.Append(digits[d]);
                    result.Append(places[p]);
                }
            }
            result.Append(myriads[g]);
        }

        return result.ToString();
    }

    private static int EditDistance(string said, string heard)
    {
        var distance = new int[said.Length + 1, heard.Length + 1];
        for (var i = 0; i <= said.Length; i++) distance[i, 0] = i;
        for (var j = 0; j <= heard.Length; j++) distance[0, j] = j;

        for (var i = 1; i <= said.Length; i++)
        {
            for (var j = 1; j <= heard.Length; j++)
            {
                var substitution = distance[i - 1, j - 1] + (said[i - 1] == heard[j - 1] ? 0 : 1);
                distance[i, j] = Math.Min(substitution, Math.Min(distance[i - 1, j] + 1, distance[i, j - 1] + 1));
            }
        }

        return distance[said.Length, heard.Length];
    }
}
