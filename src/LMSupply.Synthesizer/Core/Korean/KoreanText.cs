using System.Text;

namespace LMSupply.Synthesizer.Core.Korean;

/// <summary>A word to say, as its written syllables, or a pause, as the mark that makes it.</summary>
internal readonly record struct KoreanToken(List<Syllable>? Word, char Pause)
{
    public bool IsPause => Word is null;
}

/// <summary>
/// Reads Korean text into sentences of words and pauses. Hangul is read as written; figures are read as Sino-Korean
/// numbers (일, 이, 삼 … with 십, 백, 천, 만, 억, 조), with a decimal part, a minus and a percent sign; Latin letters
/// are read by their Korean names (AI → 에이아이). Anything else — Hanja, symbols, emoji — separates words and is not said.
/// </summary>
/// <remarks>
/// A number is always read with the Sino-Korean numerals, also where the native ones are said (세 시, 두 개): which
/// numeral a counter takes is the counter's, not the figure's. A word in Latin letters is spelled, not pronounced as
/// English.
/// </remarks>
internal static class KoreanText
{
    // Past this many digits a number is read digit by digit: nobody names the magnitude.
    private const int MaxCardinalDigits = 16;

    private const string Digits = "영일이삼사오육칠팔구";
    private static readonly string[] Places = ["", "십", "백", "천"];
    private static readonly string[] Myriads = ["", "만", "억", "조"];

    private static readonly string[] LetterNames =
    [
        "에이", "비", "씨", "디", "이", "에프", "지", "에이치", "아이", "제이", "케이", "엘", "엠",
        "엔", "오", "피", "큐", "알", "에스", "티", "유", "브이", "더블유", "엑스", "와이", "제트",
    ];

    private const string SentenceEnds = ".?!…。？！";
    private const string ClauseMarks = ",;:、，；：";

    /// <summary>The sentences of <paramref name="text"/>, each a list of words and pauses; empty sentences are dropped.</summary>
    public static List<List<KoreanToken>> Read(string text)
    {
        var sentences = new List<List<KoreanToken>>();
        var sentence = new List<KoreanToken>();
        var word = new StringBuilder();
        text = text.Normalize(NormalizationForm.FormC);

        for (var i = 0; i < text.Length; i++)
        {
            var c = text[i];
            if (Syllable.IsSyllable(c))
            {
                word.Append(c);
            }
            else if (char.IsAsciiDigit(c) || (c == '-' && word.Length == 0 && i + 1 < text.Length && char.IsAsciiDigit(text[i + 1])))
            {
                i = ReadNumber(text, i, word) - 1;
            }
            else if (char.IsAsciiLetter(c))
            {
                word.Append(LetterNames[char.ToUpperInvariant(c) - 'A']);
            }
            else if (c == '%')
            {
                word.Append("퍼센트");
            }
            else if (SentenceEnds.Contains(c) || ClauseMarks.Contains(c))
            {
                EndWord(word, sentence);
                // A run of marks makes one pause, the first one's.
                if (sentence.Count > 0 && !sentence[^1].IsPause)
                    sentence.Add(new KoreanToken(null, Ascii(c)));
                if (SentenceEnds.Contains(c))
                    EndSentence(ref sentence, sentences);
            }
            else
            {
                EndWord(word, sentence);
                if (c is '\n' or '\r')
                    EndSentence(ref sentence, sentences);
            }
        }

        EndWord(word, sentence);
        EndSentence(ref sentence, sentences);
        return sentences;
    }

    private static char Ascii(char mark) => mark switch
    {
        '…' or '。' => '.',
        '？' => '?',
        '！' => '!',
        '、' or '，' => ',',
        '；' => ';',
        '：' => ':',
        _ => mark,
    };

    private static void EndWord(StringBuilder word, List<KoreanToken> sentence)
    {
        if (word.Length == 0)
            return;

        sentence.Add(new KoreanToken([.. Syllable.Of(word.ToString())], '\0'));
        word.Clear();
    }

    private static void EndSentence(ref List<KoreanToken> sentence, List<List<KoreanToken>> sentences)
    {
        if (sentence.Exists(token => !token.IsPause))
            sentences.Add(sentence);
        sentence = [];
    }

    // Reads the number at text[start] (a figure, or a minus before one) into word; returns where it ends. Commas between
    // groups of three digits are part of it, a point followed by a digit starts its decimal part.
    private static int ReadNumber(string text, int start, StringBuilder word)
    {
        var i = start;
        if (text[i] == '-')
        {
            word.Append("마이너스");
            i++;
        }

        var whole = new StringBuilder();
        for (; i < text.Length; i++)
        {
            if (char.IsAsciiDigit(text[i]))
                whole.Append(text[i]);
            else if (text[i] == ',' && whole.Length > 0 && IsGroupOfThree(text, i + 1))
                continue;
            else
                break;
        }

        AppendWhole(whole.ToString(), word);

        if (i + 1 < text.Length && text[i] == '.' && char.IsAsciiDigit(text[i + 1]))
        {
            word.Append('점');
            for (i++; i < text.Length && char.IsAsciiDigit(text[i]); i++)
                word.Append(Digits[text[i] - '0']);
        }

        return i;
    }

    private static bool IsGroupOfThree(string text, int at) =>
        at + 3 <= text.Length
        && char.IsAsciiDigit(text[at]) && char.IsAsciiDigit(text[at + 1]) && char.IsAsciiDigit(text[at + 2])
        && (at + 3 == text.Length || !char.IsAsciiDigit(text[at + 3]));

    private static void AppendWhole(string digits, StringBuilder word)
    {
        // A leading zero (a code, a phone number) or a number too long to name is read digit by digit.
        if ((digits.Length > 1 && digits[0] == '0') || digits.Length > MaxCardinalDigits)
        {
            foreach (var digit in digits)
                word.Append(Digits[digit - '0']);
            return;
        }

        var value = long.Parse(digits, System.Globalization.CultureInfo.InvariantCulture);
        if (value == 0)
        {
            word.Append(Digits[0]);
            return;
        }

        // Groups of four digits, highest first: 1234 5678 is 천이백삼십사만 오천육백칠십팔.
        for (var myriad = Myriads.Length - 1; myriad >= 0; myriad--)
        {
            var group = (int)(value / Pow10(4 * myriad) % 10000);
            if (group == 0)
                continue;

            // 만 alone, not 일만; 일억 and 일조 keep their 일.
            if (!(group == 1 && myriad == 1))
                AppendBelowMyriad(group, word);
            word.Append(Myriads[myriad]);
        }
    }

    private static void AppendBelowMyriad(int value, StringBuilder word)
    {
        for (var place = 3; place >= 0; place--)
        {
            var digit = value / (int)Pow10(place) % 10;
            if (digit == 0)
                continue;

            // 십, 백, 천 alone, not 일십; a one is said only as the last digit. A last 육 after a final consonant takes the ㄴ
            // the standard pronunciation inserts there (십육 is said 심뉵), written so the rules nasalize what comes before.
            if (digit == 6 && place == 0 && word.Length > 0 && Syllable.IsSyllable(word[^1]) && Syllable.Of(word[^1]).Coda != Syllable.None)
                word.Append('뉵');
            else if (digit > 1 || place == 0)
                word.Append(Digits[digit]);
            word.Append(Places[place]);
        }
    }

    private static long Pow10(int exponent)
    {
        var result = 1L;
        for (var i = 0; i < exponent; i++)
            result *= 10;
        return result;
    }
}
