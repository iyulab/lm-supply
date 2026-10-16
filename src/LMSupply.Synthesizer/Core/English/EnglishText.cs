using System.Globalization;
using System.Text;

namespace LMSupply.Synthesizer.Core.English;

/// <summary>What a piece of text turns into before it is pronounced.</summary>
internal enum SpokenKind
{
    /// <summary>A word in lower-case letters, apostrophes included.</summary>
    Word,

    /// <summary>Letters to be named one by one (an initialism), in lower case.</summary>
    Letters,

    /// <summary>A break in the speech, written as the punctuation mark that makes it: one of <c>, ; : . ? !</c>.</summary>
    Pause
}

internal readonly record struct Spoken(SpokenKind Kind, string Text);

/// <summary>
/// Reads English text into sentences of words and pauses: what is written as digits, a symbol or an abbreviation becomes
/// the words that say it, and what cannot be said (quotation marks, brackets, characters of other scripts) is left out.
/// </summary>
/// <remarks>
/// Handled: whole numbers and decimals, thousands separators, ordinals written with their ending (<c>21st</c>), a
/// four-digit number from 1100 to 2099 (read in pairs, as a year is), times of day written <c>h:mm</c>, amounts in dollars,
/// percentages, a leading minus, <c>&amp; @ + =</c>, and the abbreviations of <see cref="Abbreviations"/>. Not handled,
/// and read as the digits and marks they are made of: dates, fractions, ranges, units, other currencies, Roman numerals.
/// </remarks>
internal static class EnglishText
{
    // An abbreviation written with a full stop, the words that say it, and whether that full stop can also end the
    // sentence (a title's cannot: a name follows it).
    private static readonly Dictionary<string, (string[] Words, bool CanEndSentence)> Abbreviations = new(StringComparer.Ordinal)
    {
        ["mr"] = (["mister"], false),
        ["mrs"] = (["missus"], false),
        ["ms"] = (["miz"], false),
        ["dr"] = (["doctor"], false),
        ["prof"] = (["professor"], false),
        ["st"] = (["saint"], false),
        ["vs"] = (["versus"], false),
        ["no"] = (["number"], false),
        ["jr"] = (["junior"], true),
        ["sr"] = (["senior"], true),
        ["inc"] = (["incorporated"], true),
        ["ltd"] = (["limited"], true),
        ["co"] = (["company"], true),
        ["etc"] = (["et", "cetera"], true),
    };

    // Abbreviations written as single letters with full stops between them. Any other such run is named letter by letter.
    private static readonly Dictionary<string, string[]> DottedAbbreviations = new(StringComparer.Ordinal)
    {
        ["eg"] = ["for", "example"],
        ["ie"] = ["that", "is"],
    };

    // Titles are also written without the full stop; they are recognised by their capital.
    private static readonly HashSet<string> TitlesWithoutFullStop = new(StringComparer.Ordinal) { "Mr", "Mrs", "Ms", "Dr" };

    private static readonly Dictionary<char, string> SpokenSymbols = new()
    {
        ['&'] = "and", ['@'] = "at", ['+'] = "plus", ['='] = "equals", ['%'] = "percent",
    };

    private static readonly HashSet<string> Magnitudes = new(StringComparer.Ordinal) { "thousand", "million", "billion", "trillion" };

    /// <summary>Splits <paramref name="text"/> into sentences. A sentence without a word is not returned.</summary>
    public static List<List<Spoken>> Read(string text)
    {
        var reader = new Reader(Fold(text));
        return reader.Read();
    }

    /// <summary>
    /// Brings typographic variants to the plain characters the reader knows, and letters with a diacritic to their base
    /// letter (<c>café</c> is looked up as <c>cafe</c>).
    /// </summary>
    private static string Fold(string text)
    {
        var folded = new StringBuilder(text.Length);
        foreach (var c in text)
        {
            switch (c)
            {
                case '‘' or '’' or 'ʼ' or '`':
                    folded.Append('\'');
                    break;
                case '“' or '”' or '«' or '»':
                    folded.Append('"');
                    break;
                case '–' or '—' or '―':
                    folded.Append(" - ");
                    break;
                case '…':
                    folded.Append("...");
                    break;
                case ' ' or ' ' or ' ' or '\t':
                    folded.Append(' ');
                    break;
                case '\r':
                    break;
                case > '\u007F' when char.IsLetter(c):
                    // The base letter of a decomposable letter; a letter of another script has none and is dropped.
                    var decomposed = c.ToString().Normalize(NormalizationForm.FormD);
                    if (char.IsAsciiLetter(decomposed[0]))
                        folded.Append(decomposed[0]);
                    else
                        folded.Append(c);
                    break;
                default:
                    folded.Append(c);
                    break;
            }
        }

        return folded.ToString();
    }

    private ref struct Reader(string text)
    {
        private readonly ReadOnlySpan<char> _text = text;
        private readonly List<List<Spoken>> _sentences = [];
        private List<Spoken> _sentence = [];

        // Per sentence: whether a lower-case letter was written, and which entries were written with full stops ("U.S.").
        private readonly HashSet<int> _dotted = [];
        private bool _sawLowerCase;
        private int _at;

        public List<List<Spoken>> Read()
        {
            while (_at < _text.Length)
            {
                var c = _text[_at];
                if (char.IsAsciiLetter(c))
                    ReadWord();
                else if (char.IsAsciiDigit(c) || (c == '$' && IsDigitAt(_at + 1)) || (c == '-' && IsDigitAt(_at + 1) && !FollowsWordCharacter()))
                    ReadNumber();
                else
                    ReadMark(c);
            }

            EndSentence();
            return _sentences;
        }

        private readonly char Peek(int offset = 0) => _at + offset < _text.Length ? _text[_at + offset] : '\0';

        private readonly bool IsDigitAt(int index) => index < _text.Length && char.IsAsciiDigit(_text[index]);

        private readonly bool IsLetterAt(int index) => index >= 0 && index < _text.Length && char.IsAsciiLetter(_text[index]);

        private readonly bool FollowsWordCharacter() => _at > 0 && char.IsAsciiLetterOrDigit(_text[_at - 1]);

        private void Say(string word) => _sentence.Add(new Spoken(SpokenKind.Word, word));

        private void Say(IEnumerable<string> words)
        {
            foreach (var word in words)
                Say(word);
        }

        // A pause needs something said before it, and two in a row are one: the first, unless the second ends the sentence.
        private void Pause(char mark)
        {
            if (_sentence.Count == 0)
                return;

            if (_sentence[^1].Kind == SpokenKind.Pause)
            {
                if (mark is '.' or '?' or '!')
                    _sentence[^1] = new Spoken(SpokenKind.Pause, mark.ToString());
                return;
            }

            _sentence.Add(new Spoken(SpokenKind.Pause, mark.ToString()));
        }

        private void EndSentence()
        {
            // A sentence written in capitals throughout is words, not a row of initialisms.
            if (!_sawLowerCase && _sentence.Count(s => s.Kind != SpokenKind.Pause) >= 2)
            {
                for (var i = 0; i < _sentence.Count; i++)
                {
                    if (_sentence[i].Kind == SpokenKind.Letters && !_dotted.Contains(i))
                        _sentence[i] = _sentence[i] with { Kind = SpokenKind.Word };
                }
            }

            _sawLowerCase = false;
            _dotted.Clear();

            // A pause that nothing follows, other than the one that ends the sentence, is not a pause.
            if (_sentence.Count > 0 && _sentence[^1] is { Kind: SpokenKind.Pause, Text: "," or ";" or ":" })
                _sentence.RemoveAt(_sentence.Count - 1);

            if (_sentence.Exists(s => s.Kind != SpokenKind.Pause))
                _sentences.Add(_sentence);
            _sentence = [];
        }

        private void ReadWord()
        {
            var start = _at;
            while (_at < _text.Length
                && (char.IsAsciiLetter(_text[_at]) || (_text[_at] == '\'' && IsLetterAt(_at - 1) && IsLetterAt(_at + 1))))
            {
                _at++;
            }

            var written = _text[start.._at];
            _sawLowerCase |= written.IndexOfAnyInRange('a', 'z') >= 0;

            // Single letters with full stops between them: "e.g.", "U.S.", "a.m.".
            if (written.Length == 1 && Peek() == '.' && IsLetterAt(_at + 1) && !IsLetterAt(_at + 2))
            {
                ReadDottedLetters(written[0]);
                return;
            }

            var lower = written.ToString().ToLowerInvariant();
            if (Peek() == '.' && !IsLetterAt(_at + 1) && Abbreviations.TryGetValue(lower, out var abbreviation)
                && IsAbbreviationHere(lower))
            {
                _at++;
                Say(abbreviation.Words);
                if (abbreviation.CanEndSentence && AtSentenceStart())
                {
                    Pause('.');
                    EndSentence();
                }

                return;
            }

            if (TitlesWithoutFullStop.Contains(written.ToString()))
            {
                Say(Abbreviations[lower].Words);
                return;
            }

            // Capitals throughout mark an initialism; whether it is one the dictionary reads as a word is decided there.
            var allCapitals = written.Length >= 2 && !written.ContainsAnyExceptInRange('A', 'Z');
            _sentence.Add(new Spoken(allCapitals ? SpokenKind.Letters : SpokenKind.Word, lower));
        }

        // "St." is «saint» before a name and "No." is «number» before a figure; elsewhere they are the words they spell.
        private readonly bool IsAbbreviationHere(string lower)
        {
            var next = _at + 1;
            while (next < _text.Length && _text[next] == ' ')
                next++;

            return lower switch
            {
                "st" => next < _text.Length && char.IsAsciiLetterUpper(_text[next]),
                "no" => IsDigitAt(next),
                _ => true
            };
        }

        // After a full stop: is what follows the start of a sentence (or nothing at all)?
        private readonly bool AtSentenceStart()
        {
            var next = _at;
            while (next < _text.Length && char.IsWhiteSpace(_text[next]))
                next++;
            return next >= _text.Length || char.IsAsciiLetterUpper(_text[next]) || _text[next] == '"';
        }

        private void ReadDottedLetters(char first)
        {
            var letters = new StringBuilder().Append(char.ToLowerInvariant(first));
            while (Peek() == '.' && IsLetterAt(_at + 1) && !IsLetterAt(_at + 2))
            {
                letters.Append(char.ToLowerInvariant(_text[_at + 1]));
                _at += 2;
            }

            var endsWithFullStop = Peek() == '.';
            if (endsWithFullStop)
                _at++;

            var run = letters.ToString();
            if (DottedAbbreviations.TryGetValue(run, out var words))
                Say(words);
            else
            {
                _dotted.Add(_sentence.Count);
                _sentence.Add(new Spoken(SpokenKind.Letters, run));
            }

            // The last full stop belongs to the abbreviation; it ends the sentence only where the text ends too.
            if (endsWithFullStop && _text[_at..].Trim().IsEmpty)
                Pause('.');
        }

        private void ReadNumber()
        {
            var negative = _text[_at] == '-';
            if (negative)
                _at++;

            var dollars = _text[_at] == '$';
            if (dollars)
                _at++;

            // The whole part: digits, or groups of three separated by commas.
            var start = _at;
            while (IsDigitAt(_at))
                _at++;
            var whole = _text[start.._at].ToString();
            if (whole.Length <= 3 && whole[0] != '0')
            {
                var grouped = new StringBuilder(whole);
                var at = _at;
                while (at + 3 < _text.Length && _text[at] == ',' && IsDigitAt(at + 1) && IsDigitAt(at + 2) && IsDigitAt(at + 3) && !IsDigitAt(at + 4))
                {
                    grouped.Append(_text.Slice(at + 1, 3));
                    at += 4;
                }

                if (at != _at)
                {
                    whole = grouped.ToString();
                    _at = at;
                    start = -1; // written with separators: not a year
                }
            }

            if (!negative && !dollars && start >= 0 && TryReadTime(whole))
                return;

            var fraction = ReadOnlySpan<char>.Empty;
            if (Peek() == '.' && IsDigitAt(_at + 1))
            {
                var fractionStart = ++_at;
                while (IsDigitAt(_at))
                    _at++;
                fraction = _text[fractionStart.._at];
            }

            var words = new List<string>();
            if (negative)
                words.Add("minus");

            if (dollars)
            {
                AppendDollars(whole, fraction, words);
            }
            else
            {
                var plain = start >= 0 && fraction.IsEmpty && !negative;
                AppendWhole(whole, words, inPairs: plain && Peek() != '%');
                AppendFraction(fraction, words);

                if (fraction.IsEmpty && IsOrdinalEnding())
                {
                    NumberWords.MakeOrdinal(words);
                    _at += 2;
                }
                else if (fraction.IsEmpty && IsPluralEnding(out var length))
                {
                    // "the 1990s", "the 1990's": the plural of the last word.
                    var last = words[^1];
                    words[^1] = last.EndsWith('y') ? string.Concat(last.AsSpan(0, last.Length - 1), "ies") : last + "s";
                    _at += length;
                }
            }

            if (Peek() == '%')
            {
                words.Add("percent");
                _at++;
            }

            Say(words);

            // A full stop between two digits that the number did not take is said: "3.5.1".
            if (Peek() == '.' && IsDigitAt(_at + 1))
            {
                Say("point");
                _at++;
            }
        }

        // A time of day: "4:30" «four thirty», "9:05" «nine oh five», "7:00" «seven o'clock».
        private bool TryReadTime(string hours)
        {
            if (hours.Length > 2 || Peek() != ':' || !IsDigitAt(_at + 1) || !IsDigitAt(_at + 2) || IsDigitAt(_at + 3))
                return false;

            var hour = int.Parse(hours, CultureInfo.InvariantCulture);
            var minute = int.Parse(_text.Slice(_at + 1, 2), CultureInfo.InvariantCulture);
            if (hour > 24 || minute > 59)
                return false;

            _at += 3;
            var words = new List<string>();
            NumberWords.AppendCardinal(hour, words);
            if (minute == 0)
            {
                words.Add("o'clock");
            }
            else
            {
                if (minute < 10)
                    words.Add("oh");
                NumberWords.AppendCardinal(minute, words);
            }

            Say(words);
            return true;
        }

        private readonly bool IsOrdinalEnding()
        {
            if (_at + 1 >= _text.Length || IsLetterAt(_at + 2))
                return false;

            return _text.Slice(_at, 2) is "st" or "nd" or "rd" or "th";
        }

        private readonly bool IsPluralEnding(out int length)
        {
            length = Peek() == '\'' ? 2 : 1;
            return Peek(length - 1) == 's' && !IsLetterAt(_at + length);
        }

        private static void AppendWhole(string digits, List<string> words, bool inPairs = false)
        {
            if (digits.Length > NumberWords.MaxCardinalDigits || (digits.Length > 1 && digits[0] == '0'))
            {
                NumberWords.AppendDigits(digits, words);
                return;
            }

            var value = long.Parse(digits, CultureInfo.InvariantCulture);
            if (inPairs && digits.Length == 4 && value is >= 1100 and <= 2099)
                NumberWords.AppendInPairs((int)value, words);
            else
                NumberWords.AppendCardinal(value, words);
        }

        private static void AppendFraction(ReadOnlySpan<char> fraction, List<string> words)
        {
            if (fraction.IsEmpty)
                return;

            words.Add("point");
            NumberWords.AppendDigits(fraction, words);
        }

        // "$5" five dollars, "$1" one dollar, "$2.50" two dollars and fifty cents, "$3.5 million" three point five
        // million dollars.
        private void AppendDollars(string whole, ReadOnlySpan<char> fraction, List<string> words)
        {
            var magnitude = ReadMagnitude();
            if (magnitude is null && fraction.Length == 2)
            {
                AppendWhole(whole, words);
                words.Add(whole == "1" ? "dollar" : "dollars");

                var cents = int.Parse(fraction, CultureInfo.InvariantCulture);
                if (cents > 0)
                {
                    words.Add("and");
                    NumberWords.AppendCardinal(cents, words);
                    words.Add(cents == 1 ? "cent" : "cents");
                }

                return;
            }

            AppendWhole(whole, words);
            AppendFraction(fraction, words);
            if (magnitude is not null)
                words.Add(magnitude);
            words.Add(whole == "1" && fraction.IsEmpty && magnitude is null ? "dollar" : "dollars");
        }

        // The magnitude word after an amount ("$5 million"), consumed so that the currency is said after it.
        private string? ReadMagnitude()
        {
            var next = _at;
            while (next < _text.Length && _text[next] == ' ')
                next++;
            var end = next;
            while (IsLetterAt(end))
                end++;

            var word = _text[next..end].ToString().ToLowerInvariant();
            if (next == _at || !Magnitudes.Contains(word))
                return null;

            _at = end;
            return word;
        }

        private void ReadMark(char c)
        {
            _at++;
            switch (c)
            {
                case ',' or ';' or ':':
                    Pause(c);
                    break;

                case '.' or '?' or '!':
                    // A full stop inside a name ("example.com") is said, not obeyed.
                    if (c == '.' && IsLetterAt(_at - 2) && IsLetterAt(_at))
                    {
                        Say("dot");
                        break;
                    }

                    // A run of them is one mark: a question mark if there is one, else an exclamation mark, else a full stop.
                    var mark = c;
                    for (; Peek() is '.' or '?' or '!'; _at++)
                    {
                        var next = Peek();
                        if (next == '?' || (next == '!' && mark == '.'))
                            mark = next;
                    }

                    Pause(mark);
                    EndSentence();
                    break;

                case '\n':
                    // A blank line ends the sentence; a single line break is a space.
                    if (Peek() == '\n')
                        EndSentence();
                    break;

                case '-':
                    // A hyphen joins two words and is not heard; a dash set off by spaces is a pause.
                    if (!(IsLetterAt(_at - 2) || IsDigitAt(_at - 2)) || !(IsLetterAt(_at) || IsDigitAt(_at)))
                        Pause(',');
                    break;

                case '(' or ')' or '[' or ']' or '{' or '}':
                    Pause(',');
                    break;

                default:
                    if (SpokenSymbols.TryGetValue(c, out var word))
                        Say(word);
                    break;
            }
        }
    }
}
