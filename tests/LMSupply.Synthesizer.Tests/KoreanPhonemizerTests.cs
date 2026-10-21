using AwesomeAssertions;
using LMSupply.Synthesizer.Core;
using LMSupply.Synthesizer.Core.Korean;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// The built-in Korean phonemizer: a word becomes the syllables it is said as (each rule of the standard pronunciation it
/// applies, with the textbook example), text becomes words (figures and Latin letters included), and the result is
/// written in the Korean voice's own symbols. The voice is the catalog's own config (Fixtures).
/// </summary>
public class KoreanPhonemizerTests
{
    private static string Said(string written)
    {
        var word = Syllable.Of(written).ToList();
        KoreanPronunciation.Apply(word);
        return string.Concat(word);
    }

    private static string Read(string text) =>
        string.Join(" | ", KoreanText.Read(text).Select(sentence =>
            string.Join(' ', sentence.Select(token => token.IsPause ? token.Pause.ToString() : string.Concat(token.Word!)))));

    private static KoreanPhonemizer Phonemizer() =>
        (KoreanPhonemizer)OnnxSynthesizerModel.ResolvePhonemizer(
            callers: null, VoiceFixture.Catalog(SynthesizerModels.KoKr), SynthesizerModels.KoKr.AliasName);

    // ── pronunciation ───────────────────────────────────────────────────────────────────────────────────

    [Theory]
    // A final consonant goes over to a following vowel; of a cluster, the second one.
    [InlineData("한국어를", "한구거를")]
    [InlineData("밖에", "바께")]
    [InlineData("닭이", "달기")]
    [InlineData("앉아", "안자")]
    // ...and ㅅ after a stop is said tense.
    [InlineData("값이", "갑씨")]
    [InlineData("없어요", "업써요")]
    // ㄷ and ㅌ before 이 are palatalized.
    [InlineData("굳이", "구지")]
    [InlineData("같이", "가치")]
    [InlineData("꽃이", "꼬치")]
    [InlineData("밭에", "바테")]
    // ㅎ aspirates a lax stop next to it, on either side; before ㅅ it tenses it, before ㄴ it is ㄴ, before a vowel it is not said.
    [InlineData("좋고", "조코")]
    [InlineData("축하", "추카")]
    [InlineData("입학", "이팍")]
    [InlineData("굳히다", "구치다")]
    [InlineData("앉히다", "안치다")]
    [InlineData("밝히다", "발키다")]
    [InlineData("좋습니다", "조씀니다")]
    [InlineData("놓는", "논는")]
    [InlineData("많아요", "마나요")]
    [InlineData("싫어", "시러")]
    // The seven final sounds; a cluster is reduced to one of its consonants.
    [InlineData("읽다", "익따")]
    [InlineData("맑게", "말께")]
    [InlineData("밟다", "밥따")]
    [InlineData("넓게", "널께")]
    [InlineData("앉고", "안꼬")]
    // A stop tenses the lax consonant after it, and becomes a nasal before a nasal.
    [InlineData("있습니다", "읻씀니다")]
    [InlineData("국밥", "국빱")]
    [InlineData("국물", "궁물")]
    [InlineData("받는", "반는")]
    // ㄹ after another consonant is ㄴ; ㄴ next to ㄹ is ㄹ.
    [InlineData("종로", "종노")]
    [InlineData("국립", "궁닙")]
    [InlineData("신라", "실라")]
    [InlineData("설날", "설랄")]
    // Vowels said otherwise than written.
    [InlineData("회의", "회이")]
    [InlineData("희망", "히망")]
    [InlineData("의사", "의사")]
    [InlineData("시계", "시게")]
    [InlineData("예절", "예절")]
    [InlineData("다쳐서", "다처서")]
    public void AWord_IsSaidByTheStandardPronunciation(string written, string said) =>
        Said(written).Should().Be(said);

    // ── reading text ────────────────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData("2026년", "이천이십육년")]
    [InlineData("10000원", "만원")]
    [InlineData("15000", "만오천")]
    [InlineData("100000000", "일억")]
    [InlineData("1,234,567", "백이십삼만사천오백육십칠")]
    [InlineData("3.14", "삼점일사")]
    [InlineData("-5", "마이너스오")]
    [InlineData("50%", "오십퍼센트")]
    [InlineData("010", "영일영")]
    [InlineData("0", "영")]
    [InlineData("AI 모델", "에이아이 모델")]
    public void FiguresAndLatinLetters_AreReadInKorean(string text, string read) =>
        Read(text).Should().Be(read);

    [Fact]
    public void Text_IsSplitIntoSentences_WithTheirPauses_AndWhatCannotBeSaidSeparatesWords()
    {
        Read("안녕하세요, 반갑습니다! 이건 漢字예요…\n다음 줄").Should()
            .Be("안녕하세요 , 반갑습니다 ! | 이건 예요 . | 다음 줄");
        Read("「인용」 — 그리고 😀 끝?!").Should().Be("인용 그리고 끝 ?");
        Read("  ...  ").Should().BeEmpty();
    }

    // ── writing ─────────────────────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData("오늘 날씨가 정말 좋습니다.", "onɯl nɐlsːiɡɐ tɕʌŋmɐl tɕosːɯmnidɐ.")]
    [InlineData("저는 한국어를 공부하고 있어요.", "tɕʌnɯn hɐnɡuɡʌɾɯl koŋbuhɐɡo isːʌjo.")]
    // A tap between vowels, l after ㄹ; a lax consonant voiceless at a word's start and voiced after a vowel or a nasal.
    [InlineData("우리 달리기", "uɾi tɐlliɡi")]
    public void Text_IsWrittenInTheVoicesSymbols(string text, string phonemes) =>
        Phonemizer().Phonemize(text).Should().Equal(phonemes);

    [Fact]
    public void EachSentence_IsItsOwnString()
    {
        Phonemizer().Phonemize("네. 아니요?").Should().Equal("ne.", "ɐnijo?");
    }

    [Fact]
    public void EverySymbolWritten_IsOneTheVoiceHas()
    {
        var voice = VoiceFixture.Catalog(SynthesizerModels.KoKr);
        var phonemes = Phonemizer().Phonemize(
            "꿈, 빵, 떡, 쌀, 짜다; 칼, 탈, 팔, 차, 하늘: 의자 웨이터 왜 워낙 위 얘기 예요 유리 여자 요리 야구!");

        phonemes.Should().NotBeEmpty();
        phonemes.SelectMany(s => s).Should().OnlyContain(c => voice.Encoder.Has(c));
    }

    [Fact]
    public void AVoiceWithoutTheSymbols_IsRefused_NamingWhatIsMissing()
    {
        var voice = VoiceFixture.Minimal("\"espeak\": {\"voice\": \"ko\"}", catalog: SynthesizerModels.KoKr);
        var act = () => OnnxSynthesizerModel.ResolvePhonemizer(callers: null, voice, "korean");

        var message = act.Should().Throw<NotSupportedException>().Which.Message;
        message.Should().Contain("'korean'").And.Contain("Korean phonemes").And.Contain("U+0250");
        message.Should().MatchRegex(@"^[\x20-\x7E]+$", "operational text is ASCII");
    }
}
