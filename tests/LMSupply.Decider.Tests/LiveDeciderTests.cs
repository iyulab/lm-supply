using System.Globalization;
using System.Text;
using AwesomeAssertions;
using LMSupply.Decider.Internal;
using LMSupply.Llama.Server;

namespace LMSupply.Decider.Tests;

/// <summary>
/// The decider against real GGUF models in a real llama-server. A model file comes from an environment variable and the
/// test skips when it is unset or the file is absent; nothing is downloaded. Each test appends what it measured to the
/// file named by <c>LMSUPPLY_DECIDER_REPORT</c> when that is set, and to the test output.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
[Trait("Category", "GGUF")]
public sealed class LiveDeciderTests
{
    private const string NativeVar = "LMSUPPLY_DECIDER_NATIVE_MODEL";
    private const string QwenVar = "LMSUPPLY_DECIDER_QWEN_MODEL";
    private const string GemmaVar = "LMSUPPLY_DECIDER_GEMMA_MODEL";

    private sealed record ChoiceCase(string Language, string Note, string State, string Question, string[] Options, int Expected);

    private sealed record YesNoCase(string Language, string State, string Question, bool Expected);

    // Twenty choice cases, ten English and ten Korean: easy, sarcasm, negation, arithmetic, many options, two-option
    // words, and four where no option applies (Expected = -1).
    private static readonly ChoiceCase[] ChoiceCases =
    [
        new("en", "easy", "Customer message: My card was charged twice for the same order yesterday.",
            "What does the customer want?", ["refund a duplicate charge", "change delivery address", "cancel subscription", "report a lost card"], 0),
        new("en", "sarcasm", "Review: Oh great, the 'waterproof' watch stopped working after I washed my hands. Five stars for creativity.",
            "What is the sentiment?", ["positive", "negative", "neutral", "mixed"], 1),
        new("en", "negation", "Ticket: The invoice is correct, it is NOT a billing problem — the PDF just will not open on my laptop.",
            "Which team should handle this?", ["billing", "technical support", "sales", "legal"], 1),
        new("en", "arithmetic", "Order: 3 items at $12 each, a $5 coupon, free shipping over $30.",
            "What does the customer pay?", ["$31", "$36", "$41", "$26"], 0),
        new("en", "8 options", "Server alert: certificate for api.example.com expires in 2 days; renewal job failed with DNS timeout.",
            "Which category fits best?", ["billing", "security certificate", "disk space", "network outage", "database", "user access", "deployment", "other"], 1),
        new("en", "8 options", "Email: Please move my Thursday 3pm meeting with the design team to Friday morning if the room is free.",
            "What is the request?", ["cancel a meeting", "book a new meeting", "reschedule a meeting", "invite someone", "change a room", "send minutes", "request leave", "other"], 2),
        new("en", "two words", "Message: Can someone tell me whether the office is open on the public holiday next Monday?",
            "Is this a question or a complaint?", ["question", "complaint"], 0),
        new("en", "two words", "Message: Third time this month the elevator is broken and nobody has fixed it.",
            "Is the writer urgent or relaxed?", ["urgent", "relaxed"], 0),
        new("ko", "easy", "고객 메시지: 어제 같은 주문으로 카드가 두 번 결제됐어요. 빨리 환불해 주세요.",
            "고객이 원하는 것은?", ["중복 결제 환불", "배송지 변경", "구독 해지", "분실 카드 신고"], 0),
        new("ko", "sarcasm", "리뷰: 와, 방수라더니 손 씻다가 고장 났네요. 정말 대단한 제품입니다.",
            "리뷰의 감정은?", ["긍정", "부정", "중립", "복합"], 1),
        new("ko", "negation", "문의: 청구서는 맞습니다. 결제 문제가 아니라 PDF가 노트북에서 안 열려요.",
            "어느 팀이 처리해야 하나?", ["결제팀", "기술지원팀", "영업팀", "법무팀"], 1),
        new("ko", "arithmetic", "주문: 1만 2천 원짜리 3개, 5천 원 할인 쿠폰, 3만 원 이상 무료 배송.",
            "고객이 내는 금액은?", ["3만 1천 원", "3만 6천 원", "4만 1천 원", "2만 6천 원"], 0),
        new("ko", "8 options", "알림: api.example.com 인증서가 2일 뒤 만료됩니다. 갱신 작업이 DNS 시간 초과로 실패했습니다.",
            "가장 알맞은 분류는?", ["결제", "보안 인증서", "디스크 용량", "네트워크 장애", "데이터베이스", "사용자 권한", "배포", "기타"], 1),
        new("ko", "8 options", "메일: 목요일 오후 3시 디자인팀 회의를 금요일 오전으로 옮겨 주세요. 회의실이 비어 있으면요.",
            "요청은 무엇인가?", ["회의 취소", "새 회의 예약", "회의 일정 변경", "참석자 초대", "회의실 변경", "회의록 발송", "휴가 신청", "기타"], 2),
        new("ko", "two words", "메시지: 다음 주 월요일 공휴일에 사무실 문 여는지 아시는 분 있나요?",
            "질문인가 불만인가?", ["질문", "불만"], 0),
        new("ko", "two words", "메시지: 이번 달에만 세 번째로 엘리베이터가 고장인데 아무도 안 고칩니다.",
            "급한가 여유로운가?", ["급함", "여유"], 0),
        new("en", "none applies", "Message: What time does the cafeteria open tomorrow?",
            "Which team should handle this?", ["billing", "technical support", "sales", "legal"], -1),
        new("en", "none applies", "Review: Arrived on time and works exactly as described.",
            "What went wrong?", ["late delivery", "broken item", "wrong item", "missing parts"], -1),
        new("ko", "none applies", "메시지: 내일 구내식당 몇 시에 열어요?",
            "어느 팀이 처리해야 하나?", ["결제팀", "기술지원팀", "영업팀", "법무팀"], -1),
        new("ko", "none applies", "리뷰: 제때 왔고 설명대로 잘 됩니다.",
            "무엇이 잘못됐나?", ["배송 지연", "파손", "다른 상품", "부품 누락"], -1),
    ];

    private static readonly YesNoCase[] YesNoCases =
    [
        new("en", "Customer message: My card was charged twice for the same order yesterday.", "Is the customer reporting a problem?", true),
        new("en", "Customer message: Thanks, the replacement arrived and it works great.", "Is the customer reporting a problem?", false),
        new("en", "Message: Can someone tell me whether the office is open on the public holiday next Monday?", "Is the writer asking a question?", true),
        new("en", "Message: Third time this month the elevator is broken and nobody has fixed it.", "Is the writer asking a question?", false),
        new("en", "Review: Arrived on time and works exactly as described.", "Is the review positive?", true),
        new("en", "Review: Stopped working after two days and support never answered.", "Is the review positive?", false),
        new("en", "Alert: certificate for api.example.com expires in 2 days; renewal job failed.", "Does this need action within a week?", true),
        new("en", "Alert: nightly backup finished in 14 minutes with no errors.", "Does this need action within a week?", false),
        new("ko", "고객 메시지: 어제 같은 주문으로 카드가 두 번 결제됐어요.", "고객이 문제를 알리고 있는가?", true),
        new("ko", "고객 메시지: 교체품이 잘 도착했고 아주 잘 작동합니다. 감사합니다.", "고객이 문제를 알리고 있는가?", false),
        new("ko", "메시지: 다음 주 월요일 공휴일에 사무실 문 여는지 아시는 분 있나요?", "작성자가 질문하고 있는가?", true),
        new("ko", "메시지: 이번 달에만 세 번째로 엘리베이터가 고장인데 아무도 안 고칩니다.", "작성자가 질문하고 있는가?", false),
        new("ko", "리뷰: 제때 왔고 설명대로 잘 됩니다.", "리뷰가 긍정적인가?", true),
        new("ko", "리뷰: 이틀 만에 고장 났고 고객센터는 답이 없습니다.", "리뷰가 긍정적인가?", false),
        new("ko", "알림: api.example.com 인증서가 2일 뒤 만료되고 갱신 작업이 실패했습니다.", "일주일 안에 조치가 필요한가?", true),
        new("ko", "알림: 야간 백업이 오류 없이 14분 만에 끝났습니다.", "일주일 안에 조치가 필요한가?", false),
    ];

    private static string? ModelFile(string variable)
    {
        var path = Environment.GetEnvironmentVariable(variable);
        return !string.IsNullOrWhiteSpace(path) && File.Exists(path) ? path : null;
    }

    private static string RequireModel(string variable)
    {
        var path = ModelFile(variable);
        if (path is null)
            Assert.Skip($"{variable} is unset or does not name an existing file; this test never downloads.");
        return path;
    }

    private static Task<IDeciderModel> LoadAsync(string path) =>
        LocalDecider.LoadAsync(path, new DeciderOptions { DisableAutoDownload = true }, cancellationToken: TestContext.Current.CancellationToken);

    private static void Report(string line)
    {
        TestContext.Current.TestOutputHelper?.WriteLine(line);
        if (Environment.GetEnvironmentVariable("LMSUPPLY_DECIDER_REPORT") is { Length: > 0 } file)
            File.AppendAllText(file, line + Environment.NewLine, Encoding.UTF8);
    }

    private static string F(double v) => v.ToString("0.000", CultureInfo.InvariantCulture);

    private static async Task<(int Correct, int Total, int KoCorrect, int KoTotal, double MinMass)> RunChoiceCasesAsync(
        IDeciderModel decider, string label)
    {
        var ct = TestContext.Current.CancellationToken;
        int correct = 0, koCorrect = 0, koTotal = 0;
        var minMass = 1.0;
        foreach (var c in ChoiceCases)
        {
            var decision = await decider.DecideAsync(c.State, "q", DecisionQuestion.Choice(c.Question, c.Options, allowNone: true), ct);
            var expected = c.Expected < 0 ? null : c.Options[c.Expected];
            var ok = decision.Choice == expected;
            correct += ok ? 1 : 0;
            if (c.Language == "ko")
            {
                koTotal++;
                koCorrect += ok ? 1 : 0;
            }
            minMass = Math.Min(minMass, decision.LabelMass);
            Report($"{label} choice {c.Language} {c.Note,-12} {(ok ? "ok  " : "MISS")} pick={decision.Choice ?? "(none)"} conf={F(decision.Confidence)} mass={F(decision.LabelMass)}");
        }

        Report($"{label} choice total {correct}/{ChoiceCases.Length}  ko {koCorrect}/{koTotal}  en {correct - koCorrect}/{ChoiceCases.Length - koTotal}  min label mass {F(minMass)}");
        return (correct, ChoiceCases.Length, koCorrect, koTotal, minMass);
    }

    private static async Task RunYesNoCasesAsync(IDeciderModel decider, string label)
    {
        var ct = TestContext.Current.CancellationToken;
        int correct = 0, koCorrect = 0, koTotal = 0;
        var minMass = 1.0;
        foreach (var c in YesNoCases)
        {
            var decision = await decider.DecideAsync(c.State, "q", DecisionQuestion.YesNo(c.Question), ct);
            var ok = decision.IsTrue == c.Expected;
            correct += ok ? 1 : 0;
            if (c.Language == "ko")
            {
                koTotal++;
                koCorrect += ok ? 1 : 0;
            }
            minMass = Math.Min(minMass, decision.LabelMass);
            Report($"{label} yesno {c.Language} {(ok ? "ok  " : "MISS")} p(yes)={F(decision.Probability)} mass={F(decision.LabelMass)}");
        }

        Report($"{label} yesno total {correct}/{YesNoCases.Length}  ko {koCorrect}/{koTotal}  en {correct - koCorrect}/{YesNoCases.Length - koTotal}  min label mass {F(minMass)}");
        correct.Should().BeGreaterThanOrEqualTo(YesNoCases.Length - 2, "Yes/No labels read the model's answer on clear-cut cases");
    }

    [Fact]
    public async Task ANativeModel_AnswersEnglishAndKoreanQuestions_InOneRequest()
    {
        var path = RequireModel(NativeVar);
        var ct = TestContext.Current.CancellationToken;
        await using var decider = await LoadAsync(path);
        decider.Path.Should().Be(DecisionPath.Native);

        var english = await decider.DecideAsync(
            "Customer message: My card was charged twice for the same order yesterday.",
            new Dictionary<string, DecisionQuestion>
            {
                ["intent"] = DecisionQuestion.Choice("What does the customer want?", ["refund a duplicate charge", "change delivery address", "cancel subscription", "report a lost card"]),
                ["urgent"] = DecisionQuestion.YesNo("Is this message urgent?"),
                ["anger"] = DecisionQuestion.Score("How upset is the customer?", ["calm", "mildly annoyed", "annoyed", "angry", "furious"]),
            },
            ct);
        Report($"native en intent={english["intent"].Choice} conf={F(english["intent"].Confidence)} urgent={F(english["urgent"].Probability)} anger={F(english["anger"].Score)}");
        english["intent"].Choice.Should().Be("refund a duplicate charge");
        english["urgent"].IsTrue.Should().BeTrue();
        english["anger"].Score.Should().BeInRange(0, 4);
        english["intent"].Path.Should().Be(DecisionPath.Native);
        english["intent"].LabelMass.Should().Be(1);
        english["intent"].Calibrated.Should().BeFalse();

        var korean = await decider.DecideAsync(
            "고객 메시지: 어제 같은 주문으로 카드가 두 번 결제됐어요. 빨리 환불해 주세요.",
            new Dictionary<string, DecisionQuestion>
            {
                ["intent"] = DecisionQuestion.Choice("고객이 원하는 것은?", ["중복 결제 환불", "배송지 변경", "구독 해지", "분실 카드 신고"]),
                ["urgent"] = DecisionQuestion.YesNo("급한 메시지인가?"),
            },
            ct);
        Report($"native ko intent={korean["intent"].Choice} conf={F(korean["intent"].Confidence)} urgent={F(korean["urgent"].Probability)}");
        korean["intent"].Choice.Should().Be("중복 결제 환불");

        var none = await decider.DecideAsync(
            "Message: What time does the cafeteria open tomorrow?",
            "team",
            DecisionQuestion.Choice("Which team should handle this?", ["billing", "technical support", "sales", "legal"], allowNone: true),
            ct);
        Report($"native none-case pick={none.Choice ?? "(none)"} none={F(none.NoneProbability)}");
    }

    [Fact]
    public async Task AnInstructModel_Qwen_ReadsTheChoiceCases_AndYesNo()
    {
        var path = RequireModel(QwenVar);
        await using var decider = await LoadAsync(path);
        decider.Path.Should().Be(DecisionPath.Emulated);

        var result = await RunChoiceCasesAsync(decider, "qwen3.5-4b");
        await RunYesNoCasesAsync(decider, "qwen3.5-4b");

        result.Correct.Should().BeGreaterThanOrEqualTo(18, "the scripted run of these cases got 18 of 20");
        result.MinMass.Should().BeGreaterThan(0.5);
    }

    [Fact]
    public async Task AnInstructModel_Gemma_ReadsTheChoiceCases_AndYesNo()
    {
        var path = RequireModel(GemmaVar);
        await using var decider = await LoadAsync(path);
        decider.Path.Should().Be(DecisionPath.Emulated);

        var result = await RunChoiceCasesAsync(decider, "gemma-4-e4b");
        await RunYesNoCasesAsync(decider, "gemma-4-e4b");

        // The scripted run, with its frame text in Korean, got 20 of 20; the one case that differs is the Korean arithmetic item
        // (a decision that needs a calculation, which a one-token answer is not for) — see TheFrameLanguage_OverKoreanContent_IsMeasured.
        result.Correct.Should().BeGreaterThanOrEqualTo(19);
        result.MinMass.Should().BeGreaterThan(0.5);
    }

    [Theory]
    [InlineData(QwenVar)]
    [InlineData(GemmaVar)]
    public async Task QuestionsAboutOneState_ReuseThePromptPrefix(string variable)
    {
        var path = RequireModel(variable);
        var ct = TestContext.Current.CancellationToken;
        await using var decider = await LoadAsync(path);

        // A state long enough that evaluating it dwarfs a question.
        var state = new StringBuilder("Support thread, oldest message first:\n");
        for (var i = 1; i <= 24; i++)
            state.Append(CultureInfo.InvariantCulture, $"Message {i}: The customer wrote that order number {1000 + i} was charged twice, that the invoice shows two lines for the same item, and asked when the duplicate charge would be refunded to the card.\n");

        var result = await decider.DecideAsync(
            state.ToString(),
            new Dictionary<string, DecisionQuestion>
            {
                ["route"] = DecisionQuestion.Choice("Which team should handle this?", ["billing", "shipping", "technical"], allowNone: true),
                ["angry"] = DecisionQuestion.YesNo("Is the customer angry?"),
                ["urgency"] = DecisionQuestion.Score("How urgent is this?", ["can wait", "this week", "today", "right now"]),
            },
            ct);
        result["route"].Choice.Should().Be("billing");

        var timings = ((LlamaServerDeciderModel)decider).LastTimings;
        var line = string.Join(", ", timings.Select(t => $"prompt_n={t?.PromptN} cache_n={t?.CacheN}"));
        Report($"prefix {Path.GetFileNameWithoutExtension(path)}: {line}");

        timings.Should().HaveCount(3).And.OnlyContain(t => t != null && t.PromptN != null);
        var first = timings[0]!.PromptN!.Value;
        first.Should().BeGreaterThan(500);
        // A sliding-window model keeps the whole prefix (the later questions evaluate little more than their own text); a
        // hybrid recurrent model restores from a checkpoint a batch before the end of the prompt, so it re-evaluates about
        // one batch more. Either way a later question costs well under the first one, and the server reports tokens reused.
        foreach (var later in timings.Skip(1))
        {
            later!.PromptN!.Value.Should().BeLessThan(first / 2, "a later question starts after the state the first one evaluated");
            later.CacheN.Should().BeGreaterThan(0);
        }
    }

    /// <summary>
    /// Measurement, not a gate: the Korean cases with the English frame text the decider uses, and again with the frame
    /// text in Korean. Reports both counts; asserts only that the shipped (English) frame stays at or above the other.
    /// </summary>
    [Theory]
    [InlineData(QwenVar)]
    [InlineData(GemmaVar)]
    public async Task TheFrameLanguage_OverKoreanContent_IsMeasured(string variable)
    {
        var path = RequireModel(variable);
        var ct = TestContext.Current.CancellationToken;
        await using var decider = await LoadAsync(path);
        var model = (LlamaServerDeciderModel)decider;
        var client = await model.CurrentClientAsync(ct);
        var label = Path.GetFileNameWithoutExtension(path);

        int english = 0, korean = 0, total = 0;
        foreach (var c in ChoiceCases.Where(c => c.Language == "ko"))
        {
            var question = DecisionQuestion.Choice(c.Question, c.Options, allowNone: true);
            var expected = c.Expected < 0 ? null : c.Options[c.Expected];
            total++;

            english += (await decider.DecideAsync(c.State, "q", question, ct)).Choice == expected ? 1 : 0;

            var emulated = EmulatedPrompt.Build(c.State, question);
            var prompt = emulated.Prompt
                .Replace(EmulatedPrompt.NoneOfThese, "해당 없음", StringComparison.Ordinal)
                .Replace(EmulatedPrompt.AnswerWithNumber, "숫자 하나로만 답하세요.", StringComparison.Ordinal);
            var response = await client.GenerateChatWithToolsAsync(
                [ChatCompletionMessage.User(prompt)], LlamaServerDeciderModel.RequestOptions(emulated.TopLogprobs), ct);
            var slots = EmulatedPrompt.Read("q", response.Choices?.FirstOrDefault()?.Logprobs?.Content?.FirstOrDefault(), emulated.SlotLabels, out _);
            var pick = Array.IndexOf(slots, slots.Max());
            var koreanPick = pick < c.Options.Length ? c.Options[pick] : null;
            korean += koreanPick == expected ? 1 : 0;
            Report($"{label} frame {c.Note,-12} korean-frame-pick={koreanPick ?? "(none)"} expected={expected ?? "(none)"}");
        }

        Report($"{label} frame language over Korean content: english frame {english}/{total}, korean frame {korean}/{total}");
    }
}
