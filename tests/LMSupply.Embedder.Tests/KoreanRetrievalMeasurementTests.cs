using System.Diagnostics;
using System.Globalization;
using System.Text;
using AwesomeAssertions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// Reads one embedding model on a small Korean retrieval set: twenty passages on distinct everyday topics and thirty queries
/// (twenty paraphrases, ten colloquial or short ones), each with one right passage. It reports recall at 1, the mean
/// reciprocal rank and the time, per model, and asserts nothing about them: it is the measurement a default is
/// chosen on, run beside another model with the same set. The model comes from <c>LMSUPPLY_EMBEDDER_CANDIDATE</c> (an alias
/// that is already in the cache, or the path of a .gguf file); a model that wants a prefix on the query or the passage has it in
/// <c>LMSUPPLY_EMBEDDER_QUERY_PREFIX</c> and <c>LMSUPPLY_EMBEDDER_PASSAGE_PREFIX</c>. The test skips when the model is unset, and it never
/// downloads. Each run appends its lines to the file named by <c>LMSUPPLY_EMBEDDER_REPORT</c> when that is set.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
public sealed class KoreanRetrievalMeasurementTests
{
    private static readonly string[] Passages =
    [
        "연차휴가는 입사 1년이 지나면 15일이 주어지며, 사용하려면 최소 사흘 전에 팀장에게 신청해야 한다.",
        "비밀번호를 잊었다면 로그인 화면의 '재설정' 링크를 눌러 등록된 이메일로 받은 인증 코드를 입력하면 새 비밀번호를 만들 수 있다.",
        "출장비 정산은 귀환 후 열흘 안에 영수증 사진과 함께 경비 시스템에 올려야 하며, 늦으면 다음 달 급여에 반영된다.",
        "사외에서 사내망에 접속하려면 VPN 프로그램을 설치하고 일회용 인증 앱의 숫자 여섯 자리를 함께 입력해야 한다.",
        "서버가 응답하지 않으면 먼저 상태 페이지를 확인하고, 십오 분이 지나도 복구되지 않으면 당직 담당자에게 전화로 알린다.",
        "회원이 탈퇴하면 결제 기록은 법에 따라 5년간 보관하고, 그 밖의 개인정보는 즉시 파기한다.",
        "회의실은 사내 달력에서 시간대를 골라 예약하며, 시작 후 십 분 동안 아무도 오지 않으면 예약이 자동으로 취소된다.",
        "주문한 상품이 약속한 날짜보다 일주일 넘게 늦어지면 고객센터에 알려 전액 환불을 요청할 수 있다.",
        "구독은 설정 화면의 '구독 관리'에서 언제든 해지할 수 있고, 이미 낸 달의 요금은 돌려주지 않지만 남은 기간 동안은 계속 쓸 수 있다.",
        "카드 결제가 두 번 된 경우 영수증 번호를 알려 주면 중복 청구분을 사흘 안에 취소해 준다.",
        "퇴사하는 직원은 마지막 근무일에 노트북과 출입증을 총무팀에 반납해야 하고, 반납 확인서를 받아야 퇴직 절차가 끝난다.",
        "웹사이트 보안 인증서는 만료 삼십 일 전부터 갱신할 수 있으며, 자동 갱신이 실패하면 도메인 확인 기록을 먼저 점검한다.",
        "시스템 정기 점검은 매달 둘째 주 일요일 새벽 두 시부터 네 시까지이며, 이 시간에는 서비스가 잠시 중단된다.",
        "삭제한 파일은 삼십 일 안에 백업 보관함에서 복구할 수 있고, 그 뒤에는 영구 삭제되어 되돌릴 수 없다.",
        "새 시스템의 접근 권한이 필요하면 신청서에 사유를 적어 소속 부서장의 승인을 받은 뒤 보안팀에 제출한다.",
        "사무용품은 매주 수요일까지 신청하면 금요일에 부서별로 한꺼번에 배달되며, 개별 구매는 허용되지 않는다.",
        "재택근무는 일주일에 최대 이틀까지 가능하고, 전날까지 팀 일정표에 표시하면 별도의 승인 없이 쓸 수 있다.",
        "사외 교육 수강료는 연간 백만 원까지 회사가 지원하며, 수료증을 제출하면 다음 달에 입금된다.",
        "직원은 해마다 한 번 지정 병원에서 무료로 건강검진을 받을 수 있고, 배우자는 비용의 절반을 낸다.",
        "주차를 하려면 차량 번호를 총무팀에 등록해야 하며, 등록하지 않은 차량은 지하 주차장 출입이 막힌다.",
    ];

    // (query, index of the right passage, kind)
    private static readonly (string Query, int Right, string Kind)[] Queries =
    [
        ("연차를 쓰려면 언제까지 누구에게 말해야 하나요?", 0, "paraphrase"),
        ("비밀번호가 기억나지 않을 때는 어떻게 하나요?", 1, "paraphrase"),
        ("출장 다녀온 뒤 비용은 언제까지 청구해야 하나요?", 2, "paraphrase"),
        ("집에서 회사 네트워크에 들어가려면 무엇이 필요한가요?", 3, "paraphrase"),
        ("서비스가 먹통일 때 누구에게 연락해야 하나요?", 4, "paraphrase"),
        ("탈퇴하면 내 결제 내역도 지워지나요?", 5, "paraphrase"),
        ("회의실을 예약했는데 늦게 가면 어떻게 되나요?", 6, "paraphrase"),
        ("택배가 너무 늦게 와서 돈을 돌려받고 싶어요.", 7, "paraphrase"),
        ("구독을 끊으면 이번 달 요금은 환불되나요?", 8, "paraphrase"),
        ("같은 결제가 두 번 빠져나갔습니다.", 9, "paraphrase"),
        ("회사를 그만둘 때 장비는 어디에 돌려주나요?", 10, "paraphrase"),
        ("인증서가 곧 만료되는데 갱신은 언제부터 가능한가요?", 11, "paraphrase"),
        ("서비스가 멈추는 정기 점검 시간이 언제인가요?", 12, "paraphrase"),
        ("실수로 지운 문서를 되살릴 수 있나요?", 13, "paraphrase"),
        ("새 시스템 접근권을 받으려면 누구의 승인이 필요한가요?", 14, "paraphrase"),
        ("필요한 볼펜과 종이는 어떻게 주문하나요?", 15, "paraphrase"),
        ("재택근무는 한 주에 며칠까지 할 수 있나요?", 16, "paraphrase"),
        ("외부 강의를 들으면 회사가 얼마나 내주나요?", 17, "paraphrase"),
        ("가족도 건강검진을 공짜로 받을 수 있나요?", 18, "paraphrase"),
        ("차를 가져오려면 사전에 해야 할 일이 있나요?", 19, "paraphrase"),
        ("휴가 며칠 남았지? 쓰려면 뭐부터 해야 돼?", 0, "colloquial"),
        ("계정에 못 들어가겠어요, 비번이 틀렸다고만 나와요", 1, "colloquial"),
        ("영수증 모아서 올리는 거 늦으면 월급에 영향 있나요", 2, "colloquial"),
        ("카페에서 일하는데 사내 시스템 열리나요", 3, "colloquial"),
        ("새벽에 사이트가 안 열렸는데 그게 점검 때문인가요", 12, "colloquial"),
        ("어제 삭제한 폴더 되돌리는 방법 좀", 13, "colloquial"),
        ("회의 시간 지났는데 아무도 안 와서 방이 날아갔어요", 6, "colloquial"),
        ("환불해 주세요 물건이 보름째 안 옵니다", 7, "colloquial"),
        ("인증서 자동 갱신이 실패했을 때 뭘 봐야 하죠", 11, "colloquial"),
        ("배우자 검진비는 누가 내나요", 18, "colloquial"),
    ];

    private static void Report(string line)
    {
        TestContext.Current.TestOutputHelper?.WriteLine(line);
        if (Environment.GetEnvironmentVariable("LMSUPPLY_EMBEDDER_REPORT") is { Length: > 0 } file)
            File.AppendAllText(file, line + Environment.NewLine, Encoding.UTF8);
    }

    [Fact]
    public async Task AnEmbeddingModel_ReadsTheKoreanRetrievalSet_ReportedNotGated()
    {
        var candidate = Environment.GetEnvironmentVariable("LMSUPPLY_EMBEDDER_CANDIDATE");
        if (string.IsNullOrWhiteSpace(candidate))
            Assert.Skip("LMSUPPLY_EMBEDDER_CANDIDATE is unset; this test never downloads.");
        var ct = TestContext.Current.CancellationToken;
        var queryPrefix = Environment.GetEnvironmentVariable("LMSUPPLY_EMBEDDER_QUERY_PREFIX") ?? "";
        var passagePrefix = Environment.GetEnvironmentVariable("LMSUPPLY_EMBEDDER_PASSAGE_PREFIX") ?? "";
        var label = "embedder " + Path.GetFileName(candidate);

        var load = Stopwatch.StartNew();
        await using var model = await LocalEmbedder.LoadAsync(
            candidate, new EmbedderOptions { DisableAutoDownload = true }, cancellationToken: ct);
        var loadMs = load.ElapsedMilliseconds;

        var clock = Stopwatch.StartNew();
        var passages = new float[Passages.Length][];
        for (var i = 0; i < Passages.Length; i++)
            passages[i] = await model.EmbedAsync(passagePrefix + Passages[i], ct);
        var passageMs = clock.ElapsedMilliseconds;

        clock.Restart();
        int hits = 0, hitsParaphrase = 0, hitsColloquial = 0;
        double reciprocal = 0;
        foreach (var (query, right, kind) in Queries)
        {
            var q = await model.EmbedAsync(queryPrefix + query, ct);
            var scores = passages.Select(p => LocalEmbedder.CosineSimilarity(q, p)).ToArray();
            var rank = 1 + scores.Count(s => s > scores[right]);
            reciprocal += 1.0 / rank;
            if (rank == 1)
            {
                hits++;
                if (kind == "paraphrase") hitsParaphrase++; else hitsColloquial++;
            }

            Report($"{label} {kind,-10} rank={rank} {(rank == 1 ? "ok  " : "MISS")} {query}");
        }

        var queryMs = clock.ElapsedMilliseconds;
        Report($"{label} dims={model.Dimensions} recall@1={hits}/{Queries.Length} (paraphrase {hitsParaphrase}/20, colloquial {hitsColloquial}/10) MRR={(reciprocal / Queries.Length).ToString("0.000", CultureInfo.InvariantCulture)}");
        Report($"{label} load {loadMs} ms, {Passages.Length} passages {passageMs} ms, {Queries.Length} queries {queryMs} ms");
        hits.Should().BeGreaterThanOrEqualTo(0);
    }
}
