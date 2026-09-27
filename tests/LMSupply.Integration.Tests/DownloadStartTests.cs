using LMSupply;
using LMSupply.Captioner;
using LMSupply.Detector;
using LMSupply.Embedder;
using LMSupply.Generator;
using LMSupply.Ocr;
using LMSupply.Reranker;
using LMSupply.Segmenter;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;
using LMSupply.Translator;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Tests that verify model downloads start correctly for all aliases.
/// These tests cancel quickly after download starts to avoid long download times.
/// Run locally only - requires network access.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
public class DownloadStartTests
{
    private const int DownloadStartTimeoutMs = 10000; // 10 seconds to verify download starts

    #region Embedder

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    [InlineData("quality")]
    [InlineData("large")]
    [InlineData("multilingual")]
    public async Task Embedder_DownloadStarts_ForAlias(string alias)
    {
        await VerifyDownloadStartsAsync(
            alias,
            async (progress, ct) => await LocalEmbedder.LoadAsync(alias, progress: progress, cancellationToken: ct));
    }

    #endregion

    #region Reranker

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    [InlineData("quality")]
    public async Task Reranker_DownloadStarts_ForAlias(string alias)
    {
        await VerifyDownloadStartsAsync(
            alias,
            async (progress, ct) => await LocalReranker.LoadAsync(alias, progress: progress, cancellationToken: ct));
    }

    #endregion

    #region Generator (GGUF)

    [Theory]
    [InlineData("gguf:gemma4-fast")]
    [InlineData("gguf:gemma4-default")]
    [InlineData("gguf:gemma4-quality")]
    public async Task Generator_Gguf_DownloadStarts_ForAlias(string alias)
    {
        await VerifyDownloadStartsAsync(
            alias,
            async (progress, ct) => await LocalGenerator.LoadAsync(alias, progress: progress, cancellationToken: ct));
    }

    #endregion

    #region Translator

    [Theory]
    [InlineData("default")]
    [InlineData("ko-en")]
    [InlineData("ja-en")]
    public async Task Translator_DownloadStarts_ForAlias(string alias)
    {
        await VerifyDownloadStartsAsync(
            alias,
            async (progress, ct) => await LocalTranslator.LoadAsync(alias, progress: progress, cancellationToken: ct));
    }

    #endregion

    #region Transcriber

    [Theory]
    [InlineData("default")]
    [InlineData("turbo")]  // fast model
    [InlineData("medium")]
    public async Task Transcriber_DownloadStarts_ForAlias(string alias)
    {
        await VerifyDownloadStartsAsync(
            alias,
            async (progress, ct) => await LocalTranscriber.LoadAsync(alias, progress: progress, cancellationToken: ct));
    }

    #endregion

    #region Synthesizer

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    public async Task Synthesizer_DownloadStarts_ForAlias(string alias)
    {
        await VerifyDownloadStartsAsync(
            alias,
            async (progress, ct) => await LocalSynthesizer.LoadAsync(alias, progress: progress, cancellationToken: ct));
    }

    #endregion

    #region Captioner

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    public async Task Captioner_DownloadStarts_ForAlias(string alias)
    {
        await VerifyDownloadStartsAsync(
            alias,
            async (progress, ct) => await LocalCaptioner.LoadAsync(alias, progress: progress, cancellationToken: ct));
    }

    #endregion

    #region Ocr

    [Theory]
    [InlineData("default", null)]           // Default detection + default recognition
    [InlineData("default", "crnn-en-v3")]   // English recognition
    [InlineData("default", "crnn-korean-v3")] // Korean recognition
    public async Task Ocr_DownloadStarts_ForAlias(string detectionModel, string? recognitionModel)
    {
        await VerifyDownloadStartsAsync(
            $"{detectionModel}/{recognitionModel ?? "default"}",
            async (progress, ct) => await LocalOcr.LoadAsync(detectionModel, recognitionModel, progress: progress, cancellationToken: ct));
    }

    #endregion

    #region Detector

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    public async Task Detector_DownloadStarts_ForAlias(string alias)
    {
        await VerifyDownloadStartsAsync(
            alias,
            async (progress, ct) => await LocalDetector.LoadAsync(alias, progress: progress, cancellationToken: ct));
    }

    #endregion

    #region Segmenter

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    public async Task Segmenter_DownloadStarts_ForAlias(string alias)
    {
        await VerifyDownloadStartsAsync(
            alias,
            async (progress, ct) => await LocalSegmenter.LoadAsync(alias, progress: progress, cancellationToken: ct));
    }

    #endregion

    #region Helper

    private static async Task VerifyDownloadStartsAsync<T>(
        string alias,
        Func<IProgress<DownloadProgress>, CancellationToken, Task<T>> loadFunc) where T : IAsyncDisposable
    {
        var downloadStarted = false;
        var downloadFileName = string.Empty;
        var cts = new CancellationTokenSource();

        var progress = new Progress<DownloadProgress>(p =>
        {
            if (p.Phase == DownloadPhase.Downloading && p.TotalBytes > 0)
            {
                downloadStarted = true;
                downloadFileName = p.FileName;
                cts.Cancel(); // Cancel as soon as download starts
            }
        });

        try
        {
            cts.CancelAfter(DownloadStartTimeoutMs);
            await using var model = await loadFunc(progress, cts.Token);

            // If we reach here, model was already cached - that's OK
            downloadStarted = true;
        }
        catch (OperationCanceledException)
        {
            // Expected - we cancelled after download started
        }
        catch (Exception ex) when (ex.InnerException is OperationCanceledException)
        {
            // Expected - wrapped cancellation
        }

        downloadStarted.Should().BeTrue(
            $"Download should start for alias '{alias}'. " +
            $"File: {downloadFileName}");
    }

    #endregion
}
