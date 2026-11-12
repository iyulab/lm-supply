using System.Diagnostics;
using System.Runtime.CompilerServices;
using LMSupply.ImageGenerator.Encoders;
using LMSupply.ImageGenerator.Models;
using LMSupply.ImageGenerator.Schedulers;
using LMSupply.Inference;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.ImageGenerator.Pipeline;

/// <summary>
/// Latent Consistency Model (LCM) pipeline for fast image generation.
/// Orchestrates text encoding, latent denoising, and image decoding.
/// </summary>
internal sealed class LcmPipeline : IAsyncDisposable
{
    private readonly ClipTextEncoder _textEncoder;
    private readonly UNetModel _unet;
    private readonly VaeDecoder _vaeDecoder;
    private readonly LcmScheduler _scheduler;
    private readonly int _defaultWidth;
    private readonly int _defaultHeight;
    private bool _disposed;

    private LcmPipeline(
        ClipTextEncoder textEncoder,
        UNetModel unet,
        VaeDecoder vaeDecoder,
        LcmScheduler scheduler,
        int defaultWidth,
        int defaultHeight)
    {
        _textEncoder = textEncoder;
        _unet = unet;
        _vaeDecoder = vaeDecoder;
        _scheduler = scheduler;
        _defaultWidth = defaultWidth;
        _defaultHeight = defaultHeight;
    }

    /// <summary>
    /// Loads the LCM pipeline from a model directory.
    /// </summary>
    /// <param name="modelDir">Path to model directory.</param>
    /// <param name="provider">Execution provider for the three sessions (text encoder, UNet, VAE).</param>
    /// <param name="deviceId">GPU device index for CUDA.</param>
    /// <param name="configureOptions">Session options to apply to every session (log level, threads).</param>
    /// <param name="progress">Receives the runtime download session creation may make; <see langword="null"/> when nobody is listening.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task<LcmPipeline> LoadAsync(
        string modelDir,
        ExecutionProvider provider,
        int deviceId,
        Action<SessionOptions>? configureOptions,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken = default)
    {
        // The three sessions belong to one model: a provider that fails on any of them is left by all
        // of them (see RecoverableOnnxSession).
        var blacklist = new ProviderBlacklist();

        // Load all components in parallel
        var textEncoderTask = ClipTextEncoder.LoadAsync(modelDir, provider, deviceId, configureOptions, blacklist, progress, cancellationToken);
        var unetTask = UNetModel.LoadAsync(modelDir, provider, deviceId, configureOptions, blacklist, progress, cancellationToken);
        var vaeTask = VaeDecoder.LoadAsync(modelDir, provider, deviceId, configureOptions, blacklist, progress, cancellationToken: cancellationToken);

        await Task.WhenAll(textEncoderTask, unetTask, vaeTask);

        var textEncoder = await textEncoderTask;
        var unet = await unetTask;
        var vae = await vaeTask;

        var scheduler = new LcmScheduler(LcmSchedulerConfig.ForLcm());

        return new LcmPipeline(textEncoder, unet, vae, scheduler, 512, 512);
    }

    /// <summary>
    /// Generates an image from a text prompt.
    /// </summary>
    public async Task<GeneratedImage> GenerateAsync(
        string prompt,
        ImageGenerationOptions options,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var stopwatch = Stopwatch.StartNew();
        var run = await StartRunAsync(prompt, options, cancellationToken);

        var latents = run.Latents;
        foreach (var timestep in run.Timesteps)
        {
            cancellationToken.ThrowIfCancellationRequested();
            latents = await DenoiseStepAsync(run, latents, timestep, cancellationToken);
        }

        // Decode latents to image
        var imageBytes = await _vaeDecoder.DecodeAsync(latents, cancellationToken);

        stopwatch.Stop();

        return new GeneratedImage
        {
            ImageData = imageBytes,
            Width = run.Width,
            Height = run.Height,
            Seed = run.Seed,
            Steps = run.Steps,
            Prompt = prompt,
            GenerationTime = stopwatch.Elapsed
        };
    }

    /// <summary>
    /// Generates an image with step-by-step streaming.
    /// </summary>
    public async IAsyncEnumerable<GenerationStep> GenerateStreamingAsync(
        string prompt,
        ImageGenerationOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var stopwatch = Stopwatch.StartNew();
        var run = await StartRunAsync(prompt, options, cancellationToken);

        var latents = run.Latents;
        var stepNumber = 0;
        foreach (var timestep in run.Timesteps)
        {
            cancellationToken.ThrowIfCancellationRequested();
            stepNumber++;

            latents = await DenoiseStepAsync(run, latents, timestep, cancellationToken);

            var isFinal = stepNumber == run.Steps;
            byte[]? previewData = null;
            GeneratedImage? finalImage = null;

            // Generate preview if requested or if final step
            if (options.GeneratePreviews || isFinal)
            {
                var decoded = await _vaeDecoder.DecodeAsync(latents, cancellationToken);

                if (isFinal)
                {
                    finalImage = new GeneratedImage
                    {
                        ImageData = decoded,
                        Width = run.Width,
                        Height = run.Height,
                        Seed = run.Seed,
                        Steps = run.Steps,
                        Prompt = prompt,
                        GenerationTime = stopwatch.Elapsed
                    };
                }
                else
                {
                    previewData = decoded;
                }
            }

            yield return new GenerationStep
            {
                StepNumber = stepNumber,
                TotalSteps = run.Steps,
                PreviewData = previewData,
                FinalImage = finalImage,
                Elapsed = stopwatch.Elapsed
            };
        }
    }

    // What a run needs besides the loop state: the sizes, the text conditioning and how guidance is applied.
    private sealed record Run(
        int Width,
        int Height,
        int Steps,
        int Seed,
        int[] LatentShape,
        int[] Timesteps,
        DenseTensor<float> Latents,
        DenseTensor<float> TextEmbeddings,
        float GuidanceScale,
        float[]? GuidanceEmbedding,
        bool UseClassifierFreeGuidance,
        Random Random);

    private async Task<Run> StartRunAsync(string prompt, ImageGenerationOptions options, CancellationToken cancellationToken)
    {
        var width = options.Width > 0 ? options.Width : _defaultWidth;
        var height = options.Height > 0 ? options.Height : _defaultHeight;
        var steps = options.Steps > 0 ? options.Steps : 4;
        var guidanceScale = options.GuidanceScale;
        var seed = options.Seed ?? Random.Shared.Next();

        // A model distilled with guidance takes the guidance weight as an input and runs once per step on the prompt alone
        // (a negative prompt has no place in it); any other model needs the classic second, unconditional pass.
        var embeddingDimension = _unet.GuidanceEmbeddingDimension;
        var guidanceEmbedding = embeddingDimension > 0 ? GuidanceEmbedding.Create(guidanceScale, embeddingDimension) : null;
        var useCfg = embeddingDimension == 0 && guidanceScale > 1.0f;

        var textEmbeddings = useCfg
            ? await _textEncoder.EncodeWithNegativeAsync(prompt, options.NegativePrompt, cancellationToken)
            : await _textEncoder.EncodeAsync(prompt, cancellationToken);

        // Initial noise
        var random = new Random(seed);
        var latentShape = new[] { 1, _unet.LatentChannels, height / 8, width / 8 };
        var latents = new DenseTensor<float>(LcmScheduler.CreateNoise(latentShape, random), latentShape);

        _scheduler.SetTimesteps(steps);

        return new Run(
            width, height, steps, seed, latentShape, _scheduler.Timesteps.ToArray(), latents, textEmbeddings,
            guidanceScale, guidanceEmbedding, useCfg, random);
    }

    private async Task<DenseTensor<float>> DenoiseStepAsync(
        Run run, DenseTensor<float> latents, int timestep, CancellationToken cancellationToken)
    {
        var scaledLatents = LcmScheduler.ScaleModelInput(latents.Buffer.Span, timestep);

        DenseTensor<float> modelInput;
        if (run.UseClassifierFreeGuidance)
        {
            // Double the latent for CFG [negative, positive]
            var doubleLatents = new float[scaledLatents.Length * 2];
            Array.Copy(scaledLatents, 0, doubleLatents, 0, scaledLatents.Length);
            Array.Copy(scaledLatents, 0, doubleLatents, scaledLatents.Length, scaledLatents.Length);
            modelInput = new DenseTensor<float>(doubleLatents, [2, run.LatentShape[1], run.LatentShape[2], run.LatentShape[3]]);
        }
        else
        {
            modelInput = new DenseTensor<float>(scaledLatents, run.LatentShape);
        }

        var noisePred = await _unet.ForwardAsync(modelInput, timestep, run.TextEmbeddings, run.GuidanceEmbedding, cancellationToken);

        var guidedNoise = run.UseClassifierFreeGuidance
            ? ApplyGuidance(noisePred, run.GuidanceScale)
            : noisePred.Buffer.ToArray();

        var stepResult = _scheduler.Step(guidedNoise, timestep, latents.Buffer.Span, run.Random);
        return new DenseTensor<float>(stepResult, run.LatentShape);
    }

    private static float[] ApplyGuidance(DenseTensor<float> noisePred, float guidanceScale)
    {
        var dims = noisePred.Dimensions;
        var halfLength = (int)(noisePred.Length / 2);

        var result = new float[halfLength];
        var buffer = noisePred.Buffer.Span;

        // noisePred shape: [2, channels, height, width]
        // [0] = negative (unconditional), [1] = positive (conditional)
        for (int i = 0; i < halfLength; i++)
        {
            var uncond = buffer[i];
            var cond = buffer[halfLength + i];
            // CFG formula: uncond + guidance_scale * (cond - uncond)
            result[i] = uncond + guidanceScale * (cond - uncond);
        }

        return result;
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        await _textEncoder.DisposeAsync();
        await _unet.DisposeAsync();
        await _vaeDecoder.DisposeAsync();
    }
}
