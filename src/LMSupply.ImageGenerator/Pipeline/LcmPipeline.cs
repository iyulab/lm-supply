using System.Diagnostics;
using System.Runtime.CompilerServices;
using LMSupply.ImageGenerator.Encoders;
using LMSupply.ImageGenerator.Models;
using LMSupply.ImageGenerator.Schedulers;
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
    public static async Task<LcmPipeline> LoadAsync(
        string modelDir,
        SessionOptions? sessionOptions = null,
        CancellationToken cancellationToken = default)
    {
        sessionOptions ??= new SessionOptions();

        // Load all components in parallel
        var textEncoderTask = ClipTextEncoder.LoadAsync(modelDir, sessionOptions, cancellationToken);
        var unetTask = UNetModel.LoadAsync(modelDir, sessionOptions, cancellationToken);
        var vaeTask = VaeDecoder.LoadAsync(modelDir, sessionOptions, cancellationToken: cancellationToken);

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
        GenerationOptions options,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var stopwatch = Stopwatch.StartNew();

        var width = options.Width > 0 ? options.Width : _defaultWidth;
        var height = options.Height > 0 ? options.Height : _defaultHeight;
        var steps = options.Steps > 0 ? options.Steps : 4;
        var guidanceScale = options.GuidanceScale;
        var seed = options.Seed ?? Random.Shared.Next();
        var useCfg = guidanceScale > 1.0f;

        // Encode text prompt
        // Use EncodeWithNegativeAsync for CFG (batch 2), EncodeAsync otherwise (batch 1)
        DenseTensor<float> textEmbeddings;
        if (useCfg)
        {
            textEmbeddings = await _textEncoder.EncodeWithNegativeAsync(
                prompt,
                options.NegativePrompt,
                cancellationToken);
        }
        else
        {
            textEmbeddings = await _textEncoder.EncodeAsync(prompt, cancellationToken);
        }

        // Generate initial noise
        var random = new Random(seed);
        var latentHeight = height / 8;
        var latentWidth = width / 8;
        var latentShape = new[] { 1, _unet.LatentChannels, latentHeight, latentWidth };
        var noiseData = LcmScheduler.CreateNoise(latentShape, random);
        var latents = new DenseTensor<float>(noiseData, latentShape);

        // Setup scheduler
        _scheduler.SetTimesteps(steps);

        // Denoising loop - convert to array to avoid span across await
        var timesteps = _scheduler.Timesteps.ToArray();
        foreach (var timestep in timesteps)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Scale input
            var scaledLatents = _scheduler.ScaleModelInput(latents.Buffer.Span, timestep);
            var latentInput = new DenseTensor<float>(scaledLatents, latentShape);

            // For classifier-free guidance, concatenate latents
            DenseTensor<float> modelInput;
            if (useCfg)
            {
                // Double the latent for CFG [negative, positive]
                var doubleLatents = new float[scaledLatents.Length * 2];
                Array.Copy(scaledLatents, 0, doubleLatents, 0, scaledLatents.Length);
                Array.Copy(scaledLatents, 0, doubleLatents, scaledLatents.Length, scaledLatents.Length);
                modelInput = new DenseTensor<float>(doubleLatents, [2, latentShape[1], latentShape[2], latentShape[3]]);
            }
            else
            {
                modelInput = latentInput;
            }

            // UNet forward pass
            var noisePred = await _unet.ForwardAsync(modelInput, timestep, textEmbeddings, cancellationToken);

            // Apply classifier-free guidance
            float[] guidedNoise;
            if (useCfg)
            {
                guidedNoise = ApplyGuidance(noisePred, guidanceScale);
            }
            else
            {
                guidedNoise = noisePred.Buffer.ToArray();
            }

            // Scheduler step
            var stepResult = _scheduler.Step(guidedNoise, timestep, latents.Buffer.Span, random);
            latents = new DenseTensor<float>(stepResult, latentShape);
        }

        // Decode latents to image
        var imageBytes = await _vaeDecoder.DecodeAsync(latents, cancellationToken);

        stopwatch.Stop();

        return new GeneratedImage
        {
            ImageData = imageBytes,
            Width = width,
            Height = height,
            Seed = seed,
            Steps = steps,
            Prompt = prompt,
            GenerationTime = stopwatch.Elapsed
        };
    }

    /// <summary>
    /// Generates an image with step-by-step streaming.
    /// </summary>
    public async IAsyncEnumerable<GenerationStep> GenerateStreamingAsync(
        string prompt,
        GenerationOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var stopwatch = Stopwatch.StartNew();

        var width = options.Width > 0 ? options.Width : _defaultWidth;
        var height = options.Height > 0 ? options.Height : _defaultHeight;
        var steps = options.Steps > 0 ? options.Steps : 4;
        var guidanceScale = options.GuidanceScale;
        var seed = options.Seed ?? Random.Shared.Next();
        var useCfg = guidanceScale > 1.0f;

        // Encode text prompt
        // Use EncodeWithNegativeAsync for CFG (batch 2), EncodeAsync otherwise (batch 1)
        DenseTensor<float> textEmbeddings;
        if (useCfg)
        {
            textEmbeddings = await _textEncoder.EncodeWithNegativeAsync(
                prompt,
                options.NegativePrompt,
                cancellationToken);
        }
        else
        {
            textEmbeddings = await _textEncoder.EncodeAsync(prompt, cancellationToken);
        }

        // Generate initial noise
        var random = new Random(seed);
        var latentHeight = height / 8;
        var latentWidth = width / 8;
        var latentShape = new[] { 1, _unet.LatentChannels, latentHeight, latentWidth };
        var noiseData = LcmScheduler.CreateNoise(latentShape, random);
        var latents = new DenseTensor<float>(noiseData, latentShape);

        // Setup scheduler
        _scheduler.SetTimesteps(steps);

        // Convert to array to avoid span across await
        var timesteps = _scheduler.Timesteps.ToArray();
        var stepNumber = 0;
        foreach (var timestep in timesteps)
        {
            cancellationToken.ThrowIfCancellationRequested();
            stepNumber++;

            // Scale input
            var scaledLatents = _scheduler.ScaleModelInput(latents.Buffer.Span, timestep);
            var latentInput = new DenseTensor<float>(scaledLatents, latentShape);

            // For classifier-free guidance
            DenseTensor<float> modelInput;
            if (useCfg)
            {
                var doubleLatents = new float[scaledLatents.Length * 2];
                Array.Copy(scaledLatents, 0, doubleLatents, 0, scaledLatents.Length);
                Array.Copy(scaledLatents, 0, doubleLatents, scaledLatents.Length, scaledLatents.Length);
                modelInput = new DenseTensor<float>(doubleLatents, [2, latentShape[1], latentShape[2], latentShape[3]]);
            }
            else
            {
                modelInput = latentInput;
            }

            // UNet forward pass
            var noisePred = await _unet.ForwardAsync(modelInput, timestep, textEmbeddings, cancellationToken);

            // Apply classifier-free guidance
            float[] guidedNoise;
            if (useCfg)
            {
                guidedNoise = ApplyGuidance(noisePred, guidanceScale);
            }
            else
            {
                guidedNoise = noisePred.Buffer.ToArray();
            }

            // Scheduler step
            var stepResult = _scheduler.Step(guidedNoise, timestep, latents.Buffer.Span, random);
            latents = new DenseTensor<float>(stepResult, latentShape);

            var isFinal = stepNumber == steps;
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
                        Width = width,
                        Height = height,
                        Seed = seed,
                        Steps = steps,
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
                TotalSteps = steps,
                PreviewData = previewData,
                FinalImage = finalImage,
                Elapsed = stopwatch.Elapsed
            };
        }
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
