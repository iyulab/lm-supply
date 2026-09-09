using AwesomeAssertions;
using LMSupply.Transcriber.Internal;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// The Parakeet TDT entry is reachable only by name — it must never become the automatic choice, and the family
/// dispatch must key on the architecture, not on the id.
/// </summary>
public sealed class ParakeetTdtRegistryTests
{
    [Fact]
    public void Alias_ResolvesToTheParakeetEntry()
    {
        TranscriberModelRegistry.Default.TryResolve("parakeet-tdt", out var info).Should().BeTrue();

        info!.Id.Should().Be("istupakov/parakeet-tdt-0.6b-v3-onnx");
        info.Architecture.Should().Be(TranscriberArchitectures.ParakeetTdt);
        info.NumMelBins.Should().Be(128);
        info.HiddenSize.Should().Be(1024);
        info.EncoderFile.Should().Be("encoder-model.int8.onnx");
        info.DecoderFile.Should().Be("decoder_joint-model.int8.onnx");
        info.SupportedLanguages.Should().HaveCount(25).And.Contain("en").And.NotContain("ko");
    }

    [Fact]
    public void EveryOtherBuiltInEntry_IsStillWhisper()
    {
        DefaultModels.All.Where(m => !TranscriberArchitectures.IsParakeetTdt(m.Architecture))
            .Should().HaveCount(DefaultModels.All.Count - 1)
            .And.OnlyContain(m => m.Architecture == TranscriberArchitectures.Whisper);
    }

    [Fact]
    public void AutoSelection_NeverPicksParakeet()
    {
        // "auto" ranks a fixed Whisper-only candidate list by available VRAM; whatever this machine has,
        // the answer must come from that family — Parakeet is opt-in by name only.
        TranscriberModelRegistry.Default.TryResolve("auto", out var auto).Should().BeTrue();

        auto!.Architecture.Should().Be(TranscriberArchitectures.Whisper);
    }

    [Theory]
    [InlineData("""{"model_type":"nemo-conformer-tdt","features_size":128,"subsampling_factor":8}""", true, 128, 8)]
    [InlineData("""{"model_type":"nemo-conformer-ctc","features_size":80}""", false, 80, null)]
    public void NemoConfigReader_ParsesNemoExports(string json, bool isTdt, int? features, int? subsampling)
    {
        var cfg = NemoConfigReader.ParseConfig(json);

        cfg.Should().NotBeNull();
        cfg!.IsTdt.Should().Be(isTdt);
        cfg.FeaturesSize.Should().Be(features);
        cfg.SubsamplingFactor.Should().Be(subsampling);
    }

    [Theory]
    [InlineData("""{"model_type":"whisper","num_mel_bins":80}""")]
    [InlineData("""{"d_model":512}""")]
    public void NemoConfigReader_IgnoresNonNemoConfigs(string json)
    {
        NemoConfigReader.ParseConfig(json).Should().BeNull();
    }
}
