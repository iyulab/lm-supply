using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Generator.Tests;

/// <summary>
/// A consumer that ships its own llama-server (<see cref="LlamaServerUpdateOptions.ServerBinaryPath"/>)
/// must be able to load a GGUF on whichever build it ships. llama.cpp b11146 (v0.5.0) removed
/// <c>--mmap</c>/<c>--no-mmap</c>/<c>--mlock</c>, and the presets set <c>UseMemoryMap = true</c>,
/// so a load that treated the external binary as "unknown build" died on "invalid argument: --mmap"
/// before <c>/health</c>.
/// Point <c>LMSUPPLY_EXTERNAL_LLAMA_SERVER</c> at a llama-server binary (run it once against a pre-b10105
/// build and once against b11146+). <c>LMSUPPLY_TEST_GGUF</c> overrides the model file.
/// </summary>
[Trait("Category", "Integration")]
public class ExternalLlamaServerBinaryIntegrationTests
{
    private const string ServerEnvVar = "LMSUPPLY_EXTERNAL_LLAMA_SERVER";
    private const string GgufEnvVar = "LMSUPPLY_TEST_GGUF";

    private static readonly string DefaultGguf = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".cache", "huggingface", "hub",
        "models--bartowski--microsoft_Phi-4-mini-instruct-GGUF",
        "snapshots", "main", "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf");

    [Fact]
    public async Task ExternalBinary_LoadsAndGenerates_OnItsOwnBuild()
    {
        await LoadAndGenerateAsync(llamaOptions: null);
    }

    [Fact]
    public async Task ExternalBinary_NonDefaultMemoryLock_ReachesTheServerInItsOwnSpelling()
    {
        // Positive control for the --version probe. Memory mapping on is also covered by "send
        // nothing" when the build is unknown, so the test above passes whether or not the probe
        // worked. A lock request has no spelling that parses on both sides of b10105/b11146: it
        // loads only if the probe read the build ("--load-mode mlock" on b11146, "--mlock" before).
        await LoadAndGenerateAsync(new LlamaOptions { UseMemoryLock = true });
    }

    private static async Task LoadAndGenerateAsync(LlamaOptions? llamaOptions)
    {
        var server = Environment.GetEnvironmentVariable(ServerEnvVar);
        if (string.IsNullOrEmpty(server))
            Assert.Skip($"{ServerEnvVar} is not set.");

        var gguf = Environment.GetEnvironmentVariable(GgufEnvVar) ?? DefaultGguf;
        Assert.True(File.Exists(gguf), $"GGUF not found at {gguf}. Set {GgufEnvVar}.");

        var options = new GeneratorOptions
        {
            Provider = ExecutionProvider.Cpu,
            DisableAutoDownload = true,
            LlamaOptions = llamaOptions,
            ServerUpdateOptions = new LlamaServerUpdateOptions
            {
                ServerBinaryPath = server,
                AutoDownloadUpdates = false,
            },
        };

        await using var model = await LocalGenerator.LoadFromPathAsync(gguf, options);

        var result = await model.GenerateCompleteAsync(
            "What is 2+2? Answer with just the number.",
            new Models.GenerationOptions { MaxTokens = 16 },
            TestContext.Current.CancellationToken);

        result.Should().NotBeNullOrWhiteSpace();
    }
}
