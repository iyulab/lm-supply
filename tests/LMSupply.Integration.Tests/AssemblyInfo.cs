using Xunit;

// Disable cross-collection test parallelization for this assembly.
//
// The Functional suites are [Category=LocalOnly] tests that each call LocalGenerator.LoadAsync
// (or the embedder/ocr/reranker/etc. equivalents), spinning up a llama-server / ONNX runtime that
// holds GPU memory for the test's duration. Under xUnit's default cross-class parallelism, several
// model-loading classes (GeneratorFunctionalTests, Gemma4EmptyChatProbeTests, EmbedderFunctionalTests,
// ...) spawn concurrent backends on a single GPU, exhausting VRAM and aborting in-flight requests
// (System.Net.Http.HttpRequestException: "An established connection was aborted by the software in
// your host machine"). That is a harness artifact, not a product defect.
//
// Serial execution makes the local suite deterministic on a single-GPU box. These tests are
// CI-excluded (LocalOnly), so the wall-clock cost is local only. Mirrors the same mitigation in
// LMSupply.Generator.Tests/AssemblyInfo.cs.
[assembly: Xunit.v3.Parallelization(Mode = Xunit.Sdk.ParallelMode.None)]
