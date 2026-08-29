using Xunit;

// Disable cross-collection test parallelization for this assembly.
//
// Several suites assert on process-global mutable state — System.Diagnostics.Trace.Listeners
// (LlamaOffloadTraceHelperTests' CapturingTraceListener) and the LMSUPPLY_VRAM_BUDGET_MB env var.
// Under xUnit's default parallel execution, a trace-emitting test on one thread pollutes a
// capturing listener on another, making assertions like `listener.Information.Should().BeEmpty()`
// fail nondeterministically (~50% locally) and risk reddening the publish gate.
//
// The suite runs in ~1s, so serial execution is a negligible cost for full determinism.
// This is a blunt CI-risk mitigation; a surgical correlation-token isolation + regression test
// remains an open follow-up (trace-listener parallel pollution flake).
[assembly: Xunit.v3.Parallelization(Mode = Xunit.Sdk.ParallelMode.None)]
