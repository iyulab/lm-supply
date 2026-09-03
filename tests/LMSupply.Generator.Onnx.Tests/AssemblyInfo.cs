using Xunit;

// Disable cross-collection test parallelization for this assembly.
//
// CachePathResolutionTests mutates the HF_HUB_CACHE environment variable, which is
// process-global. Under xUnit's default parallel execution, that mutation could race with
// another collection reading the same env var mid-test. This mirrors the same guard applied to
// LMSupply.Generator.Tests (the project these files were split out of), for the same reason.
[assembly: Xunit.v3.Parallelization(Mode = Xunit.Sdk.ParallelMode.None)]
