using Microsoft.ML.OnnxRuntime;

namespace LMSupply.Inference;

/// <summary>
/// The ONNX Runtime session settings every LMSupply model takes from <see cref="LMSupplyOptionsBase"/>, applied in one
/// place so that no model type reads them differently.
/// </summary>
public static class SessionOptionsExtensions
{
    /// <summary>ONNX Runtime session configuration key: whether intra-op worker threads spin while waiting for work.</summary>
    public const string IntraOpAllowSpinningKey = "session.intra_op.allow_spinning";

    /// <summary>ONNX Runtime session configuration key: whether inter-op worker threads spin while waiting for work.</summary>
    public const string InterOpAllowSpinningKey = "session.inter_op.allow_spinning";

    /// <summary>
    /// Applies <see cref="LMSupplyOptionsBase.LogLevel"/> and <see cref="LMSupplyOptionsBase.ThreadCount"/> to
    /// <paramref name="sessionOptions"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// With <see cref="LMSupplyOptionsBase.ThreadCount"/> set, the session uses that many intra-op and inter-op threads,
    /// and its threads stop spin-waiting for work: ONNX Runtime's pool otherwise keeps worker threads busy-waiting after
    /// each run, which on a many-core machine shows as several cores at full load for seconds after a short call.
    /// Leaving CPU headroom and saving power is what the option is for, so it turns that off too.
    /// </para>
    /// <para>
    /// Without it, ONNX Runtime's defaults apply (a thread per physical core, spinning on) — the throughput setting.
    /// </para>
    /// </remarks>
    public static SessionOptions ApplyCommonOptions(this SessionOptions sessionOptions, LMSupplyOptionsBase options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return sessionOptions.ApplyCommonOptions(options.LogLevel, options.ThreadCount);
    }

    /// <summary>
    /// Applies a log level and a thread count to <paramref name="sessionOptions"/> — the same rules as
    /// <see cref="ApplyCommonOptions(SessionOptions, LMSupplyOptionsBase)"/>, for an options type that carries these two
    /// values without deriving from <see cref="LMSupplyOptionsBase"/>.
    /// </summary>
    public static SessionOptions ApplyCommonOptions(this SessionOptions sessionOptions, OrtLogLevel logLevel, int? threadCount)
    {
        ArgumentNullException.ThrowIfNull(sessionOptions);

        sessionOptions.LogSeverityLevel = (OrtLoggingLevel)(int)logLevel;

        if (threadCount is { } threads)
        {
            if (threads < 1)
                throw new ArgumentOutOfRangeException(nameof(threadCount), threads, "ThreadCount must be at least 1, or null for the ONNX Runtime default.");

            sessionOptions.IntraOpNumThreads = threads;
            sessionOptions.InterOpNumThreads = threads;
            sessionOptions.AddSessionConfigEntry(IntraOpAllowSpinningKey, "0");
            sessionOptions.AddSessionConfigEntry(InterOpAllowSpinningKey, "0");
        }

        return sessionOptions;
    }
}
