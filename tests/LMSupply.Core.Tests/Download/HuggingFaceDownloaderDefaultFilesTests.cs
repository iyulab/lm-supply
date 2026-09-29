using LMSupply.Download;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// Regression teeth for the curated default file list used by the alias download path
/// (<c>DownloadModelAsync</c> without an explicit file list).
/// </summary>
public class HuggingFaceDownloaderDefaultFilesTests
{
    /// <summary>
    /// External-weight ONNX models (e.g. BAAI/bge-m3 — the 'default'/'quality' embedder
    /// alias) ship <c>model.onnx</c> as a small graph shell whose weights live in a
    /// companion data file. Omitting the companion from the curated list downloads a
    /// model that crashes at session init ("file_size: ... model.onnx_data").
    /// </summary>
    [Fact]
    public void DefaultModelFiles_Include_ExternalWeightCompanions()
    {
        var files = HuggingFaceDownloader.GetDefaultModelFiles().ToList();

        Assert.Contains("model.onnx", files);
        Assert.Contains("model.onnx_data", files);  // HF underscore convention (bge-m3)
        Assert.Contains("model.onnx.data", files);  // HF dot convention
    }

    /// <summary>
    /// Companions must come after the graph shell so progress reporting counts the
    /// critical file first, and must not displace the critical file itself.
    /// </summary>
    [Fact]
    public void DefaultModelFiles_GraphShell_Precedes_Companions()
    {
        var files = HuggingFaceDownloader.GetDefaultModelFiles().ToList();

        Assert.True(files.IndexOf("model.onnx") < files.IndexOf("model.onnx_data"));
        Assert.True(files.IndexOf("model.onnx") < files.IndexOf("model.onnx.data"));
    }
}
