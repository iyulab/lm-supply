// TextGeneratorBuilder Sample - Tests the fix for onnxruntime-genai DLL loading
//
// This sample validates that TextGeneratorBuilder correctly downloads and loads
// the ONNX Runtime GenAI native binaries on-demand, fixing the DllNotFoundException
// that occurred when the runtime wasn't ensured before model creation.

using LMSupply;
using LMSupply.Generator;
using LMSupply.Generator.Models;

// Registers the ONNX Runtime GenAI backend so the "Default" preset can route to it on a discrete
// non-NVIDIA Windows GPU (LMSupply.Generator itself has no compile-time ONNX dependency).
LMSupply.Generator.Onnx.OnnxGeneratorBackend.Register();

Console.WriteLine("=== TextGeneratorBuilder Sample ===");
Console.WriteLine("Testing automatic runtime download and loading...\n");

// Progress reporter for downloads
var progress = new Progress<DownloadProgress>(p =>
{
    if (p.TotalBytes > 0)
    {
        var percent = (double)p.BytesDownloaded / p.TotalBytes;
        Console.Write($"\rDownloading {p.FileName}: {percent:P0}");
    }
});

try
{
    // =========================================================================
    // Test 1: TextGeneratorBuilder with Default Preset
    // =========================================================================
    Console.WriteLine("[Test 1] TextGeneratorBuilder with Default Preset");
    Console.WriteLine("This should automatically download onnxruntime-genai runtime...\n");

    // This was failing with DllNotFoundException before the fix
    await using var model = await TextGeneratorBuilder.Create()
        .WithModel(GeneratorModelPreset.Default)
        .BuildAsync();

    Console.WriteLine("\n[SUCCESS] Model loaded successfully!");

    // Display model info
    var info = model.GetModelInfo();
    Console.WriteLine($"  Model ID: {info.ModelId}");
    Console.WriteLine($"  Path: {info.ModelPath}");
    Console.WriteLine($"  Max Context: {info.MaxContextLength}");
    Console.WriteLine($"  Chat Format: {info.ChatFormat}");
    Console.WriteLine($"  Provider: {info.ExecutionProvider}");

    // =========================================================================
    // Test 2: Simple Generation
    // =========================================================================
    Console.WriteLine("\n[Test 2] Simple Text Generation");
    Console.WriteLine(new string('-', 50));

    Console.Write("Prompt: Hello, I am\nResponse: ");
    await foreach (var token in model.GenerateAsync(
        "Hello, I am",
        new GenerationOptions { MaxTokens = 30, Temperature = 0.7f }))
    {
        Console.Write(token);
    }
    Console.WriteLine();

    // =========================================================================
    // Test 3: Chat Completion
    // =========================================================================
    Console.WriteLine("\n[Test 3] Chat Completion");
    Console.WriteLine(new string('-', 50));

    var messages = new[]
    {
        ChatMessage.System("You are a helpful assistant. Keep responses brief."),
        ChatMessage.User("What is 2 + 2?")
    };

    Console.WriteLine("System: You are a helpful assistant. Keep responses brief.");
    Console.WriteLine("User: What is 2 + 2?");
    Console.Write("Assistant: ");

    await foreach (var token in model.GenerateChatAsync(
        messages,
        new GenerationOptions { MaxTokens = 50 }))
    {
        Console.Write(token);
    }
    Console.WriteLine();

    // =========================================================================
    // Summary
    // =========================================================================
    Console.WriteLine("\n" + new string('=', 50));
    Console.WriteLine("[ALL TESTS PASSED]");
    Console.WriteLine("TextGeneratorBuilder correctly downloads and loads");
    Console.WriteLine("ONNX Runtime GenAI native binaries on-demand.");
    Console.WriteLine(new string('=', 50));
}
catch (DllNotFoundException ex)
{
    Console.WriteLine($"\n[FAILED] DllNotFoundException: {ex.Message}");
    Console.WriteLine("\nThis indicates the runtime download/loading fix is not working.");
    Console.WriteLine("Expected: Runtime should be downloaded automatically before model creation.");
    Environment.Exit(1);
}
catch (Exception ex)
{
    Console.WriteLine($"\n[ERROR] {ex.GetType().Name}: {ex.Message}");
    if (ex.InnerException != null)
    {
        Console.WriteLine($"  Inner: {ex.InnerException.Message}");
    }
    Environment.Exit(1);
}
