import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { ImageGenerationExtendedResponse, ImageModelInfo } from '../api/types';
import { Loader2, Wand2, Download, RefreshCw, Settings2 } from 'lucide-react';

const SIZE_PRESETS = [
  { label: '256x256', value: '256x256' },
  { label: '512x512', value: '512x512' },
  { label: '768x768', value: '768x768' },
  { label: '1024x1024', value: '1024x1024' },
  { label: '1024x1792 (Portrait)', value: '1024x1792' },
  { label: '1792x1024 (Landscape)', value: '1792x1024' },
];

export function ImageGenerate() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [modelId, setModelId] = useState('default');
  const [size, setSize] = useState('512x512');
  const [steps, setSteps] = useState(4);
  const [guidanceScale, setGuidanceScale] = useState(1.0);
  const [seed, setSeed] = useState<number | undefined>(undefined);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [result, setResult] = useState<ImageGenerationExtendedResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ImageModelInfo[]>([]);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const modelList = await api.getImageModels();
      setModels(modelList);
    } catch (err) {
      console.error('Failed to load models:', err);
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.generateImage({
        prompt: prompt.trim(),
        model: modelId,
        size,
        steps,
        guidance_scale: guidanceScale,
        seed,
        negative_prompt: negativePrompt.trim() || undefined,
      });
      setResult(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result?.data[0]?.b64_json) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${result.data[0].b64_json}`;
    link.download = `generated-${result.id}.png`;
    link.click();
  };

  const handleModelChange = (id: string) => {
    setModelId(id);
    const model = models.find(m => m.id === id);
    if (model) {
      setSteps(model.recommended_steps);
      setGuidanceScale(model.recommended_guidance_scale);
    }
  };

  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 2147483647));
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Image Generation</h1>
      <p className="text-muted-foreground">
        Generate images from text using Latent Consistency Models (LCM) for fast, high-quality results.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Panel */}
        <div className="space-y-4">
          {/* Prompt */}
          <div>
            <label className="block text-sm font-medium mb-1">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full px-3 py-2 bg-muted border border-border rounded resize-none h-24"
              placeholder="A serene lake at sunset with mountains in the background..."
              disabled={isLoading}
            />
          </div>

          {/* Model & Size Row */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                value={modelId}
                onChange={(e) => handleModelChange(e.target.value)}
                className="w-full px-3 py-2 bg-muted border border-border rounded"
                disabled={isLoading}
              >
                {models.length === 0 ? (
                  <option value="default">default</option>
                ) : (
                  models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.id}
                    </option>
                  ))
                )}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Size</label>
              <select
                value={size}
                onChange={(e) => setSize(e.target.value)}
                className="w-full px-3 py-2 bg-muted border border-border rounded"
                disabled={isLoading}
              >
                {SIZE_PRESETS.map((preset) => (
                  <option key={preset.value} value={preset.value}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Advanced Settings Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <Settings2 className="w-4 h-4" />
            {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
          </button>

          {/* Advanced Settings */}
          {showAdvanced && (
            <div className="space-y-4 p-4 bg-muted/50 rounded-lg border border-border">
              {/* Negative Prompt */}
              <div>
                <label className="block text-sm font-medium mb-1">Negative Prompt</label>
                <textarea
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  className="w-full px-3 py-2 bg-muted border border-border rounded resize-none h-16"
                  placeholder="blurry, low quality, distorted..."
                  disabled={isLoading}
                />
              </div>

              {/* Steps & Guidance */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    Steps ({steps})
                  </label>
                  <input
                    type="range"
                    min={1}
                    max={8}
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value))}
                    className="w-full"
                    disabled={isLoading}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    LCM models work best with 2-8 steps
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    Guidance Scale ({guidanceScale.toFixed(1)})
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={3}
                    step={0.1}
                    value={guidanceScale}
                    onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                    className="w-full"
                    disabled={isLoading}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Lower values (~1.0) recommended for LCM
                  </p>
                </div>
              </div>

              {/* Seed */}
              <div>
                <label className="block text-sm font-medium mb-1">Seed</label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    value={seed ?? ''}
                    onChange={(e) => setSeed(e.target.value ? parseInt(e.target.value) : undefined)}
                    className="flex-1 px-3 py-2 bg-muted border border-border rounded"
                    placeholder="Random"
                    disabled={isLoading}
                  />
                  <button
                    onClick={randomizeSeed}
                    className="px-3 py-2 bg-secondary rounded hover:bg-secondary/80"
                    disabled={isLoading}
                    title="Randomize seed"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Use same seed for reproducible results
                </p>
              </div>
            </div>
          )}

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={isLoading || !prompt.trim()}
            className="w-full px-4 py-3 bg-primary text-primary-foreground rounded-lg disabled:opacity-50 flex items-center justify-center gap-2 font-medium"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Wand2 className="w-5 h-5" />
                Generate Image
              </>
            )}
          </button>

          {/* Error */}
          {error && (
            <div className="p-4 bg-destructive/10 text-destructive rounded-lg">
              {error}
            </div>
          )}
        </div>

        {/* Result Panel */}
        <div className="space-y-4">
          <div className="bg-card border border-border rounded-lg p-4 min-h-[400px] flex items-center justify-center">
            {isLoading ? (
              <div className="text-center">
                <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
                <p className="text-muted-foreground">Generating your image...</p>
                <p className="text-sm text-muted-foreground">This may take a moment</p>
              </div>
            ) : result?.data[0]?.b64_json ? (
              <img
                src={`data:image/png;base64,${result.data[0].b64_json}`}
                alt="Generated"
                className="max-w-full max-h-[500px] object-contain rounded-lg"
              />
            ) : (
              <div className="text-center text-muted-foreground">
                <Wand2 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Your generated image will appear here</p>
              </div>
            )}
          </div>

          {/* Result Info */}
          {result && (
            <div className="bg-card border border-border rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">Generation Details</h3>
                <button
                  onClick={handleDownload}
                  className="px-3 py-1.5 bg-primary text-primary-foreground rounded flex items-center gap-2 text-sm"
                >
                  <Download className="w-4 h-4" />
                  Download
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-muted-foreground">Model:</span>{' '}
                  <span className="font-medium">{result.model}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Time:</span>{' '}
                  <span className="font-medium">{(result.generation_time_ms / 1000).toFixed(2)}s</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Size:</span>{' '}
                  <span className="font-medium">
                    {result.data[0]?.width}x{result.data[0]?.height}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Seed:</span>{' '}
                  <span className="font-medium">{result.data[0]?.seed}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Steps:</span>{' '}
                  <span className="font-medium">{result.data[0]?.steps}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">ID:</span>{' '}
                  <span className="font-medium font-mono text-xs">{result.id}</span>
                </div>
              </div>
              {result.data[0]?.prompt && (
                <div className="text-sm">
                  <span className="text-muted-foreground">Prompt:</span>{' '}
                  <span className="italic">{result.data[0].prompt}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
