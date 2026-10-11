import { useState, useRef } from 'react';
import { api } from '../api/client';
import type { CaptionResponse, VqaResponse } from '../api/types';
import { Loader2, Upload, Image, MessageSquare } from 'lucide-react';
import { CurlExample } from '../components/CurlExample';

type Mode = 'caption' | 'vqa';

export function Caption() {
  const [mode, setMode] = useState<Mode>('caption');
  const [modelId, setModelId] = useState('default');
  const [question, setQuestion] = useState('');
  const [captionResult, setCaptionResult] = useState<CaptionResponse | null>(null);
  const [vqaResult, setVqaResult] = useState<VqaResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedFile = useRef<File | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    selectedFile.current = file;
    setFileName(file.name);
    setPreviewUrl(URL.createObjectURL(file));
    setCaptionResult(null);
    setVqaResult(null);
    setError(null);

    if (mode === 'caption') {
      handleCaption(file);
    }
  };

  const handleCaption = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setCaptionResult(null);
    setElapsedMs(null);

    const start = performance.now();
    try {
      const response = await api.caption(file, modelId);
      setElapsedMs(Math.round(performance.now() - start));
      setCaptionResult(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVqa = async () => {
    if (!selectedFile.current || !question.trim()) return;

    setIsLoading(true);
    setError(null);
    setVqaResult(null);
    setElapsedMs(null);

    const start = performance.now();
    try {
      const response = await api.vqa(selectedFile.current, question, modelId);
      setElapsedMs(Math.round(performance.now() - start));
      setVqaResult(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Image Captioning & VQA</h1>

      {/* Mode Selector */}
      <div className="flex gap-2">
        <button
          onClick={() => setMode('caption')}
          className={`px-4 py-2 rounded flex items-center gap-2 ${
            mode === 'caption'
              ? 'bg-primary text-primary-foreground'
              : 'bg-secondary hover:bg-secondary/80'
          }`}
        >
          <Image className="w-4 h-4" />
          Caption
        </button>
        <button
          onClick={() => setMode('vqa')}
          className={`px-4 py-2 rounded flex items-center gap-2 ${
            mode === 'vqa'
              ? 'bg-primary text-primary-foreground'
              : 'bg-secondary hover:bg-secondary/80'
          }`}
        >
          <MessageSquare className="w-4 h-4" />
          Visual QA
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Model ID</label>
          <input
            type="text"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="w-full max-w-md px-3 py-2 bg-muted border border-border rounded"
            placeholder="default"
          />
        </div>

        <div
          className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-primary transition-colors"
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
          {isLoading && mode === 'caption' ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-12 h-12 animate-spin text-primary" />
              <p>Generating caption...</p>
            </div>
          ) : previewUrl ? (
            <div className="flex flex-col items-center gap-2">
              <img
                src={previewUrl}
                alt="Preview"
                className="max-w-md max-h-64 object-contain rounded"
              />
              <p className="text-sm text-muted-foreground">{fileName}</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-12 h-12 text-muted-foreground" />
              <p className="font-medium">Click to upload image</p>
              <p className="text-sm text-muted-foreground">
                Supports JPG, PNG, WebP, and other image formats
              </p>
            </div>
          )}
        </div>

        {mode === 'vqa' && previewUrl && (
          <div className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="flex-1 px-3 py-2 bg-muted border border-border rounded"
              placeholder="Ask a question about the image..."
              onKeyDown={(e) => e.key === 'Enter' && handleVqa()}
            />
            <button
              onClick={handleVqa}
              disabled={isLoading || !question.trim()}
              className="px-4 py-2 bg-primary text-primary-foreground rounded disabled:opacity-50 flex items-center gap-2"
            >
              {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
              Ask
            </button>
          </div>
        )}
      </div>

      <CurlExample commands={[
        { label: 'Caption', command: `curl -X POST http://localhost:5000/v1/images/caption \\\n  -F file=@image.jpg \\\n  -F model=${modelId}` },
        { label: 'Visual QA', command: `curl -X POST http://localhost:5000/v1/images/vqa \\\n  -F file=@image.jpg \\\n  -F question="What is in this image?" \\\n  -F model=${modelId}` },
      ]} />

      {error && (
        <div className="p-4 bg-destructive/10 text-destructive rounded">
          {error}
        </div>
      )}

      {captionResult && (
        <div className="bg-card border border-border rounded-lg p-4 space-y-2">
          <h2 className="text-lg font-semibold">
            Caption
            {elapsedMs != null && (
              <span className="text-sm font-normal text-muted-foreground ml-2">
                {elapsedMs.toLocaleString()} ms
              </span>
            )}
          </h2>
          <p className="text-lg">{captionResult.choices[0]?.caption}</p>
          {captionResult.choices[0]?.confidence != null && (
            <p className="text-sm text-muted-foreground">
              Confidence: {(captionResult.choices[0].confidence * 100).toFixed(1)}%
            </p>
          )}
          {captionResult.choices.length > 1 && (
            <div>
              <h3 className="text-sm font-medium mt-2">Alternatives:</h3>
              <ul className="list-disc list-inside text-sm text-muted-foreground">
                {captionResult.choices.slice(1).map((alt) => (
                  <li key={alt.index}>{alt.caption}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {vqaResult && (
        <div className="bg-card border border-border rounded-lg p-4 space-y-2">
          <h2 className="text-lg font-semibold">
            Answer
            {elapsedMs != null && (
              <span className="text-sm font-normal text-muted-foreground ml-2">
                {elapsedMs.toLocaleString()} ms
              </span>
            )}
          </h2>
          <p className="text-sm text-muted-foreground">Q: {vqaResult.question}</p>
          <p className="text-lg">A: {vqaResult.answer}</p>
          {vqaResult.confidence != null && (
            <p className="text-sm text-muted-foreground">
              Confidence: {(vqaResult.confidence * 100).toFixed(1)}%
            </p>
          )}
        </div>
      )}
    </div>
  );
}
