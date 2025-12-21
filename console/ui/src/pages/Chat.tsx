import { useState, useRef, useEffect } from 'react';
import { api } from '../api/client';
import type { ChatMessage } from '../api/types';
import { ModelSelector } from '../components/ModelSelector';
import { Send, Bot, User, Square } from 'lucide-react';

export function Chat() {
  const [modelId, setModelId] = useState('');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent]);

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !modelId) return;

    const userMessage: ChatMessage = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setStreamingContent('');

    // Create new abort controller for this request
    abortControllerRef.current = new AbortController();

    const allMessages = [...messages, userMessage];
    let fullContent = '';

    try {
      for await (const token of api.chatStream({
        modelId,
        messages: allMessages,
        signal: abortControllerRef.current.signal,
      })) {
        fullContent += token;
        setStreamingContent(fullContent);
      }

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: fullContent },
      ]);
      setStreamingContent('');
    } catch (error) {
      // Handle abort separately from other errors
      if (error instanceof Error && error.name === 'AbortError') {
        // User cancelled - save partial content if any
        if (fullContent) {
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: fullContent + '\n\n(cancelled)' },
          ]);
        }
        setStreamingContent('');
      } else {
        console.error('Chat error:', error);
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Error: ${(error as Error).message}` },
        ]);
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-bold">Chat</h1>
          <div className="flex items-center gap-2">
            <label className="text-sm text-muted-foreground">Model:</label>
            <ModelSelector
              modelType="chat"
              value={modelId}
              onChange={setModelId}
              disabled={isLoading}
            />
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-muted-foreground py-8">
            <Bot className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>Start a conversation with the AI model</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}
          >
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                <Bot className="w-4 h-4 text-primary-foreground" />
              </div>
            )}
            <div
              className={`max-w-[70%] p-3 rounded-lg ${
                msg.role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted'
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}

        {/* Streaming content */}
        {streamingContent && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
              <Bot className="w-4 h-4 text-primary-foreground" />
            </div>
            <div className="max-w-[70%] p-3 rounded-lg bg-muted">
              <p className="whitespace-pre-wrap">{streamingContent}</p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-border">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 px-4 py-2 bg-muted border border-border rounded-lg"
            disabled={isLoading}
          />
          {isLoading ? (
            <button
              type="button"
              onClick={handleCancel}
              className="px-4 py-2 bg-destructive text-destructive-foreground rounded-lg flex items-center gap-2 hover:bg-destructive/90"
              title="Cancel generation"
            >
              <Square className="w-4 h-4" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim() || !modelId}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50 flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
