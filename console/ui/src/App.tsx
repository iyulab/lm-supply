import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import {
  Dashboard,
  Chat,
  Models,
  Embed,
  Rerank,
  Transcribe,
  Synthesize,
  Caption,
  Ocr,
  Detect,
  Segment,
  Translate,
  ImageGenerate,
} from './pages';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/embed" element={<Embed />} />
          <Route path="/rerank" element={<Rerank />} />
          <Route path="/transcribe" element={<Transcribe />} />
          <Route path="/synthesize" element={<Synthesize />} />
          <Route path="/caption" element={<Caption />} />
          <Route path="/ocr" element={<Ocr />} />
          <Route path="/detect" element={<Detect />} />
          <Route path="/segment" element={<Segment />} />
          <Route path="/translate" element={<Translate />} />
          <Route path="/image-generate" element={<ImageGenerate />} />
          <Route path="/models" element={<Models />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
