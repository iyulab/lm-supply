# Test fixtures

Checked in, never generated at test time — a fixture produced by a platform-specific encoder or
voice at test time only exists on the machine that has that encoder or voice.

| File | Origin | Used by |
|---|---|---|
| `tone-440hz-1s.mp3` | 1 s 440 Hz sine, MP3 | `AudioProcessorTests` (real MP3 decode path) |
| `korean-meeting-notice-11s.wav` | 11.4 s Korean speech, 16 kHz mono 16-bit PCM. Synthesized once, offline, with a locally installed Korean text-to-speech voice from a short original sentence (a meeting notice). No third-party recording. | `WhisperLanguageAutodetectConformanceTests` (language auto-detection without a hint) |
| `english-deploy-notice-21s.wav` | 21.5 s English speech, 16 kHz mono 16-bit PCM. Synthesized once, offline, with a locally installed English text-to-speech voice from an original paragraph (a deployment notice: "…deployment window for the reporting service opens at nine o'clock on Thursday… release notes… page the on-call engineer…"). No third-party recording. | `ParakeetTdtConformanceTests` (Parakeet TDT transcription, and the informational Whisper A/B) |
| `english-meeting-notes-16s.wav` | 16.3 s English speech, 16 kHz mono 16-bit PCM. Synthesized once, offline, with a locally installed English text-to-speech voice from an original paragraph (meeting notes: "…ship the translation feature on Friday… update the budget sheet… The next meeting is on Tuesday at ten in the morning."). No third-party recording. | `WhisperTrailingSentenceConformanceTests` (the final sentence of a single-window clip survives decoding) |
