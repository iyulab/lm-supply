# Test fixtures

Checked in, never generated at test time — a fixture produced by a platform-specific encoder or
voice at test time only exists on the machine that has that encoder or voice.

| File | Origin | Used by |
|---|---|---|
| `tone-440hz-1s.mp3` | 1 s 440 Hz sine, MP3 | `AudioProcessorTests` (real MP3 decode path) |
| `korean-meeting-notice-11s.wav` | 11.4 s Korean speech, 16 kHz mono 16-bit PCM. Synthesized once, offline, with a locally installed Korean text-to-speech voice from a short original sentence (a meeting notice). No third-party recording. | `WhisperLanguageAutodetectConformanceTests` (language auto-detection without a hint) |
| `english-deploy-notice-21s.wav` | 21.5 s English speech, 16 kHz mono 16-bit PCM. Synthesized once, offline, with a locally installed English text-to-speech voice from an original paragraph (a deployment notice: "…deployment window for the reporting service opens at nine o'clock on Thursday… release notes… page the on-call engineer…"). No third-party recording. | `ParakeetTdtConformanceTests` (Parakeet TDT transcription, and the informational Whisper A/B) |
| `english-meeting-notes-16s.wav` | 16.3 s English speech, 16 kHz mono 16-bit PCM. Synthesized once, offline, with a locally installed English text-to-speech voice from an original paragraph (meeting notes: "…ship the translation feature on Friday… update the budget sheet… The next meeting is on Tuesday at ten in the morning."). No third-party recording. | `WhisperTrailingSentenceConformanceTests` (the final sentence of a single-window clip survives decoding) |
| `korean-meeting-minutes-131s.mp3` | 131 s Korean speech, 16 kHz mono MP3 (32 kbps). Synthesized once, offline, with a locally installed Korean text-to-speech voice from an original 26-sentence script (weekly marketing meeting minutes: agenda, figures, action items, next meeting date). No third-party recording. | `WhisperLongFormConformanceTests` (speech spoken across 30 s window boundaries survives long-form decoding) |
| `meeting-three-voices.mp3` | 86 s English speech, 16 kHz mono MP3 (48 kbps). Synthesized once, offline, with three Windows OneCore text-to-speech voices (Microsoft David, Zira and Mark: two male, one female) from an original 18-line planning-meeting script, one synthesis per line with 0.7 s of silence between; truth `A A A A A B B C C C A A A C A B A A`. Regenerate with `make-meeting-fixtures.ps1` (Windows PowerShell 5.1 + ffmpeg). No third-party recording. | `DiarizationMeetingConformanceTests` (speaker counts, `NumSpeakers`, short recordings) |
| `meeting-one-voice.mp3` | The same script read by one voice (Microsoft David), 87 s, same format and generator (`-Mono`). | `DiarizationMeetingConformanceTests` (one voice stays one speaker, under `MaxSpeakers` too) |
