# Test fixtures

Checked in, never fetched at test time.

| File | Origin | Used by |
|---|---|---|
| `en_US-ljspeech-medium.onnx.json` | The config of the catalog's default voice (`rhasspy/piper-voices`, `en/en_US/ljspeech/medium`; the LJ Speech voice, public domain), cut down to the keys the synthesizer reads: `audio.sample_rate`, `espeak`, `language.code`, `inference`, `phoneme_type`, `num_speakers` and the whole `phoneme_id_map`. Values are unchanged. | `VoiceConfigTests`, `PhonemeIdEncoderTests` (a voice's declared contract, read by the library's own loader) |
