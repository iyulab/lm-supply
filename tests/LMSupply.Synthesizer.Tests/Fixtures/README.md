# Test fixtures

Checked in, never fetched at test time.

| File | Origin | Used by |
|---|---|---|
| `en_US-ljspeech-medium.onnx.json` | The config of the catalog's default voice (`rhasspy/piper-voices`, `en/en_US/ljspeech/medium`; the LJ Speech voice, public domain), cut down to the keys the synthesizer reads: `audio.sample_rate`, `espeak`, `language.code`, `inference`, `phoneme_type`, `num_speakers` and the whole `phoneme_id_map`. Values are unchanged. | `VoiceConfigTests`, `PhonemeIdEncoderTests`, `EnglishPhonemizerTests` (a voice's declared contract, read by the library's own loader) |
| `en_US-amy-medium.onnx.json`, `en_US-lessac-medium.onnx.json`, `en_US-ryan-low.onnx.json`, `en_GB-semaine-medium.onnx.json` | The configs of the catalog's other English voices (`rhasspy/piper-voices`, the voice folders of `SynthesizerModels`), cut down the same way. Values are unchanged; the Ryan voice's config has no `phoneme_type`, and so neither has its fixture. | `EnglishPhonemizerTests` (the built-in English phonemizer writes only symbols each voice's own map has, in each voice's accent) |
