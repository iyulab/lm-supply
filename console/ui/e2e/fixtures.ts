import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/** A page of two printed lines in English («Please keep your receipt for seven days.» / «The library is closed on Mondays.»): an image with text in it. */
export const TEXT_PAGE_PNG = readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'fixtures/text-page.png'));

/** A valid 1x1 red PNG, for multipart image uploads. */
export const PNG_BUFFER = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC',
  'base64'
);

/** The same image as a data URL, for chat messages. */
export const PNG_DATA_URL = `data:image/png;base64,${PNG_BUFFER.toString('base64')}`;

/**
 * A PNG whose header reads (1x1) but whose pixel data is damaged: it identifies as an image and does not decode.
 * An endpoint that takes an upload must answer a client error for it, not a server error.
 */
export const DAMAGED_PNG_BUFFER = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==',
  'base64'
);

/** A valid WAV buffer: a 44-byte header and one second of 8 kHz mono silence. */
export function minimalWav(): Buffer {
  const sampleRate = 8000;
  const numSamples = sampleRate;
  const dataSize = numSamples * 2;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + dataSize, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(dataSize, 40);
  return buf;
}

/** Korean for "hello", written as escapes so that the source stays ASCII: the registered translation models read Korean. */
export const KOREAN_HELLO = '\uC548\uB155\uD558\uC138\uC694';
/** Korean for "thank you". */
export const KOREAN_THANKS = '\uAC10\uC0AC\uD569\uB2C8\uB2E4';
/** Korean for "one", "two", "three". */
export const KOREAN_COUNT = ['\uD558\uB098', '\uB458', '\uC14B'];

/** A valid PCM WAV header (mono, 8 kHz, 16 bit) with no samples after it: a recording whose header was written and whose audio never was. */
export function headerOnlyWav(): Buffer {
  const buf = Buffer.alloc(44);
  buf.write('RIFF', 0); buf.writeUInt32LE(36, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22); buf.writeUInt32LE(8000, 24);
  buf.writeUInt32LE(16000, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(0, 40);
  return buf;
}
