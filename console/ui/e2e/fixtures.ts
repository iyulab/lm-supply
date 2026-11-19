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
