/**
 * Record a short clip from the microphone straight to 16 kHz mono WAV.
 *
 * Wake-word training needs raw PCM the server can read without a codec, so
 * this taps the audio graph instead of using MediaRecorder (which produces
 * webm/mp4). The clip stops by itself after the speaker pauses.
 */

export const WAV_SAMPLE_RATE = 16000;

export interface ClipOptions {
  /** Stop once this much quiet follows speech. */
  silenceMs?: number;
  /** Give up if no speech starts within this time. */
  startTimeoutMs?: number;
  maxMs?: number;
  onLevel?: (level: number) => void;
  signal?: AbortSignal;
}

export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

export function downsample(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const length = Math.floor(input.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    // Average the source samples that fold into this one (cheap low-pass).
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}

export class NoSpeechError extends Error {
  constructor() {
    super("Didn't hear anything. Check the microphone and try again.");
    this.name = 'NoSpeechError';
  }
}

/** Record one utterance; resolves with a WAV blob. Must start from a click. */
export async function recordClip(options: ClipOptions = {}): Promise<Blob> {
  const silenceMs = options.silenceMs ?? 700;
  const startTimeoutMs = options.startTimeoutMs ?? 5000;
  const maxMs = options.maxMs ?? 3000;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const Ctor = window.AudioContext
    ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new Ctor();
  const source = ctx.createMediaStreamSource(stream);
  // ScriptProcessor is deprecated but universally available and plenty for a
  // few seconds of capture; an AudioWorklet would need a separate module file.
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];
  const sink = ctx.createGain();
  sink.gain.value = 0;

  return new Promise<Blob>((resolve, reject) => {
    let started = 0;
    let lastVoice = 0;
    let heard = false;
    let noise = 0.004;
    let finished = false;

    const finish = (error?: Error) => {
      if (finished) return;
      finished = true;
      processor.onaudioprocess = null;
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close().catch(() => {});
      options.onLevel?.(0);
      if (error) {
        reject(error);
        return;
      }
      const total = chunks.reduce((n, c) => n + c.length, 0);
      const joined = new Float32Array(total);
      let offset = 0;
      for (const c of chunks) {
        joined.set(c, offset);
        offset += c.length;
      }
      resolve(encodeWav(downsample(joined, ctx.sampleRate, WAV_SAMPLE_RATE), WAV_SAMPLE_RATE));
    };

    options.signal?.addEventListener('abort', () =>
      finish(new DOMException('Recording cancelled', 'AbortError')),
    );

    processor.onaudioprocess = (event) => {
      const data = new Float32Array(event.inputBuffer.getChannelData(0));
      chunks.push(data);
      const now = performance.now();
      if (!started) started = now;
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
      const rms = Math.sqrt(sum / data.length);
      options.onLevel?.(Math.min(1, rms * 6));
      const elapsed = now - started;
      if (elapsed < 250) {
        noise = Math.max(noise, rms);
        return;
      }
      if (rms > Math.max(0.015, noise * 3)) {
        heard = true;
        lastVoice = now;
      }
      if (!heard && elapsed > startTimeoutMs) finish(new NoSpeechError());
      else if (heard && now - lastVoice > silenceMs) finish();
      else if (elapsed > maxMs) finish();
    };
    source.connect(processor);
    processor.connect(sink);
    sink.connect(ctx.destination);
    void ctx.resume();
  });
}
