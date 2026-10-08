import { create } from 'zustand';
import { fetchSpeechHealth, transcribeAudio } from './api';
import { useTtsStore } from './tts';
import type { ListenerState } from './voice-api';

/**
 * Voice input is one shared resource, like voice output (see tts.ts): one
 * microphone session for the whole app. Keeping it in a store rather than in a
 * component hook lets every part of the UI -- the input bar, the Jarvis orb,
 * the tab title -- show the same live state, and lets the level meter update
 * many times a second without re-rendering the whole chat input.
 */
export type VoicePhase = 'idle' | 'requesting' | 'listening' | 'transcribing';

export type StopReason =
  | 'manual'
  | 'silence'
  | 'no-speech'
  | 'max-duration'
  | 'cancelled'
  | 'device-lost'
  | 'error';

export interface StartOptions {
  /** Receives the transcript of a finished, non-empty recording. */
  onTranscript: (text: string) => void;
  /** Stop on its own once the speaker pauses. */
  autoStop: boolean;
  /** Play short tones when listening starts and stops. */
  earcons: boolean;
}

interface VoiceStore {
  phase: VoicePhase;
  /** Smoothed microphone level, 0..1, while listening. */
  level: number;
  /** Date.now() when listening began, for the elapsed timer. */
  startedAt: number | null;
  /** Whether speech (not just background noise) was heard this session. */
  heardSpeech: boolean;
  error: string | null;
  /** Why and when the last session ended, so the UI can say so. */
  lastStop: { reason: StopReason; at: number } | null;
  /** State of the background listener (`jarvis listen`), when it reports. */
  listener: { state: ListenerState; detail: string | null } | null;
  /** A spoken request from the listener waiting to be sent to the model. */
  pendingCommand: { id: string; text: string } | null;
  /** The server answered 404 to the voice routes: it predates this page. */
  serverOutdated: boolean;
  setListener: (state: ListenerState | null, detail?: string | null) => void;
  setPendingCommand: (command: { id: string; text: string } | null) => void;
  /** null until the speech backend health probe has answered. */
  available: boolean | null;
  ensureHealth: (force?: boolean) => Promise<void>;
  start: (options: StartOptions) => Promise<void>;
  stop: (reason?: StopReason) => void;
  cancel: () => void;
}

// Tuning for the silence detector. RMS of the time-domain signal, 0..1.
export const VOICE_TUNING = {
  /** Interval of the level/VAD loop. setInterval, not rAF: rAF pauses in a
   * background tab, which would leave the mic open forever. */
  tickMs: 50,
  /** Ignore the first moments: the start tone and mic auto-gain settle here. */
  calibrateFromMs: 250,
  calibrateUntilMs: 600,
  minSpeechRms: 0.02,
  noiseMultiplier: 2.5,
  /** Pause length that ends an utterance once speech was heard. */
  silenceMs: 1500,
  /** Give up if nothing but noise arrives for this long. */
  noSpeechMs: 8000,
  maxDurationMs: 120_000,
};

// Recording handles are not render state; see the same pattern in tts.ts.
let stream: MediaStream | null = null;
let recorder: MediaRecorder | null = null;
let chunks: Blob[] = [];
let audioCtx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
let loop: ReturnType<typeof setInterval> | null = null;
let session = 0;
let options: StartOptions | null = null;
let healthProbe: Promise<void> | null = null;

let earconCtx: AudioContext | null = null;

/** Two short sine blips, rising for "listening", falling for "stopped". */
export function playEarcon(kind: 'start' | 'stop' | 'error'): void {
  try {
    const Ctor = window.AudioContext
      ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return;
    earconCtx ??= new Ctor();
    const ctx = earconCtx;
    if (ctx.state === 'suspended') void ctx.resume();
    const notes = kind === 'start' ? [660, 880] : kind === 'stop' ? [880, 587] : [330, 262];
    notes.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const t = ctx.currentTime + i * 0.09;
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(0.12, t + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.08);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t);
      osc.stop(t + 0.09);
    });
  } catch {
    // Audio cues are a nicety; never let them break recording.
  }
}

/** File extension the speech backend can use to pick a decoder. */
export function extensionForMime(mime: string): string {
  if (mime.includes('mp4') || mime.includes('aac')) return 'mp4';
  if (mime.includes('ogg')) return 'ogg';
  if (mime.includes('wav')) return 'wav';
  return 'webm';
}

function rms(buffer: Uint8Array): number {
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) {
    const v = (buffer[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / buffer.length);
}

function releaseInput(): void {
  if (loop) {
    clearInterval(loop);
    loop = null;
  }
  stream?.getTracks().forEach((track) => {
    track.onended = null;
    track.stop();
  });
  stream = null;
  if (audioCtx) {
    void audioCtx.close().catch(() => {});
    audioCtx = null;
  }
  analyser = null;
}

function describeMicError(err: unknown): string {
  const name = err instanceof DOMException ? err.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Microphone access was blocked. Allow it in the browser address bar and try again.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return 'No microphone was found. Plug one in and try again.';
  }
  if (name === 'NotReadableError') {
    return 'The microphone is in use by another application.';
  }
  return 'Could not start the microphone.';
}

export const useVoiceStore = create<VoiceStore>((set, get) => {
  function finish(reason: StopReason, mine: number): void {
    if (mine !== session) return;
    const rec = recorder;
    const opts = options;
    const transcribe = reason !== 'cancelled' && reason !== 'no-speech' && reason !== 'error';
    recorder = null;
    if (opts?.earcons) playEarcon(reason === 'error' || reason === 'device-lost' ? 'error' : 'stop');

    if (!rec || rec.state === 'inactive' || !transcribe) {
      if (rec && rec.state !== 'inactive') {
        rec.onstop = null;
        rec.stop();
      }
      releaseInput();
      chunks = [];
      set({ phase: 'idle', level: 0, startedAt: null, lastStop: { reason, at: Date.now() } });
      return;
    }

    set({ phase: 'transcribing', level: 0, lastStop: { reason, at: Date.now() } });
    rec.onstop = async () => {
      if (mine !== session) return;
      const mime = rec.mimeType || 'audio/webm';
      const blob = new Blob(chunks, { type: mime });
      chunks = [];
      try {
        const result = await transcribeAudio(blob, `recording.${extensionForMime(mime)}`);
        if (mine !== session) return;
        set({ phase: 'idle', startedAt: null });
        const text = result?.text?.trim() ?? '';
        if (text) opts?.onTranscript(text);
        else set({ error: "Didn't catch that. Try again a little closer to the microphone." });
      } catch (err) {
        if (mine !== session) return;
        set({
          phase: 'idle',
          startedAt: null,
          error: err instanceof Error ? err.message : 'Transcription failed',
        });
      }
    };
    // Stop the recorder before its tracks so the final chunk is flushed.
    rec.stop();
    releaseInput();
  }

  return {
    phase: 'idle',
    level: 0,
    startedAt: null,
    heardSpeech: false,
    error: null,
    lastStop: null,
    available: null,
    listener: null,
    pendingCommand: null,
    serverOutdated: false,

    setListener: (state, detail = null) =>
      set({ listener: state ? { state, detail: detail ?? null } : null }),
    setPendingCommand: (command) => set({ pendingCommand: command }),

    ensureHealth: (force = false) => {
      if (healthProbe && !force) return healthProbe;
      healthProbe = fetchSpeechHealth()
        .then((health) => {
          set({ available: health.available });
          // Retry later when unavailable: the API may still be starting.
          if (!health.available) healthProbe = null;
        })
        .catch(() => {
          set({ available: false });
          healthProbe = null;
        });
      return healthProbe;
    },

    start: async (opts: StartOptions) => {
      if (get().phase !== 'idle') return;
      session += 1;
      const mine = session;
      options = opts;
      set({ phase: 'requesting', error: null, heardSpeech: false, level: 0 });

      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
        set({
          phase: 'idle',
          error: window.isSecureContext
            ? 'Microphone recording is not supported in this browser.'
            : 'The microphone only works on https:// or http://localhost pages.',
        });
        return;
      }

      // Barge-in: talking to Jarvis should silence Jarvis.
      useTtsStore.getState().stop();

      let media: MediaStream;
      try {
        media = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });
      } catch (err) {
        if (mine !== session) return;
        set({ phase: 'idle', error: describeMicError(err) });
        if (opts.earcons) playEarcon('error');
        return;
      }
      if (mine !== session) {
        media.getTracks().forEach((t) => t.stop());
        return;
      }
      stream = media;

      try {
        recorder = new MediaRecorder(media);
      } catch {
        releaseInput();
        set({ phase: 'idle', error: 'This browser cannot record audio from the microphone.' });
        return;
      }
      chunks = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      recorder.onerror = () => {
        set({ error: 'Recording failed.' });
        finish('error', mine);
      };
      media.getAudioTracks().forEach((track) => {
        track.onended = () => {
          set({ error: 'The microphone was disconnected.' });
          finish('device-lost', mine);
        };
      });

      // Level meter and silence detection. Optional: without Web Audio the
      // session still records, it just needs a manual stop.
      try {
        const Ctor = window.AudioContext
          ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (Ctor) {
          audioCtx = new Ctor();
          analyser = audioCtx.createAnalyser();
          analyser.fftSize = 1024;
          audioCtx.createMediaStreamSource(media).connect(analyser);
        }
      } catch {
        audioCtx = null;
        analyser = null;
      }

      // Start chunking so a long session does not hold one giant buffer.
      recorder.start(250);
      const startedAt = Date.now();
      set({ phase: 'listening', startedAt });
      if (opts.earcons) playEarcon('start');

      const buffer = analyser ? new Uint8Array(analyser.fftSize) : null;
      let noiseSum = 0;
      let noiseCount = 0;
      let threshold = VOICE_TUNING.minSpeechRms;
      let lastSpeechAt = 0;
      let smoothed = 0;

      loop = setInterval(() => {
        if (mine !== session) return;
        const now = Date.now();
        const elapsed = now - startedAt;
        if (elapsed >= VOICE_TUNING.maxDurationMs) {
          finish('max-duration', mine);
          return;
        }
        if (!analyser || !buffer) return;
        analyser.getByteTimeDomainData(buffer);
        const value = rms(buffer);

        if (elapsed >= VOICE_TUNING.calibrateFromMs && elapsed < VOICE_TUNING.calibrateUntilMs) {
          noiseSum += value;
          noiseCount += 1;
          threshold = Math.max(
            VOICE_TUNING.minSpeechRms,
            (noiseSum / noiseCount) * VOICE_TUNING.noiseMultiplier,
          );
        }

        smoothed = Math.max(value, smoothed * 0.8);
        const level = Math.min(1, smoothed * 5);
        if (Math.abs(level - get().level) > 0.01) set({ level });

        if (elapsed < VOICE_TUNING.calibrateUntilMs) return;
        if (value > threshold) {
          lastSpeechAt = now;
          if (!get().heardSpeech) set({ heardSpeech: true });
        }
        if (!options?.autoStop) return;
        if (lastSpeechAt && now - lastSpeechAt >= VOICE_TUNING.silenceMs) {
          finish('silence', mine);
        } else if (!lastSpeechAt && elapsed >= VOICE_TUNING.noSpeechMs) {
          set({ error: "I didn't hear anything, so I stopped listening." });
          finish('no-speech', mine);
        }
      }, VOICE_TUNING.tickMs);
    },

    stop: (reason: StopReason = 'manual') => {
      const { phase } = get();
      if (phase === 'requesting') {
        // Permission prompt still open: abandon the session.
        session += 1;
        set({ phase: 'idle', lastStop: { reason: 'cancelled', at: Date.now() } });
        return;
      }
      if (phase !== 'listening') return;
      finish(reason, session);
    },

    cancel: () => {
      const { phase } = get();
      if (phase === 'listening') {
        finish('cancelled', session);
      } else if (phase === 'requesting' || phase === 'transcribing') {
        session += 1;
        releaseInput();
        recorder = null;
        chunks = [];
        set({ phase: 'idle', level: 0, startedAt: null, lastStop: { reason: 'cancelled', at: Date.now() } });
      }
    },
  };
});

/** Test seam: drop module-level recording handles between cases. */
export function __resetVoiceForTests(): void {
  session += 1;
  releaseInput();
  recorder = null;
  chunks = [];
  options = null;
  healthProbe = null;
  useVoiceStore.setState({
    phase: 'idle',
    level: 0,
    startedAt: null,
    heardSpeech: false,
    error: null,
    lastStop: null,
    available: null,
    listener: null,
    pendingCommand: null,
    serverOutdated: false,
  });
}
