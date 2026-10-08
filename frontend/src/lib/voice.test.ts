import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./api', () => ({
  transcribeAudio: vi.fn(),
  fetchSpeechHealth: vi.fn(),
  synthesizeSpeech: vi.fn(),
  fetchTtsHealth: vi.fn(),
}));

// presence.ts reads the app store, which touches localStorage on import.
vi.mock('./store', () => ({ useAppStore: vi.fn() }));

import { transcribeAudio, fetchSpeechHealth } from './api';
import {
  useVoiceStore,
  __resetVoiceForTests,
  extensionForMime,
  VOICE_TUNING,
} from './voice';
import { derivePresence } from './presence';

const transcribe = vi.mocked(transcribeAudio);
const health = vi.mocked(fetchSpeechHealth);

/** Amplitude the fake analyser reports; 0 is silence, ~0.3 is speech. */
let amplitude = 0;

class FakeTrack {
  onended: (() => void) | null = null;
  stopped = false;
  stop(): void {
    this.stopped = true;
  }
}

class FakeStream {
  track = new FakeTrack();
  getTracks() {
    return [this.track];
  }
  getAudioTracks() {
    return [this.track];
  }
}

class FakeRecorder {
  static instances: FakeRecorder[] = [];
  state: 'inactive' | 'recording' = 'inactive';
  mimeType = 'audio/webm';
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor() {
    FakeRecorder.instances.push(this);
  }
  start(): void {
    this.state = 'recording';
  }
  stop(): void {
    this.state = 'inactive';
    this.ondataavailable?.({ data: new Blob(['audio']) });
    queueMicrotask(() => this.onstop?.());
  }
}

class FakeAnalyser {
  fftSize = 1024;
  getByteTimeDomainData(buffer: Uint8Array): void {
    // Square wave of the current amplitude: its RMS equals the amplitude.
    for (let i = 0; i < buffer.length; i++) {
      buffer[i] = 128 + (i % 2 === 0 ? 1 : -1) * Math.round(amplitude * 127);
    }
  }
}

class FakeAudioContext {
  state = 'running';
  currentTime = 0;
  destination = {};
  createAnalyser() {
    return new FakeAnalyser();
  }
  createMediaStreamSource() {
    return { connect: () => undefined };
  }
  createOscillator() {
    return {
      type: 'sine',
      frequency: { value: 0 },
      connect: (node: unknown) => node,
      start: () => undefined,
      stop: () => undefined,
    };
  }
  createGain() {
    return {
      gain: {
        setValueAtTime: () => undefined,
        exponentialRampToValueAtTime: () => undefined,
      },
      connect: (node: unknown) => node,
    };
  }
  resume() {
    return Promise.resolve();
  }
  close() {
    return Promise.resolve();
  }
}

const getUserMedia = vi.fn();

beforeEach(() => {
  vi.useFakeTimers();
  amplitude = 0;
  FakeRecorder.instances = [];
  getUserMedia.mockReset();
  getUserMedia.mockImplementation(async () => new FakeStream());
  transcribe.mockReset();
  health.mockReset();
  vi.stubGlobal('window', { AudioContext: FakeAudioContext, isSecureContext: true });
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia } });
  vi.stubGlobal('MediaRecorder', FakeRecorder);
  __resetVoiceForTests();
});

afterEach(() => {
  __resetVoiceForTests();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function flush(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

function start(onTranscript = vi.fn(), autoStop = true) {
  return useVoiceStore.getState().start({ onTranscript, autoStop, earcons: false });
}

describe('voice store', () => {
  it('shows listening while the mic is open and delivers the transcript on stop', async () => {
    transcribe.mockResolvedValue({ text: 'hello jarvis' } as never);
    const onTranscript = vi.fn();
    await start(onTranscript);
    expect(useVoiceStore.getState().phase).toBe('listening');
    expect(useVoiceStore.getState().startedAt).not.toBeNull();

    useVoiceStore.getState().stop('manual');
    expect(useVoiceStore.getState().phase).toBe('transcribing');
    await flush();

    expect(onTranscript).toHaveBeenCalledWith('hello jarvis');
    const state = useVoiceStore.getState();
    expect(state.phase).toBe('idle');
    expect(state.lastStop?.reason).toBe('manual');
    expect(transcribe.mock.calls[0][1]).toBe('recording.webm');
  });

  it('stops by itself after the speaker pauses', async () => {
    transcribe.mockResolvedValue({ text: 'turn on the lights' } as never);
    const onTranscript = vi.fn();
    await start(onTranscript);

    vi.advanceTimersByTime(VOICE_TUNING.calibrateUntilMs);
    amplitude = 0.3;
    vi.advanceTimersByTime(1000);
    const listening = useVoiceStore.getState();
    expect(listening.phase).toBe('listening');
    expect(listening.heardSpeech).toBe(true);
    expect(listening.level).toBeGreaterThan(0.5);

    amplitude = 0;
    vi.advanceTimersByTime(VOICE_TUNING.silenceMs + VOICE_TUNING.tickMs);
    expect(useVoiceStore.getState().lastStop?.reason).toBe('silence');
    await flush();
    expect(onTranscript).toHaveBeenCalledWith('turn on the lights');
  });

  it('keeps listening through pauses when auto-stop is off', async () => {
    await start(vi.fn(), false);
    vi.advanceTimersByTime(VOICE_TUNING.calibrateUntilMs);
    amplitude = 0.3;
    vi.advanceTimersByTime(500);
    amplitude = 0;
    vi.advanceTimersByTime(VOICE_TUNING.noSpeechMs + 1000);
    expect(useVoiceStore.getState().phase).toBe('listening');
  });

  it('gives up without transcribing when it hears nothing', async () => {
    await start();
    vi.advanceTimersByTime(VOICE_TUNING.noSpeechMs + VOICE_TUNING.tickMs);
    await flush();
    const state = useVoiceStore.getState();
    expect(state.phase).toBe('idle');
    expect(state.lastStop?.reason).toBe('no-speech');
    expect(state.error).toMatch(/didn't hear anything/);
    expect(transcribe).not.toHaveBeenCalled();
  });

  it('cancel discards the recording and releases the microphone', async () => {
    await start();
    const stream = await getUserMedia.mock.results[0].value;
    useVoiceStore.getState().cancel();
    await flush();
    expect(useVoiceStore.getState().phase).toBe('idle');
    expect(useVoiceStore.getState().lastStop?.reason).toBe('cancelled');
    expect(stream.track.stopped).toBe(true);
    expect(transcribe).not.toHaveBeenCalled();
  });

  it('explains a blocked microphone', async () => {
    getUserMedia.mockRejectedValue(new DOMException('denied', 'NotAllowedError'));
    await start();
    const state = useVoiceStore.getState();
    expect(state.phase).toBe('idle');
    expect(state.error).toMatch(/blocked/);
  });

  it('reports an unplugged microphone and keeps what was said', async () => {
    transcribe.mockResolvedValue({ text: 'half a sentence' } as never);
    const onTranscript = vi.fn();
    await start(onTranscript);
    const stream = await getUserMedia.mock.results[0].value;
    stream.track.onended?.();
    await flush();
    const state = useVoiceStore.getState();
    expect(state.phase).toBe('idle');
    expect(state.lastStop?.reason).toBe('device-lost');
    expect(state.error).toMatch(/disconnected/);
    expect(onTranscript).toHaveBeenCalledWith('half a sentence');
  });

  it('re-probes speech health after an unavailable answer', async () => {
    health.mockResolvedValueOnce({ available: false });
    await useVoiceStore.getState().ensureHealth();
    expect(useVoiceStore.getState().available).toBe(false);
    health.mockResolvedValueOnce({ available: true });
    await useVoiceStore.getState().ensureHealth();
    expect(useVoiceStore.getState().available).toBe(true);
    expect(health).toHaveBeenCalledTimes(2);
  });
});

describe('helpers', () => {
  it('names the upload after the recorded container', () => {
    expect(extensionForMime('audio/webm;codecs=opus')).toBe('webm');
    expect(extensionForMime('audio/mp4')).toBe('mp4');
    expect(extensionForMime('audio/ogg')).toBe('ogg');
  });

  it('puts an open microphone ahead of everything else', () => {
    expect(derivePresence('listening', 'speaking', true)).toBe('listening');
    expect(derivePresence('transcribing', 'idle', false)).toBe('transcribing');
    expect(derivePresence('idle', 'speaking', false)).toBe('speaking');
    expect(derivePresence('idle', 'idle', true)).toBe('thinking');
    expect(derivePresence('idle', 'idle', false)).toBe('idle');
  });
});
