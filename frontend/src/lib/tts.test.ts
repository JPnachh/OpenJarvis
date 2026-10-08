import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./api', () => ({
  synthesizeSpeech: vi.fn(),
  fetchTtsHealth: vi.fn(),
}));

import { synthesizeSpeech, fetchTtsHealth } from './api';
import { useTtsStore, __resetTtsForTests, shouldAutoplayFinishedReply } from './tts';

const synth = vi.mocked(synthesizeSpeech);
const health = vi.mocked(fetchTtsHealth);

/** Minimal HTMLAudioElement stand-in; jsdom cannot decode or play real audio. */
class FakeAudio {
  static instances: FakeAudio[] = [];
  onended: (() => void) | null = null;
  onerror: (() => void) | null = null;
  src: string;
  paused = false;
  playCalls = 0;

  constructor(src: string) {
    this.src = src;
    FakeAudio.instances.push(this);
  }

  play(): Promise<void> {
    this.playCalls += 1;
    return Promise.resolve();
  }

  pause(): void {
    this.paused = true;
  }
}

const created: string[] = [];
const revoked: string[] = [];
let urlCounter = 0;

beforeEach(() => {
  vi.stubGlobal('Audio', FakeAudio as unknown as typeof Audio);
  vi.stubGlobal('URL', {
    createObjectURL: () => {
      const url = `blob:fake/${++urlCounter}`;
      created.push(url);
      return url;
    },
    revokeObjectURL: (url: string) => {
      revoked.push(url);
    },
  });

  synth.mockReset();
  health.mockReset();

  // Reset the store first: it tears down whatever the previous test left
  // playing, and that teardown revokes a URL. Clearing the ledgers afterwards
  // keeps that bookkeeping out of this test's assertions.
  __resetTtsForTests();
  FakeAudio.instances = [];
  created.length = 0;
  revoked.length = 0;
  urlCounter = 0;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('shared voice output', () => {
  it('plays one utterance and reports which message owns it', async () => {
    synth.mockResolvedValue(new Blob(['wav']));

    await useTtsStore.getState().speak('m1', 'Hallo');

    expect(FakeAudio.instances).toHaveLength(1);
    expect(FakeAudio.instances[0].playCalls).toBe(1);
    expect(useTtsStore.getState().state).toBe('speaking');
    expect(useTtsStore.getState().speakingId).toBe('m1');
  });

  it('starting a second message stops the first instead of overlapping it', async () => {
    synth.mockResolvedValue(new Blob(['wav']));

    await useTtsStore.getState().speak('m1', 'Erste');
    const first = FakeAudio.instances[0];

    await useTtsStore.getState().speak('m2', 'Zweite');

    expect(first.paused).toBe(true);
    expect(useTtsStore.getState().speakingId).toBe('m2');
    expect(revoked).toContain(created[0]);
  });

  it('aborts a synthesis that is superseded before it resolves', async () => {
    const pending = deferred<Blob>();
    synth.mockReturnValueOnce(pending.promise);
    synth.mockResolvedValueOnce(new Blob(['wav']));

    const firstCall = useTtsStore.getState().speak('m1', 'Erste');
    const signal = synth.mock.calls[0][1]?.signal as AbortSignal;
    expect(signal.aborted).toBe(false);

    await useTtsStore.getState().speak('m2', 'Zweite');
    expect(signal.aborted).toBe(true);

    // The stale response must not start playback or leak its blob URL.
    pending.resolve(new Blob(['stale']));
    await firstCall;

    expect(FakeAudio.instances).toHaveLength(1);
    expect(useTtsStore.getState().speakingId).toBe('m2');
  });

  it('stop() halts playback and revokes the blob URL', async () => {
    synth.mockResolvedValue(new Blob(['wav']));

    await useTtsStore.getState().speak('m1', 'Hallo');
    useTtsStore.getState().stop();

    expect(FakeAudio.instances[0].paused).toBe(true);
    expect(revoked).toEqual(created);
    expect(useTtsStore.getState().state).toBe('idle');
    expect(useTtsStore.getState().speakingId).toBeNull();
  });

  it('finishing playback leaves no error behind', async () => {
    synth.mockResolvedValue(new Blob(['wav']));

    await useTtsStore.getState().speak('m1', 'Hallo');
    const el = FakeAudio.instances[0];

    el.onended?.();
    // Clearing src fires a media error; the handler must already be detached.
    el.onerror?.();

    expect(useTtsStore.getState().error).toBeNull();
    expect(useTtsStore.getState().state).toBe('idle');
    expect(revoked).toEqual(created);
  });

  it('surfaces a synthesis failure', async () => {
    synth.mockRejectedValue(new Error('No text-to-speech backend available'));

    await useTtsStore.getState().speak('m1', 'Hallo');

    expect(useTtsStore.getState().error).toBe('No text-to-speech backend available');
    expect(useTtsStore.getState().errorId).toBe('m1');
    expect(useTtsStore.getState().state).toBe('idle');
  });

  it('ignores empty text', async () => {
    await useTtsStore.getState().speak('m1', '   ');
    expect(synth).not.toHaveBeenCalled();
  });

  it('probes the backend once no matter how many callers ask', () => {
    health.mockResolvedValue({ available: true });

    useTtsStore.getState().ensureHealth();
    useTtsStore.getState().ensureHealth();
    useTtsStore.getState().ensureHealth();

    expect(health).toHaveBeenCalledTimes(1);
  });

  it('shares a pending health probe so autoplay can wait for model loading', async () => {
    const pending = deferred<{ available: boolean }>();
    health.mockReturnValue(pending.promise);

    const first = useTtsStore.getState().ensureHealth();
    const second = useTtsStore.getState().ensureHealth();
    expect(second).toBe(first);
    expect(health).toHaveBeenCalledTimes(1);

    pending.resolve({ available: true });
    await first;
    expect(useTtsStore.getState().available).toBe(true);
  });

  it('retries health after an unavailable response', async () => {
    health.mockResolvedValueOnce({ available: false });
    health.mockResolvedValueOnce({ available: true });

    await useTtsStore.getState().ensureHealth();
    expect(useTtsStore.getState().available).toBe(false);
    await useTtsStore.getState().ensureHealth();

    expect(health).toHaveBeenCalledTimes(2);
    expect(useTtsStore.getState().available).toBe(true);
  });

  it('retries health after a network error', async () => {
    health.mockRejectedValueOnce(new Error('offline'));
    health.mockResolvedValueOnce({ available: true });

    await useTtsStore.getState().ensureHealth();
    expect(useTtsStore.getState().available).toBe(false);
    await useTtsStore.getState().ensureHealth();

    expect(health).toHaveBeenCalledTimes(2);
    expect(useTtsStore.getState().available).toBe(true);
  });
});

describe('autoplay gating', () => {
  it('stays silent when a finished conversation is merely restored', () => {
    expect(
      shouldAutoplayFinishedReply(null, 'chat-1', false, { role: 'assistant', id: 'old' }, null),
    ).toBe(false);
  });

  it('speaks when a stream finishes', () => {
    expect(
      shouldAutoplayFinishedReply('chat-1', 'chat-1', false, { role: 'assistant', id: 'fresh' }, null),
    ).toBe(true);
  });

  it('stays silent while the reply is still streaming', () => {
    expect(
      shouldAutoplayFinishedReply('chat-1', 'chat-1', true, { role: 'assistant', id: 'fresh' }, null),
    ).toBe(false);
  });

  it('never repeats a message it already spoke', () => {
    expect(
      shouldAutoplayFinishedReply('chat-1', 'chat-1', false, { role: 'assistant', id: 'fresh' }, 'fresh'),
    ).toBe(false);
  });

  it('ignores a trailing user message', () => {
    expect(
      shouldAutoplayFinishedReply('chat-1', 'chat-1', false, { role: 'user', id: 'u1' }, null),
    ).toBe(false);
  });

  it('does not speak an old reply when the user switches conversations mid-stream', () => {
    expect(
      shouldAutoplayFinishedReply('chat-1', 'chat-2', true, { role: 'assistant', id: 'old' }, null),
    ).toBe(false);
  });
});

describe('built-in computer voice fallback', () => {
  class FakeUtterance {
    voice: unknown = null;
    lang = '';
    onend: (() => void) | null = null;
    onerror: ((e: { error: string }) => void) | null = null;
    constructor(public text: string) {}
  }
  const spoken: FakeUtterance[] = [];
  let cancels = 0;

  beforeEach(() => {
    spoken.length = 0;
    cancels = 0;
    const speechSynthesis = {
      speak: (u: FakeUtterance) => spoken.push(u),
      cancel: () => {
        cancels += 1;
      },
      getVoices: () => [
        { name: 'Microsoft David', lang: 'en-US', localService: true },
        { name: 'Microsoft Sabina', lang: 'es-MX', localService: true },
      ],
    };
    vi.stubGlobal('window', { speechSynthesis, SpeechSynthesisUtterance: FakeUtterance });
    vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance);
  });

  it('is used when the server has no TTS backend', async () => {
    health.mockResolvedValue({ available: false });
    await useTtsStore.getState().ensureHealth();
    expect(useTtsStore.getState().available).toBe(true);
    expect(useTtsStore.getState().engine).toBe('browser');

    await useTtsStore.getState().speak('m1', '**Hola**, ¿qué tal? Mira [esto](https://x.y)');
    expect(synth).not.toHaveBeenCalled();
    expect(spoken).toHaveLength(1);
    expect(spoken[0].text).toBe('Hola, ¿qué tal? Mira esto');
    expect(spoken[0].lang).toBe('es-MX'); // Spanish text, Spanish voice
    expect(useTtsStore.getState().state).toBe('speaking');

    spoken[0].onend?.();
    expect(useTtsStore.getState().state).toBe('idle');
  });

  it('takes over when the server fails to synthesize', async () => {
    health.mockResolvedValue({ available: true });
    await useTtsStore.getState().ensureHealth();
    synth.mockRejectedValue(new Error('No text-to-speech backend available'));

    await useTtsStore.getState().speak('m1', 'Hello there');
    expect(spoken).toHaveLength(1);
    expect(spoken[0].lang).toBe('en-US');
    expect(useTtsStore.getState().engine).toBe('browser');
    expect(useTtsStore.getState().error).toBeNull();
  });

  it('stop cancels the browser voice', async () => {
    health.mockResolvedValue({ available: false });
    await useTtsStore.getState().ensureHealth();
    await useTtsStore.getState().speak('m1', 'Hola');
    const before = cancels;
    useTtsStore.getState().stop();
    expect(cancels).toBeGreaterThan(before);
    expect(useTtsStore.getState().state).toBe('idle');
  });
});

describe('speakableText', () => {
  it('drops markdown that would be read out as symbols', async () => {
    const { speakableText } = await import('./tts');
    expect(speakableText('# Título\n- uno\n- **dos**\n```js\ncode()\n```\nVer `x`')).toBe(
      'Título uno dos Ver x',
    );
  });
});
