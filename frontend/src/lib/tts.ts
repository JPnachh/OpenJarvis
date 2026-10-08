import { create } from 'zustand';
import { synthesizeSpeech, fetchTtsHealth } from './api';

export type TtsState = 'idle' | 'loading' | 'speaking';

/** Which engine speaks: the server's TTS backend, or the browser's own voices. */
export type TtsEngine = 'server' | 'browser';

/**
 * The browser (and on Windows, the system) ships voices of its own. They are
 * the fallback whenever the server has no TTS backend installed, so replies
 * can always be read aloud.
 */
export function browserTtsAvailable(): boolean {
  return (
    typeof window !== 'undefined'
    && 'speechSynthesis' in window
    && typeof (window as unknown as { SpeechSynthesisUtterance?: unknown }).SpeechSynthesisUtterance !== 'undefined'
  );
}

const SPANISH_HINT = /[áéíóúñ¿¡]|\b(el|la|los|las|que|de|es|una?|por|para|con|hola|gracias)\b/i;

/** Turn markdown into something worth hearing: no asterisks, code or links. */
export function speakableText(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/[*_~>|#]+/g, '')
    .replace(/https?:\/\/\S+/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function pickVoice(text: string): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return undefined;
  const lang = SPANISH_HINT.test(text) ? 'es' : 'en';
  const matching = voices.filter((v) => v.lang.toLowerCase().startsWith(lang));
  return (
    matching.find((v) => v.localService && /natural|neural|online/i.test(v.name))
    ?? matching.find((v) => v.localService)
    ?? matching[0]
  );
}

/**
 * Voice output is a single shared resource: one utterance at a time, for the
 * whole app. The state lives in one store rather than per component, because a
 * per-component hook would give every message its own audio element -- two
 * replies would then talk over each other, and autoplay would make that the
 * normal case rather than the exception.
 */
interface TtsStore {
  state: TtsState;
  /** id of the message currently loading or speaking, if any. */
  speakingId: string | null;
  error: string | null;
  /** Message whose read-aloud attempt failed, if any. */
  errorId: string | null;
  /** null until the health probe has answered. */
  available: boolean | null;
  /** Which engine speaks once available. */
  engine: TtsEngine | null;
  /** Last message spoken by autoplay, so a re-render never repeats it. */
  autoSpokenId: string | null;
  speak: (id: string, text: string) => Promise<void>;
  stop: () => void;
  ensureHealth: () => Promise<void>;
  markAutoSpoken: (id: string) => void;
}

// Playback handles are not render state -- keeping them out of the store avoids
// re-rendering every subscriber when an audio element is swapped.
let audio: HTMLAudioElement | null = null;
let objectUrl: string | null = null;
let controller: AbortController | null = null;
let token = 0;
let healthProbe: Promise<void> | null = null;
let browserUtterance: SpeechSynthesisUtterance | null = null;

/** Only a stream ending in the active conversation may trigger autoplay. */
export function shouldAutoplayFinishedReply(
  previousStreamingConversationId: string | null,
  activeId: string | null,
  streamIsActive: boolean,
  lastMessage: { id: string; role: string } | undefined,
  autoSpokenId: string | null,
): boolean {
  return previousStreamingConversationId !== null
    && previousStreamingConversationId === activeId
    && !streamIsActive
    && lastMessage !== undefined
    && lastMessage.role === 'assistant'
    && lastMessage.id !== autoSpokenId;
}

function teardown(): void {
  if (browserUtterance) {
    browserUtterance.onend = null;
    browserUtterance.onerror = null;
    browserUtterance = null;
    try {
      window.speechSynthesis.cancel();
    } catch {
      // nothing to cancel
    }
  }
  if (audio) {
    // Detach first: clearing src re-runs the media load algorithm, which fails
    // on an empty source and dispatches an `error` event. With the handler
    // still attached that surfaces as a bogus "Playback failed" after every
    // successful utterance.
    audio.onended = null;
    audio.onerror = null;
    audio.pause();
    audio.src = '';
    audio = null;
  }
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  if (controller) {
    controller.abort();
    controller = null;
  }
}

export const useTtsStore = create<TtsStore>((set, get) => ({
  state: 'idle',
  speakingId: null,
  error: null,
  errorId: null,
  available: null,
  engine: null,
  autoSpokenId: null,

  ensureHealth: () => {
    if (healthProbe) return healthProbe;
    const fallback = () => {
      const browser = browserTtsAvailable();
      set({ available: browser, engine: browser ? 'browser' : null });
      // Keep probing: the server may get a backend (or finish starting).
      healthProbe = null;
    };
    healthProbe = fetchTtsHealth()
      .then((health) => {
        if (health.available) set({ available: true, engine: 'server' });
        else fallback();
      })
      .catch(fallback);
    return healthProbe;
  },

  markAutoSpoken: (id: string) => set({ autoSpokenId: id }),

  speak: async (id: string, text: string) => {
    const trimmed = speakableText(text);
    if (!trimmed) return;

    // Bump before teardown so a synthesis still in flight is both aborted and
    // fenced off by the token, even if the abort loses the race.
    token += 1;
    const mine = token;
    teardown();
    set({ state: 'loading', speakingId: id, error: null, errorId: null });

    const speakInBrowser = () => {
      const utterance = new SpeechSynthesisUtterance(trimmed);
      const voice = pickVoice(trimmed);
      if (voice) {
        utterance.voice = voice;
        utterance.lang = voice.lang;
      }
      utterance.onend = () => {
        if (mine !== token) return;
        browserUtterance = null;
        set({ state: 'idle', speakingId: null });
      };
      utterance.onerror = (event) => {
        if (mine !== token) return;
        browserUtterance = null;
        const cancelled = event.error === 'interrupted' || event.error === 'canceled';
        set({
          state: 'idle',
          speakingId: null,
          ...(cancelled ? {} : { error: 'Playback failed', errorId: id }),
        });
      };
      browserUtterance = utterance;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
      set({ state: 'speaking', speakingId: id });
    };

    if (get().engine === 'browser') {
      speakInBrowser();
      return;
    }

    const ac = new AbortController();
    controller = ac;

    try {
      const blob = await synthesizeSpeech(trimmed, { signal: ac.signal });
      if (mine !== token) return;

      const url = URL.createObjectURL(blob);
      objectUrl = url;
      const el = new Audio(url);
      audio = el;

      el.onended = () => {
        if (mine !== token) return;
        teardown();
        set({ state: 'idle', speakingId: null });
      };
      el.onerror = () => {
        if (mine !== token) return;
        teardown();
        set({ state: 'idle', speakingId: null, error: 'Playback failed', errorId: id });
      };

      await el.play();
      if (mine === token) set({ state: 'speaking', speakingId: id });
    } catch (err) {
      if (mine !== token) return;
      if (err instanceof DOMException && err.name === 'AbortError') return;
      teardown();
      // The server could not speak (no backend, synthesis error): use the
      // browser's own voice rather than staying silent.
      if (browserTtsAvailable()) {
        set({ engine: 'browser' });
        speakInBrowser();
        return;
      }
      set({
        state: 'idle',
        speakingId: null,
        error: err instanceof Error ? err.message : 'Speech synthesis failed',
        errorId: id,
      });
    }
  },

  stop: () => {
    token += 1;
    teardown();
    set({ state: 'idle', speakingId: null, error: null, errorId: null });
  },
}));

/** Test seam: reset module-level playback handles between cases. */
export function __resetTtsForTests(): void {
  teardown();
  token = 0;
  healthProbe = null;
  useTtsStore.setState({
    state: 'idle',
    speakingId: null,
    error: null,
    errorId: null,
    available: null,
    engine: null,
    autoSpokenId: null,
  });
}
