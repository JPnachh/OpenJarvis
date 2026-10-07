import { create } from 'zustand';
import { authHeaders, getBase } from './api';
import { useAppStore } from './store';
import type { VoiceTurn } from './store';

/**
 * Live state of the hands-free voice assistant (the listener that runs inside
 * the Jarvis server). The server pushes events over SSE; we show the state and
 * mirror each spoken exchange into the chat history.
 */
export type VoiceState =
  | 'offline'
  | 'listening'
  | 'heard'
  | 'recording'
  | 'follow_up'
  | 'thinking'
  | 'speaking'
  | 'denied'
  | 'paused';

interface VoiceStore {
  state: VoiceState;
  connected: boolean;
  trigger: string | null;
  setConnected: (connected: boolean) => void;
  setState: (state: VoiceState, trigger?: string | null) => void;
}

export const useVoiceStore = create<VoiceStore>((set) => ({
  state: 'offline',
  connected: false,
  trigger: null,
  setConnected: (connected) => set({ connected }),
  setState: (state, trigger = null) => set({ state, trigger }),
}));

interface ServerEvent {
  type: string;
  id?: string;
  ts?: number;
  state?: string | { state?: string; trigger?: string };
  trigger?: string;
  user?: string;
  assistant?: string;
  model?: string;
  turns?: ServerEvent[];
}

function asTurn(event: ServerEvent): VoiceTurn | null {
  if (!event.id || (!event.user && !event.assistant)) return null;
  return {
    id: event.id,
    user: event.user ?? '',
    assistant: event.assistant ?? '',
    model: event.model ?? '',
    ts: event.ts ?? Date.now() / 1000,
  };
}

/** Apply one server event to the stores (exported for tests). */
export function handleVoiceEvent(event: ServerEvent): void {
  const voice = useVoiceStore.getState();
  if (event.type === 'snapshot') {
    const snap = event.state;
    if (snap && typeof snap === 'object') {
      voice.setState((snap.state as VoiceState) ?? 'offline', snap.trigger ?? null);
    }
    for (const t of event.turns ?? []) {
      const turn = asTurn(t);
      if (turn) useAppStore.getState().addVoiceTurn(turn);
    }
  } else if (event.type === 'state' && typeof event.state === 'string') {
    voice.setState(event.state as VoiceState, event.trigger ?? null);
  } else if (event.type === 'turn') {
    const turn = asTurn(event);
    if (turn) useAppStore.getState().addVoiceTurn(turn);
  }
}

/** Connect to ``/v1/voice/events`` and keep reconnecting. Returns a stopper. */
export function startVoiceStream(): () => void {
  let stopped = false;
  let controller: AbortController | null = null;

  const run = async () => {
    let delay = 1000;
    while (!stopped) {
      controller = new AbortController();
      try {
        const res = await fetch(`${getBase()}/v1/voice/events`, {
          headers: authHeaders({ Accept: 'text/event-stream' }),
          signal: controller.signal,
        });
        if (!res.ok || !res.body) throw new Error(`voice stream ${res.status}`);
        useVoiceStore.getState().setConnected(true);
        delay = 1000;
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (!stopped) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const chunks = buffer.split('\n\n');
          buffer = chunks.pop() ?? '';
          for (const chunk of chunks) {
            const line = chunk.split('\n').find((l) => l.startsWith('data: '));
            if (!line) continue; // keep-alive comment
            try {
              handleVoiceEvent(JSON.parse(line.slice(6)) as ServerEvent);
            } catch {
              /* ignore a malformed event */
            }
          }
        }
      } catch {
        /* offline or aborted: fall through to retry */
      }
      useVoiceStore.getState().setConnected(false);
      useVoiceStore.getState().setState('offline');
      if (stopped) break;
      await new Promise((r) => setTimeout(r, delay));
      delay = Math.min(delay * 2, 15000);
    }
  };

  void run();
  return () => {
    stopped = true;
    controller?.abort();
  };
}
