import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

class MemoryStorage {
  private store = new Map<string, string>();
  getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.store.set(key, String(value));
  }
  removeItem(key: string): void {
    this.store.delete(key);
  }
}

beforeEach(() => {
  vi.resetModules();
  (globalThis as unknown as { localStorage: MemoryStorage }).localStorage =
    new MemoryStorage();
});

afterEach(() => {
  (globalThis as unknown as { localStorage?: MemoryStorage }).localStorage =
    undefined;
});

async function fresh() {
  const store = (await import('./store')).useAppStore;
  const voice = await import('./voice');
  return { store, ...voice };
}

const turn = (id: string, user: string, assistant: string) => ({
  type: 'turn',
  id,
  ts: 1_700_000_000,
  user,
  assistant,
  model: 'qwen3:8b',
});

describe('voice events', () => {
  it('state events update the pill state', async () => {
    const { useVoiceStore, handleVoiceEvent } = await fresh();
    handleVoiceEvent({ type: 'state', state: 'thinking' });
    expect(useVoiceStore.getState().state).toBe('thinking');
    handleVoiceEvent({ type: 'state', state: 'heard', trigger: 'Hey Jarvis' });
    expect(useVoiceStore.getState().trigger).toBe('Hey Jarvis');
  });

  it('mirrors a spoken turn into a dedicated voice conversation', async () => {
    const { store, handleVoiceEvent } = await fresh();
    handleVoiceEvent(turn('a-1', '¿qué es la fotosíntesis?', 'Es el proceso…'));
    const conv = store
      .getState()
      .conversations.find((c) => c.title.includes('voz'));
    expect(conv).toBeDefined();
    expect(conv!.messages.map((m) => m.role)).toEqual(['user', 'assistant']);
    expect(conv!.messages[1].telemetry?.model_id).toBe('qwen3:8b');
  });

  it('ignores a turn delivered twice (second tab / reconnect snapshot)', async () => {
    const { store, handleVoiceEvent } = await fresh();
    handleVoiceEvent(turn('a-1', 'hola', 'hola'));
    handleVoiceEvent(turn('a-1', 'hola', 'hola'));
    handleVoiceEvent({
      type: 'snapshot',
      state: { state: 'listening' },
      turns: [turn('a-1', 'hola', 'hola'), turn('a-2', 'otra', 'ok')],
    });
    const conv = store.getState().conversations.find((c) => c.title.includes('voz'))!;
    expect(conv.messages.map((m) => m.id)).toEqual([
      'a-1-u',
      'a-1-a',
      'a-2-u',
      'a-2-a',
    ]);
  });

  it('does not steal the chat the user is looking at', async () => {
    const { store, handleVoiceEvent } = await fresh();
    const mine = store.getState().createConversation('m');
    handleVoiceEvent(turn('a-1', 'hola', 'hola'));
    expect(store.getState().activeId).toBe(mine);
    expect(store.getState().messages).toEqual([]);
  });
});
