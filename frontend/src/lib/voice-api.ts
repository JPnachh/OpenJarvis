import { apiFetch, authHeaders, getBase } from './api';

// ---------------------------------------------------------------------------
// Hands-free voice: events from the background listener (`jarvis listen`)
// ---------------------------------------------------------------------------

export type ListenerState =
  | 'idle'
  | 'listening'
  | 'transcribing'
  | 'thinking'
  | 'speaking'
  | 'offline';

export type VoiceEvent =
  | { type: 'hello'; states: Record<string, { state: ListenerState; detail?: string | null }> }
  | { type: 'state'; source: 'ui' | 'listener'; state: ListenerState; detail?: string | null }
  | { type: 'command'; id: string; text: string; source: string };

/**
 * Follow /v1/voice/events until aborted. Uses fetch rather than EventSource so
 * the Bearer token is sent when the server has an API key.
 */
export async function streamVoiceEvents(
  onEvent: (event: VoiceEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`${getBase()}/v1/voice/events`, {
    headers: authHeaders({ Accept: 'text/event-stream' }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`Voice events unavailable: ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let split: number;
    while ((split = buffer.indexOf('\n\n')) !== -1) {
      const chunk = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      for (const line of chunk.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        try {
          onEvent(JSON.parse(line.slice(6)) as VoiceEvent);
        } catch {
          // Ignore a malformed frame rather than dropping the stream.
        }
      }
    }
  }
}

export async function postUiVoiceState(state: ListenerState): Promise<void> {
  await apiFetch('/v1/voice/state', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: 'ui', state }),
  });
}

export async function postVoiceReply(id: string, text: string): Promise<void> {
  await apiFetch('/v1/voice/reply', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, text }),
  });
}

// ---------------------------------------------------------------------------
// Wake word training
// ---------------------------------------------------------------------------

export interface WakeWordStats {
  positives: number;
  negatives: number;
  quality: 'good' | 'fair' | 'poor' | 'untested';
  advice: string;
  margin: number | null;
}

export interface WakeWordStatus {
  directory: string;
  profile_path: string;
  positives: number;
  negatives: number;
  min_positives: number;
  recommended_positives: number;
  trained: boolean;
  phrase: string;
  trained_at: number | null;
  stats: WakeWordStats | null;
  samples: Array<{ id: string; kind: 'positive' | 'negative'; seconds: number }>;
}

async function errorDetail(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    if (body?.detail) return String(body.detail);
  } catch {
    // fall through
  }
  return `${fallback} (${res.status})`;
}

export async function fetchWakeWordStatus(): Promise<WakeWordStatus> {
  const res = await apiFetch('/v1/wakeword');
  if (!res.ok) throw new Error(await errorDetail(res, 'Could not load wake word status'));
  return res.json();
}

export async function uploadWakeWordSample(
  wav: Blob,
  kind: 'positive' | 'negative',
): Promise<{ id: string; seconds: number }> {
  const form = new FormData();
  form.append('kind', kind);
  form.append('file', wav, 'sample.wav');
  const res = await apiFetch('/v1/wakeword/samples', { method: 'POST', body: form });
  if (!res.ok) throw new Error(await errorDetail(res, 'Could not save the recording'));
  return res.json();
}

export async function deleteWakeWordSample(id: string): Promise<void> {
  const res = await apiFetch(`/v1/wakeword/samples/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(await errorDetail(res, 'Could not delete the recording'));
}

export async function fetchWakeWordSampleAudio(id: string): Promise<Blob> {
  const res = await apiFetch(`/v1/wakeword/samples/${encodeURIComponent(id)}/audio`);
  if (!res.ok) throw new Error(await errorDetail(res, 'Could not load the recording'));
  return res.blob();
}

export async function trainWakeWord(
  phrase: string,
): Promise<{ phrase: string; threshold: number; stats: WakeWordStats }> {
  const res = await apiFetch('/v1/wakeword/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ phrase }),
  });
  if (!res.ok) throw new Error(await errorDetail(res, 'Training failed'));
  return res.json();
}
