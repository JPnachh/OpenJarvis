import { apiFetch } from './api';
import { errorDetail } from './voice-api';
import { generateId, useAppStore } from './store';
import type { ChatMessage } from '../types';

export interface ActionRun {
  handled: boolean;
  reply?: string;
  ok?: boolean;
  intent?: string;
}

/** Ask the server to run *text* as a direct command (volume, music, apps…). */
export async function runAction(text: string): Promise<ActionRun> {
  const res = await apiFetch('/v1/actions/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) return { handled: false };
  return res.json();
}

/**
 * Show a command and Jarvis's answer in the active chat. Direct commands do
 * not go through the model, so this is the only record of them.
 */
export function recordActionExchange(text: string, reply: string): void {
  const app = useAppStore.getState();
  const convId = app.activeId ?? app.createConversation(app.selectedModel || 'jarvis');
  const now = Date.now();
  const user: ChatMessage = { id: generateId(), role: 'user', content: text, timestamp: now };
  const assistant: ChatMessage = {
    id: generateId(),
    role: 'assistant',
    content: reply,
    timestamp: now + 1,
    telemetry: { engine: 'jarvis actions', model_id: 'direct command', total_ms: 0 },
  };
  app.addMessage(convId, user);
  app.addMessage(convId, assistant);
}

// ---------------------------------------------------------------------------
// User-taught commands
// ---------------------------------------------------------------------------

export type CustomAction = 'open' | 'spotify' | 'keys' | 'say' | 'shell';

export interface CustomCommand {
  id: string;
  phrases: string[];
  action: CustomAction;
  target: string;
  reply: string;
}

export interface CommandsInfo {
  path: string;
  allow_shell: boolean;
  actions: CustomAction[];
  keys: string[];
  commands: CustomCommand[];
  examples: Array<{ group: string; phrases: string[] }>;
}

const detail = errorDetail;

export async function fetchCommands(): Promise<CommandsInfo> {
  const res = await apiFetch('/v1/actions/commands');
  if (!res.ok) throw new Error(await detail(res, 'Could not load commands'));
  return res.json();
}

export async function addCommand(command: Omit<CustomCommand, 'id'>): Promise<void> {
  const res = await apiFetch('/v1/actions/commands', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(command),
  });
  if (!res.ok) throw new Error(await detail(res, 'Could not save the command'));
}

export async function deleteCommand(id: string): Promise<void> {
  const res = await apiFetch(`/v1/actions/commands/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(await detail(res, 'Could not delete the command'));
}
