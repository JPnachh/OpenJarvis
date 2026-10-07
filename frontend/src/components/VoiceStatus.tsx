import { useVoiceStore } from '../lib/voice';
import type { VoiceState } from '../lib/voice';

interface Look {
  text: string;
  color: string;
  pulse?: boolean;
}

const LOOKS: Record<VoiceState, Look> = {
  offline: { text: 'Voz apagada', color: 'var(--color-text-tertiary)' },
  listening: { text: 'Escuchando · di «Hey Jarvis»', color: 'var(--color-success)' },
  heard: { text: 'Te escucho…', color: 'var(--color-accent)', pulse: true },
  recording: { text: 'Te escucho…', color: 'var(--color-accent)', pulse: true },
  follow_up: { text: 'Sigo escuchando…', color: 'var(--color-accent)', pulse: true },
  thinking: { text: 'Pensando…', color: 'var(--color-warning, #d4a017)', pulse: true },
  speaking: { text: 'Hablando…', color: 'var(--color-accent)', pulse: true },
  denied: { text: 'Voz no reconocida', color: 'var(--color-error)' },
  paused: { text: 'Escucha en pausa', color: 'var(--color-text-tertiary)' },
};

/** Small always-visible pill showing what the voice assistant is doing. */
export function VoiceStatus() {
  const state = useVoiceStore((s) => s.state);
  const connected = useVoiceStore((s) => s.connected);
  // Not connected or the listener is not running: stay out of the way.
  if (!connected || state === 'offline') return null;

  const look = LOOKS[state] ?? LOOKS.listening;
  return (
    <div
      role="status"
      aria-live="polite"
      data-voice-state={state}
      className="fixed left-1/2 -translate-x-1/2 z-40 flex items-center gap-2 px-3 py-1 rounded-full text-xs select-none"
      style={{
        top: '8px',
        background: 'var(--color-bg-secondary)',
        border: `1px solid color-mix(in srgb, ${look.color} 45%, transparent)`,
        color: 'var(--color-text)',
        boxShadow: '0 1px 6px rgba(0,0,0,0.25)',
      }}
    >
      <span
        className={`w-2 h-2 rounded-full shrink-0 ${look.pulse ? 'animate-pulse' : ''}`}
        style={{ background: look.color }}
      />
      <span>{look.text}</span>
    </div>
  );
}
