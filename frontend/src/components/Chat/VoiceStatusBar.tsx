import { useEffect, useState } from 'react';
import { Check, Square, X } from 'lucide-react';
import { JarvisOrb } from '../Jarvis/JarvisOrb';
import { useVoiceStore, type StopReason } from '../../lib/voice';
import { useTtsStore } from '../../lib/tts';
import { useAppStore } from '../../lib/store';

// How long the "stopped listening" confirmation stays on screen.
const STOP_NOTICE_MS = 2500;

const STOP_MESSAGE: Partial<Record<StopReason, string>> = {
  manual: 'Stopped listening',
  silence: 'Stopped listening — you paused',
  'max-duration': 'Stopped listening — time limit reached',
  'no-speech': "Stopped listening — didn't hear anything",
  cancelled: 'Listening cancelled',
  'device-lost': 'Stopped listening — microphone disconnected',
  error: 'Stopped listening — recording failed',
};

function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

function LevelMeter() {
  const level = useVoiceStore((s) => s.level);
  const bars = [0.55, 0.8, 1, 0.8, 0.55];
  return (
    <span className="voice-meter" aria-hidden="true">
      {bars.map((weight, i) => (
        <span key={i} style={{ height: `${Math.max(3, Math.min(16, 3 + level * weight * 16))}px` }} />
      ))}
    </span>
  );
}

/**
 * Always-visible answer to "is Jarvis listening right now?": shows while the
 * microphone is open, while the recording is transcribed, while Jarvis talks,
 * and briefly after listening stops, saying why it stopped.
 */
export function VoiceStatusBar() {
  const phase = useVoiceStore((s) => s.phase);
  const startedAt = useVoiceStore((s) => s.startedAt);
  const heardSpeech = useVoiceStore((s) => s.heardSpeech);
  const lastStop = useVoiceStore((s) => s.lastStop);
  const ttsState = useTtsStore((s) => s.state);
  const autoStop = useAppStore((s) => s.settings.voiceAutoStop);
  const [now, setNow] = useState(() => Date.now());

  const listening = phase === 'listening';
  const showStopNotice =
    phase === 'idle' && lastStop !== null && now - lastStop.at < STOP_NOTICE_MS;

  // Tick the elapsed timer while listening.
  useEffect(() => {
    if (!listening) return;
    setNow(Date.now());
    const interval = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(interval);
  }, [listening]);

  // Re-render once when the stop notice should disappear.
  useEffect(() => {
    if (phase !== 'idle' || !lastStop) return;
    setNow(Date.now());
    const remaining = lastStop.at + STOP_NOTICE_MS - Date.now();
    if (remaining <= 0) return;
    const timer = setTimeout(() => setNow(Date.now()), remaining + 20);
    return () => clearTimeout(timer);
  }, [phase, lastStop]);

  // Esc cancels listening from anywhere on the page.
  useEffect(() => {
    if (phase !== 'listening' && phase !== 'requesting') return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') useVoiceStore.getState().cancel();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [phase]);

  // Mark the browser tab too, so an open mic is visible from other tabs.
  useEffect(() => {
    if (phase !== 'listening') return;
    const original = document.title;
    document.title = `● Listening — ${original}`;
    return () => {
      document.title = original;
    };
  }, [phase]);

  let content: React.ReactNode = null;
  let tone = 'var(--color-text-secondary)';
  let announce = '';

  if (phase === 'requesting') {
    announce = 'Waiting for microphone permission';
    content = <span>Waiting for microphone permission… allow it in the browser prompt.</span>;
  } else if (listening) {
    tone = 'var(--color-error)';
    announce = 'Listening';
    content = (
      <>
        <span className="font-medium">Listening</span>
        <span className="font-mono tabular-nums">{formatElapsed(now - (startedAt ?? now))}</span>
        <LevelMeter />
        <span className="hidden sm:inline" style={{ color: 'var(--color-text-tertiary)' }}>
          {!heardSpeech
            ? 'Go ahead, I’m listening…'
            : autoStop
              ? 'Stops when you pause · Esc to cancel'
              : 'Click Done when finished · Esc to cancel'}
        </span>
        <span className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={() => useVoiceStore.getState().stop('manual')}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs cursor-pointer"
            style={{ background: 'var(--color-error)', color: 'white' }}
          >
            <Check size={12} /> Done
          </button>
          <button
            type="button"
            onClick={() => useVoiceStore.getState().cancel()}
            className="p-1 rounded-md cursor-pointer"
            style={{ color: 'var(--color-text-tertiary)' }}
            title="Cancel (Esc)"
            aria-label="Cancel listening"
          >
            <X size={14} />
          </button>
        </span>
      </>
    );
  } else if (phase === 'transcribing') {
    tone = 'var(--color-warning)';
    announce = 'Stopped listening. Transcribing';
    content = <span>Stopped listening · transcribing what you said…</span>;
  } else if (ttsState === 'loading' || ttsState === 'speaking') {
    tone = 'var(--color-success)';
    announce = ttsState === 'speaking' ? 'Jarvis is speaking' : 'Preparing voice';
    content = (
      <>
        <span>{ttsState === 'speaking' ? 'Jarvis is speaking' : 'Preparing voice…'}</span>
        <button
          type="button"
          onClick={() => useTtsStore.getState().stop()}
          className="ml-auto inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs cursor-pointer"
          style={{ border: '1px solid var(--color-border)', color: 'var(--color-text-secondary)' }}
        >
          <Square size={10} /> Stop
        </button>
      </>
    );
  } else if (showStopNotice && lastStop) {
    announce = STOP_MESSAGE[lastStop.reason] ?? 'Stopped listening';
    content = <span>{announce}</span>;
  }

  return (
    <>
      <div className="sr-only" role="status" aria-live="polite">
        {announce}
      </div>
      {content && (
        <div
          className="mb-2 flex items-center gap-2.5 rounded-xl px-3 py-1.5 text-xs"
          style={{
            color: tone,
            background: `color-mix(in srgb, ${tone} 8%, transparent)`,
            border: `1px solid color-mix(in srgb, ${tone} 25%, transparent)`,
          }}
          data-testid="voice-status"
        >
          <JarvisOrb size={22} />
          {content}
        </div>
      )}
    </>
  );
}
