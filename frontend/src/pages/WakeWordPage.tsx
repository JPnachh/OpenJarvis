import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, Copy, Mic, Play, Sparkles, Trash2 } from 'lucide-react';
import { toast } from 'sonner';
import { JarvisOrb } from '../components/Jarvis/JarvisOrb';
import {
  deleteWakeWordSample,
  fetchWakeWordSampleAudio,
  fetchWakeWordStatus,
  trainWakeWord,
  uploadWakeWordSample,
  type WakeWordStats,
  type WakeWordStatus,
  OUTDATED_SERVER_MESSAGE,
} from '../lib/voice-api';
import { recordClip } from '../lib/wav-recorder';
import { useVoiceStore } from '../lib/voice';

type Kind = 'positive' | 'negative';

const NEGATIVE_IDEAS = [
  'What time is it?',
  'Hey there, how are you?',
  'Hey Travis',
  'Turn on the lights',
];

const QUALITY_COLOR: Record<WakeWordStats['quality'], string> = {
  good: 'var(--color-success)',
  fair: 'var(--color-warning)',
  poor: 'var(--color-error)',
  untested: 'var(--color-text-tertiary)',
};

const LISTEN_COMMAND = 'uv run --extra desktop jarvis listen --autostart';
const TEST_COMMAND = 'uv run --extra desktop jarvis listen --test';

function Card({ step, title, children }: { step: string; title: string; children: React.ReactNode }) {
  return (
    <section
      className="rounded-xl p-5 mb-4"
      style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
    >
      <div className="flex items-baseline gap-2 mb-3">
        <span className="text-[11px] font-mono" style={{ color: 'var(--color-accent)' }}>{step}</span>
        <h2 className="text-sm font-semibold" style={{ color: 'var(--color-text)' }}>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function CommandLine({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div
      className="flex items-center gap-2 rounded-lg px-3 py-2 font-mono text-xs"
      style={{ background: 'var(--color-bg)', border: '1px solid var(--color-border)', color: 'var(--color-text)' }}
    >
      <span className="flex-1 overflow-x-auto whitespace-nowrap">{command}</span>
      <button
        type="button"
        className="p-1 rounded cursor-pointer shrink-0"
        style={{ color: 'var(--color-text-tertiary)' }}
        aria-label="Copy command"
        onClick={() => {
          void navigator.clipboard?.writeText(command).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          });
        }}
      >
        {copied ? <Check size={13} /> : <Copy size={13} />}
      </button>
    </div>
  );
}

export function WakeWordPage() {
  const [status, setStatus] = useState<WakeWordStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [recording, setRecording] = useState<Kind | null>(null);
  const [level, setLevel] = useState(0);
  const [training, setTraining] = useState(false);
  const [phrase, setPhrase] = useState('Hey Jarvis');
  const listener = useVoiceStore((s) => s.listener);
  const listenerRunning = listener !== null && listener.state !== 'offline';
  const abortRef = useRef<AbortController | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await fetchWakeWordStatus();
      setStatus(next);
      setLoadError(null);
      return next;
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Could not reach OpenJarvis');
      return null;
    }
  }, []);

  useEffect(() => {
    void refresh().then((s) => {
      if (s?.trained) setPhrase(s.phrase);
    });
    return () => abortRef.current?.abort();
  }, [refresh]);

  const record = async (kind: Kind) => {
    if (recording) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setRecording(kind);
    try {
      const wav = await recordClip({
        maxMs: kind === 'positive' ? 3000 : 8000,
        silenceMs: kind === 'positive' ? 700 : 1000,
        onLevel: setLevel,
        signal: controller.signal,
      });
      const saved = await uploadWakeWordSample(wav, kind);
      toast.success(`Saved (${saved.seconds.toFixed(1)}s)`);
      await refresh();
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        const name = err instanceof DOMException ? err.name : '';
        toast.error(
          name === 'NotAllowedError'
            ? 'Microphone access was blocked. Allow it in the address bar.'
            : err instanceof Error ? err.message : 'Recording failed',
        );
      }
    } finally {
      setRecording(null);
      setLevel(0);
      abortRef.current = null;
    }
  };

  const play = async (id: string) => {
    try {
      const blob = await fetchWakeWordSampleAudio(id);
      audioRef.current?.pause();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.onended = () => URL.revokeObjectURL(url);
      audioRef.current = audio;
      await audio.play();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Playback failed');
    }
  };

  const remove = async (id: string) => {
    try {
      await deleteWakeWordSample(id);
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Delete failed');
    }
  };

  const train = async () => {
    setTraining(true);
    try {
      const result = await trainWakeWord(phrase.trim() || 'Hey Jarvis');
      toast.success(`Trained · quality: ${result.stats.quality}`);
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Training failed');
    } finally {
      setTraining(false);
    }
  };

  const positives = status?.samples.filter((s) => s.kind === 'positive') ?? [];
  const negatives = status?.samples.filter((s) => s.kind === 'negative') ?? [];
  const minPositives = status?.min_positives ?? 3;
  const recommended = status?.recommended_positives ?? 6;
  const stats = status?.stats;

  const recordButton = (kind: Kind, label: string) => {
    const active = recording === kind;
    return (
      <button
        type="button"
        onClick={() => (active ? abortRef.current?.abort() : void record(kind))}
        disabled={recording !== null && !active}
        className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium cursor-pointer disabled:opacity-40 disabled:cursor-default"
        style={{
          background: active ? 'var(--color-error)' : 'var(--color-accent)',
          color: 'white',
        }}
      >
        <Mic size={15} />
        {active ? 'Listening… (click to cancel)' : label}
        {active && (
          <span className="voice-meter" aria-hidden="true">
            {[0.6, 1, 0.6].map((w, i) => (
              <span key={i} style={{ height: `${Math.max(3, Math.min(16, 3 + level * w * 16))}px` }} />
            ))}
          </span>
        )}
      </button>
    );
  };

  const sampleList = (items: typeof positives) => (
    <ul className="mt-3 flex flex-wrap gap-2">
      {items.map((s, i) => (
        <li
          key={s.id}
          className="flex items-center gap-1 rounded-lg pl-2.5 pr-1 py-1 text-xs"
          style={{ background: 'var(--color-bg)', border: '1px solid var(--color-border)', color: 'var(--color-text-secondary)' }}
        >
          #{i + 1} · {s.seconds.toFixed(1)}s
          {s.id.startsWith('auto-') && (
            <span title="Learned automatically from a 'Hey Jarvis' that worked" style={{ color: 'var(--color-accent)' }}>
              · learned
            </span>
          )}
          <button type="button" className="p-1 cursor-pointer" aria-label="Play" onClick={() => void play(s.id)}>
            <Play size={12} />
          </button>
          <button type="button" className="p-1 cursor-pointer" aria-label="Delete" onClick={() => void remove(s.id)}>
            <Trash2 size={12} />
          </button>
        </li>
      ))}
    </ul>
  );

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-10">
      <div className="max-w-3xl mx-auto w-full">
        <header className="flex items-center gap-4 mb-6">
          <JarvisOrb size={56} presence={recording ? 'listening' : training ? 'thinking' : 'idle'} />
          <div>
            <h1 className="text-lg font-semibold" style={{ color: 'var(--color-text)' }}>
              Train “Hey Jarvis” with your voice
            </h1>
            <p className="text-sm mt-1" style={{ color: 'var(--color-text-secondary)' }}>
              Record the phrase a few times. Jarvis learns how <em>you</em> say it — any language or
              accent — and then answers to it hands-free. Everything stays on this computer.
            </p>
          </div>
        </header>

        {loadError && (
          <div className="mb-4 rounded-lg px-4 py-3 text-sm" style={{ background: 'color-mix(in srgb, var(--color-error) 8%, transparent)', color: 'var(--color-text)' }}>
            {loadError === OUTDATED_SERVER_MESSAGE ? (
              loadError
            ) : (
              <>
                {loadError}. Is the OpenJarvis server running (<code>jarvis gui</code>)?
              </>
            )}
          </div>
        )}

        <Card step="1" title={`Say “${phrase || 'Hey Jarvis'}” — ${positives.length}/${recommended}`}>
          <div className="flex flex-wrap items-center gap-3 mb-2">
            <label className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
              Phrase
              <input
                value={phrase}
                onChange={(e) => setPhrase(e.target.value)}
                maxLength={60}
                className="ml-2 px-2 py-1 rounded-md text-sm outline-none"
                style={{ background: 'var(--color-bg)', border: '1px solid var(--color-border)', color: 'var(--color-text)' }}
              />
            </label>
          </div>
          <p className="text-xs mb-3" style={{ color: 'var(--color-text-tertiary)' }}>
            Click, say the phrase once the way you normally would, and pause — recording stops on its
            own. Do it at least {minPositives} times ({recommended}+ is best), from where you usually
            sit. You can use your own phrase, e.g. “Oye Jarvis”. Jarvis also keeps learning: each
            time it recognises you and you ask something, that take is added (marked “learned”).
          </p>
          {recordButton('positive', positives.length ? 'Record another' : 'Record')}
          {positives.length > 0 && sampleList(positives)}
        </Card>

        <Card step="2" title={`Say other things — ${negatives.length} recorded (recommended: 3)`}>
          <p className="text-xs mb-3" style={{ color: 'var(--color-text-tertiary)' }}>
            Teach Jarvis what is <strong>not</strong> the wake word, so it does not wake up on
            ordinary talk. Ideas: {NEGATIVE_IDEAS.map((idea) => `“${idea}”`).join(', ')}.
          </p>
          {recordButton('negative', 'Record a different sentence')}
          {negatives.length > 0 && sampleList(negatives)}
        </Card>

        <Card step="3" title="Train">
          <button
            type="button"
            onClick={() => void train()}
            disabled={training || positives.length < minPositives}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium cursor-pointer disabled:opacity-40 disabled:cursor-default"
            style={{ background: 'var(--color-accent)', color: 'white' }}
          >
            <Sparkles size={15} /> {training ? 'Training…' : status?.trained ? 'Retrain' : 'Train my voice'}
          </button>
          {positives.length < minPositives && (
            <span className="ml-3 text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
              Record {minPositives - positives.length} more first.
            </span>
          )}
          {status?.trained && stats && (
            <div className="mt-4 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
              <span className="font-medium" style={{ color: QUALITY_COLOR[stats.quality] }}>
                Quality: {stats.quality}
              </span>
              {' · '}
              {stats.positives} samples of your voice, {stats.negatives} other sentences.
              <p className="mt-1 text-xs">{stats.advice}</p>
            </div>
          )}
        </Card>

        <Card step="4" title="Use it">
          <p className="text-xs mb-3 flex items-center gap-2" style={{ color: 'var(--color-text-secondary)' }}>
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: listenerRunning ? 'var(--color-success)' : 'var(--color-text-tertiary)' }}
            />
            {listenerRunning
              ? 'The background listener is running — say the phrase any time.'
              : 'The background listener is not running yet.'}
          </p>
          <p className="text-xs mb-2" style={{ color: 'var(--color-text-tertiary)' }}>
            The listener runs in the background on this computer. Run this once in the OpenJarvis
            folder; it then starts by itself every time you log in:
          </p>
          <CommandLine command={LISTEN_COMMAND} />
          <p className="text-xs mt-3 mb-2" style={{ color: 'var(--color-text-tertiary)' }}>
            Then say “{phrase || 'Hey Jarvis'}” (or clap twice), wait for the chirp, and ask your
            question. To check how well it hears you first, run:
          </p>
          <CommandLine command={TEST_COMMAND} />
          {status && (
            <p className="text-xs mt-3" style={{ color: 'var(--color-text-tertiary)' }}>
              Training files: <code>{status.directory}</code> — your recordings are in{' '}
              <code>samples/</code>, the trained profile is <code>profile.json</code>. Open the folder
              with <code>jarvis wake folder</code>.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}
