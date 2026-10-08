import { useCallback, useEffect, useState } from 'react';
import { Play, Plus, Trash2, Wand2 } from 'lucide-react';
import { toast } from 'sonner';
import {
  addCommand,
  deleteCommand,
  fetchCommands,
  recordActionExchange,
  runAction,
  type CommandsInfo,
  type CustomAction,
} from '../lib/actions';

const ACTION_LABELS: Record<CustomAction, string> = {
  open: 'Open an app, site or link',
  spotify: 'Play on Spotify',
  keys: 'Press a media/volume key',
  say: 'Just answer',
  shell: 'Run a program (advanced)',
};

const TARGET_HINT: Record<CustomAction, string> = {
  open: 'e.g. spotify, https://calendar.google.com, C:\\Games\\game.exe',
  spotify: 'e.g. lofi para estudiar, Bad Bunny, my playlist name',
  keys: '',
  say: '',
  shell: 'Command line to run',
};

const inputStyle = {
  background: 'var(--color-bg)',
  border: '1px solid var(--color-border)',
  color: 'var(--color-text)',
};

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section
      className="rounded-xl p-5 mb-4"
      style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
    >
      <h2 className="text-sm font-semibold mb-3" style={{ color: 'var(--color-text)' }}>{title}</h2>
      {children}
    </section>
  );
}

export function CommandsPage() {
  const [info, setInfo] = useState<CommandsInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tryText, setTryText] = useState('');
  const [trying, setTrying] = useState(false);
  const [phrases, setPhrases] = useState('');
  const [action, setAction] = useState<CustomAction>('open');
  const [target, setTarget] = useState('');
  const [reply, setReply] = useState('');
  const [saving, setSaving] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setInfo(await fetchCommands());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not reach OpenJarvis');
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const tryCommand = async (text: string) => {
    if (!text.trim()) return;
    setTrying(true);
    try {
      const result = await runAction(text.trim());
      if (!result.handled) {
        toast('Not a direct command; Jarvis would ask the model for this one.');
      } else {
        recordActionExchange(text.trim(), result.reply ?? '');
        (result.ok ? toast.success : toast.error)(result.reply ?? 'Done');
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed');
    } finally {
      setTrying(false);
    }
  };

  const save = async () => {
    setSaving(true);
    try {
      await addCommand({
        phrases: phrases.split('\n').map((p) => p.trim()).filter(Boolean),
        action,
        target: action === 'say' ? '' : target.trim(),
        reply: reply.trim(),
      });
      toast.success('Jarvis learned a new command');
      setPhrases('');
      setTarget('');
      setReply('');
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Could not save');
    } finally {
      setSaving(false);
    }
  };

  const remove = async (id: string) => {
    try {
      await deleteCommand(id);
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Could not delete');
    }
  };

  const actions = (info?.actions ?? ['open', 'spotify', 'keys', 'say']).filter(
    (a) => a !== 'shell' || info?.allow_shell,
  );

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-10">
      <div className="max-w-3xl mx-auto w-full">
        <header className="mb-6">
          <h1 className="text-lg font-semibold" style={{ color: 'var(--color-text)' }}>Commands</h1>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-secondary)' }}>
            Things Jarvis does instantly when you say them — no model, no waiting. Say “Hey Jarvis”
            (or clap twice), then the command. Anything else goes to the model as a normal question.
          </p>
        </header>

        {error && (
          <div className="mb-4 rounded-lg px-4 py-3 text-sm" style={{ background: 'color-mix(in srgb, var(--color-error) 8%, transparent)', color: 'var(--color-text)' }}>
            {error}
          </div>
        )}

        <Card title="Try a command">
          <div className="flex gap-2">
            <input
              value={tryText}
              onChange={(e) => setTryText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && void tryCommand(tryText)}
              placeholder="sube el volumen · pon Bad Bunny en Spotify · ¿qué tengo hoy?"
              className="flex-1 px-3 py-2 rounded-lg text-sm outline-none"
              style={inputStyle}
            />
            <button
              type="button"
              onClick={() => void tryCommand(tryText)}
              disabled={trying || !tryText.trim()}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium cursor-pointer disabled:opacity-40"
              style={{ background: 'var(--color-accent)', color: 'white' }}
            >
              <Play size={14} /> Run
            </button>
          </div>
        </Card>

        <Card title="Built in">
          <div className="grid sm:grid-cols-2 gap-4">
            {(info?.examples ?? []).map((group) => (
              <div key={group.group}>
                <div className="text-[11px] uppercase tracking-wide mb-1.5" style={{ color: 'var(--color-text-tertiary)' }}>
                  {group.group}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {group.phrases.map((phrase) => (
                    <button
                      key={phrase}
                      type="button"
                      onClick={() => setTryText(phrase)}
                      className="px-2 py-1 rounded-md text-xs cursor-pointer"
                      style={{ background: 'var(--color-bg)', border: '1px solid var(--color-border)', color: 'var(--color-text-secondary)' }}
                      title="Put in the try box"
                    >
                      “{phrase}”
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs mt-4" style={{ color: 'var(--color-text-tertiary)' }}>
            Spotify playback needs <code>jarvis connect spotify</code> (Premium); without it, play/pause/next
            use your keyboard's media keys and searches open in the Spotify app. Calendar needs{' '}
            <code>jarvis connect gdrive</code>. English works too: “next song”, “turn up the volume”.
          </p>
        </Card>

        <Card title="Teach Jarvis a command">
          <div className="grid gap-3">
            <label className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
              When I say (one phrase per line)
              <textarea
                value={phrases}
                onChange={(e) => setPhrases(e.target.value)}
                rows={2}
                placeholder={'modo trabajo\nwork mode'}
                className="mt-1 w-full px-3 py-2 rounded-lg text-sm outline-none resize-y"
                style={inputStyle}
              />
            </label>
            <label className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
              Jarvis should
              <select
                value={action}
                onChange={(e) => setAction(e.target.value as CustomAction)}
                className="mt-1 w-full px-3 py-2 rounded-lg text-sm outline-none"
                style={inputStyle}
              >
                {actions.map((a) => (
                  <option key={a} value={a}>{ACTION_LABELS[a]}</option>
                ))}
              </select>
            </label>
            {action === 'keys' ? (
              <label className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                Key
                <select
                  value={target}
                  onChange={(e) => setTarget(e.target.value)}
                  className="mt-1 w-full px-3 py-2 rounded-lg text-sm outline-none"
                  style={inputStyle}
                >
                  <option value="">Choose…</option>
                  {(info?.keys ?? []).map((k) => (
                    <option key={k} value={k}>{k.replace('_', ' ')}</option>
                  ))}
                </select>
              </label>
            ) : action !== 'say' ? (
              <label className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                What
                <input
                  value={target}
                  onChange={(e) => setTarget(e.target.value)}
                  placeholder={TARGET_HINT[action]}
                  className="mt-1 w-full px-3 py-2 rounded-lg text-sm outline-none"
                  style={inputStyle}
                />
              </label>
            ) : null}
            <label className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
              And answer {action === 'say' ? '' : '(optional)'}
              <input
                value={reply}
                onChange={(e) => setReply(e.target.value)}
                placeholder="Listo, modo trabajo activado"
                className="mt-1 w-full px-3 py-2 rounded-lg text-sm outline-none"
                style={inputStyle}
              />
            </label>
            <div>
              <button
                type="button"
                onClick={() => void save()}
                disabled={saving || !phrases.trim()}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium cursor-pointer disabled:opacity-40"
                style={{ background: 'var(--color-accent)', color: 'white' }}
              >
                <Plus size={14} /> Save command
              </button>
            </div>
          </div>
        </Card>

        <Card title={`Your commands (${info?.commands.length ?? 0})`}>
          {info?.commands.length ? (
            <ul className="divide-y" style={{ borderColor: 'var(--color-border)' }}>
              {info.commands.map((c) => (
                <li key={c.id} className="flex items-center gap-3 py-2 text-sm">
                  <Wand2 size={14} style={{ color: 'var(--color-accent)' }} />
                  <div className="flex-1 min-w-0">
                    <div style={{ color: 'var(--color-text)' }}>{c.phrases.map((p) => `“${p}”`).join(' · ')}</div>
                    <div className="text-xs truncate" style={{ color: 'var(--color-text-tertiary)' }}>
                      {ACTION_LABELS[c.action]}{c.target ? `: ${c.target}` : ''}{c.reply ? ` → “${c.reply}”` : ''}
                    </div>
                  </div>
                  <button type="button" className="p-1.5 rounded cursor-pointer" aria-label="Try" onClick={() => void tryCommand(c.phrases[0])}>
                    <Play size={13} />
                  </button>
                  <button type="button" className="p-1.5 rounded cursor-pointer" aria-label="Delete" onClick={() => void remove(c.id)}>
                    <Trash2 size={13} />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
              None yet. Ideas: “modo trabajo” → open Google Calendar; “pon lofi” → Spotify “lofi beats”;
              “abre mi juego” → the game's .exe.
            </p>
          )}
          {info && (
            <p className="text-xs mt-3" style={{ color: 'var(--color-text-tertiary)' }}>
              Saved in <code>{info.path}</code>.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}
