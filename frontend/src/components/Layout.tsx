import { useEffect, useState } from 'react';
import { Outlet, useNavigate } from 'react-router';
import { ApprovalBell } from './ApprovalBell';
import { Sidebar } from './Sidebar/Sidebar';
import { SystemPulse } from './SystemPulse';
import { useAppStore } from '../lib/store';
import { checkHealth } from '../lib/api';
import { useVoiceStore } from '../lib/voice';
import { useTtsStore } from '../lib/tts';
import { toast } from 'sonner';

const HEALTHY_POLL_MS = 30000;
const UNREACHABLE_POLL_MS = 3000;

export function Layout() {
  const sidebarOpen = useAppStore((s) => s.sidebarOpen);
  const [apiReachable, setApiReachable] = useState<boolean | null>(null);

  // Poll fast while the backend is unreachable -- `jarvis gui` opens the page
  // while the API may still be starting -- and slowly once it answers.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let previous: boolean | null = null;
    let inFlight = false;
    const check = async () => {
      if (inFlight) return;
      inFlight = true;
      if (timer) clearTimeout(timer);
      const reachable = await checkHealth();
      inFlight = false;
      if (cancelled) return;
      setApiReachable(reachable);
      if (reachable && previous === false) {
        toast.success('Connected to OpenJarvis');
      }
      if (reachable && previous !== true) {
        // Voice backends may have been probed before the server was ready.
        void useVoiceStore.getState().ensureHealth(true);
        void useTtsStore.getState().ensureHealth();
      }
      previous = reachable;
      timer = setTimeout(check, reachable ? HEALTHY_POLL_MS : UNREACHABLE_POLL_MS);
    };
    void check();
    const onFocus = () => void check();
    window.addEventListener('focus', onFocus);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      window.removeEventListener('focus', onFocus);
    };
  }, []);

  const navigate = useNavigate();

  return (
    <div className="flex flex-col h-full w-full overflow-hidden relative" style={{ paddingTop: '3px' }}>
      <div className="hud-backdrop" aria-hidden="true" />
      <SystemPulse apiReachable={apiReachable} />
      <ApprovalBell />

      {/* Health check banner */}
      {apiReachable === false && (
        <div
          className="flex items-center gap-3 px-4 py-2 text-sm shrink-0"
          style={{
            background: 'color-mix(in srgb, var(--color-error) 8%, transparent)',
            borderBottom: '1px solid color-mix(in srgb, var(--color-error) 15%, transparent)',
            color: 'var(--color-text)',
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full shrink-0"
            style={{ background: 'var(--color-error)' }}
          />
          <span>Cannot reach OpenJarvis backend — retrying every few seconds…</span>
          <button
            onClick={() => navigate('/settings')}
            className="text-sm underline cursor-pointer ml-auto shrink-0"
            style={{ color: 'var(--color-accent)' }}
          >
            Change URL
          </button>
        </div>
      )}

      <div className="flex flex-1 min-h-0 relative z-10">
        <Sidebar />
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-20 bg-black/40 md:hidden"
            onClick={() => useAppStore.getState().setSidebarOpen(false)}
          />
        )}
        <main className="flex-1 flex flex-col min-w-0 h-full relative overflow-hidden" style={{ background: 'transparent' }}>
          <div className="flex-1 flex flex-col min-w-0 min-h-0 relative z-[2]">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
