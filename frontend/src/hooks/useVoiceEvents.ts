import { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router';
import { toast } from 'sonner';
import { useVoiceStore } from '../lib/voice';
import { useTtsStore } from '../lib/tts';
import { recordActionExchange } from '../lib/actions';
import {
  OutdatedServerError,
  postUiVoiceState,
  streamVoiceEvents,
  type VoiceEvent,
} from '../lib/voice-api';

const RETRY_MIN_MS = 1000;
const RETRY_MAX_MS = 10000;

/**
 * Keep the page in sync with the background listener (`jarvis listen`):
 * show when it is listening, receive what the user said, and tell it when
 * the page itself is speaking so it does not mistake Jarvis for the user.
 */
export function useVoiceEvents(): void {
  const navigate = useNavigate();
  const location = useLocation();

  // Spoken requests are sent from the chat page; bring it up when one arrives.
  const pending = useVoiceStore((s) => s.pendingCommand);
  useEffect(() => {
    if (pending && location.pathname !== '/') navigate('/');
  }, [pending, location.pathname, navigate]);

  useEffect(() => {
    const controller = new AbortController();
    let delay = RETRY_MIN_MS;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const onEvent = (event: VoiceEvent) => {
      const voice = useVoiceStore.getState();
      delay = RETRY_MIN_MS;
      if (event.type === 'hello') {
        const listener = event.states.listener;
        voice.setListener(listener ? listener.state : null, listener?.detail);
      } else if (event.type === 'state' && event.source === 'listener') {
        voice.setListener(event.state, event.detail);
        if (event.detail?.startsWith('error: ')) {
          toast.error(event.detail.slice('error: '.length), { duration: 8000 });
        }
      } else if (event.type === 'action') {
        // Ran directly on this computer; record it in the chat.
        recordActionExchange(event.text, event.reply);
        if (!event.ok) toast.error(event.reply, { duration: 6000 });
      } else if (event.type === 'command') {
        toast(`You said: “${event.text}”`, { duration: 5000 });
        voice.setPendingCommand({ id: event.id, text: event.text });
      }
    };

    const connect = async () => {
      try {
        await streamVoiceEvents((event) => {
          useVoiceStore.setState({ serverOutdated: false });
          onEvent(event);
        }, controller.signal);
      } catch (err) {
        // Server down or restarting; retry below. A 404 means an old server
        // is still running; say so instead of failing silently.
        useVoiceStore.setState({ serverOutdated: err instanceof OutdatedServerError });
      }
      if (controller.signal.aborted) return;
      useVoiceStore.getState().setListener(null);
      timer = setTimeout(connect, delay);
      delay = Math.min(RETRY_MAX_MS, delay * 2);
    };
    void connect();

    // Report our own speech so the listener ignores the mic meanwhile.
    let last = useTtsStore.getState().state;
    const unsubscribe = useTtsStore.subscribe((s) => {
      if (s.state === last) return;
      last = s.state;
      void postUiVoiceState(s.state === 'speaking' ? 'speaking' : 'idle').catch(() => {});
    });

    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
      unsubscribe();
    };
  }, []);
}
