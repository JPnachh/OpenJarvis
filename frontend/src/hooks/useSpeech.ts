import { useCallback, useEffect, useRef } from 'react';
import { useVoiceStore, type VoicePhase } from '../lib/voice';
import { useAppStore } from '../lib/store';

export type SpeechState = 'idle' | 'requesting' | 'recording' | 'transcribing';

const PHASE_TO_STATE: Record<VoicePhase, SpeechState> = {
  idle: 'idle',
  requesting: 'requesting',
  listening: 'recording',
  transcribing: 'transcribing',
};

// While the speech backend reports unavailable, look again this often. The
// page commonly opens before the API has finished starting, and a single
// probe at mount used to leave the mic disabled until a full reload.
const HEALTH_RETRY_MS = 15000;

/**
 * Microphone controls for the chat input, backed by the shared voice store.
 *
 * `onTranscript` receives the text of each finished recording, whether the
 * user stopped it or it stopped on its own after a pause.
 */
export function useSpeech(onTranscript: (text: string) => void) {
  const phase = useVoiceStore((s) => s.phase);
  const error = useVoiceStore((s) => s.error);
  const available = useVoiceStore((s) => s.available);
  const speechEnabled = useAppStore((s) => s.settings.speechEnabled);
  const autoStop = useAppStore((s) => s.settings.voiceAutoStop);
  const earcons = useAppStore((s) => s.settings.voiceEarcons);

  const onTranscriptRef = useRef(onTranscript);
  onTranscriptRef.current = onTranscript;

  useEffect(() => {
    const voice = useVoiceStore.getState();
    void voice.ensureHealth();
    if (!speechEnabled) return;
    const retry = () => {
      if (useVoiceStore.getState().available !== true) {
        void useVoiceStore.getState().ensureHealth(true);
      }
    };
    const interval = setInterval(retry, HEALTH_RETRY_MS);
    window.addEventListener('focus', retry);
    return () => {
      clearInterval(interval);
      window.removeEventListener('focus', retry);
    };
  }, [speechEnabled]);

  // Never leave the microphone open behind a page the user navigated away from.
  useEffect(() => () => useVoiceStore.getState().cancel(), []);

  const startRecording = useCallback(async () => {
    await useVoiceStore.getState().start({
      onTranscript: (text) => onTranscriptRef.current(text),
      autoStop,
      earcons,
    });
  }, [autoStop, earcons]);

  const stopRecording = useCallback(() => {
    useVoiceStore.getState().stop('manual');
  }, []);

  const cancelRecording = useCallback(() => {
    useVoiceStore.getState().cancel();
  }, []);

  const state = PHASE_TO_STATE[phase];
  return {
    state,
    error,
    available: available === true,
    startRecording,
    stopRecording,
    cancelRecording,
    isRecording: state === 'recording',
    isTranscribing: state === 'transcribing',
  };
}
