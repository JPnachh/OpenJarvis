import { useAppStore } from './store';
import { useTtsStore, type TtsState } from './tts';
import { useVoiceStore, type VoicePhase } from './voice';

/** What Jarvis is doing right now, as one value the whole UI can show. */
export type JarvisPresence = 'idle' | 'listening' | 'transcribing' | 'thinking' | 'speaking';

export function derivePresence(
  voicePhase: VoicePhase,
  ttsState: TtsState,
  isStreaming: boolean,
): JarvisPresence {
  // The microphone wins: if it is open, the user must be able to see that.
  if (voicePhase === 'listening' || voicePhase === 'requesting') return 'listening';
  if (voicePhase === 'transcribing') return 'transcribing';
  if (ttsState === 'speaking') return 'speaking';
  if (isStreaming || ttsState === 'loading') return 'thinking';
  return 'idle';
}

export function useJarvisPresence(): JarvisPresence {
  const voicePhase = useVoiceStore((s) => s.phase);
  const ttsState = useTtsStore((s) => s.state);
  const isStreaming = useAppStore((s) => s.streamState.isStreaming);
  return derivePresence(voicePhase, ttsState, isStreaming);
}

export const PRESENCE_LABEL: Record<JarvisPresence, string> = {
  idle: 'Ready',
  listening: 'Listening',
  transcribing: 'Transcribing',
  thinking: 'Thinking',
  speaking: 'Speaking',
};
