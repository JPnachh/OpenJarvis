import { useId } from 'react';
import { useVoiceStore } from '../../lib/voice';
import { useJarvisPresence, PRESENCE_LABEL, type JarvisPresence } from '../../lib/presence';

interface JarvisOrbProps {
  size?: number;
  /** Override the live presence, e.g. for previews. */
  presence?: JarvisPresence;
  className?: string;
}

const PRESENCE_COLOR: Record<JarvisPresence, string> = {
  idle: 'var(--color-accent)',
  listening: 'var(--color-error)',
  transcribing: 'var(--color-warning, #f59e0b)',
  thinking: 'var(--color-accent)',
  speaking: 'var(--color-success)',
};

/**
 * Jarvis's face: a glowing core with rings whose motion says what Jarvis is
 * doing. While listening the core follows the live microphone level, so the
 * user can see that their voice is being picked up.
 */
export function JarvisOrb({ size = 40, presence: override, className }: JarvisOrbProps) {
  const live = useJarvisPresence();
  const presence = override ?? live;
  const level = useVoiceStore((s) => (presence === 'listening' ? s.level : 0));
  const gradientId = useId().replace(/:/g, '');
  const color = PRESENCE_COLOR[presence];
  const coreScale = presence === 'listening' ? 0.85 + level * 0.45 : 1;

  return (
    <div
      className={`jarvis-orb jarvis-orb--${presence} ${className ?? ''}`}
      style={{ width: size, height: size, color }}
      role="img"
      aria-label={`Jarvis: ${PRESENCE_LABEL[presence]}`}
      data-presence={presence}
    >
      <svg viewBox="0 0 100 100" width={size} height={size} aria-hidden="true">
        <defs>
          <radialGradient id={`core-${gradientId}`} cx="50%" cy="45%" r="55%">
            <stop offset="0%" stopColor="white" stopOpacity="0.95" />
            <stop offset="35%" stopColor="currentColor" stopOpacity="0.9" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
          </radialGradient>
        </defs>
        <circle className="jarvis-orb__halo" cx="50" cy="50" r="46" fill="currentColor" opacity="0.08" />
        <circle
          className="jarvis-orb__ring jarvis-orb__ring--outer"
          cx="50" cy="50" r="40"
          fill="none" stroke="currentColor" strokeWidth="2"
          strokeDasharray="60 22 8 22" strokeLinecap="round" opacity="0.7"
        />
        <circle
          className="jarvis-orb__ring jarvis-orb__ring--inner"
          cx="50" cy="50" r="31"
          fill="none" stroke="currentColor" strokeWidth="1.5"
          strokeDasharray="4 6" opacity="0.5"
        />
        <g style={{ transform: `scale(${coreScale})`, transformOrigin: '50px 50px', transition: 'transform 60ms linear' }}>
          <circle className="jarvis-orb__core" cx="50" cy="50" r="22" fill={`url(#core-${gradientId})`} />
        </g>
      </svg>
    </div>
  );
}
