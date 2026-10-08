import { describe, expect, it } from 'vitest';
import { downsample, encodeWav } from './wav-recorder';

describe('wav recorder helpers', () => {
  it('encodes 16-bit mono PCM with a valid header', async () => {
    const blob = encodeWav(new Float32Array([0, 0.5, -0.5, 1]), 16000);
    const view = new DataView(await blob.arrayBuffer());
    const text = (offset: number) =>
      String.fromCharCode(...[0, 1, 2, 3].map((i) => view.getUint8(offset + i)));
    expect(text(0)).toBe('RIFF');
    expect(text(8)).toBe('WAVE');
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(16000);
    expect(view.getUint16(34, true)).toBe(16); // bits
    expect(view.getUint32(40, true)).toBe(8); // 4 samples * 2 bytes
    expect(view.getInt16(46, true)).toBe(Math.trunc(0.5 * 0x7fff));
    expect(view.getInt16(50, true)).toBe(0x7fff);
  });

  it('downsamples by averaging', () => {
    const out = downsample(new Float32Array([1, 1, 1, 0, 0, 0]), 48000, 16000);
    expect(Array.from(out)).toEqual([1, 0]);
    const same = new Float32Array([0.1, 0.2]);
    expect(downsample(same, 16000, 16000)).toBe(same);
  });
});
