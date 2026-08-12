/**
 * Seeded xorshift32 PRNG — the same algorithm qsim uses, so seeded runs are
 * reproducible across implementations.
 *
 * Lives in its own module rather than in `circuit.ts` so backends can seed
 * themselves without importing the circuit layer, which would be a cycle.
 */
export function makePrng(seed?: number): () => number {
  let s = seed !== undefined ? ((seed >>> 0) || 1) : ((Date.now() & 0xffffffff) >>> 0) || 1
  return () => {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5
    return (s >>> 0) / 0x100000000
  }
}
