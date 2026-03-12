/**
 * Lazy-loaded worker_threads shim.
 *
 * Top-level await resolves once at module initialization. In Node.js ≥22 this
 * succeeds and parallel workers are enabled. In browsers or environments where
 * node:worker_threads is unavailable the catch returns null and the workers
 * path in runMps is silently skipped.
 *
 * The dynamic import (rather than a static import) prevents bundlers from
 * analyzing 'node:worker_threads' as a hard dependency.
 */
import type { Worker, receiveMessageOnPort } from 'node:worker_threads'

type WorkerThreads = {
  Worker: typeof Worker
  receiveMessageOnPort: typeof receiveMessageOnPort
}

export const wt: WorkerThreads | null = await (
  import('node:worker_threads') as Promise<WorkerThreads>
).catch(() => null)
