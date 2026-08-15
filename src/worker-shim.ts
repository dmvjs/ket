/**
 * worker_threads shim.
 *
 * Resolves once at module initialization to either the real module (Node) or
 * null (everywhere else), so the workers path in runMps and runStabilizerRank
 * can check one synchronous value.
 *
 * `process.getBuiltinModule` rather than `await import(...)` for two reasons:
 *
 * - **No top-level await.** A TLA anywhere in the graph makes the whole bundle
 *   un-buildable as IIFE, which is the format a `<script src>` tag needs. This
 *   file is the only reason ket could not ship a global build.
 * - **Nothing is requested in a browser.** A browser cannot resolve
 *   'node:worker_threads' and rejects it at the network layer, logging a CORS
 *   error whether or not the rejection is caught. Not asking is the only way to
 *   stay quiet, and a synchronous lookup never asks.
 *
 * `getBuiltinModule` landed in Node 22.3. On 22.0–22.2 this yields null and the
 * workers option falls back to the single-threaded path, which is the same
 * behaviour as any non-Node host.
 */
import type { Worker, receiveMessageOnPort } from 'node:worker_threads'

type WorkerThreads = {
  Worker: typeof Worker
  receiveMessageOnPort: typeof receiveMessageOnPort
}

/** Node's synchronous builtin loader, absent on other hosts and before Node 22.3. */
type BuiltinLoader = { getBuiltinModule?: (id: string) => unknown }

function loadWorkerThreads(): WorkerThreads | null {
  if (typeof process === 'undefined') return null
  const load = (process as unknown as BuiltinLoader).getBuiltinModule
  if (typeof load !== 'function') return null
  try {
    return load.call(process, 'node:worker_threads') as WorkerThreads
  } catch {
    return null   // permission-restricted or stubbed runtime
  }
}

export const wt: WorkerThreads | null = loadWorkerThreads()
