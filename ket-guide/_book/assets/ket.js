// src/complex.ts
var ZERO = { re: 0, im: 0 };
var ONE = { re: 1, im: 0 };
var I = { re: 0, im: 1 };
var c = (re, im = 0) => ({ re, im });
var add = (a, b) => ({ re: a.re + b.re, im: a.im + b.im });
var mul = (a, b) => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
var scale = (s, a) => ({ re: s * a.re, im: s * a.im });
var conj = (a) => ({ re: a.re, im: -a.im });
var norm2 = (a) => a.re * a.re + a.im * a.im;
var isNegligible = (a) => norm2(a) < 1e-14;

// src/gates.ts
var sq2 = 1 / Math.sqrt(2);
var Id = [[ONE, ZERO], [ZERO, ONE]];
var H = [[c(sq2), c(sq2)], [c(sq2), c(-sq2)]];
var X = [[ZERO, ONE], [ONE, ZERO]];
var Y = [[ZERO, c(0, -1)], [c(0, 1), ZERO]];
var Z = [[ONE, ZERO], [ZERO, c(-1)]];
var S = [[ONE, ZERO], [ZERO, c(0, 1)]];
var Si = [[ONE, ZERO], [ZERO, c(0, -1)]];
var T = [[ONE, ZERO], [ZERO, c(sq2, sq2)]];
var Ti = [[ONE, ZERO], [ZERO, c(sq2, -sq2)]];
var V = [[c(0.5, 0.5), c(0.5, -0.5)], [c(0.5, -0.5), c(0.5, 0.5)]];
var Vi = [[c(0.5, -0.5), c(0.5, 0.5)], [c(0.5, 0.5), c(0.5, -0.5)]];
var Rx = (theta) => {
  const cos = Math.cos(theta / 2);
  const sin = Math.sin(theta / 2);
  return [[c(cos), c(0, -sin)], [c(0, -sin), c(cos)]];
};
var Ry = (theta) => {
  const cos = Math.cos(theta / 2);
  const sin = Math.sin(theta / 2);
  return [[c(cos, 0), c(-sin)], [c(sin), c(cos)]];
};
var Rz = (theta) => {
  const cos = Math.cos(theta / 2);
  const sin = Math.sin(theta / 2);
  return [[c(cos, -sin), ZERO], [ZERO, c(cos, sin)]];
};
var R2 = Rz(Math.PI / 2);
var R4 = Rz(Math.PI / 4);
var R8 = Rz(Math.PI / 8);
var U3 = (theta, phi, lambda) => {
  const cos = Math.cos(theta / 2);
  const sin = Math.sin(theta / 2);
  return [
    [c(cos), c(-sin * Math.cos(lambda), -sin * Math.sin(lambda))],
    [c(sin * Math.cos(phi), sin * Math.sin(phi)), c(cos * Math.cos(phi + lambda), cos * Math.sin(phi + lambda))]
  ];
};
var U2 = (phi, lambda) => U3(Math.PI / 2, phi, lambda);
var U1 = (lambda) => [[ONE, ZERO], [ZERO, c(Math.cos(lambda), Math.sin(lambda))]];
var Xx = (theta) => {
  const co = c(Math.cos(theta / 2)), ni = c(0, -Math.sin(theta / 2));
  return [
    [co, ZERO, ZERO, ni],
    [ZERO, co, ni, ZERO],
    [ZERO, ni, co, ZERO],
    [ni, ZERO, ZERO, co]
  ];
};
var Yy = (theta) => {
  const co = c(Math.cos(theta / 2));
  const ni = c(0, -Math.sin(theta / 2));
  const pi = c(0, Math.sin(theta / 2));
  return [
    [co, ZERO, ZERO, pi],
    [ZERO, co, ni, ZERO],
    [ZERO, ni, co, ZERO],
    [pi, ZERO, ZERO, co]
  ];
};
var Zz = (theta) => {
  const cos = Math.cos(theta / 2), sin = Math.sin(theta / 2);
  const m = c(cos, -sin), p = c(cos, sin);
  return [
    [m, ZERO, ZERO, ZERO],
    [ZERO, p, ZERO, ZERO],
    [ZERO, ZERO, p, ZERO],
    [ZERO, ZERO, ZERO, m]
  ];
};
var Xy = (theta) => {
  const co = c(Math.cos(theta / 2)), is = c(0, Math.sin(theta / 2));
  return [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, co, is, ZERO],
    [ZERO, is, co, ZERO],
    [ZERO, ZERO, ZERO, ONE]
  ];
};
var ISwap = Xy(Math.PI);
var SrSwap = Xy(Math.PI / 2);
var Gpi = (phi) => {
  const cos = Math.cos(phi), sin = Math.sin(phi);
  return [[ZERO, c(cos, -sin)], [c(cos, sin), ZERO]];
};
var Gpi2 = (phi) => {
  const cos = Math.cos(phi), sin = Math.sin(phi);
  return [[c(sq2), c(-sq2 * sin, -sq2 * cos)], [c(sq2 * sin, -sq2 * cos), c(sq2)]];
};
var Ms = (phi0, phi1) => {
  const sp = Math.sin(phi0 + phi1), cp = Math.cos(phi0 + phi1);
  const sd = Math.sin(phi0 - phi1), cd = Math.cos(phi0 - phi1);
  return [
    [c(sq2), ZERO, ZERO, c(-sp * sq2, -cp * sq2)],
    [ZERO, c(sq2), c(-sd * sq2, -cd * sq2), ZERO],
    [ZERO, c(sd * sq2, -cd * sq2), c(sq2), ZERO],
    [c(sp * sq2, -cp * sq2), ZERO, ZERO, c(sq2)]
  ];
};

// src/statevector.ts
var zero = (n) => /* @__PURE__ */ new Map([[0n, { re: 1, im: 0 }]]);
function accumulate(sv, idx, amp) {
  const existing = sv.get(idx);
  const next = existing ? add(existing, amp) : amp;
  if (!isNegligible(next)) {
    sv.set(idx, next);
  } else {
    sv.delete(idx);
  }
}
function applySingle(sv, q, [[a, b], [c2, d]]) {
  const next = /* @__PURE__ */ new Map();
  const mask = 1n << BigInt(q);
  const seen = /* @__PURE__ */ new Set();
  for (const idx of sv.keys()) {
    const base = idx & ~mask;
    if (seen.has(base)) continue;
    seen.add(base);
    const amp0 = sv.get(base) ?? ZERO;
    const amp1 = sv.get(base | mask) ?? ZERO;
    accumulate(next, base, add(mul(a, amp0), mul(b, amp1)));
    accumulate(next, base | mask, add(mul(c2, amp0), mul(d, amp1)));
  }
  return next;
}
function applyCNOT(sv, control, target) {
  const next = /* @__PURE__ */ new Map();
  const cmask = 1n << BigInt(control);
  const tmask = 1n << BigInt(target);
  for (const [idx, amp] of sv) {
    next.set((idx & cmask) !== 0n ? idx ^ tmask : idx, amp);
  }
  return next;
}
function applySWAP(sv, a, b) {
  const next = /* @__PURE__ */ new Map();
  const amask = 1n << BigInt(a);
  const bmask = 1n << BigInt(b);
  for (const [idx, amp] of sv) {
    const bitA = (idx & amask) !== 0n;
    const bitB = (idx & bmask) !== 0n;
    if (bitA === bitB) {
      next.set(idx, amp);
    } else {
      next.set(idx ^ amask ^ bmask, amp);
    }
  }
  return next;
}
function applyTwo(sv, a, b, gate) {
  const next = /* @__PURE__ */ new Map();
  const ma = 1n << BigInt(a);
  const mb = 1n << BigInt(b);
  const seen = /* @__PURE__ */ new Set();
  for (const idx of sv.keys()) {
    const ctx = idx & ~(ma | mb);
    if (seen.has(ctx)) continue;
    seen.add(ctx);
    const bases = [ctx, ctx | mb, ctx | ma, ctx | ma | mb];
    const amps = bases.map((i) => sv.get(i) ?? ZERO);
    for (let r = 0; r < 4; r++) {
      let out = ZERO;
      for (let c2 = 0; c2 < 4; c2++) out = add(out, mul(gate[r][c2], amps[c2]));
      accumulate(next, bases[r], out);
    }
  }
  return next;
}
function applyToffoli(sv, c1, c2, target) {
  const next = /* @__PURE__ */ new Map();
  const c1mask = 1n << BigInt(c1);
  const c2mask = 1n << BigInt(c2);
  const tmask = 1n << BigInt(target);
  for (const [idx, amp] of sv) {
    next.set((idx & c1mask) !== 0n && (idx & c2mask) !== 0n ? idx ^ tmask : idx, amp);
  }
  return next;
}
function applyCSwap(sv, control, a, b) {
  const next = /* @__PURE__ */ new Map();
  const cmask = 1n << BigInt(control);
  const amask = 1n << BigInt(a);
  const bmask = 1n << BigInt(b);
  for (const [idx, amp] of sv) {
    if ((idx & cmask) === 0n) {
      next.set(idx, amp);
    } else {
      const bitA = (idx & amask) !== 0n;
      const bitB = (idx & bmask) !== 0n;
      next.set(bitA === bitB ? idx : idx ^ amask ^ bmask, amp);
    }
  }
  return next;
}
function applyCsrSwap(sv, control, a, b) {
  const next = /* @__PURE__ */ new Map();
  const cmask = 1n << BigInt(control);
  const amask = 1n << BigInt(a);
  const bmask = 1n << BigInt(b);
  const seen = /* @__PURE__ */ new Set();
  const sq22 = 1 / Math.sqrt(2);
  for (const idx of sv.keys()) {
    const ctx = idx & ~(cmask | amask | bmask);
    if (seen.has(ctx)) continue;
    seen.add(ctx);
    for (const cc of [0n, cmask]) {
      const bases = [ctx | cc, ctx | cc | bmask, ctx | cc | amask, ctx | cc | amask | bmask];
      const amps = bases.map((i) => sv.get(i) ?? ZERO);
      if (cc === 0n) {
        for (let r = 0; r < 4; r++) accumulate(next, bases[r], amps[r]);
      } else {
        accumulate(next, bases[0], amps[0]);
        accumulate(next, bases[3], amps[3]);
        const [a01, a10] = [amps[1], amps[2]];
        accumulate(next, bases[1], add(mul({ re: sq22, im: 0 }, a01), mul({ re: 0, im: sq22 }, a10)));
        accumulate(next, bases[2], add(mul({ re: 0, im: sq22 }, a01), mul({ re: sq22, im: 0 }, a10)));
      }
    }
  }
  return next;
}
function applyControlled(sv, control, target, [[a, b], [c2, d]]) {
  if (control === target) throw new TypeError(`control and target qubits must differ (got ${control})`);
  const next = /* @__PURE__ */ new Map();
  const cmask = 1n << BigInt(control);
  const tmask = 1n << BigInt(target);
  const seen = /* @__PURE__ */ new Set();
  for (const [idx, amp] of sv) {
    if ((idx & cmask) === 0n) {
      accumulate(next, idx, amp);
      continue;
    }
    const base = idx & ~tmask;
    if (seen.has(base)) continue;
    seen.add(base);
    const amp0 = sv.get(base) ?? ZERO;
    const amp1 = sv.get(base | tmask) ?? ZERO;
    accumulate(next, base, add(mul(a, amp0), mul(b, amp1)));
    accumulate(next, base | tmask, add(mul(c2, amp0), mul(d, amp1)));
  }
  return next;
}
function applyUnitary(sv, qs, matrix) {
  const n = qs.length;
  const dim = 1 << n;
  const masks = qs.map((q) => 1n << BigInt(q));
  const allMask = masks.reduce((a, b) => a | b, 0n);
  const next = /* @__PURE__ */ new Map();
  const seen = /* @__PURE__ */ new Set();
  for (const idx of sv.keys()) {
    const ctx = idx & ~allMask;
    if (seen.has(ctx)) continue;
    seen.add(ctx);
    const bases = Array.from({ length: dim }, (_, i) => {
      let b = ctx;
      for (let bit = 0; bit < n; bit++) {
        if (i >> n - 1 - bit & 1) b |= masks[bit];
      }
      return b;
    });
    const amps = bases.map((b) => sv.get(b) ?? ZERO);
    for (let r = 0; r < dim; r++) {
      let out = ZERO;
      for (let c2 = 0; c2 < dim; c2++) out = add(out, mul(matrix[r][c2], amps[c2]));
      accumulate(next, bases[r], out);
    }
  }
  return next;
}
function probabilities(sv) {
  const probs = /* @__PURE__ */ new Map();
  for (const [idx, amp] of sv) {
    const p = amp.re * amp.re + amp.im * amp.im;
    if (p > 1e-14) probs.set(idx, p);
  }
  return probs;
}

// src/mps.ts
var JUMP_THRESHOLD = 1e-15;
var CNOT4 = [
  [ONE, ZERO, ZERO, ZERO],
  [ZERO, ONE, ZERO, ZERO],
  [ZERO, ZERO, ZERO, ONE],
  [ZERO, ZERO, ONE, ZERO]
];
var SWAP4 = [
  [ONE, ZERO, ZERO, ZERO],
  [ZERO, ZERO, ONE, ZERO],
  [ZERO, ONE, ZERO, ZERO],
  [ZERO, ZERO, ZERO, ONE]
];
function controlledGate([[a, b], [cc, d]]) {
  return [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, a, b],
    [ZERO, ZERO, cc, d]
  ];
}
function swapQubits([[a, b, c2, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]]) {
  return [
    [a, c2, b, d],
    [i, k, j, l],
    [e, g, f, h],
    [m, o, n, p]
  ];
}
var MpsTrajectory = class {
  n;
  /**
   * Current bond dimension cap. Starts at the value passed to the constructor and
   * grows automatically (via `growTo`) whenever a gate would require a larger bond.
   * Read this after simulation to see the peak χ actually used.
   */
  _maxChi;
  get maxChi() {
    return this._maxChi;
  }
  /**
   * Relative Schmidt truncation threshold. Singular values σ_k < truncErr · σ_max
   * are discarded during SVD.
   *
   * 0 (default) means truncate by absolute tolerance only (1e-14).
   * Typical values: 1e-8 for chemistry/VQE circuits with rapidly decaying Schmidt spectra.
   */
  truncErr;
  // Per-site tensor storage. data[q] holds up to maxChi × 2 × maxChi complex values.
  // Layout: data[q][((l*2+p)*chiR+r)*2] = re, +1 = im, where chiR = this.chiR[q].
  //
  // Vidal canonical (Γ-Λ) form:
  //   data[q]       = Γ[q]      — left-isometric in the Vidal sense: Γ†Γ = I after
  //                               weighting by the boundary lambdas (not plain U†U=I)
  //   bondLambda[b] = Λ[b]      — Schmidt values at bond (b, b+1), sorted descending
  //
  // The state amplitude is: ψ = Γ[0] · Λ[0] · Γ[1] · Λ[1] · ... · Γ[n-1]
  // (boundary lambdas Λ[-1] = Λ[n] = 1 are implicit).
  //
  // This form ensures:
  //   • apply2Adjacent includes all three boundary lambdas in theta → optimal truncation
  //   • bondLambda holds the true Schmidt spectrum at every bond → correct bondEntropies()
  //   • sample() weighted by bondLambda[q] gives exact marginals (right environment = Λ²)
  data;
  chiL;
  chiR;
  bondLambda;
  // n-1 entries; bondLambda[b][k] = σ_k at bond b
  // Workspace for apply2Adjacent — sized for worst-case maxChi bonds.
  // Not readonly: reallocated by growTo() when maxChi increases.
  mBuf;
  // contracted θ matrix, rows × cols
  aBuf;
  // column-major working copy for Jacobi
  qBuf;
  // U·Σ output, row-major: rows × bond
  rBuf;
  // V†   output, row-major: bond × cols
  // Workspace for SVD (one-sided complex Jacobi).
  // vBuf: V matrix (cols × cols, column-major), cols = maxChi * 2.
  // sigmaBuf: singular values per column, length maxChi * 2.
  // orderBuf: column index permutation sorted by descending sigma, length maxChi * 2.
  vBuf;
  sigmaBuf;
  orderBuf;
  // Workspace for sample — one allocation per maxChi, reused across all qubits.
  sv0re;
  sv0im;
  sv1re;
  sv1im;
  stRe;
  stIm;
  // Workspace for expectation() — reused across all terms in a Hamiltonian loop.
  // ogBuf/fBuf: OΓ and F intermediates [l/p'][p'/r][r/l'], sized maxChi*2 × maxChi*2.
  // exCur/exNxt: current and next E matrix, sized maxChi × maxChi.
  // exDiag: diagonal of E when it is known to be diagonal (leading identity fast path).
  exOgBuf;
  exFBuf;
  exCurBuf;
  exNxtBuf;
  exDiag;
  /** Set to true by svdDecompose whenever a physically significant singular value is discarded. */
  wasTruncated_ = false;
  /** True if any SVD during this trajectory discarded a singular value above 1e-14 (due to truncErr threshold). */
  get wasTruncated() {
    return this.wasTruncated_;
  }
  constructor(n, maxChi, truncErr = 0) {
    this.n = n;
    this._maxChi = maxChi;
    this.truncErr = truncErr;
    const maxRows = maxChi * 2;
    const maxCols = maxChi * 2;
    this.data = Array.from({ length: n }, () => new Float64Array(maxRows * maxCols * 2));
    this.chiL = new Int32Array(n);
    this.chiR = new Int32Array(n);
    this.bondLambda = Array.from({ length: Math.max(n - 1, 0) }, () => new Float64Array(maxChi));
    this.mBuf = new Float64Array(maxRows * maxCols * 2);
    this.aBuf = new Float64Array(maxCols * maxRows * 2);
    this.qBuf = new Float64Array(maxRows * maxChi * 2);
    this.rBuf = new Float64Array(maxChi * maxCols * 2);
    this.vBuf = new Float64Array(maxCols * maxCols * 2);
    this.sigmaBuf = new Float64Array(maxCols);
    this.orderBuf = new Int32Array(maxCols);
    this.sv0re = new Float64Array(maxChi);
    this.sv0im = new Float64Array(maxChi);
    this.sv1re = new Float64Array(maxChi);
    this.sv1im = new Float64Array(maxChi);
    this.stRe = new Float64Array(maxChi);
    this.stIm = new Float64Array(maxChi);
    this.exOgBuf = new Float64Array(maxCols * maxCols * 2);
    this.exFBuf = new Float64Array(maxCols * maxCols * 2);
    this.exCurBuf = new Float64Array(maxChi * maxChi * 2);
    this.exNxtBuf = new Float64Array(maxChi * maxChi * 2);
    this.exDiag = new Float64Array(maxChi);
    this.reset();
  }
  /**
   * Grow all buffers to support a larger bond dimension.
   *
   * Tensor data is preserved: each site tensor uses its actual chiR[q] as the row stride,
   * so elements sit at the same offsets in the new (larger) Float64Array — a plain set()
   * copy is sufficient with no layout remapping.
   *
   * Called automatically by apply2Adjacent when min(rows, cols) > maxChi.
   */
  growTo(newMaxChi) {
    const maxRows = newMaxChi * 2;
    const maxCols = newMaxChi * 2;
    for (let q = 0; q < this.n; q++) {
      const grown = new Float64Array(maxRows * maxCols * 2);
      grown.set(this.data[q]);
      this.data[q] = grown;
    }
    for (let b = 0; b < this.n - 1; b++) {
      const grown = new Float64Array(newMaxChi);
      grown.set(this.bondLambda[b]);
      this.bondLambda[b] = grown;
    }
    this.mBuf = new Float64Array(maxRows * maxCols * 2);
    this.aBuf = new Float64Array(maxCols * maxRows * 2);
    this.qBuf = new Float64Array(maxRows * newMaxChi * 2);
    this.rBuf = new Float64Array(newMaxChi * maxCols * 2);
    this.vBuf = new Float64Array(maxCols * maxCols * 2);
    this.sigmaBuf = new Float64Array(maxCols);
    this.orderBuf = new Int32Array(maxCols);
    this.sv0re = new Float64Array(newMaxChi);
    this.sv0im = new Float64Array(newMaxChi);
    this.sv1re = new Float64Array(newMaxChi);
    this.sv1im = new Float64Array(newMaxChi);
    this.stRe = new Float64Array(newMaxChi);
    this.stIm = new Float64Array(newMaxChi);
    this.exOgBuf = new Float64Array(maxCols * maxCols * 2);
    this.exFBuf = new Float64Array(maxCols * maxCols * 2);
    this.exCurBuf = new Float64Array(newMaxChi * newMaxChi * 2);
    this.exNxtBuf = new Float64Array(newMaxChi * newMaxChi * 2);
    this.exDiag = new Float64Array(newMaxChi);
    this._maxChi = newMaxChi;
  }
  /** Reset to |0...0⟩. Sets all tensors to T[0][0][0]=1 with chiL=chiR=1. */
  reset() {
    for (let q = 0; q < this.n; q++) {
      this.data[q].fill(0);
      this.data[q][0] = 1;
      this.chiL[q] = 1;
      this.chiR[q] = 1;
    }
    for (let b = 0; b < this.n - 1; b++) {
      this.bondLambda[b].fill(0);
      this.bondLambda[b][0] = 1;
    }
  }
  /** Apply a single-qubit gate in place. O(chiL · chiR). */
  apply1(q, [[a, b], [c2, d]]) {
    const are = a.re, aim = a.im;
    const bre = b.re, bim = b.im;
    const cre = c2.re, cim = c2.im;
    const dre = d.re, dim = d.im;
    const data = this.data[q];
    const chiL = this.chiL[q], chiR = this.chiR[q];
    for (let l = 0; l < chiL; l++) {
      for (let r = 0; r < chiR; r++) {
        const i0 = ((l * 2 + 0) * chiR + r) * 2;
        const i1 = ((l * 2 + 1) * chiR + r) * 2;
        const t0re = data[i0], t0im = data[i0 + 1];
        const t1re = data[i1], t1im = data[i1 + 1];
        data[i0] = are * t0re - aim * t0im + bre * t1re - bim * t1im;
        data[i0 + 1] = are * t0im + aim * t0re + bre * t1im + bim * t1re;
        data[i1] = cre * t0re - cim * t0im + dre * t1re - dim * t1im;
        data[i1 + 1] = cre * t0im + cim * t0re + dre * t1im + dim * t1re;
      }
    }
  }
  /**
   * Apply a two-qubit gate in place.
   *
   * `a` and `b` may be given in either order — if `a > b` the gate matrix is
   * qubit-transposed so the physical operation is identical regardless of argument order.
   * Non-adjacent pairs are handled via a SWAP network at O(|b-a|) adjacent applications.
   */
  apply2(a, b, gate) {
    if (a === b) throw new RangeError(`apply2: a and b must differ (got ${a})`);
    if (a > b) {
      this.apply2(b, a, swapQubits(gate));
      return;
    }
    if (b === a + 1) {
      this.apply2Adjacent(a, gate);
      return;
    }
    for (let q = b - 1; q > a; q--) this.apply2Adjacent(q, SWAP4);
    this.apply2Adjacent(a, gate);
    for (let q = a + 1; q < b; q++) this.apply2Adjacent(q, SWAP4);
  }
  apply2Adjacent(a, gate) {
    const b = a + 1;
    const chiL = this.chiL[a];
    const chiR = this.chiR[b];
    const rows = chiL * 2;
    const cols = chiR * 2;
    const needed = Math.min(rows, cols);
    if (needed > this._maxChi) {
      this.growTo(Math.max(needed, Math.ceil(this._maxChi * 1.5)));
    }
    const chiM = this.chiR[a];
    const dA = this.data[a];
    const dB = this.data[b];
    const mBuf = this.mBuf;
    const lambdaL = a > 0 ? this.bondLambda[a - 1] : null;
    const lambdaM = this.bondLambda[a];
    const lambdaR = b < this.n - 1 ? this.bondLambda[b] : null;
    const mSize = rows * cols * 2;
    for (let i = 0; i < mSize; i++) mBuf[i] = 0;
    for (let la = 0; la < chiL; la++) {
      const scaleL = lambdaL ? lambdaL[la] : 1;
      for (let pa = 0; pa < 2; pa++) {
        for (let pb = 0; pb < 2; pb++) {
          for (let rb = 0; rb < chiR; rb++) {
            const scaleR = lambdaR ? lambdaR[rb] : 1;
            let tre = 0, tim = 0;
            for (let m = 0; m < chiM; m++) {
              const scaleM = lambdaM[m];
              const ai = ((la * 2 + pa) * chiM + m) * 2;
              const bi = ((m * 2 + pb) * chiR + rb) * 2;
              tre += scaleM * (dA[ai] * dB[bi] - dA[ai + 1] * dB[bi + 1]);
              tim += scaleM * (dA[ai] * dB[bi + 1] + dA[ai + 1] * dB[bi]);
            }
            const scale2 = scaleL * scaleR;
            tre *= scale2;
            tim *= scale2;
            for (let pa2 = 0; pa2 < 2; pa2++) {
              for (let pb2 = 0; pb2 < 2; pb2++) {
                const g = gate[pa2 * 2 + pb2][pa * 2 + pb];
                const mi = ((la * 2 + pa2) * cols + pb2 * chiR + rb) * 2;
                mBuf[mi] = (mBuf[mi] ?? 0) + g.re * tre - g.im * tim;
                mBuf[mi + 1] = (mBuf[mi + 1] ?? 0) + g.re * tim + g.im * tre;
              }
            }
          }
        }
      }
    }
    const bond = this.svdDecompose(rows, cols);
    const qBuf = this.qBuf;
    for (let la = 0; la < chiL; la++) {
      const invL = lambdaL ? lambdaL[la] > 1e-14 ? 1 / lambdaL[la] : 0 : 1;
      for (let p = 0; p < 2; p++) {
        for (let k = 0; k < bond; k++) {
          const i = ((la * 2 + p) * bond + k) * 2;
          dA[i] = qBuf[i] * invL;
          dA[i + 1] = qBuf[i + 1] * invL;
        }
      }
    }
    const rBuf = this.rBuf;
    for (let k = 0; k < bond; k++) {
      for (let p = 0; p < 2; p++) {
        for (let rb = 0; rb < chiR; rb++) {
          const invR = lambdaR ? lambdaR[rb] > 1e-14 ? 1 / lambdaR[rb] : 0 : 1;
          const i = (k * cols + p * chiR + rb) * 2;
          dB[i] = rBuf[i] * invR;
          dB[i + 1] = rBuf[i + 1] * invR;
        }
      }
    }
    this.chiR[a] = bond;
    this.chiL[b] = bond;
    const lambda = this.bondLambda[a];
    for (let k = 0; k < bond; k++) lambda[k] = this.sigmaBuf[this.orderBuf[k]];
    for (let k = bond; k < this.maxChi; k++) lambda[k] = 0;
  }
  /**
   * One-sided complex Jacobi SVD on this.mBuf (rows × cols, row-major).
   *
   * Decomposes M ≈ U · diag(σ) · V† where:
   *   qBuf  — U   (rows × bond, row-major): left singular vectors
   *   rBuf  — σ·V† (bond × cols, row-major): scaled right singular vectors
   *
   * Algorithm: sweep Jacobi rotations over column pairs until orthogonal,
   * then sort by descending singular value and truncate to maxChi.
   * Gives the optimal low-rank approximation at each MPS bond cut (Schmidt decomp).
   *
   * All buffers pre-allocated; zero heap allocation in this method.
   */
  svdDecompose(rows, cols) {
    const mBuf = this.mBuf;
    const aBuf = this.aBuf;
    const vBuf = this.vBuf;
    const maxChi = this.maxChi;
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        const src = (i * cols + j) * 2;
        const dst = (j * rows + i) * 2;
        aBuf[dst] = mBuf[src];
        aBuf[dst + 1] = mBuf[src + 1];
      }
    }
    const vSize = cols * cols * 2;
    for (let k = 0; k < vSize; k++) vBuf[k] = 0;
    for (let k = 0; k < cols; k++) vBuf[(k * cols + k) * 2] = 1;
    let frobSq = 0;
    for (let k = 0; k < cols; k++) {
      for (let i = 0; i < rows; i++) {
        const re = aBuf[(k * rows + i) * 2], im = aBuf[(k * rows + i) * 2 + 1];
        frobSq += re * re + im * im;
      }
    }
    const maxSweeps = 20;
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let maxOff = 0;
      for (let p = 0; p < cols - 1; p++) {
        for (let q = p + 1; q < cols; q++) {
          let Gpp = 0, Gqq = 0, Gpqre = 0, Gpqim = 0;
          for (let i = 0; i < rows; i++) {
            const Apre = aBuf[(p * rows + i) * 2], Apim = aBuf[(p * rows + i) * 2 + 1];
            const Aqre = aBuf[(q * rows + i) * 2], Aqim = aBuf[(q * rows + i) * 2 + 1];
            Gpp += Apre * Apre + Apim * Apim;
            Gqq += Aqre * Aqre + Aqim * Aqim;
            Gpqre += Apre * Aqre + Apim * Aqim;
            Gpqim += Apre * Aqim - Apim * Aqre;
          }
          const r = Math.sqrt(Gpqre * Gpqre + Gpqim * Gpqim);
          if (r > maxOff) maxOff = r;
          if (r < 1e-14 * Math.sqrt(Gpp * Gqq) + 1e-28) continue;
          const epre = Gpqre / r, epim = -Gpqim / r;
          const tau = (Gqq - Gpp) / (2 * r);
          const t = tau >= 0 ? -1 / (tau + Math.sqrt(1 + tau * tau)) : 1 / (-tau + Math.sqrt(1 + tau * tau));
          const c2 = 1 / Math.sqrt(1 + t * t);
          const s = t * c2;
          for (let i = 0; i < rows; i++) {
            const pi = (p * rows + i) * 2, qi = (q * rows + i) * 2;
            const Apr = aBuf[pi], Api = aBuf[pi + 1];
            const Aqr = aBuf[qi], Aqi = aBuf[qi + 1];
            const seqr = s * (epre * Aqr - epim * Aqi);
            const seqi = s * (epre * Aqi + epim * Aqr);
            const ceqr = c2 * (epre * Aqr - epim * Aqi);
            const ceqi = c2 * (epre * Aqi + epim * Aqr);
            aBuf[pi] = c2 * Apr + seqr;
            aBuf[pi + 1] = c2 * Api + seqi;
            aBuf[qi] = -s * Apr + ceqr;
            aBuf[qi + 1] = -s * Api + ceqi;
          }
          for (let i = 0; i < cols; i++) {
            const pi = (p * cols + i) * 2, qi = (q * cols + i) * 2;
            const Vpr = vBuf[pi], Vpi = vBuf[pi + 1];
            const Vqr = vBuf[qi], Vqi = vBuf[qi + 1];
            const seVr = s * (epre * Vqr - epim * Vqi);
            const seVi = s * (epre * Vqi + epim * Vqr);
            const ceVr = c2 * (epre * Vqr - epim * Vqi);
            const ceVi = c2 * (epre * Vqi + epim * Vqr);
            vBuf[pi] = c2 * Vpr + seVr;
            vBuf[pi + 1] = c2 * Vpi + seVi;
            vBuf[qi] = -s * Vpr + ceVr;
            vBuf[qi + 1] = -s * Vpi + ceVi;
          }
        }
      }
      if (frobSq < 1e-28 || maxOff < 1e-14 * frobSq) break;
    }
    const sigmaBuf = this.sigmaBuf;
    for (let k = 0; k < cols; k++) {
      let s2 = 0;
      for (let i = 0; i < rows; i++) {
        const re = aBuf[(k * rows + i) * 2], im = aBuf[(k * rows + i) * 2 + 1];
        s2 += re * re + im * im;
      }
      sigmaBuf[k] = Math.sqrt(s2);
    }
    const orderBuf = this.orderBuf;
    for (let k = 0; k < cols; k++) orderBuf[k] = k;
    for (let i = 1; i < cols; i++) {
      let j = i;
      while (j > 0 && sigmaBuf[orderBuf[j - 1]] < sigmaBuf[orderBuf[j]]) {
        const tmp = orderBuf[j - 1];
        orderBuf[j - 1] = orderBuf[j];
        orderBuf[j] = tmp;
        j--;
      }
    }
    const sigma0 = sigmaBuf[orderBuf[0]];
    const cutoff = sigma0 > 0 ? Math.max(1e-14, this.truncErr * sigma0) : 1e-14;
    let bond = 0;
    while (bond < cols && bond < maxChi && sigmaBuf[orderBuf[bond]] > cutoff) bond++;
    if (bond === 0) bond = 1;
    if (bond < cols && sigmaBuf[orderBuf[bond]] > 1e-14) {
      this.wasTruncated_ = true;
    }
    const qBuf = this.qBuf;
    for (let k = 0; k < bond; k++) {
      const col = orderBuf[k];
      const sigma = sigmaBuf[col];
      const inv = sigma > 1e-14 ? 1 / sigma : 0;
      for (let i = 0; i < rows; i++) {
        const src = (col * rows + i) * 2;
        qBuf[(i * bond + k) * 2] = aBuf[src] * inv;
        qBuf[(i * bond + k) * 2 + 1] = aBuf[src + 1] * inv;
      }
    }
    const rBuf = this.rBuf;
    for (let k = 0; k < bond; k++) {
      const col = orderBuf[k];
      for (let j = 0; j < cols; j++) {
        const vi = (col * cols + j) * 2;
        rBuf[(k * cols + j) * 2] = vBuf[vi];
        rBuf[(k * cols + j) * 2 + 1] = -vBuf[vi + 1];
      }
    }
    return bond;
  }
  /**
   * Sample a basis state by sequential left-to-right marginal collapse.
   * No allocations — uses pre-allocated sv0/sv1/st workspace.
   */
  sample(rand) {
    const stRe = this.stRe, stIm = this.stIm;
    stRe[0] = 1;
    stIm[0] = 0;
    let result = 0n;
    for (let q = 0; q < this.n; q++) {
      const data = this.data[q];
      const chiL = this.chiL[q], chiR = this.chiR[q];
      const v0re = this.sv0re, v0im = this.sv0im;
      const v1re = this.sv1re, v1im = this.sv1im;
      for (let r = 0; r < chiR; r++) {
        v0re[r] = 0;
        v0im[r] = 0;
        v1re[r] = 0;
        v1im[r] = 0;
      }
      for (let l = 0; l < chiL; l++) {
        const sre = stRe[l], sim = stIm[l];
        for (let r = 0; r < chiR; r++) {
          const i0 = ((l * 2 + 0) * chiR + r) * 2;
          const i1 = ((l * 2 + 1) * chiR + r) * 2;
          v0re[r] = (v0re[r] ?? 0) + sre * data[i0] - sim * data[i0 + 1];
          v0im[r] = (v0im[r] ?? 0) + sre * data[i0 + 1] + sim * data[i0];
          v1re[r] = (v1re[r] ?? 0) + sre * data[i1] - sim * data[i1 + 1];
          v1im[r] = (v1im[r] ?? 0) + sre * data[i1 + 1] + sim * data[i1];
        }
      }
      if (q < this.n - 1) {
        const lambda = this.bondLambda[q];
        for (let r = 0; r < chiR; r++) {
          const s = lambda[r];
          v0re[r] = (v0re[r] ?? 0) * s;
          v0im[r] = (v0im[r] ?? 0) * s;
          v1re[r] = (v1re[r] ?? 0) * s;
          v1im[r] = (v1im[r] ?? 0) * s;
        }
      }
      let p0 = 0, p1 = 0;
      for (let r = 0; r < chiR; r++) {
        p0 += v0re[r] * v0re[r] + v0im[r] * v0im[r];
        p1 += v1re[r] * v1re[r] + v1im[r] * v1im[r];
      }
      const total = p0 + p1;
      const bit = total > 0 && rand() >= p0 / total ? 1 : 0;
      if (bit === 1) result |= 1n << BigInt(q);
      const chosen = bit === 0 ? p0 : p1;
      const inv = chosen > 0 ? 1 / Math.sqrt(chosen) : 0;
      const vRe = bit === 0 ? v0re : v1re;
      const vIm = bit === 0 ? v0im : v1im;
      for (let r = 0; r < chiR; r++) {
        stRe[r] = vRe[r] * inv;
        stIm[r] = vIm[r] * inv;
      }
    }
    return result;
  }
  /**
   * Project qubit `q` onto a computational basis state in-place.
   *
   * Probability is computed exactly from the Vidal form bond lambdas.
   * The site tensor Γ[q] is projected and renormalized; all other tensors
   * are unchanged, so Vidal canonical form is preserved.
   *
   * O(chiL · chiR) — no SVD, no allocation.
   * Returns the measurement outcome (0 or 1).
   */
  measure(q, rand) {
    const data = this.data[q];
    const chiL = this.chiL[q], chiR = this.chiR[q];
    const lamL = q > 0 ? this.bondLambda[q - 1] : null;
    const lamR = q < this.n - 1 ? this.bondLambda[q] : null;
    let p1 = 0;
    for (let l = 0; l < chiL; l++) {
      const wL = lamL ? lamL[l] * lamL[l] : 1;
      for (let r = 0; r < chiR; r++) {
        const wR = lamR ? lamR[r] * lamR[r] : 1;
        const i1 = ((l * 2 + 1) * chiR + r) * 2;
        p1 += wL * wR * (data[i1] * data[i1] + data[i1 + 1] * data[i1 + 1]);
      }
    }
    const bit = rand() < p1 ? 1 : 0;
    const pKeep = bit === 1 ? p1 : 1 - p1;
    const inv = pKeep > 0 ? 1 / Math.sqrt(pKeep) : 0;
    for (let l = 0; l < chiL; l++) {
      for (let r = 0; r < chiR; r++) {
        const i0 = ((l * 2 + 0) * chiR + r) * 2;
        const i1 = ((l * 2 + 1) * chiR + r) * 2;
        const s0 = bit === 0 ? inv : 0;
        const s1 = bit === 1 ? inv : 0;
        data[i0] *= s0;
        data[i0 + 1] *= s0;
        data[i1] *= s1;
        data[i1 + 1] *= s1;
      }
    }
    return bit;
  }
  /**
   * Probability of qubit q being in state |1⟩ without collapsing.
   * Same formula as `measure()` but read-only: O(chiL·chiR).
   */
  pQubit(q) {
    const data = this.data[q];
    const chiL = this.chiL[q], chiR = this.chiR[q];
    const lamL = q > 0 ? this.bondLambda[q - 1] : null;
    const lamR = q < this.n - 1 ? this.bondLambda[q] : null;
    let p1 = 0;
    for (let l = 0; l < chiL; l++) {
      const wL = lamL ? lamL[l] * lamL[l] : 1;
      for (let r = 0; r < chiR; r++) {
        const wR = lamR ? lamR[r] * lamR[r] : 1;
        const i1 = ((l * 2 + 1) * chiR + r) * 2;
        p1 += wL * wR * (data[i1] * data[i1] + data[i1 + 1] * data[i1 + 1]);
      }
    }
    return p1;
  }
  /** Maximum bond dimension currently in use across all sites. */
  maxBondUsed() {
    let max = 1;
    for (let q = 0; q < this.n; q++) {
      if (this.chiL[q] > max) max = this.chiL[q];
      if (this.chiR[q] > max) max = this.chiR[q];
    }
    return max;
  }
  /**
   * Von Neumann entanglement entropies S_b = -Σ_k σ_k² log₂(σ_k²) at each bond.
   * Returns n-1 values. Uses the stored bondLambda Schmidt values (left-canonical).
   * O(n · χ) — suitable for circuit monitoring and truncation diagnostics.
   */
  bondEntropies() {
    const result = new Array(this.n - 1);
    for (let b = 0; b < this.n - 1; b++) {
      const lambda = this.bondLambda[b];
      const chi = this.chiR[b];
      let S2 = 0;
      for (let k = 0; k < chi; k++) {
        const s = lambda[k];
        if (s > 1e-14) S2 -= s * s * Math.log2(s * s);
      }
      result[b] = S2;
    }
    return result;
  }
  /**
   * Single-site expectation value ⟨ψ|I⊗…⊗O_q⊗…⊗I|ψ⟩ directly from MPS tensors.
   *
   * Exploits the Vidal canonical form: the left environment at site q is Λ[q-1]² and
   * the right environment is Λ[q]², so the result is exact with no sampling variance.
   * O(χ²) per call.
   *
   * @param q  Site index (0-based).
   * @param op 2×2 gate matrix representing the observable.
   */
  expect1(q, op) {
    const data = this.data[q];
    const chiL = this.chiL[q], chiR = this.chiR[q];
    const lamL = q > 0 ? this.bondLambda[q - 1] : null;
    const lamR = q < this.n - 1 ? this.bondLambda[q] : null;
    const [[a, b], [c2, d]] = op;
    const are = a.re, aim = a.im, bre = b.re, bim = b.im;
    const cre = c2.re, cim = c2.im, dre = d.re, dim = d.im;
    let re = 0, im = 0;
    for (let l = 0; l < chiL; l++) {
      const wl = lamL !== null ? lamL[l] * lamL[l] : 1;
      for (let r = 0; r < chiR; r++) {
        const w = wl * (lamR !== null ? lamR[r] * lamR[r] : 1);
        const i0 = ((l * 2 + 0) * chiR + r) * 2;
        const i1 = ((l * 2 + 1) * chiR + r) * 2;
        const g0re = data[i0], g0im = data[i0 + 1];
        const g1re = data[i1], g1im = data[i1 + 1];
        const og0re = are * g0re - aim * g0im + bre * g1re - bim * g1im;
        const og0im = are * g0im + aim * g0re + bre * g1im + bim * g1re;
        const og1re = cre * g0re - cim * g0im + dre * g1re - dim * g1im;
        const og1im = cre * g0im + cim * g0re + dre * g1im + dim * g1re;
        re += w * (g0re * og0re + g0im * og0im + g1re * og1re + g1im * og1im);
        im += w * (g0re * og0im - g0im * og0re + g1re * og1im - g1im * og1re);
      }
    }
    return { re, im };
  }
  /**
   * Expectation value ⟨ψ|O₀⊗O₁⊗…⊗O_{n-1}|ψ⟩ for a product observable.
   *
   * Pass `null` for identity at a site. Uses a left-to-right transfer matrix sweep in the
   * Vidal Γ-Λ basis. O(n·χ³) — far cheaper than building a density matrix or sampling.
   *
   * Fast path: leading identity sites cost O(χ) each (E stays diagonal).
   * After the first non-identity site the full O(χ³) contraction runs for remaining sites.
   *
   * Typical use: Hamiltonian terms in VQE, two-point correlators, Pauli strings.
   *
   * @param ops Array of length n. null = identity at that site.
   */
  expectation(ops) {
    if (ops.length !== this.n)
      throw new TypeError(`ops.length (${ops.length}) must equal n (${this.n})`);
    const n = this.n;
    const ogBuf = this.exOgBuf;
    const fBuf = this.exFBuf;
    const curBuf = this.exCurBuf;
    const nxtBuf = this.exNxtBuf;
    const diagBuf = this.exDiag;
    diagBuf[0] = 1;
    let isDiag = true;
    let curDim = 1;
    for (let q = 0; q < n; q++) {
      const op = ops[q] ?? null;
      const data = this.data[q];
      const chiL = this.chiL[q], chiR = this.chiR[q];
      const lam = q < n - 1 ? this.bondLambda[q] : null;
      if (op === null && isDiag) {
        if (lam !== null) {
          for (let r = 0; r < chiR; r++) {
            const s = lam[r];
            diagBuf[r] = s * s;
          }
        } else {
          diagBuf[0] = 1;
        }
        curDim = chiR;
        continue;
      }
      if (op !== null) {
        const [[a, b], [c2, d]] = op;
        const are = a.re, aim = a.im, bre = b.re, bim = b.im;
        const cre = c2.re, cim = c2.im, dre = d.re, dim = d.im;
        for (let l = 0; l < chiL; l++) {
          for (let r = 0; r < chiR; r++) {
            const i0 = ((l * 2 + 0) * chiR + r) * 2;
            const i1 = ((l * 2 + 1) * chiR + r) * 2;
            const g0re = data[i0], g0im = data[i0 + 1];
            const g1re = data[i1], g1im = data[i1 + 1];
            ogBuf[i0] = are * g0re - aim * g0im + bre * g1re - bim * g1im;
            ogBuf[i0 + 1] = are * g0im + aim * g0re + bre * g1im + bim * g1re;
            ogBuf[i1] = cre * g0re - cim * g0im + dre * g1re - dim * g1im;
            ogBuf[i1 + 1] = cre * g0im + cim * g0re + dre * g1im + dim * g1re;
          }
        }
      }
      const src = op !== null ? ogBuf : data;
      for (let pp = 0; pp < 2; pp++) {
        for (let r = 0; r < chiR; r++) {
          for (let lp = 0; lp < chiL; lp++) {
            let fRe = 0, fIm = 0;
            if (isDiag) {
              const si = ((lp * 2 + pp) * chiR + r) * 2;
              fRe = diagBuf[lp] * src[si];
              fIm = diagBuf[lp] * src[si + 1];
            } else {
              for (let l = 0; l < chiL; l++) {
                const eIdx = (l * curDim + lp) * 2;
                const eRe = curBuf[eIdx], eIm = curBuf[eIdx + 1];
                const si = ((l * 2 + pp) * chiR + r) * 2;
                const sRe = src[si], sIm = src[si + 1];
                fRe += eRe * sRe - eIm * sIm;
                fIm += eRe * sIm + eIm * sRe;
              }
            }
            const fi = ((pp * chiR + r) * chiL + lp) * 2;
            fBuf[fi] = fRe;
            fBuf[fi + 1] = fIm;
          }
        }
      }
      nxtBuf.fill(0, 0, chiR * chiR * 2);
      for (let r = 0; r < chiR; r++) {
        const lamR = lam !== null ? lam[r] : 1;
        for (let rp = 0; rp < chiR; rp++) {
          const w = lamR * (lam !== null ? lam[rp] : 1);
          let accRe = 0, accIm = 0;
          for (let pp = 0; pp < 2; pp++) {
            for (let lp = 0; lp < chiL; lp++) {
              const fi = ((pp * chiR + r) * chiL + lp) * 2;
              const fRe = fBuf[fi], fIm = fBuf[fi + 1];
              const gi = ((lp * 2 + pp) * chiR + rp) * 2;
              const gRe = data[gi], gIm = -data[gi + 1];
              accRe += fRe * gRe - fIm * gIm;
              accIm += fRe * gIm + fIm * gRe;
            }
          }
          const ni = (r * chiR + rp) * 2;
          nxtBuf[ni] = w * accRe;
          nxtBuf[ni + 1] = w * accIm;
        }
      }
      curBuf.set(nxtBuf.subarray(0, chiR * chiR * 2));
      isDiag = false;
      curDim = chiR;
    }
    if (isDiag) return { re: diagBuf[0], im: 0 };
    let re = 0, im = 0;
    for (let r = 0; r < curDim; r++) {
      re += curBuf[(r * curDim + r) * 2];
      im += curBuf[(r * curDim + r) * 2 + 1];
    }
    return { re, im };
  }
};
var TWO_PAULI_TRAJ = [
  [null, X],
  [null, Y],
  [null, Z],
  [X, null],
  [X, X],
  [X, Y],
  [X, Z],
  [Y, null],
  [Y, X],
  [Y, Y],
  [Y, Z],
  [Z, null],
  [Z, X],
  [Z, Y],
  [Z, Z]
];
function dep1Traj(traj, q, p, rand) {
  if (rand >= p) return;
  const r = rand / p;
  if (r < 1 / 3) traj.apply1(q, X);
  else if (r < 2 / 3) traj.apply1(q, Y);
  else traj.apply1(q, Z);
}
function dampAmpTraj(traj, q, gamma, rng) {
  if (gamma <= 0) return;
  const p1 = traj.pQubit(q);
  const pJump = gamma * p1;
  if (pJump < JUMP_THRESHOLD) return;
  if (rng() < pJump) {
    traj.measure(q, () => 0);
    traj.apply1(q, X);
  } else {
    const sqG = Math.sqrt(1 - gamma);
    const invN = 1 / Math.sqrt(1 - pJump);
    const K0 = [
      [{ re: invN, im: 0 }, { re: 0, im: 0 }],
      [{ re: 0, im: 0 }, { re: sqG * invN, im: 0 }]
    ];
    traj.apply1(q, K0);
  }
}
function dampPhaseTraj(traj, q, lambda, rng) {
  if (lambda <= 0) return;
  const p1 = traj.pQubit(q);
  const pJump = lambda * p1;
  if (pJump < JUMP_THRESHOLD) return;
  if (rng() < pJump) {
    traj.measure(q, () => 0);
  } else {
    const sqL = Math.sqrt(1 - lambda);
    const invN = 1 / Math.sqrt(1 - pJump);
    const K0 = [
      [{ re: invN, im: 0 }, { re: 0, im: 0 }],
      [{ re: 0, im: 0 }, { re: sqL * invN, im: 0 }]
    ];
    traj.apply1(q, K0);
  }
}
function dep2Traj(traj, a, b, p, rand) {
  if (rand >= p) return;
  const [pa, pb] = TWO_PAULI_TRAJ[Math.min(Math.floor(rand / p * 15), 14)];
  if (pa) traj.apply1(a, pa);
  if (pb) traj.apply1(b, pb);
}
function applyTrajOps(traj, ops, p1, p2, rng, shotCregs, pMeas = 0, gamma = 0, lambda = 0) {
  for (const op of ops) {
    switch (op.kind) {
      case "single":
        traj.apply1(op.q, op.gate);
        if (p1) dep1Traj(traj, op.q, p1, rng());
        if (gamma) dampAmpTraj(traj, op.q, gamma, rng);
        if (lambda) dampPhaseTraj(traj, op.q, lambda, rng);
        break;
      case "cnot":
        traj.apply2(op.control, op.target, CNOT4);
        if (p2) dep2Traj(traj, op.control, op.target, p2, rng());
        if (gamma) {
          dampAmpTraj(traj, op.control, gamma, rng);
          dampAmpTraj(traj, op.target, gamma, rng);
        }
        if (lambda) {
          dampPhaseTraj(traj, op.control, lambda, rng);
          dampPhaseTraj(traj, op.target, lambda, rng);
        }
        break;
      case "swap":
        traj.apply2(op.a, op.b, SWAP4);
        if (p2) dep2Traj(traj, op.a, op.b, p2, rng());
        if (gamma) {
          dampAmpTraj(traj, op.a, gamma, rng);
          dampAmpTraj(traj, op.b, gamma, rng);
        }
        if (lambda) {
          dampPhaseTraj(traj, op.a, lambda, rng);
          dampPhaseTraj(traj, op.b, lambda, rng);
        }
        break;
      case "two":
        traj.apply2(op.a, op.b, op.gate);
        if (p2) dep2Traj(traj, op.a, op.b, p2, rng());
        if (gamma) {
          dampAmpTraj(traj, op.a, gamma, rng);
          dampAmpTraj(traj, op.b, gamma, rng);
        }
        if (lambda) {
          dampPhaseTraj(traj, op.a, lambda, rng);
          dampPhaseTraj(traj, op.b, lambda, rng);
        }
        break;
      case "measure": {
        const raw = traj.measure(op.q, rng);
        const reported = pMeas && rng() < pMeas ? raw ^ 1 : raw;
        const reg = shotCregs?.get(op.creg);
        if (reg) reg[op.bit] = reported === 1;
        break;
      }
      case "reset":
        if (traj.measure(op.q, rng) === 1) traj.apply1(op.q, X);
        break;
      case "if": {
        const val = (shotCregs?.get(op.creg) ?? []).reduce((acc, b, i) => b ? acc | 1 << i : acc, 0);
        if (val === op.value) applyTrajOps(traj, op.ops, p1, p2, rng, shotCregs, pMeas, gamma, lambda);
        break;
      }
      case "barrier":
        break;
      default: {
        const _exhaustive = op;
        break;
      }
    }
  }
}

// src/worker-shim.ts
var wt = await import("node:worker_threads").catch(() => null);

// src/density.ts
function dmGet(dm, shift, r, c2) {
  return dm.get(r << shift | c2) ?? ZERO;
}
function dmSet(dm, shift, r, c2, v) {
  const k = r << shift | c2;
  if (!isNegligible(v)) dm.set(k, v);
  else dm.delete(k);
}
function dmAcc(dm, shift, r, c2, v) {
  const k = r << shift | c2;
  const ex = dm.get(k);
  const nx = ex ? add(ex, v) : v;
  if (!isNegligible(nx)) dm.set(k, nx);
  else dm.delete(k);
}
function applySingle2(dm, n, q, [[a, b], [c2, d]]) {
  const next = /* @__PURE__ */ new Map();
  const shift = BigInt(n);
  const dimMask = (1n << shift) - 1n;
  const qMask = 1n << BigInt(q);
  const keyRMask = qMask << shift;
  const keyCMask = qMask;
  const seen = /* @__PURE__ */ new Set();
  const ca = conj(a), cb = conj(b), cc = conj(c2), cd = conj(d);
  for (const k of dm.keys()) {
    const ctx = k & ~keyRMask & ~keyCMask;
    if (seen.has(ctx)) continue;
    seen.add(ctx);
    const r0 = ctx >> shift, c0 = ctx & dimMask;
    const r1 = r0 | qMask, c1 = c0 | qMask;
    const p00 = dmGet(dm, shift, r0, c0), p01 = dmGet(dm, shift, r0, c1);
    const p10 = dmGet(dm, shift, r1, c0), p11 = dmGet(dm, shift, r1, c1);
    const t00 = add(mul(a, p00), mul(b, p10)), t01 = add(mul(a, p01), mul(b, p11));
    const t10 = add(mul(c2, p00), mul(d, p10)), t11 = add(mul(c2, p01), mul(d, p11));
    dmSet(next, shift, r0, c0, add(mul(ca, t00), mul(cb, t01)));
    dmSet(next, shift, r0, c1, add(mul(cc, t00), mul(cd, t01)));
    dmSet(next, shift, r1, c0, add(mul(ca, t10), mul(cb, t11)));
    dmSet(next, shift, r1, c1, add(mul(cc, t10), mul(cd, t11)));
  }
  return next;
}
function applyTwo2(dm, n, a, b, gate) {
  const next = /* @__PURE__ */ new Map();
  const shift = BigInt(n);
  const dimMask = (1n << shift) - 1n;
  const ma = 1n << BigInt(a), mb = 1n << BigInt(b);
  const seen = /* @__PURE__ */ new Set();
  for (const k of dm.keys()) {
    const ctx = k & ~(ma << shift) & ~(mb << shift) & ~ma & ~mb;
    if (seen.has(ctx)) continue;
    seen.add(ctx);
    const rb = ctx >> shift, cb = ctx & dimMask;
    const rowB = [rb, rb | mb, rb | ma, rb | ma | mb];
    const colB = [cb, cb | mb, cb | ma, cb | ma | mb];
    const p = Array.from(
      { length: 4 },
      (_, ri) => Array.from({ length: 4 }, (_2, ci) => dmGet(dm, shift, rowB[ri], colB[ci]))
    );
    const t = Array.from(
      { length: 4 },
      (_, ri) => Array.from({ length: 4 }, (_2, ci) => {
        let v = ZERO;
        for (let k2 = 0; k2 < 4; k2++) v = add(v, mul(gate[ri][k2], p[k2][ci]));
        return v;
      })
    );
    for (let ri = 0; ri < 4; ri++) {
      for (let ci = 0; ci < 4; ci++) {
        let v = ZERO;
        for (let k2 = 0; k2 < 4; k2++) v = add(v, mul(t[ri][k2], conj(gate[ci][k2])));
        dmSet(next, shift, rowB[ri], colB[ci], v);
      }
    }
  }
  return next;
}
function applyPerm(dm, n, f) {
  const shift = BigInt(n), dimMask = (1n << shift) - 1n;
  const next = /* @__PURE__ */ new Map();
  for (const [k, v] of dm) {
    const r = k >> shift, c2 = k & dimMask;
    next.set(f(r) << shift | f(c2), v);
  }
  return next;
}
function applyUnitaryN(dm, n, qs, matrix) {
  const next = /* @__PURE__ */ new Map();
  const shift = BigInt(n);
  const dimMask = (1n << shift) - 1n;
  const localDim = 1 << qs.length;
  const masks = qs.map((q) => 1n << BigInt(q));
  const allMask = masks.reduce((a, b) => a | b, 0n);
  const seen = /* @__PURE__ */ new Set();
  const buildBases = (base) => Array.from({ length: localDim }, (_, i) => {
    let b = base;
    for (let bit = 0; bit < qs.length; bit++) {
      if (i >> qs.length - 1 - bit & 1) b |= masks[bit];
    }
    return b;
  });
  for (const key of dm.keys()) {
    const ctx = key & ~(allMask << shift) & ~allMask;
    if (seen.has(ctx)) continue;
    seen.add(ctx);
    const rowBases = buildBases(ctx >> shift);
    const colBases = buildBases(ctx & dimMask);
    const p = Array.from(
      { length: localDim },
      (_, ri) => Array.from({ length: localDim }, (_2, ci) => dmGet(dm, shift, rowBases[ri], colBases[ci]))
    );
    const t = Array.from(
      { length: localDim },
      (_, ri) => Array.from({ length: localDim }, (_2, ci) => {
        let v = ZERO;
        for (let j = 0; j < localDim; j++) v = add(v, mul(matrix[ri][j], p[j][ci]));
        return v;
      })
    );
    for (let ri = 0; ri < localDim; ri++) {
      for (let ci = 0; ci < localDim; ci++) {
        let v = ZERO;
        for (let j = 0; j < localDim; j++) v = add(v, mul(t[ri][j], conj(matrix[ci][j])));
        dmSet(next, shift, rowBases[ri], colBases[ci], v);
      }
    }
  }
  return next;
}
function depolarize1(dm, n, q, p) {
  if (p <= 0) return dm;
  const next = /* @__PURE__ */ new Map();
  const shift = BigInt(n);
  const dimMask = (1n << shift) - 1n;
  const qMask = 1n << BigInt(q);
  const keyRMask = qMask << shift;
  const keyCMask = qMask;
  const seen = /* @__PURE__ */ new Set();
  const sa = 1 - 2 * p / 3, sb = 2 * p / 3;
  const cf = 1 - 4 * p / 3;
  for (const k of dm.keys()) {
    const ctx = k & ~keyRMask & ~keyCMask;
    if (seen.has(ctx)) continue;
    seen.add(ctx);
    const r0 = ctx >> shift, c0 = ctx & dimMask;
    const r1 = r0 | qMask, c1 = c0 | qMask;
    const p00 = dmGet(dm, shift, r0, c0), p11 = dmGet(dm, shift, r1, c1);
    dmSet(next, shift, r0, c0, { re: sa * p00.re + sb * p11.re, im: sa * p00.im + sb * p11.im });
    dmSet(next, shift, r1, c1, { re: sa * p11.re + sb * p00.re, im: sa * p11.im + sb * p00.im });
    const p01 = dmGet(dm, shift, r0, c1), p10 = dmGet(dm, shift, r1, c0);
    dmSet(next, shift, r0, c1, { re: cf * p01.re, im: cf * p01.im });
    dmSet(next, shift, r1, c0, { re: cf * p10.re, im: cf * p10.im });
  }
  return next;
}
var PAULI15 = [
  [0, 1, 0, 0],
  [0, 1, 0, 1],
  [0, 0, 0, 1],
  // I⊗X  I⊗Y  I⊗Z
  [1, 0, 0, 0],
  [1, 1, 0, 0],
  [1, 1, 0, 1],
  [1, 0, 0, 1],
  // X⊗I  X⊗X  X⊗Y  X⊗Z
  [1, 0, 1, 0],
  [1, 1, 1, 0],
  [1, 1, 1, 1],
  [1, 0, 1, 1],
  // Y⊗I  Y⊗X  Y⊗Y  Y⊗Z
  [0, 0, 1, 0],
  [0, 1, 1, 0],
  [0, 1, 1, 1],
  [0, 0, 1, 1]
  // Z⊗I  Z⊗X  Z⊗Y  Z⊗Z
];
function depolarize2(dm, n, a, b, p) {
  if (p <= 0) return dm;
  const next = /* @__PURE__ */ new Map();
  const shift = BigInt(n);
  const dimMask = (1n << shift) - 1n;
  const ma = 1n << BigInt(a), mb = 1n << BigInt(b);
  const w1 = 1 - p, w2 = p / 15;
  for (const [k, v] of dm) {
    const r = k >> shift, c2 = k & dimMask;
    dmAcc(next, shift, r, c2, { re: w1 * v.re, im: w1 * v.im });
    const ba_r = Number(r >> BigInt(a) & 1n), bb_r = Number(r >> BigInt(b) & 1n);
    const ba_c = Number(c2 >> BigInt(a) & 1n), bb_c = Number(c2 >> BigInt(b) & 1n);
    for (const [fa, fb, za, zb] of PAULI15) {
      const perm = (fa ? ma : 0n) | (fb ? mb : 0n);
      const parity = za * (ba_r ^ ba_c) ^ zb * (bb_r ^ bb_c);
      const scale2 = parity ? -w2 : w2;
      dmAcc(next, shift, r ^ perm, c2 ^ perm, { re: scale2 * v.re, im: scale2 * v.im });
    }
  }
  return next;
}
function amplitudeDamping1(dm, n, q, gamma) {
  if (gamma <= 0) return dm;
  const next = /* @__PURE__ */ new Map();
  const shift = BigInt(n);
  const dimMsk = (1n << shift) - 1n;
  const qMask = 1n << BigInt(q);
  const sqG = Math.sqrt(1 - gamma);
  const seen = /* @__PURE__ */ new Set();
  for (const k of dm.keys()) {
    const ctx = k & ~(qMask << shift) & ~qMask;
    if (seen.has(ctx)) continue;
    seen.add(ctx);
    const r0 = ctx >> shift, c0 = ctx & dimMsk;
    const r1 = r0 | qMask, c1 = c0 | qMask;
    const v00 = dmGet(dm, shift, r0, c0), v11 = dmGet(dm, shift, r1, c1);
    dmSet(next, shift, r0, c0, { re: v00.re + gamma * v11.re, im: v00.im + gamma * v11.im });
    dmSet(next, shift, r1, c1, { re: (1 - gamma) * v11.re, im: (1 - gamma) * v11.im });
    const v01 = dmGet(dm, shift, r0, c1), v10 = dmGet(dm, shift, r1, c0);
    dmSet(next, shift, r0, c1, { re: sqG * v01.re, im: sqG * v01.im });
    dmSet(next, shift, r1, c0, { re: sqG * v10.re, im: sqG * v10.im });
  }
  return next;
}
function phaseDamping1(dm, n, q, lambda) {
  if (lambda <= 0) return dm;
  const sqL = Math.sqrt(1 - lambda);
  const next = /* @__PURE__ */ new Map();
  const shift = BigInt(n);
  const dimMsk = (1n << shift) - 1n;
  const qMask = 1n << BigInt(q);
  for (const [k, v] of dm) {
    const r = k >> shift, c2 = k & dimMsk;
    const rq = r >> BigInt(q) & 1n;
    const cq = c2 >> BigInt(q) & 1n;
    const s = rq === cq ? 1 : sqL;
    next.set(k, { re: s * v.re, im: s * v.im });
  }
  return next;
}
function applyKraus1DM(dm, n, q, kraus) {
  const shift = BigInt(n);
  const dimMsk = (1n << shift) - 1n;
  const qMask = 1n << BigInt(q);
  const next = /* @__PURE__ */ new Map();
  const seen = /* @__PURE__ */ new Set();
  for (const k of dm.keys()) {
    const ctx = k & ~(qMask << shift) & ~qMask;
    if (seen.has(ctx)) continue;
    seen.add(ctx);
    const r0 = ctx >> shift, c0 = ctx & dimMsk;
    const r1 = r0 | qMask, c1 = c0 | qMask;
    const v00 = dmGet(dm, shift, r0, c0), v01 = dmGet(dm, shift, r0, c1);
    const v10 = dmGet(dm, shift, r1, c0), v11 = dmGet(dm, shift, r1, c1);
    const rho = [[v00, v01], [v10, v11]];
    for (let iq = 0; iq < 2; iq++) {
      for (let jq = 0; jq < 2; jq++) {
        let re = 0, im = 0;
        for (const K of kraus) {
          for (let ip = 0; ip < 2; ip++) {
            for (let jp = 0; jp < 2; jp++) {
              const kip = K[iq][ip], kjp = K[jq][jp];
              const kRe = kip.re * kjp.re + kip.im * kjp.im;
              const kIm = kip.im * kjp.re - kip.re * kjp.im;
              const rij = rho[ip][jp];
              re += kRe * rij.re - kIm * rij.im;
              im += kRe * rij.im + kIm * rij.re;
            }
          }
        }
        const rOut = iq === 0 ? r0 : r1;
        const cOut = jq === 0 ? c0 : c1;
        dmAcc(next, shift, rOut, cOut, { re, im });
      }
    }
  }
  return next;
}
function applyKraus2DM(dm, n, a, b, kraus) {
  const shift = BigInt(n);
  const dimMsk = (1n << shift) - 1n;
  const ma = 1n << BigInt(a), mb = 1n << BigInt(b);
  const contexts = /* @__PURE__ */ new Set();
  const next = /* @__PURE__ */ new Map();
  for (const k of dm.keys()) {
    const r = k >> shift, c2 = k & dimMsk;
    const ctx = (r & ~ma & ~mb) << shift | c2 & ~ma & ~mb;
    contexts.add(ctx);
  }
  for (const ctx of contexts) {
    const rBase = ctx >> shift, cBase = ctx & dimMsk;
    const rho = [];
    for (let ri = 0; ri < 4; ri++) {
      const ra = ri >> 1 & 1, rb = ri & 1;
      const rIdx = rBase | (ra ? ma : 0n) | (rb ? mb : 0n);
      for (let ci = 0; ci < 4; ci++) {
        const ca = ci >> 1 & 1, cb = ci & 1;
        const cIdx = cBase | (ca ? ma : 0n) | (cb ? mb : 0n);
        const v = dmGet(dm, shift, rIdx, cIdx);
        if (!rho[ri]) rho[ri] = [];
        rho[ri][ci * 2] = v.re;
        rho[ri][ci * 2 + 1] = v.im;
      }
    }
    for (let ri = 0; ri < 4; ri++) {
      const ra = ri >> 1 & 1, rb = ri & 1;
      const rIdx = rBase | (ra ? ma : 0n) | (rb ? mb : 0n);
      for (let ci = 0; ci < 4; ci++) {
        const ca = ci >> 1 & 1, cb = ci & 1;
        const cIdx = cBase | (ca ? ma : 0n) | (cb ? mb : 0n);
        let re = 0, im = 0;
        for (const K of kraus) {
          for (let rp = 0; rp < 4; rp++) {
            for (let cp = 0; cp < 4; cp++) {
              const kri = K[ri][rp], kci = K[ci][cp];
              const kRe = kri.re * kci.re + kri.im * kci.im;
              const kIm = kri.im * kci.re - kri.re * kci.im;
              const rhoRe = rho[rp][cp * 2], rhoIm = rho[rp][cp * 2 + 1];
              re += kRe * rhoRe - kIm * rhoIm;
              im += kRe * rhoIm + kIm * rhoRe;
            }
          }
        }
        dmAcc(next, shift, rIdx, cIdx, { re, im });
      }
    }
  }
  return next;
}
var DensityMatrix = class {
  qubits;
  #dm;
  #shift;
  #dimMask;
  /** @internal */
  constructor(qubits, dm) {
    this.qubits = qubits;
    this.#dm = dm;
    this.#shift = BigInt(qubits);
    this.#dimMask = (1n << this.#shift) - 1n;
  }
  /** ρ[row][col]. */
  get(row, col) {
    return dmGet(this.#dm, this.#shift, row, col);
  }
  /**
   * Diagonal probabilities: P(bitstring) = ρ[bs][bs].
   *
   * Keys are standard bitstrings (q0 leftmost).  Only non-negligible values
   * (> 1e-14) are included.
   */
  probabilities() {
    const out = {};
    for (const [k, v] of this.#dm) {
      const r = k >> this.#shift, c2 = k & this.#dimMask;
      if (r === c2 && v.re > 1e-14) out[r.toString(2).padStart(this.qubits, "0").split("").reverse().join("")] = v.re;
    }
    return Object.freeze(out);
  }
  /**
   * Purity Tr(ρ²) = Σ_{r,c} |ρ[r][c]|².
   *
   * Equals 1 for a pure state; equals 1/2ⁿ for the maximally mixed state.
   * Values below 1 indicate entanglement-induced or noise-induced mixing.
   */
  purity() {
    let p = 0;
    for (const v of this.#dm.values()) p += v.re * v.re + v.im * v.im;
    return p;
  }
  /**
   * Von Neumann entropy S = −Tr(ρ log₂ ρ) in bits.
   *
   * Computed by diagonalising the full 2ⁿ × 2ⁿ density matrix via Jacobi
   * iteration.  Practical for n ≤ 8 (matrix size ≤ 256 × 256).
   *
   * @throws RangeError for n > 12 (4096 × 4096 matrix — too expensive).
   */
  entropy() {
    const dim = 1 << this.qubits;
    if (dim > 4096) throw new RangeError(`entropy(): circuit too large (n=${this.qubits}, dim=${dim})`);
    const re = new Float64Array(dim * dim);
    const im = new Float64Array(dim * dim);
    for (const [k, v] of this.#dm) {
      const r = Number(k >> this.#shift), c2 = Number(k & this.#dimMask);
      re[r * dim + c2] = v.re;
      im[r * dim + c2] = v.im;
    }
    const \u03BB = jacobiEigenvalues(re, im, dim);
    let S2 = 0;
    for (const lam of \u03BB) {
      if (lam > 1e-14) S2 -= lam * Math.log2(lam);
    }
    return S2;
  }
  /**
   * Bloch sphere coordinates (θ, φ) for qubit q from the reduced density matrix.
   *
   * Computes ρ_q = Tr_{others}(ρ) then extracts:
   *   rx = 2·Re(ρ_q[0][1]),  ry = −2·Im(ρ_q[0][1]),  rz = ρ_q[0][0] − ρ_q[1][1]
   *
   * - θ = arccos(rz)  ∈ [0, π]
   * - φ = atan2(ry, rx)  ∈ (−π, π]
   */
  blochAngles(q) {
    const qMask = 1n << BigInt(q);
    let rho00 = 0, rho11 = 0, rho01re = 0, rho01im = 0;
    for (const [k, v] of this.#dm) {
      const r = k >> this.#shift, c2 = k & this.#dimMask;
      if (r === c2) {
        if ((r & qMask) === 0n) rho00 += v.re;
        else rho11 += v.re;
      } else if ((r & qMask) === 0n && c2 === (r | qMask)) {
        rho01re += v.re;
        rho01im += v.im;
      }
    }
    const rz = rho00 - rho11, rx = 2 * rho01re, ry = -2 * rho01im;
    return { theta: Math.acos(Math.max(-1, Math.min(1, rz))), phi: Math.atan2(ry, rx) };
  }
};
function jacobiEigenvalues(re, im, n) {
  const R = new Float64Array(re), Im = new Float64Array(im);
  for (let sweep = 0; sweep < 30 * n; sweep++) {
    let maxOff = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const oRe = R[p * n + q], oIm = Im[p * n + q];
        const off2 = oRe * oRe + oIm * oIm;
        if (off2 > maxOff) maxOff = off2;
      }
    }
    if (maxOff < 1e-28) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const oRe = R[p * n + q], oIm = Im[p * n + q];
        const off2 = oRe * oRe + oIm * oIm;
        if (off2 < 1e-28) continue;
        const phi = Math.atan2(oIm, oRe);
        const mag = Math.sqrt(off2);
        const tau = (R[q * n + q] - R[p * n + p]) / 2;
        const t = mag / (Math.abs(tau) + Math.sqrt(tau * tau + mag * mag)) * (tau < 0 ? -1 : 1);
        const cg = 1 / Math.sqrt(1 + t * t);
        const sg = t * cg;
        const sre = sg * Math.cos(phi), sim = sg * Math.sin(phi);
        for (let k = 0; k < n; k++) {
          const xRe = R[p * n + k], xIm = Im[p * n + k];
          const yRe = R[q * n + k], yIm = Im[q * n + k];
          R[p * n + k] = cg * xRe + sre * yRe - sim * yIm;
          Im[p * n + k] = cg * xIm + sre * yIm + sim * yRe;
          R[q * n + k] = -sre * xRe + sim * xIm + cg * yRe;
          Im[q * n + k] = -sre * xIm - sim * xRe + cg * yIm;
        }
        for (let k = 0; k < n; k++) {
          const xRe = R[k * n + p], xIm = Im[k * n + p];
          const yRe = R[k * n + q], yIm = Im[k * n + q];
          R[k * n + p] = cg * xRe + sre * yRe + sim * yIm;
          Im[k * n + p] = cg * xIm + sre * yIm - sim * yRe;
          R[k * n + q] = -sre * xRe + sim * xIm + cg * yRe;
          Im[k * n + q] = -sre * xIm - sim * xRe + cg * yIm;
        }
      }
    }
  }
  return Array.from({ length: n }, (_, i) => R[i * n + i]);
}
var DM_DEVICE_NOISE = {
  "aria-1": { p1: 3e-4, p2: 5e-3 },
  "forte-1": { p1: 1e-4, p2: 2e-3 },
  "harmony": { p1: 1e-3, p2: 0.015 }
};
function runDM(ops, qubits, noise) {
  const p1 = noise?.p1 ?? 0;
  const p2 = noise?.p2 ?? 0;
  const gamma = noise?.gamma ?? 0;
  const lambda = noise?.lambda ?? 0;
  const kraus1 = noise?.kraus1;
  const kraus2 = noise?.kraus2;
  let dm = /* @__PURE__ */ new Map([[0n, { re: 1, im: 0 }]]);
  const n = qubits;
  const sq22 = 1 / Math.sqrt(2);
  const SRISW = [
    [{ re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }],
    [{ re: 0, im: 0 }, { re: sq22, im: 0 }, { re: 0, im: sq22 }, { re: 0, im: 0 }],
    [{ re: 0, im: 0 }, { re: 0, im: sq22 }, { re: sq22, im: 0 }, { re: 0, im: 0 }],
    [{ re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }]
  ];
  for (const op of ops) {
    switch (op.kind) {
      case "single":
        dm = applySingle2(dm, n, op.q, op.gate);
        if (p1) dm = depolarize1(dm, n, op.q, p1);
        if (gamma) dm = amplitudeDamping1(dm, n, op.q, gamma);
        if (lambda) dm = phaseDamping1(dm, n, op.q, lambda);
        if (kraus1) dm = applyKraus1DM(dm, n, op.q, kraus1);
        break;
      case "cnot": {
        const cm = 1n << BigInt(op.control), tm = 1n << BigInt(op.target);
        dm = applyPerm(dm, n, (i) => (i & cm) !== 0n ? i ^ tm : i);
        if (p2) dm = depolarize2(dm, n, op.control, op.target, p2);
        if (gamma) {
          dm = amplitudeDamping1(dm, n, op.control, gamma);
          dm = amplitudeDamping1(dm, n, op.target, gamma);
        }
        if (lambda) {
          dm = phaseDamping1(dm, n, op.control, lambda);
          dm = phaseDamping1(dm, n, op.target, lambda);
        }
        if (kraus2) dm = applyKraus2DM(dm, n, op.control, op.target, kraus2);
        break;
      }
      case "swap": {
        const am = 1n << BigInt(op.a), bm = 1n << BigInt(op.b);
        dm = applyPerm(dm, n, (i) => {
          const ba = (i & am) !== 0n, bb = (i & bm) !== 0n;
          return ba === bb ? i : i ^ am ^ bm;
        });
        if (p2) dm = depolarize2(dm, n, op.a, op.b, p2);
        if (gamma) {
          dm = amplitudeDamping1(dm, n, op.a, gamma);
          dm = amplitudeDamping1(dm, n, op.b, gamma);
        }
        if (lambda) {
          dm = phaseDamping1(dm, n, op.a, lambda);
          dm = phaseDamping1(dm, n, op.b, lambda);
        }
        if (kraus2) dm = applyKraus2DM(dm, n, op.a, op.b, kraus2);
        break;
      }
      case "two":
        dm = applyTwo2(dm, n, op.a, op.b, op.gate);
        if (p2) dm = depolarize2(dm, n, op.a, op.b, p2);
        if (gamma) {
          dm = amplitudeDamping1(dm, n, op.a, gamma);
          dm = amplitudeDamping1(dm, n, op.b, gamma);
        }
        if (lambda) {
          dm = phaseDamping1(dm, n, op.a, lambda);
          dm = phaseDamping1(dm, n, op.b, lambda);
        }
        if (kraus2) dm = applyKraus2DM(dm, n, op.a, op.b, kraus2);
        break;
      case "controlled":
        dm = applyTwo2(dm, n, op.control, op.target, controlledGate(op.gate));
        if (p2) dm = depolarize2(dm, n, op.control, op.target, p2);
        if (gamma) {
          dm = amplitudeDamping1(dm, n, op.control, gamma);
          dm = amplitudeDamping1(dm, n, op.target, gamma);
        }
        if (lambda) {
          dm = phaseDamping1(dm, n, op.control, lambda);
          dm = phaseDamping1(dm, n, op.target, lambda);
        }
        if (kraus2) dm = applyKraus2DM(dm, n, op.control, op.target, kraus2);
        break;
      case "toffoli": {
        const c1m = 1n << BigInt(op.c1), c2m = 1n << BigInt(op.c2), tm = 1n << BigInt(op.target);
        dm = applyPerm(dm, n, (i) => (i & c1m) !== 0n && (i & c2m) !== 0n ? i ^ tm : i);
        break;
      }
      case "cswap": {
        const cm = 1n << BigInt(op.control), am = 1n << BigInt(op.a), bm = 1n << BigInt(op.b);
        dm = applyPerm(dm, n, (i) => {
          if ((i & cm) === 0n) return i;
          const ba = (i & am) !== 0n, bb = (i & bm) !== 0n;
          return ba === bb ? i : i ^ am ^ bm;
        });
        break;
      }
      case "csrswap": {
        const cm = 1n << BigInt(op.control);
        const shift = BigInt(n);
        const dimMsk = (1n << shift) - 1n;
        const blocks = [/* @__PURE__ */ new Map(), /* @__PURE__ */ new Map(), /* @__PURE__ */ new Map(), /* @__PURE__ */ new Map()];
        for (const [k, v] of dm) {
          const r = k >> shift, c2 = k & dimMsk;
          const idx = ((r & cm) !== 0n ? 2 : 0) | ((c2 & cm) !== 0n ? 1 : 0);
          blocks[idx].set(k, v);
        }
        let dm11 = applyTwo2(blocks[3], n, op.a, op.b, SRISW);
        const dm01t = /* @__PURE__ */ new Map();
        for (const [k, v] of blocks[1]) {
          const r = k >> shift, c2 = k & dimMsk;
          dm01t.set(c2 << shift | r, v);
        }
        let dm01tApplied = applyTwo2(dm01t, n, op.a, op.b, SRISW);
        const dm01 = /* @__PURE__ */ new Map();
        for (const [k, v] of dm01tApplied) {
          const r = k >> shift, c2 = k & dimMsk;
          dm01.set(c2 << shift | r, v);
        }
        let dm10 = applyTwo2(blocks[2], n, op.a, op.b, SRISW);
        const dmNext = new Map(blocks[0]);
        for (const [k, v] of dm01) dmAcc(dmNext, shift, k >> shift, k & dimMsk, v);
        for (const [k, v] of dm10) dmAcc(dmNext, shift, k >> shift, k & dimMsk, v);
        for (const [k, v] of dm11) dmAcc(dmNext, shift, k >> shift, k & dimMsk, v);
        dm = dmNext;
        break;
      }
      case "unitary":
        dm = applyUnitaryN(dm, n, op.qubits, op.matrix);
        break;
      default: {
        const _exhaustive = op;
      }
    }
  }
  return new DensityMatrix(qubits, dm);
}

// src/clifford.ts
function popcount32(v) {
  let x = v | 0;
  x = x - (x >>> 1 & 1431655765);
  x = (x & 858993459) + (x >>> 2 & 858993459);
  x = x + (x >>> 4) & 252645135;
  return Math.imul(x, 16843009) >>> 24;
}
var CliffordSim = class {
  n;
  W;
  // words per row = ⌈n/32⌉
  _x;
  // (2n+1)×W packed x-bits
  _z;
  // (2n+1)×W packed z-bits
  _r;
  // 2n+1 phase bits
  constructor(n) {
    this.n = n;
    this.W = n + 31 >> 5;
    const W = this.W;
    const rows = 2 * n + 1;
    this._x = new Int32Array(rows * W);
    this._z = new Int32Array(rows * W);
    this._r = new Array(rows).fill(0);
    for (let i = 0; i < n; i++) {
      this._x[i * W + (i >> 5)] = 1 << (i & 31);
      this._z[(i + n) * W + (i >> 5)] = 1 << (i & 31);
    }
  }
  /**
   * Tableau row multiply: row_i ← row_i · row_h.
   * Phase accumulated via vectorized popcount on packed Pauli word pairs (O(n/32)):
   *   +1 contributions: Y·Z, X·Y, Z·X
   *   -1 contributions: Y·X, X·Z, Z·Y
   * XOR update in the same pass to avoid re-reading words.
   */
  rowmul(i, h) {
    const { W, _x, _z, _r } = this;
    const iW = i * W, hW = h * W;
    let pos = 0, neg = 0;
    for (let w = 0; w < W; w++) {
      const xh = _x[hW + w] ?? 0, zh = _z[hW + w] ?? 0;
      const xi = _x[iW + w] ?? 0, zi = _z[iW + w] ?? 0;
      pos += popcount32(xh & zh & ~xi & zi | xh & ~zh & xi & zi | ~xh & zh & xi & ~zi);
      neg += popcount32(xh & zh & xi & ~zi | xh & ~zh & ~xi & zi | ~xh & zh & xi & zi);
      _x[iW + w] = xi ^ xh;
      _z[iW + w] = zi ^ zh;
    }
    const sum = 2 * ((_r[h] ?? 0) + (_r[i] ?? 0)) + pos - neg;
    _r[i] = (sum % 4 + 4) % 4 >> 1 & 1;
  }
  // ── Single-qubit gates ────────────────────────────────────────────────────
  /** H: swap x↔z for column a, r[i] ^= x[i,a] & z[i,a]. */
  h(a) {
    const { n, W, _x, _z, _r } = this;
    const w = a >> 5, sh = a & 31;
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W;
      const xv = _x[iW + w] ?? 0, zv = _z[iW + w] ?? 0;
      const xi = xv >>> sh & 1, zi = zv >>> sh & 1;
      _r[i] = ((_r[i] ?? 0) ^ xi & zi) & 1;
      const flip = (xi ^ zi) << sh;
      _x[iW + w] = xv ^ flip;
      _z[iW + w] = zv ^ flip;
    }
  }
  /** S: z[i,a] ^= x[i,a], r[i] ^= x[i,a] & z_old[i,a]. */
  s(a) {
    const { n, W, _x, _z, _r } = this;
    const w = a >> 5, sh = a & 31;
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W;
      const xv = _x[iW + w] ?? 0, zv = _z[iW + w] ?? 0;
      const xi = xv >>> sh & 1, zi = zv >>> sh & 1;
      _r[i] = ((_r[i] ?? 0) ^ xi & zi) & 1;
      _z[iW + w] = zv ^ xi << sh;
    }
  }
  /** S†: same z update as S, but r[i] ^= x[i,a] & ~z_old[i,a]. */
  si(a) {
    const { n, W, _x, _z, _r } = this;
    const w = a >> 5, sh = a & 31;
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W;
      const xv = _x[iW + w] ?? 0, zv = _z[iW + w] ?? 0;
      const xi = xv >>> sh & 1, zi = zv >>> sh & 1;
      _r[i] = ((_r[i] ?? 0) ^ xi & (zi ^ 1)) & 1;
      _z[iW + w] = zv ^ xi << sh;
    }
  }
  /** X: r[i] ^= z[i,a]. (X anticommutes with Z, commutes with X.) */
  x(a) {
    const { n, W, _z, _r } = this;
    const w = a >> 5, sh = a & 31;
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W;
      _r[i] = ((_r[i] ?? 0) ^ (_z[iW + w] ?? 0) >>> sh & 1) & 1;
    }
  }
  /** Y: r[i] ^= x[i,a] ^ z[i,a]. (Y anticommutes with X and Z, commutes with Y.) */
  y(a) {
    const { n, W, _x, _z, _r } = this;
    const w = a >> 5, sh = a & 31;
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W;
      const xi = (_x[iW + w] ?? 0) >>> sh & 1;
      const zi = (_z[iW + w] ?? 0) >>> sh & 1;
      _r[i] = ((_r[i] ?? 0) ^ (xi ^ zi)) & 1;
    }
  }
  /** Z: r[i] ^= x[i,a]. (Z anticommutes with X, commutes with Z.) */
  z(a) {
    const { n, W, _x, _r } = this;
    const w = a >> 5, sh = a & 31;
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W;
      _r[i] = ((_r[i] ?? 0) ^ (_x[iW + w] ?? 0) >>> sh & 1) & 1;
    }
  }
  // ── Two-qubit gates ───────────────────────────────────────────────────────
  /** CNOT: x[i,b] ^= x[i,a], z[i,a] ^= z[i,b], r update per CHP §3. */
  cnot(a, b) {
    const { n, W, _x, _z, _r } = this;
    const wA = a >> 5, shA = a & 31;
    const wB = b >> 5, shB = b & 31;
    for (let i = 0; i < 2 * n; i++) {
      const iW = i * W;
      const xvA = _x[iW + wA] ?? 0, xvB = _x[iW + wB] ?? 0;
      const zvA = _z[iW + wA] ?? 0, zvB = _z[iW + wB] ?? 0;
      const xa = xvA >>> shA & 1, xb = xvB >>> shB & 1;
      const za = zvA >>> shA & 1, zb = zvB >>> shB & 1;
      _r[i] = ((_r[i] ?? 0) ^ xa & zb & ((xb ^ za ^ 1) & 1)) & 1;
      _x[iW + wB] = xvB ^ xa << shB;
      _z[iW + wA] = zvA ^ zb << shA;
    }
  }
  /**
   * CZ: z[i,a] ^= x[i,b], z[i,b] ^= x[i,a], r[i] ^= x[i,a] & x[i,b] & (z[i,a] ^ z[i,b]).
   * Derived from H_b · CNOT(a,b) · H_b.
   * Same-word case (a and b share a 32-bit word) handled with a single combined write.
   */
  cz(a, b) {
    const { n, W, _x, _z, _r } = this;
    const wA = a >> 5, shA = a & 31;
    const wB = b >> 5, shB = b & 31;
    if (wA === wB) {
      for (let i = 0; i < 2 * n; i++) {
        const iW = i * W;
        const xv = _x[iW + wA] ?? 0, zv = _z[iW + wA] ?? 0;
        const xa = xv >>> shA & 1, xb = xv >>> shB & 1;
        const za = zv >>> shA & 1, zb = zv >>> shB & 1;
        _r[i] = ((_r[i] ?? 0) ^ xa & xb & (za ^ zb)) & 1;
        _z[iW + wA] = zv ^ xb << shA ^ xa << shB;
      }
    } else {
      for (let i = 0; i < 2 * n; i++) {
        const iW = i * W;
        const xvA = _x[iW + wA] ?? 0, xvB = _x[iW + wB] ?? 0;
        const zvA = _z[iW + wA] ?? 0, zvB = _z[iW + wB] ?? 0;
        const xa = xvA >>> shA & 1, xb = xvB >>> shB & 1;
        const za = zvA >>> shA & 1, zb = zvB >>> shB & 1;
        _r[i] = ((_r[i] ?? 0) ^ xa & xb & (za ^ zb)) & 1;
        _z[iW + wA] = zvA ^ xb << shA;
        _z[iW + wB] = zvB ^ xa << shB;
      }
    }
  }
  /** CY: S†_b · CNOT(a,b) · S_b. */
  cy(a, b) {
    this.si(b);
    this.cnot(a, b);
    this.s(b);
  }
  /**
   * SWAP: swap columns a and b in _x and _z. No phase change.
   * XOR-swap trick handles both same-word and different-word cases correctly.
   */
  swap(a, b) {
    const { n, W, _x, _z } = this;
    const wA = a >> 5, shA = a & 31;
    const wB = b >> 5, shB = b & 31;
    if (wA === wB) {
      for (let i = 0; i < 2 * n; i++) {
        const iW = i * W;
        const xv = _x[iW + wA] ?? 0, zv = _z[iW + wA] ?? 0;
        const xd = (xv >>> shA ^ xv >>> shB) & 1;
        const zd = (zv >>> shA ^ zv >>> shB) & 1;
        _x[iW + wA] = xv ^ (xd << shA | xd << shB);
        _z[iW + wA] = zv ^ (zd << shA | zd << shB);
      }
    } else {
      for (let i = 0; i < 2 * n; i++) {
        const iW = i * W;
        const xvA = _x[iW + wA] ?? 0, xvB = _x[iW + wB] ?? 0;
        const zvA = _z[iW + wA] ?? 0, zvB = _z[iW + wB] ?? 0;
        const xd = (xvA >>> shA ^ xvB >>> shB) & 1;
        const zd = (zvA >>> shA ^ zvB >>> shB) & 1;
        _x[iW + wA] = xvA ^ xd << shA;
        _x[iW + wB] = xvB ^ xd << shB;
        _z[iW + wA] = zvA ^ zd << shA;
        _z[iW + wB] = zvB ^ zd << shB;
      }
    }
  }
  // ── Measurement ───────────────────────────────────────────────────────────
  /** Measure qubit a. rand uniform in [0,1). Returns 0 or 1. */
  measure(a, rand) {
    const { n, W, _x, _z, _r } = this;
    const aw = a >> 5, ash = a & 31;
    let p = -1;
    for (let i = n; i < 2 * n; i++) {
      if ((_x[i * W + aw] ?? 0) >>> ash & 1) {
        p = i;
        break;
      }
    }
    if (p !== -1) {
      for (let i = 0; i < 2 * n; i++) {
        if (i !== p && (_x[i * W + aw] ?? 0) >>> ash & 1) this.rowmul(i, p);
      }
      const d = (p - n) * W, pW = p * W;
      for (let w = 0; w < W; w++) {
        _x[d + w] = _x[pW + w] ?? 0;
        _z[d + w] = _z[pW + w] ?? 0;
      }
      _r[p - n] = _r[p] ?? 0;
      _x.fill(0, pW, pW + W);
      _z.fill(0, pW, pW + W);
      _z[pW + aw] = 1 << ash;
      _r[p] = rand < 0.5 ? 0 : 1;
      return _r[p] ?? 0;
    } else {
      const sW = 2 * n * W;
      _x.fill(0, sW, sW + W);
      _z.fill(0, sW, sW + W);
      _r[2 * n] = 0;
      for (let i = 0; i < n; i++) {
        if ((_x[i * W + aw] ?? 0) >>> ash & 1) this.rowmul(2 * n, i + n);
      }
      return _r[2 * n] ?? 0;
    }
  }
  /**
   * Return the n stabilizer generators as signed Pauli strings, e.g. `['+XZZXI', '-IXZZX']`.
   * Rows n..2n-1 of the tableau; sign from the phase bit (0 → '+', 1 → '-').
   */
  stabilizerGenerators() {
    const { n, W, _x, _z, _r } = this;
    const out = [];
    for (let i = n; i < 2 * n; i++) {
      const iW = i * W;
      let s = _r[i] ?? 0 ? "-" : "+";
      for (let q = 0; q < n; q++) {
        const w = q >> 5, sh = q & 31;
        const xb = (_x[iW + w] ?? 0) >>> sh & 1;
        const zb = (_z[iW + w] ?? 0) >>> sh & 1;
        s += xb && zb ? "Y" : xb ? "X" : zb ? "Z" : "I";
      }
      out.push(s);
    }
    return out;
  }
};

// src/circuit.ts
function collapseQubit(sv, q, rand) {
  const mask = 1n << BigInt(q);
  let p1 = 0;
  for (const [idx, amp] of sv) {
    if ((idx & mask) !== 0n) p1 += amp.re * amp.re + amp.im * amp.im;
  }
  const outcome = rand < p1 ? 1 : 0;
  const invNorm = 1 / Math.sqrt(outcome === 1 ? p1 : 1 - p1);
  const next = /* @__PURE__ */ new Map();
  for (const [idx, amp] of sv) {
    if (outcome === 1 ? (idx & mask) !== 0n : (idx & mask) === 0n) {
      next.set(idx, { re: amp.re * invNorm, im: amp.im * invNorm });
    }
  }
  return { outcome, sv: next };
}
function sampleSV(sv, rand) {
  const sorted = Array.from(sv.entries()).toSorted(([a], [b]) => a < b ? -1 : 1);
  let cum = 0;
  for (const [idx, amp] of sorted) {
    cum += amp.re * amp.re + amp.im * amp.im;
    if (rand <= cum) return idx;
  }
  return sorted.at(-1)?.[0] ?? 0n;
}
function remapOp(op, qmap) {
  const q = (i) => qmap[i];
  switch (op.kind) {
    case "single":
      return { ...op, q: q(op.q) };
    case "cnot":
      return { ...op, control: q(op.control), target: q(op.target) };
    case "swap":
      return { ...op, a: q(op.a), b: q(op.b) };
    case "two":
      return { ...op, a: q(op.a), b: q(op.b) };
    case "controlled":
      return { ...op, control: q(op.control), target: q(op.target) };
    case "toffoli":
      return { ...op, c1: q(op.c1), c2: q(op.c2), target: q(op.target) };
    case "cswap":
      return { ...op, control: q(op.control), a: q(op.a), b: q(op.b) };
    case "csrswap":
      return { ...op, control: q(op.control), a: q(op.a), b: q(op.b) };
    case "subcircuit":
      return { ...op, qubits: op.qubits.map((i) => q(i)) };
    case "unitary":
      return { ...op, qubits: op.qubits.map((i) => q(i)) };
    case "barrier":
      return { ...op, qubits: op.qubits.map((i) => q(i)) };
    case "measure":
      return op;
    case "reset":
      return op;
    case "if":
      return op;
    case "parametric":
      return op;
    default: {
      const _exhaustive = op;
      return _exhaustive;
    }
  }
}
function flattenOps(ops) {
  let hasSubcircuit = false;
  for (const op of ops) {
    if (op.kind === "parametric") {
      const names = [...collectParams(ops)].join(", ");
      throw new TypeError(`Circuit has unbound parameters: [${names}]. Call bind({ ${names} }) first.`);
    }
    if (op.kind === "subcircuit") hasSubcircuit = true;
  }
  if (!hasSubcircuit) return ops;
  const result = [];
  for (const op of ops) {
    if (op.kind === "subcircuit") {
      result.push(...flattenOps(op.def.map((inner) => remapOp(inner, op.qubits))));
    } else {
      result.push(op);
    }
  }
  return result;
}
function svFromBitstring(s, qubits) {
  if (s.length !== qubits || !/^[01]+$/.test(s))
    throw new TypeError(`initialState '${s}' must be a ${qubits}-character binary string`);
  return /* @__PURE__ */ new Map([[BigInt("0b" + [...s].reverse().join("")), { re: 1, im: 0 }]]);
}
function simulatePure(ops, qubits, init) {
  let sv = init ?? zero(qubits);
  for (const op of flattenOps(ops)) {
    switch (op.kind) {
      case "single":
        sv = applySingle(sv, op.q, op.gate);
        break;
      case "cnot":
        sv = applyCNOT(sv, op.control, op.target);
        break;
      case "controlled":
        sv = applyControlled(sv, op.control, op.target, op.gate);
        break;
      case "swap":
        sv = applySWAP(sv, op.a, op.b);
        break;
      case "toffoli":
        sv = applyToffoli(sv, op.c1, op.c2, op.target);
        break;
      case "cswap":
        sv = applyCSwap(sv, op.control, op.a, op.b);
        break;
      case "csrswap":
        sv = applyCsrSwap(sv, op.control, op.a, op.b);
        break;
      case "two":
        sv = applyTwo(sv, op.a, op.b, op.gate);
        break;
      case "unitary":
        sv = applyUnitary(sv, op.qubits, op.matrix);
        break;
      case "barrier":
      case "measure":
      case "reset":
      case "if":
        break;
      default: {
        const _exhaustive = op;
      }
    }
  }
  return sv;
}
function cregValue(shotCregs, name) {
  return (shotCregs.get(name) ?? []).reduce((acc, b, i) => acc | (b ? 1 << i : 0), 0);
}
function applyOps(ops, svIn, shotCregs, rng, noise) {
  let sv = svIn;
  const p1 = noise?.p1 ?? 0;
  const p2 = noise?.p2 ?? 0;
  const pM = noise?.pMeas ?? 0;
  const gamma = noise?.gamma ?? 0;
  const lambda = noise?.lambda ?? 0;
  const kraus1 = noise?.kraus1;
  const kraus2 = noise?.kraus2;
  for (const op of flattenOps(ops)) {
    switch (op.kind) {
      case "single":
        sv = applySingle(sv, op.q, op.gate);
        if (p1) sv = dep1(sv, op.q, p1, rng());
        if (gamma) sv = dampAmp1(sv, op.q, gamma, rng());
        if (lambda) sv = dampPhase1(sv, op.q, lambda, rng());
        if (kraus1) sv = applyKraus1Channel(sv, op.q, kraus1, rng);
        break;
      case "cnot":
        sv = applyCNOT(sv, op.control, op.target);
        if (p2) sv = dep2(sv, op.control, op.target, p2, rng());
        if (gamma) {
          sv = dampAmp1(sv, op.control, gamma, rng());
          sv = dampAmp1(sv, op.target, gamma, rng());
        }
        if (lambda) {
          sv = dampPhase1(sv, op.control, lambda, rng());
          sv = dampPhase1(sv, op.target, lambda, rng());
        }
        if (kraus2) sv = applyKraus2Channel(sv, op.control, op.target, kraus2, rng);
        break;
      case "controlled":
        sv = applyControlled(sv, op.control, op.target, op.gate);
        if (p2) sv = dep2(sv, op.control, op.target, p2, rng());
        if (gamma) {
          sv = dampAmp1(sv, op.control, gamma, rng());
          sv = dampAmp1(sv, op.target, gamma, rng());
        }
        if (lambda) {
          sv = dampPhase1(sv, op.control, lambda, rng());
          sv = dampPhase1(sv, op.target, lambda, rng());
        }
        if (kraus2) sv = applyKraus2Channel(sv, op.control, op.target, kraus2, rng);
        break;
      case "swap":
        sv = applySWAP(sv, op.a, op.b);
        if (p2) sv = dep2(sv, op.a, op.b, p2, rng());
        if (gamma) {
          sv = dampAmp1(sv, op.a, gamma, rng());
          sv = dampAmp1(sv, op.b, gamma, rng());
        }
        if (lambda) {
          sv = dampPhase1(sv, op.a, lambda, rng());
          sv = dampPhase1(sv, op.b, lambda, rng());
        }
        if (kraus2) sv = applyKraus2Channel(sv, op.a, op.b, kraus2, rng);
        break;
      case "toffoli":
        sv = applyToffoli(sv, op.c1, op.c2, op.target);
        break;
      case "cswap":
        sv = applyCSwap(sv, op.control, op.a, op.b);
        break;
      case "csrswap":
        sv = applyCsrSwap(sv, op.control, op.a, op.b);
        break;
      case "two":
        sv = applyTwo(sv, op.a, op.b, op.gate);
        if (p2) sv = dep2(sv, op.a, op.b, p2, rng());
        if (gamma) {
          sv = dampAmp1(sv, op.a, gamma, rng());
          sv = dampAmp1(sv, op.b, gamma, rng());
        }
        if (lambda) {
          sv = dampPhase1(sv, op.a, lambda, rng());
          sv = dampPhase1(sv, op.b, lambda, rng());
        }
        if (kraus2) sv = applyKraus2Channel(sv, op.a, op.b, kraus2, rng);
        break;
      case "unitary":
        sv = applyUnitary(sv, op.qubits, op.matrix);
        break;
      case "measure": {
        const { outcome, sv: next } = collapseQubit(sv, op.q, rng());
        const reported = pM && rng() < pM ? outcome === 1 ? 0 : 1 : outcome;
        sv = next;
        const reg = shotCregs.get(op.creg);
        if (reg) reg[op.bit] = reported === 1;
        break;
      }
      case "reset": {
        const { outcome, sv: next } = collapseQubit(sv, op.q, rng());
        sv = next;
        if (outcome === 1) sv = applySingle(sv, op.q, X);
        break;
      }
      case "if":
        if (cregValue(shotCregs, op.creg) === op.value) sv = applyOps(op.ops, sv, shotCregs, rng, noise);
        break;
      case "barrier":
        break;
      default: {
        const _exhaustive = op;
      }
    }
  }
  return sv;
}
var JUMP_THRESHOLD2 = 1e-15;
var DEVICES = {
  // ── IonQ ──────────────────────────────────────────────────────────────────
  "aria-1": { qubits: 25, nativeGates: ["gpi", "gpi2", "ms", "vz"], noise: { p1: 3e-4, p2: 5e-3, pMeas: 4e-3 } },
  "forte-1": { qubits: 36, nativeGates: ["gpi", "gpi2", "ms", "vz", "zz"], noise: { p1: 1e-4, p2: 2e-3, pMeas: 2e-3 } },
  "harmony": { qubits: 11, nativeGates: ["gpi", "gpi2", "ms", "vz"], noise: { p1: 1e-3, p2: 0.015, pMeas: 0.01 } },
  // ── IBM Quantum ───────────────────────────────────────────────────────────
  "ibm_sherbrooke": { qubits: 127, noise: { p1: 24e-5, p2: 74e-4, pMeas: 0.0135 } },
  "ibm_brisbane": { qubits: 127, noise: { p1: 24e-5, p2: 76e-4, pMeas: 0.0135 } },
  "ibm_torino": { qubits: 133, noise: { p1: 2e-4, p2: 3e-3, pMeas: 0.01 } },
  // ── Quantinuum ────────────────────────────────────────────────────────────
  "h1-1": { qubits: 20, noise: { p1: 18e-6, p2: 97e-5, pMeas: 23e-4 } },
  "h2-1": { qubits: 56, noise: { p1: 19e-6, p2: 11e-4, pMeas: 1e-3 } }
};
var IONQ_DEVICES = Object.fromEntries(
  Object.entries(DEVICES).filter(([, d]) => d.nativeGates != null)
);
var DEVICE_NOISE = Object.fromEntries(Object.entries(DEVICES).map(([k, v]) => [k, v.noise]));
var TWO_PAULI = [
  [null, X],
  [null, Y],
  [null, Z],
  [X, null],
  [X, X],
  [X, Y],
  [X, Z],
  [Y, null],
  [Y, X],
  [Y, Y],
  [Y, Z],
  [Z, null],
  [Z, X],
  [Z, Y],
  [Z, Z]
];
var TWO_PAULI_IDX = [
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 0],
  [1, 1],
  [1, 2],
  [1, 3],
  [2, 0],
  [2, 1],
  [2, 2],
  [2, 3],
  [3, 0],
  [3, 1],
  [3, 2],
  [3, 3]
];
function dep1(sv, q, p, rand) {
  if (rand >= p) return sv;
  const r = rand / p;
  if (r < 1 / 3) return applySingle(sv, q, X);
  if (r < 2 / 3) return applySingle(sv, q, Y);
  return applySingle(sv, q, Z);
}
function dep2(sv, a, b, p, rand) {
  if (rand >= p) return sv;
  const [pa, pb] = TWO_PAULI[Math.min(Math.floor(rand / p * 15), 14)];
  if (pa) sv = applySingle(sv, a, pa);
  if (pb) sv = applySingle(sv, b, pb);
  return sv;
}
function dampAmp1(sv, q, gamma, rand) {
  const mask = 1n << BigInt(q);
  let p1 = 0;
  for (const [idx, amp] of sv) {
    if (idx & mask) p1 += amp.re * amp.re + amp.im * amp.im;
  }
  const pJump = gamma * p1;
  if (pJump < JUMP_THRESHOLD2) return sv;
  if (rand < pJump) {
    const scale2 = 1 / Math.sqrt(p1);
    const next = /* @__PURE__ */ new Map();
    for (const [idx, amp] of sv) {
      if (idx & mask) {
        const flipped = idx ^ mask;
        const cur = next.get(flipped);
        if (cur) next.set(flipped, { re: cur.re + amp.re * scale2, im: cur.im + amp.im * scale2 });
        else next.set(flipped, { re: amp.re * scale2, im: amp.im * scale2 });
      }
    }
    return next;
  } else {
    const sqG = Math.sqrt(1 - gamma);
    const inv = 1 / Math.sqrt(1 - pJump);
    const next = /* @__PURE__ */ new Map();
    for (const [idx, amp] of sv) {
      const s = idx & mask ? sqG * inv : inv;
      next.set(idx, { re: amp.re * s, im: amp.im * s });
    }
    return next;
  }
}
function dampPhase1(sv, q, lambda, rand) {
  const mask = 1n << BigInt(q);
  let p1 = 0;
  for (const [idx, amp] of sv) {
    if (idx & mask) p1 += amp.re * amp.re + amp.im * amp.im;
  }
  const pJump = lambda * p1;
  if (pJump < JUMP_THRESHOLD2) return sv;
  if (rand < pJump) {
    const scale2 = 1 / Math.sqrt(p1);
    const next = /* @__PURE__ */ new Map();
    for (const [idx, amp] of sv) {
      if (idx & mask) next.set(idx, { re: amp.re * scale2, im: amp.im * scale2 });
    }
    return next;
  } else {
    const sqL = Math.sqrt(1 - lambda);
    const inv = 1 / Math.sqrt(1 - pJump);
    const next = /* @__PURE__ */ new Map();
    for (const [idx, amp] of sv) {
      const s = idx & mask ? sqL * inv : inv;
      next.set(idx, { re: amp.re * s, im: amp.im * s });
    }
    return next;
  }
}
function applyKraus1Channel(sv, q, kraus, rng) {
  let cumP = 0;
  const probs = [];
  const results = [];
  for (const K of kraus) {
    const out = applySingle(sv, q, K);
    let p = 0;
    for (const amp of out.values()) p += amp.re * amp.re + amp.im * amp.im;
    probs.push(p);
    results.push(out);
    cumP += p;
  }
  let r = rng() * cumP;
  for (let k = 0; k < results.length; k++) {
    r -= probs[k];
    if (r <= 0) {
      const inv = probs[k] > 0 ? 1 / Math.sqrt(probs[k]) : 0;
      const next = /* @__PURE__ */ new Map();
      for (const [idx, amp] of results[k]) next.set(idx, { re: amp.re * inv, im: amp.im * inv });
      return next;
    }
  }
  const last = results.length - 1;
  const invLast = probs[last] > 0 ? 1 / Math.sqrt(probs[last]) : 0;
  const fallback = /* @__PURE__ */ new Map();
  for (const [idx, amp] of results[last]) fallback.set(idx, { re: amp.re * invLast, im: amp.im * invLast });
  return fallback;
}
function applyKraus2Channel(sv, a, b, kraus, rng) {
  let cumP = 0;
  const probs = [];
  const results = [];
  for (const K of kraus) {
    const out = applyTwo(sv, a, b, K);
    let p = 0;
    for (const amp of out.values()) p += amp.re * amp.re + amp.im * amp.im;
    probs.push(p);
    results.push(out);
    cumP += p;
  }
  let r = rng() * cumP;
  for (let k = 0; k < results.length; k++) {
    r -= probs[k];
    if (r <= 0) {
      const inv = probs[k] > 0 ? 1 / Math.sqrt(probs[k]) : 0;
      const next = /* @__PURE__ */ new Map();
      for (const [idx, amp] of results[k]) next.set(idx, { re: amp.re * inv, im: amp.im * inv });
      return next;
    }
  }
  const last = results.length - 1;
  const invLast = probs[last] > 0 ? 1 / Math.sqrt(probs[last]) : 0;
  const fallback = /* @__PURE__ */ new Map();
  for (const [idx, amp] of results[last]) fallback.set(idx, { re: amp.re * invLast, im: amp.im * invLast });
  return fallback;
}
function fmtAngle(r, piToken) {
  if (Math.abs(r) < 1e-14) return "0";
  const f = r / Math.PI;
  for (const d of [1, 2, 3, 4, 6, 8, 12, 16]) {
    for (let n = -16; n <= 16; n++) {
      if (n === 0) continue;
      if (Math.abs(f - n / d) < 1e-12) {
        const sign = n < 0 ? "-" : "";
        const a = Math.abs(n);
        if (d === 1) return a === 1 ? `${sign}${piToken}` : `${sign}${a}*${piToken}`;
        return a === 1 ? `${sign}${piToken}/${d}` : `${sign}${a}*${piToken}/${d}`;
      }
    }
  }
  return String(r);
}
function qasmAngle(r) {
  return fmtAngle(r, "pi");
}
function pyAngle(r) {
  return fmtAngle(r, "math.pi");
}
function parseAngle(expr) {
  const s = expr.replace(/\s/g, "");
  let i = 0;
  function parseFactor() {
    if (s[i] === "-") {
      i++;
      return -parseFactor();
    }
    if (s[i] === "+") {
      i++;
      return parseFactor();
    }
    if (s[i] === "(") {
      i++;
      const v = parseExpr();
      if (s[i] === ")") i++;
      return v;
    }
    if (s.startsWith("pi", i)) {
      i += 2;
      return Math.PI;
    }
    const j = i;
    while (i < s.length && /[0-9.]/.test(s[i])) i++;
    return parseFloat(s.slice(j, i));
  }
  function parseTerm() {
    let v = parseFactor();
    while (i < s.length && (s[i] === "*" || s[i] === "/")) {
      const op = s[i++];
      v = op === "*" ? v * parseFactor() : v / parseFactor();
    }
    return v;
  }
  function parseExpr() {
    let v = parseTerm();
    while (i < s.length && (s[i] === "+" || s[i] === "-")) {
      const op = s[i++];
      v = op === "+" ? v + parseTerm() : v - parseTerm();
    }
    return v;
  }
  const result = parseExpr();
  if (i !== s.length || !isFinite(result)) throw new TypeError(`Invalid angle expression: "${expr}"`);
  return result;
}
function qasmGateName(meta) {
  switch (meta.name) {
    case "si":
      return { qname: "sdg", qparams: [] };
    case "ti":
      return { qname: "tdg", qparams: [] };
    case "v":
      return { qname: "sx", qparams: [] };
    case "vi":
      return { qname: "sxdg", qparams: [] };
    case "r2":
      return { qname: "rz", qparams: [Math.PI / 2] };
    case "r4":
      return { qname: "rz", qparams: [Math.PI / 4] };
    case "r8":
      return { qname: "rz", qparams: [Math.PI / 8] };
    case "vz":
      return { qname: "rz", qparams: [...meta.params ?? []] };
    case "cr2":
      return { qname: "crz", qparams: [Math.PI / 2] };
    case "cr4":
      return { qname: "crz", qparams: [Math.PI / 4] };
    case "cr8":
      return { qname: "crz", qparams: [Math.PI / 8] };
    case "cs":
      return { qname: "cu1", qparams: [Math.PI / 2] };
    case "ct":
      return { qname: "cu1", qparams: [Math.PI / 4] };
    case "csdg":
      return { qname: "cu1", qparams: [-Math.PI / 2] };
    case "ctdg":
      return { qname: "cu1", qparams: [-Math.PI / 4] };
    default:
      return { qname: meta.name, qparams: [...meta.params ?? []] };
  }
}
function applyQASMGate(c2, name, params, qs) {
  const [a, b, d] = qs;
  const [p0, p1, p2] = params;
  switch (name) {
    case "id":
      return c2.id(a);
    case "barrier":
      return qs.length ? c2.barrier(...qs) : c2.barrier();
    case "h":
      return c2.h(a);
    case "x":
      return c2.x(a);
    case "y":
      return c2.y(a);
    case "z":
      return c2.z(a);
    case "s":
      return c2.s(a);
    case "sdg":
      return c2.si(a);
    case "t":
      return c2.t(a);
    case "tdg":
      return c2.ti(a);
    case "sx":
      return c2.v(a);
    case "sxdg":
      return c2.vi(a);
    case "srn":
      return c2.srn(a);
    case "srndg":
      return c2.srndg(a);
    case "p":
      return c2.p(p0, a);
    case "rx":
      return c2.rx(p0, a);
    case "ry":
      return c2.ry(p0, a);
    case "rz":
      return c2.rz(p0, a);
    case "u1":
      return c2.u1(p0, a);
    case "u2":
      return c2.u2(p0, p1, a);
    case "u3":
      return c2.u3(p0, p1, p2, a);
    case "cx":
      return c2.cnot(a, b);
    case "cy":
      return c2.cy(a, b);
    case "cz":
      return c2.cz(a, b);
    case "ch":
      return c2.ch(a, b);
    case "crx":
      return c2.crx(p0, a, b);
    case "cry":
      return c2.cry(p0, a, b);
    case "crz":
      return c2.crz(p0, a, b);
    case "cu1":
      return c2.cu1(p0, a, b);
    case "cu2":
      return c2.cu2(p0, p1, a, b);
    case "cu3":
      return c2.cu3(p0, p1, p2, a, b);
    case "swap":
      return c2.swap(a, b);
    case "ccx":
      return c2.ccx(a, b, d);
    case "cswap":
      return c2.cswap(a, b, d);
    case "csrn":
      return c2.csrn(a, b);
    default:
      throw new TypeError(`Unknown QASM gate: '${name}'`);
  }
}
function gate2x2FromMeta(name, params) {
  const p = params;
  switch (name) {
    case "h":
      return H;
    case "x":
      return X;
    case "y":
      return Y;
    case "z":
      return Z;
    case "s":
      return S;
    case "si":
      return Si;
    case "sdg":
      return Si;
    case "t":
      return T;
    case "ti":
      return Ti;
    case "tdg":
      return Ti;
    case "v":
      return V;
    case "vi":
      return Vi;
    case "srn":
      return V;
    case "srndg":
      return Vi;
    case "sx":
      return V;
    case "sxdg":
      return Vi;
    case "id":
      return Id;
    case "rx":
      return Rx(p[0]);
    case "ry":
      return Ry(p[0]);
    case "rz":
      return Rz(p[0]);
    case "vz":
      return Rz(p[0]);
    case "r2":
      return R2;
    case "r4":
      return R4;
    case "r8":
      return R8;
    case "u1":
      return U1(p[0]);
    case "p":
      return U1(p[0]);
    case "u2":
      return U2(p[0], p[1]);
    case "u3":
      return U3(p[0], p[1], p[2]);
    case "gpi":
      return Gpi(p[0]);
    case "gpi2":
      return Gpi2(p[0]);
    default:
      throw new TypeError(`fromJSON: unknown single-qubit gate '${name}'`);
  }
}
function gate4x4FromMeta(name, params) {
  const p = params;
  switch (name) {
    case "xx":
      return Xx(p[0]);
    case "yy":
      return Yy(p[0]);
    case "zz":
      return Zz(p[0]);
    case "xy":
      return Xy(p[0]);
    case "iswap":
      return ISwap;
    case "srswap":
      return SrSwap;
    case "ms":
      return Ms(p[0], p[1]);
    default:
      throw new TypeError(`fromJSON: unknown two-qubit gate '${name}'`);
  }
}
function ctrlGate2x2FromMeta(name, params) {
  const p = params;
  switch (name) {
    case "cx":
      return X;
    case "cy":
      return Y;
    case "cz":
      return Z;
    case "ch":
      return H;
    case "crx":
      return Rx(p[0]);
    case "cry":
      return Ry(p[0]);
    case "crz":
      return Rz(p[0]);
    case "cu1":
      return U1(p[0]);
    case "cu2":
      return U2(p[0], p[1]);
    case "cu3":
      return U3(p[0], p[1], p[2]);
    case "cs":
      return S;
    case "ct":
      return T;
    case "csdg":
      return Si;
    case "ctdg":
      return Ti;
    case "cr2":
      return R2;
    case "cr4":
      return R4;
    case "cr8":
      return R8;
    case "csrn":
      return V;
    default:
      throw new TypeError(`fromJSON: unknown controlled gate '${name}'`);
  }
}
function opsFromJSON(raw) {
  return raw.map((o) => {
    const kind = o["kind"];
    switch (kind) {
      case "single": {
        const meta = o["meta"];
        return { kind: "single", q: o["q"], gate: gate2x2FromMeta(meta.name, meta.params ?? []), meta };
      }
      case "cnot":
        return { kind: "cnot", control: o["control"], target: o["target"] };
      case "swap":
        return { kind: "swap", a: o["a"], b: o["b"] };
      case "two": {
        const meta = o["meta"];
        return { kind: "two", a: o["a"], b: o["b"], gate: gate4x4FromMeta(meta.name, meta.params ?? []), meta };
      }
      case "controlled": {
        const meta = o["meta"];
        return { kind: "controlled", control: o["control"], target: o["target"], gate: ctrlGate2x2FromMeta(meta.name, meta.params ?? []), meta };
      }
      case "toffoli":
        return { kind: "toffoli", c1: o["c1"], c2: o["c2"], target: o["target"] };
      case "cswap":
        return { kind: "cswap", control: o["control"], a: o["a"], b: o["b"] };
      case "csrswap":
        return { kind: "csrswap", control: o["control"], a: o["a"], b: o["b"] };
      case "measure":
        return { kind: "measure", q: o["q"], creg: o["creg"], bit: o["bit"] };
      case "reset":
        return { kind: "reset", q: o["q"] };
      case "if":
        return { kind: "if", creg: o["creg"], value: o["value"], ops: opsFromJSON(o["ops"]) };
      case "subcircuit":
        return { kind: "subcircuit", name: o["name"], qubits: o["qubits"], def: opsFromJSON(o["def"]) };
      case "barrier":
        return { kind: "barrier", qubits: o["qubits"] };
      case "parametric":
        return { kind: "parametric", name: o["name"], params: o["params"], qubits: o["qubits"] };
      case "unitary": {
        const rawMatrix = o["matrix"];
        const matrix = rawMatrix.map((row) => row.map(([re, im]) => ({ re, im })));
        return { kind: "unitary", qubits: o["qubits"], matrix };
      }
      default:
        throw new TypeError(`fromJSON: unknown op kind '${kind}'`);
    }
  });
}
function opsToJSON(ops) {
  return ops.map((op) => {
    switch (op.kind) {
      case "single":
        return { kind: "single", q: op.q, meta: op.meta };
      case "cnot":
        return { kind: "cnot", control: op.control, target: op.target };
      case "swap":
        return { kind: "swap", a: op.a, b: op.b };
      case "two":
        return { kind: "two", a: op.a, b: op.b, meta: op.meta };
      case "controlled":
        return { kind: "controlled", control: op.control, target: op.target, meta: op.meta };
      case "toffoli":
        return { kind: "toffoli", c1: op.c1, c2: op.c2, target: op.target };
      case "cswap":
        return { kind: "cswap", control: op.control, a: op.a, b: op.b };
      case "csrswap":
        return { kind: "csrswap", control: op.control, a: op.a, b: op.b };
      case "measure":
        return { kind: "measure", q: op.q, creg: op.creg, bit: op.bit };
      case "reset":
        return { kind: "reset", q: op.q };
      case "if":
        return { kind: "if", creg: op.creg, value: op.value, ops: opsToJSON(op.ops) };
      case "subcircuit":
        return { kind: "subcircuit", name: op.name, qubits: [...op.qubits], def: opsToJSON(op.def) };
      case "barrier":
        return { kind: "barrier", qubits: [...op.qubits] };
      case "unitary":
        return { kind: "unitary", qubits: [...op.qubits], matrix: op.matrix.map((row) => row.map(({ re, im }) => [re, im])) };
      case "parametric":
        return { kind: "parametric", name: op.name, params: [...op.params], qubits: [...op.qubits] };
      default: {
        const _exhaustive = op;
        return _exhaustive;
      }
    }
  });
}
function latexAngle(r) {
  if (Math.abs(r) < 1e-14) return "0";
  const f = r / Math.PI;
  for (const d of [1, 2, 3, 4, 6, 8, 12, 16]) {
    for (let n = -16; n <= 16; n++) {
      if (n === 0) continue;
      if (Math.abs(f - n / d) < 1e-12) {
        const sign = n < 0 ? "-" : "";
        const a = Math.abs(n);
        if (d === 1) return a === 1 ? `${sign}\\pi` : `${sign}${a}\\pi`;
        return a === 1 ? `${sign}\\frac{\\pi}{${d}}` : `${sign}\\frac{${a}\\pi}{${d}}`;
      }
    }
  }
  return String(r);
}
function latexSingleLabel(op) {
  const nm = op.meta?.name, p = op.meta?.params ?? [];
  const a = (i) => latexAngle(p[i] ?? 0);
  switch (nm) {
    case "h":
      return "H";
    case "x":
      return "X";
    case "y":
      return "Y";
    case "z":
      return "Z";
    case "s":
      return "S";
    case "si":
      return "S^\\dagger";
    case "t":
      return "T";
    case "ti":
      return "T^\\dagger";
    case "v":
      return "\\sqrt{X}";
    case "vi":
      return "\\sqrt{X}^\\dagger";
    case "id":
      return "I";
    case "rx":
      return `R_x(${a(0)})`;
    case "ry":
      return `R_y(${a(0)})`;
    case "rz":
    case "vz":
      return `R_z(${a(0)})`;
    case "r2":
      return "R_2";
    case "r4":
      return "R_4";
    case "r8":
      return "R_8";
    case "u1":
    case "p":
      return `U_1(${a(0)})`;
    case "u2":
      return `U_2(${a(0)},${a(1)})`;
    case "u3":
      return `U_3(${a(0)},${a(1)},${a(2)})`;
    case "gpi":
      return `\\text{GPI}(${a(0)})`;
    case "gpi2":
      return `\\text{GPI2}(${a(0)})`;
    default:
      return nm ? nm.toUpperCase() : "U";
  }
}
function latexTwoLabel(op) {
  const nm = op.meta?.name, p = op.meta?.params ?? [];
  const a = (i) => latexAngle(p[i] ?? 0);
  switch (nm) {
    case "xx":
      return `XX(${a(0)})`;
    case "yy":
      return `YY(${a(0)})`;
    case "zz":
      return `ZZ(${a(0)})`;
    case "xy":
      return `XY(${a(0)})`;
    case "iswap":
      return "\\text{iSWAP}";
    case "srswap":
      return "\\sqrt{\\text{iSWAP}}";
    case "ms":
      return `\\text{MS}(${a(0)},${a(1)})`;
    default:
      return nm ? nm.toUpperCase() : "U";
  }
}
function latexCtrlTargetLabel(op) {
  const nm = op.meta?.name, p = op.meta?.params ?? [];
  const a = (i) => latexAngle(p[i] ?? 0);
  switch (nm) {
    case "cx":
      return "X";
    case "cy":
      return "Y";
    case "cz":
      return "Z";
    case "ch":
      return "H";
    case "crx":
      return `R_x(${a(0)})`;
    case "cry":
      return `R_y(${a(0)})`;
    case "crz":
      return `R_z(${a(0)})`;
    case "cu1":
      return `U_1(${a(0)})`;
    case "cu2":
      return `U_2(${a(0)},${a(1)})`;
    case "cu3":
      return `U_3(${a(0)},${a(1)},${a(2)})`;
    case "cs":
      return "S";
    case "ct":
      return "T";
    case "csdg":
      return "S^\\dagger";
    case "ctdg":
      return "T^\\dagger";
    case "cr2":
      return "R_2";
    case "cr4":
      return "R_4";
    case "cr8":
      return "R_8";
    default:
      return nm ? nm.slice(1).toUpperCase() : "U";
  }
}
function drawAngle(r) {
  return fmtAngle(r, "\u03C0");
}
function opQubits(op) {
  switch (op.kind) {
    case "single":
      return [op.q];
    case "cnot":
      return [op.control, op.target];
    case "swap":
      return [op.a, op.b];
    case "two":
      return [op.a, op.b];
    case "controlled":
      return [op.control, op.target];
    case "toffoli":
      return [op.c1, op.c2, op.target];
    case "cswap":
      return [op.control, op.a, op.b];
    case "csrswap":
      return [op.control, op.a, op.b];
    case "measure":
      return [op.q];
    case "reset":
      return [op.q];
    case "barrier":
      return [...op.qubits];
    case "subcircuit":
      return [...op.qubits];
    case "unitary":
      return [...op.qubits];
    case "if":
      return [];
    case "parametric":
      return [...op.qubits];
    default: {
      const _exhaustive = op;
      return [];
    }
  }
}
function opLabel(op, q) {
  const a = (p, i) => drawAngle(p[i] ?? 0);
  switch (op.kind) {
    case "single": {
      const name = op.meta?.name;
      const p = op.meta?.params ?? [];
      switch (name) {
        case "h":
          return "H";
        case "x":
          return "X";
        case "y":
          return "Y";
        case "z":
          return "Z";
        case "s":
          return "S";
        case "si":
          return "S\u2020";
        case "t":
          return "T";
        case "ti":
          return "T\u2020";
        case "v":
          return "V";
        case "vi":
          return "V\u2020";
        case "id":
          return "I";
        case "rx":
          return `Rx(${a(p, 0)})`;
        case "ry":
          return `Ry(${a(p, 0)})`;
        case "rz":
        case "vz":
          return `Rz(${a(p, 0)})`;
        case "r2":
          return "R\u2082";
        case "r4":
          return "R\u2084";
        case "r8":
          return "R\u2088";
        case "u1":
        case "p":
          return `U1(${a(p, 0)})`;
        case "u2":
          return `U2(${a(p, 0)},${a(p, 1)})`;
        case "u3":
          return `U3(${a(p, 0)},${a(p, 1)},${a(p, 2)})`;
        case "gpi":
          return `GPI(${a(p, 0)})`;
        case "gpi2":
          return `GPI2(${a(p, 0)})`;
        default:
          return name ? name.toUpperCase() : "U";
      }
    }
    case "cnot":
      return op.control === q ? "\u25CF" : "\u2295";
    case "swap":
      return "\u2573";
    case "two": {
      const name = op.meta?.name;
      const p = op.meta?.params ?? [];
      switch (name) {
        case "xx":
          return `XX(${a(p, 0)})`;
        case "yy":
          return `YY(${a(p, 0)})`;
        case "zz":
          return `ZZ(${a(p, 0)})`;
        case "xy":
          return `XY(${a(p, 0)})`;
        case "iswap":
          return "iSWAP";
        case "srswap":
          return "\u221AiSWAP";
        case "ms":
          return `MS(${a(p, 0)},${a(p, 1)})`;
        default:
          return name ? name.toUpperCase() : "U";
      }
    }
    case "controlled": {
      if (op.control === q) return "\u25CF";
      const name = op.meta?.name;
      const p = op.meta?.params ?? [];
      switch (name) {
        case "cx":
          return "X";
        case "cy":
          return "Y";
        case "cz":
          return "Z";
        case "ch":
          return "H";
        case "crx":
          return `Rx(${a(p, 0)})`;
        case "cry":
          return `Ry(${a(p, 0)})`;
        case "crz":
          return `Rz(${a(p, 0)})`;
        case "cu1":
          return `U1(${a(p, 0)})`;
        case "cu2":
          return `U2(${a(p, 0)},${a(p, 1)})`;
        case "cu3":
          return `U3(${a(p, 0)},${a(p, 1)},${a(p, 2)})`;
        case "cs":
          return "S";
        case "ct":
          return "T";
        case "csdg":
          return "S\u2020";
        case "ctdg":
          return "T\u2020";
        case "cr2":
          return "R\u2082";
        case "cr4":
          return "R\u2084";
        case "cr8":
          return "R\u2088";
        default:
          return name ? name.slice(1).toUpperCase() : "U";
      }
    }
    case "toffoli":
      return op.target === q ? "\u2295" : "\u25CF";
    case "cswap":
      return op.control === q ? "\u25CF" : "\u2573";
    case "csrswap":
      return op.control === q ? "\u25CF" : "\u221ASW";
    case "measure":
      return "M";
    case "reset":
      return "|0\u27E9";
    case "barrier":
      return "\u2591";
    case "subcircuit":
      return op.name;
    case "unitary":
      return "U";
    case "if":
      return "?";
    case "parametric":
      return `${op.name}(${op.params.join(",")})`;
    default: {
      const _exhaustive = op;
      return "?";
    }
  }
}
var _pool = null;
function acquirePool(size, workerUrl, WorkerClass) {
  const href = workerUrl.href;
  if (_pool !== null && _pool.size === size && _pool.url === href) return _pool.ws;
  _pool?.ws.forEach((w) => void w.terminate());
  _pool = { ws: Array.from({ length: size }, () => new WorkerClass(workerUrl)), size, url: href };
  return _pool.ws;
}
function toTrajOps(flatOps) {
  const out = [];
  for (const op of flatOps) {
    switch (op.kind) {
      case "single":
        out.push({ kind: "single", q: op.q, gate: op.gate });
        break;
      case "cnot":
        out.push({ kind: "cnot", control: op.control, target: op.target });
        break;
      case "swap":
        out.push({ kind: "swap", a: op.a, b: op.b });
        break;
      case "two":
        out.push({ kind: "two", a: op.a, b: op.b, gate: op.gate });
        break;
      case "controlled":
        out.push({ kind: "two", a: op.control, b: op.target, gate: controlledGate(op.gate) });
        break;
      case "unitary": {
        const n = op.qubits.length;
        if (n === 1) out.push({ kind: "single", q: op.qubits[0], gate: op.matrix });
        else if (n === 2) out.push({ kind: "two", a: op.qubits[0], b: op.qubits[1], gate: op.matrix });
        else throw new TypeError(`unitary gate with ${n} qubits is not supported in MPS mode; use run() instead`);
        break;
      }
      case "toffoli": {
        const { c1, c2, target: t } = op;
        out.push(
          { kind: "single", q: t, gate: H },
          { kind: "cnot", control: c2, target: t },
          { kind: "single", q: t, gate: Ti },
          { kind: "cnot", control: c1, target: t },
          { kind: "single", q: t, gate: T },
          { kind: "cnot", control: c2, target: t },
          { kind: "single", q: t, gate: Ti },
          { kind: "cnot", control: c1, target: t },
          { kind: "single", q: c2, gate: T },
          { kind: "single", q: t, gate: T },
          { kind: "single", q: t, gate: H },
          { kind: "cnot", control: c1, target: c2 },
          { kind: "single", q: c1, gate: T },
          { kind: "single", q: c2, gate: Ti },
          { kind: "cnot", control: c1, target: c2 }
        );
        break;
      }
      case "cswap": {
        const { control: ctrl, a, b } = op;
        out.push({ kind: "cnot", control: b, target: a });
        out.push(
          { kind: "single", q: b, gate: H },
          { kind: "cnot", control: a, target: b },
          { kind: "single", q: b, gate: Ti },
          { kind: "cnot", control: ctrl, target: b },
          { kind: "single", q: b, gate: T },
          { kind: "cnot", control: a, target: b },
          { kind: "single", q: b, gate: Ti },
          { kind: "cnot", control: ctrl, target: b },
          { kind: "single", q: a, gate: T },
          { kind: "single", q: b, gate: T },
          { kind: "single", q: b, gate: H },
          { kind: "cnot", control: ctrl, target: a },
          { kind: "single", q: ctrl, gate: T },
          { kind: "single", q: a, gate: Ti },
          { kind: "cnot", control: ctrl, target: a }
        );
        out.push({ kind: "cnot", control: b, target: a });
        break;
      }
      case "csrswap":
        throw new TypeError("csrswap not supported in MPS mode; decompose into CX gates");
      case "measure":
        out.push({ kind: "measure", q: op.q, creg: op.creg, bit: op.bit });
        break;
      case "reset":
        out.push({ kind: "reset", q: op.q });
        break;
      case "if":
        out.push({ kind: "if", creg: op.creg, value: op.value, ops: toTrajOps(flattenOps(op.ops)) });
        break;
      case "barrier":
        out.push({ kind: "barrier" });
        break;
      default: {
        const _exhaustive = op;
        break;
      }
    }
  }
  return out;
}
function distributeShots(shots, n) {
  const base = Math.floor(shots / n);
  const rem = shots % n;
  return Array.from({ length: n }, (_, i) => base + (i < rem ? 1 : 0));
}
function makePrng(seed) {
  let s = seed !== void 0 ? seed >>> 0 || 1 : (Date.now() & 4294967295) >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}
var Distribution = class _Distribution {
  qubits;
  shots;
  probs;
  histogram;
  /** Classical register results: `cregs[name][bit]` = fraction of shots where that bit was 1. */
  cregs;
  /**
   * `true` if `truncErr` caused one or more physically significant singular values to be
   * discarded during MPS simulation. Results are approximate when this is `true`.
   *
   * Always `false` for `run()` and `runClifford()`, and for `runMps()` with `truncErr = 0`
   * (the default) — the bond dimension grows automatically in that case.
   */
  truncated;
  /**
   * Which simulation backend produced this result.
   * Set by `simulate()` and the individual `run*` methods.
   */
  backend;
  /**
   * Peak bond dimension χ used during MPS simulation.
   * Only defined when `backend === 'mps'`.
   */
  peakChi;
  constructor(qubits, shots, counts, cregCounts = /* @__PURE__ */ new Map(), truncated = false, backend, peakChi) {
    this.qubits = qubits;
    this.shots = shots;
    this.truncated = truncated;
    this.backend = backend;
    this.peakChi = peakChi;
    const probs = {};
    const histogram = {};
    for (const [idx, count] of counts) {
      const prob = count / shots;
      const bitstring = idx.toString(2).padStart(qubits, "0").split("").reverse().join("");
      probs[bitstring] = prob;
      histogram[String(idx)] = prob;
    }
    this.probs = Object.freeze(probs);
    this.histogram = Object.freeze(histogram);
    const cregs = {};
    for (const [name, bitCounts] of cregCounts) {
      cregs[name] = Object.freeze(bitCounts.map((c2) => c2 / shots));
    }
    this.cregs = Object.freeze(cregs);
  }
  /** Most probable bitstring. */
  get most() {
    let best = "", bestP = -1;
    for (const [bs, p] of Object.entries(this.probs)) {
      if (p > bestP) {
        best = bs;
        bestP = p;
      }
    }
    return best;
  }
  /** Shannon entropy of the distribution (in bits). */
  get entropy() {
    let h = 0;
    for (const p of Object.values(this.probs)) {
      if (p > 0) h -= p * Math.log2(p);
    }
    return h;
  }
  /** ASCII bar chart of measurement outcomes. */
  render() {
    const entries = Object.entries(this.probs).toSorted(([a], [b]) => a.localeCompare(b));
    const maxP = Math.max(...entries.map(([, p]) => p));
    const width = 40;
    const lines = entries.map(([bs, p]) => {
      const bar = "\u2588".repeat(Math.round(p / maxP * width));
      const pct = (p * 100).toFixed(1).padStart(5);
      return `|${bs}\u27E9 ${pct}%  ${bar}`;
    });
    return lines.join("\n");
  }
  /**
   * SVG bar chart of measurement outcomes — same visual style as the QAOA Max-Cut diagram.
   *
   * Bars are sorted by bitstring. Dominant peaks (≥ 80 % of the max probability) are
   * highlighted in blue with a percentage label; the rest render in slate.
   *
   * @param opts.title     Override the subtitle line (default: `"measurement outcomes"`).
   * @param opts.highlight Explicit set of bitstrings to highlight instead of auto-detecting.
   *
   * @example
   * import fs from 'fs'
   * const result = new Circuit(2).h(0).cnot(0, 1).run({ shots: 1024, seed: 42 })
   * fs.writeFileSync('bell.svg', result.toSVG())
   */
  toSVG(opts = {}) {
    const entries = Object.entries(this.probs).toSorted(([a], [b]) => a.localeCompare(b));
    if (entries.length === 0) {
      return '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="240" viewBox="0 0 300 240"><rect width="300" height="240" fill="#fff"/></svg>';
    }
    const n = entries.length;
    const barW = 14;
    const barStep = 20;
    const barsSpan = n * barStep - 6;
    const totalW = Math.max(300, barsSpan + 120);
    const ml = Math.round((totalW - barsSpan) / 2);
    const baseline = 170;
    const maxBarH = 115;
    const totalH = 240;
    const maxP = entries.reduce((m, [, p]) => p > m ? p : m, 0);
    const highlighted = opts.highlight ? new Set(opts.highlight) : new Set(entries.filter(([, p]) => p >= maxP * 0.8).map(([bs]) => bs));
    const possible = 2 ** this.qubits;
    const countLabel = n === possible ? `all ${n}` : `${n} of ${possible}`;
    const subtitle = `${opts.title ?? "measurement outcomes"} (${countLabel} state${n !== 1 ? "s" : ""})`;
    const els = [];
    els.push(`<rect width="${totalW}" height="${totalH}" fill="#fff"/>`);
    els.push(
      `<text x="${(totalW / 2).toFixed(0)}" y="18" text-anchor="middle" font-family="ui-sans-serif,sans-serif" font-size="11" fill="#94a3b8">${subtitle}</text>`
    );
    for (let i = 0; i < entries.length; i++) {
      const [bs, p] = entries[i];
      const x = ml + i * barStep;
      const cx = x + barW / 2;
      const barH = p / maxP * maxBarH;
      const y = baseline - barH;
      const isHi = highlighted.has(bs);
      const fill = isHi ? "#3b82f6" : "#e2e8f0";
      els.push(`<rect x="${x}" y="${y.toFixed(2)}" width="${barW}" height="${barH.toFixed(2)}" rx="2" fill="${fill}"/>`);
      if (isHi) {
        const pct = `${(p * 100).toFixed(0)}%`;
        els.push(
          `<text x="${cx.toFixed(0)}" y="${(y - 8).toFixed(0)}" text-anchor="middle" font-family="ui-sans-serif,sans-serif" font-size="10" font-weight="600" fill="#2563eb">${pct}</text>`
        );
        els.push(
          `<text x="${cx.toFixed(0)}" y="${baseline + 18}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9" fill="#2563eb" transform="rotate(-45 ${cx.toFixed(0)} ${baseline + 18})">${bs}</text>`
        );
      }
    }
    els.push(`<line x1="${ml - 1}" y1="${baseline}" x2="${ml + barsSpan + 2}" y2="${baseline}" stroke="#e2e8f0" stroke-width="1"/>`);
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${totalW}" height="${totalH}" viewBox="0 0 ${totalW} ${totalH}">
${els.join("\n")}
</svg>`;
  }
  /**
   * Apply inverse per-qubit readout error mitigation.
   *
   * Corrects measurement results for independent per-qubit bit-flip errors at rate `p`
   * (the `pMeas` parameter used during simulation). Applies the exact inverse of the
   * per-qubit confusion matrix: A_q = [[1−p, p], [p, 1−p]] → A_q⁻¹ = diag correction.
   *
   * The inverse is applied qubit-by-qubit in sequence, equivalent to the tensor-product
   * inverse A⁻¹ = A_0⁻¹ ⊗ … ⊗ A_{n−1}⁻¹. Negative probabilities are clipped to 0 and
   * the result is renormalized — standard practice for near-threshold error rates.
   *
   * @param p Readout error probability per qubit (the same `pMeas` used in `.run()`).
   *
   * @example
   * const d = circuit.run({ shots: 8192, noise: { pMeas: 0.02 } })
   * const corrected = d.mitigateReadout(0.02)
   */
  mitigateReadout(p) {
    if (p <= 0 || p >= 0.5) return this;
    const inv = 1 / (1 - 2 * p);
    let corrected = { ...this.probs };
    for (let q = 0; q < this.qubits; q++) {
      const next = {};
      for (const [bs, prob] of Object.entries(corrected)) {
        const flipped = bs.slice(0, q) + (bs[q] === "0" ? "1" : "0") + bs.slice(q + 1);
        const pFl = corrected[flipped] ?? 0;
        next[bs] = (next[bs] ?? 0) + inv * (prob - p * pFl);
      }
      corrected = next;
    }
    let total = 0;
    const clipped = {};
    for (const [bs, prob] of Object.entries(corrected)) {
      if (prob > 0) {
        clipped[bs] = prob;
        total += prob;
      }
    }
    if (total === 0) return this;
    const counts = /* @__PURE__ */ new Map();
    for (const [bs, prob] of Object.entries(clipped)) {
      let idx = 0n;
      for (let i = 0; i < bs.length; i++) if (bs[i] === "1") idx |= 1n << BigInt(i);
      counts.set(idx, Math.round(prob / total * this.shots));
    }
    return new _Distribution(this.qubits, this.shots, counts, /* @__PURE__ */ new Map(), this.truncated, this.backend, this.peakChi);
  }
};
function resolveParametric(name, ps, qubits) {
  switch (name) {
    case "rx":
      return { kind: "single", q: qubits[0], gate: Rx(ps[0]), meta: { name, params: ps } };
    case "ry":
      return { kind: "single", q: qubits[0], gate: Ry(ps[0]), meta: { name, params: ps } };
    case "rz":
      return { kind: "single", q: qubits[0], gate: Rz(ps[0]), meta: { name, params: ps } };
    case "vz":
      return { kind: "single", q: qubits[0], gate: Rz(ps[0]), meta: { name, params: ps } };
    case "u1":
      return { kind: "single", q: qubits[0], gate: U1(ps[0]), meta: { name, params: ps } };
    case "p":
      return { kind: "single", q: qubits[0], gate: U1(ps[0]), meta: { name, params: ps } };
    case "u2":
      return { kind: "single", q: qubits[0], gate: U2(ps[0], ps[1]), meta: { name, params: ps } };
    case "u3":
      return { kind: "single", q: qubits[0], gate: U3(ps[0], ps[1], ps[2]), meta: { name, params: ps } };
    case "gpi":
      return { kind: "single", q: qubits[0], gate: Gpi(ps[0]), meta: { name, params: ps } };
    case "gpi2":
      return { kind: "single", q: qubits[0], gate: Gpi2(ps[0]), meta: { name, params: ps } };
    case "xx":
      return { kind: "two", a: qubits[0], b: qubits[1], gate: Xx(ps[0]), meta: { name, params: ps } };
    case "yy":
      return { kind: "two", a: qubits[0], b: qubits[1], gate: Yy(ps[0]), meta: { name, params: ps } };
    case "zz":
      return { kind: "two", a: qubits[0], b: qubits[1], gate: Zz(ps[0]), meta: { name, params: ps } };
    case "xy":
      return { kind: "two", a: qubits[0], b: qubits[1], gate: Xy(ps[0]), meta: { name, params: ps } };
    case "ms":
      return { kind: "two", a: qubits[0], b: qubits[1], gate: Ms(ps[0], ps[1]), meta: { name, params: ps } };
    case "crx":
      return { kind: "controlled", control: qubits[0], target: qubits[1], gate: Rx(ps[0]), meta: { name, params: ps } };
    case "cry":
      return { kind: "controlled", control: qubits[0], target: qubits[1], gate: Ry(ps[0]), meta: { name, params: ps } };
    case "crz":
      return { kind: "controlled", control: qubits[0], target: qubits[1], gate: Rz(ps[0]), meta: { name, params: ps } };
    case "cu1":
      return { kind: "controlled", control: qubits[0], target: qubits[1], gate: U1(ps[0]), meta: { name, params: ps } };
    case "cu2":
      return { kind: "controlled", control: qubits[0], target: qubits[1], gate: U2(ps[0], ps[1]), meta: { name, params: ps } };
    case "cu3":
      return { kind: "controlled", control: qubits[0], target: qubits[1], gate: U3(ps[0], ps[1], ps[2]), meta: { name, params: ps } };
    default:
      throw new TypeError(`bind: unknown parametric gate '${name}'`);
  }
}
function collectParams(ops) {
  const names = /* @__PURE__ */ new Set();
  for (const op of ops) {
    if (op.kind === "parametric") {
      for (const p of op.params) if (typeof p === "string") names.add(p);
    } else if (op.kind === "if") {
      for (const n of collectParams(op.ops)) names.add(n);
    }
  }
  return names;
}
var Circuit = class _Circuit {
  qubits;
  #ops;
  #cregs;
  // name → declared size
  #gates;
  // name → registered sub-circuit
  constructor(qubits, ops = [], cregs = /* @__PURE__ */ new Map(), gates = /* @__PURE__ */ new Map()) {
    this.qubits = qubits;
    this.#ops = ops;
    this.#cregs = cregs;
    this.#gates = gates;
  }
  #add(op) {
    this.#checkOp(op);
    return new _Circuit(this.qubits, [...this.#ops, op], this.#cregs, this.#gates);
  }
  #checkOp(op) {
    const q = (i) => {
      if (i < 0 || i >= this.qubits)
        throw new RangeError(`qubit index ${i} is out of range for a ${this.qubits}-qubit circuit`);
    };
    const diff = (a, b) => {
      if (a === b) throw new TypeError(`control and target qubits must differ (got ${a})`);
    };
    switch (op.kind) {
      case "single":
        q(op.q);
        break;
      case "cnot":
        q(op.control);
        q(op.target);
        diff(op.control, op.target);
        break;
      case "swap":
        q(op.a);
        q(op.b);
        break;
      case "two":
        q(op.a);
        q(op.b);
        break;
      case "controlled":
        q(op.control);
        q(op.target);
        diff(op.control, op.target);
        break;
      case "toffoli":
        q(op.c1);
        q(op.c2);
        q(op.target);
        break;
      case "cswap":
        q(op.control);
        q(op.a);
        q(op.b);
        break;
      case "csrswap":
        q(op.control);
        q(op.a);
        q(op.b);
        break;
      case "measure":
        q(op.q);
        break;
      case "reset":
        q(op.q);
        break;
      case "barrier":
        op.qubits.forEach(q);
        break;
      case "parametric":
        op.qubits.forEach(q);
        break;
      case "unitary": {
        if (op.qubits.length === 0) throw new TypeError("unitary: qubits must be non-empty");
        const expected = 1 << op.qubits.length;
        if (op.matrix.length !== expected || op.matrix.some((row) => row.length !== expected))
          throw new TypeError(`unitary: matrix must be ${expected}\xD7${expected} for ${op.qubits.length} qubit(s), got ${op.matrix.length}\xD7${op.matrix[0]?.length ?? 0}`);
        op.qubits.forEach(q);
        break;
      }
      case "if":
        op.ops.forEach((inner) => this.#checkOp(inner));
        break;
      case "subcircuit":
        op.qubits.forEach(q);
        break;
      default: {
        const _ = op;
      }
    }
  }
  #ctrl(control, target, gate, meta) {
    return this.#add({ kind: "controlled", control, target, gate, ...meta !== void 0 && { meta } });
  }
  // ── IonQ single-qubit gates ──────────────────────────────────────────────
  /** Identity gate — no-op on the statevector; preserved by name through import/export. */
  id(q) {
    return this.#add({ kind: "single", q, gate: Id, meta: { name: "id" } });
  }
  h(q) {
    return this.#add({ kind: "single", q, gate: H, meta: { name: "h" } });
  }
  x(q) {
    return this.#add({ kind: "single", q, gate: X, meta: { name: "x" } });
  }
  y(q) {
    return this.#add({ kind: "single", q, gate: Y, meta: { name: "y" } });
  }
  z(q) {
    return this.#add({ kind: "single", q, gate: Z, meta: { name: "z" } });
  }
  s(q) {
    return this.#add({ kind: "single", q, gate: S, meta: { name: "s" } });
  }
  si(q) {
    return this.#add({ kind: "single", q, gate: Si, meta: { name: "si" } });
  }
  /** S† — alias `sdg` (Qiskit / OpenQASM convention). */
  sdg(q) {
    return this.#add({ kind: "single", q, gate: Si, meta: { name: "sdg" } });
  }
  t(q) {
    return this.#add({ kind: "single", q, gate: T, meta: { name: "t" } });
  }
  ti(q) {
    return this.#add({ kind: "single", q, gate: Ti, meta: { name: "ti" } });
  }
  /** T† — alias `tdg` (Qiskit / OpenQASM convention). */
  tdg(q) {
    return this.#add({ kind: "single", q, gate: Ti, meta: { name: "tdg" } });
  }
  v(q) {
    return this.#add({ kind: "single", q, gate: V, meta: { name: "v" } });
  }
  vi(q) {
    return this.#add({ kind: "single", q, gate: Vi, meta: { name: "vi" } });
  }
  /** √NOT — alias `srn`; same as `v`. */
  srn(q) {
    return this.#add({ kind: "single", q, gate: V, meta: { name: "srn" } });
  }
  /** (√NOT)† — alias `srndg`; same as `vi`. */
  srndg(q) {
    return this.#add({ kind: "single", q, gate: Vi, meta: { name: "srndg" } });
  }
  // ── Rotation gates ───────────────────────────────────────────────────────
  rx(theta, q) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "rx", params: [theta], qubits: [q] });
    return this.#add({ kind: "single", q, gate: Rx(theta), meta: { name: "rx", params: [theta] } });
  }
  ry(theta, q) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "ry", params: [theta], qubits: [q] });
    return this.#add({ kind: "single", q, gate: Ry(theta), meta: { name: "ry", params: [theta] } });
  }
  rz(theta, q) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "rz", params: [theta], qubits: [q] });
    return this.#add({ kind: "single", q, gate: Rz(theta), meta: { name: "rz", params: [theta] } });
  }
  /**
   * VirtualZ(θ) — named Rz alias common in superconducting hardware native gate sets
   * (IBM, Rigetti). Functionally identical to `rz(θ)` but carries the `vz` name through
   * import/export for hardware compilation pass awareness.
   */
  vz(theta, q) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "vz", params: [theta], qubits: [q] });
    return this.#add({ kind: "single", q, gate: Rz(theta), meta: { name: "vz", params: [theta] } });
  }
  // ── Named phase rotation gates ───────────────────────────────────────────
  /** Rz(π/2) — phase rotation by a half-turn; S up to global phase. */
  r2(q) {
    return this.#add({ kind: "single", q, gate: R2, meta: { name: "r2" } });
  }
  /** Rz(π/4) — phase rotation by a quarter-turn; T up to global phase. */
  r4(q) {
    return this.#add({ kind: "single", q, gate: R4, meta: { name: "r4" } });
  }
  /** Rz(π/8) — phase rotation by an eighth-turn. */
  r8(q) {
    return this.#add({ kind: "single", q, gate: R8, meta: { name: "r8" } });
  }
  // ── OpenQASM basis gates ─────────────────────────────────────────────────
  /** U1(λ) — phase gate; equal to Rz(λ) up to global phase. */
  u1(lambda, q) {
    if (typeof lambda === "string") return this.#add({ kind: "parametric", name: "u1", params: [lambda], qubits: [q] });
    return this.#add({ kind: "single", q, gate: U1(lambda), meta: { name: "u1", params: [lambda] } });
  }
  /** P(λ) — phase gate alias for U1(λ); Qiskit 1.0+ name. P(π) = Z, P(π/2) = S, P(π/4) = T. */
  p(lambda, q) {
    if (typeof lambda === "string") return this.#add({ kind: "parametric", name: "p", params: [lambda], qubits: [q] });
    return this.#add({ kind: "single", q, gate: U1(lambda), meta: { name: "p", params: [lambda] } });
  }
  /** U2(φ, λ) = U3(π/2, φ, λ) — equatorial gate. U2(0, π) = H. */
  u2(phi, lambda, q) {
    if (typeof phi === "string" || typeof lambda === "string")
      return this.#add({ kind: "parametric", name: "u2", params: [phi, lambda], qubits: [q] });
    return this.#add({ kind: "single", q, gate: U2(phi, lambda), meta: { name: "u2", params: [phi, lambda] } });
  }
  /** U3(θ, φ, λ) — general single-qubit unitary; OpenQASM 2.0 basis gate. */
  u3(theta, phi, lambda, q) {
    if (typeof theta === "string" || typeof phi === "string" || typeof lambda === "string")
      return this.#add({ kind: "parametric", name: "u3", params: [theta, phi, lambda], qubits: [q] });
    return this.#add({ kind: "single", q, gate: U3(theta, phi, lambda), meta: { name: "u3", params: [theta, phi, lambda] } });
  }
  // ── Two-qubit gates ──────────────────────────────────────────────────────
  /** Controlled-NOT. IonQ name: cnot. */
  cnot(control, target) {
    return this.#add({ kind: "cnot", control, target });
  }
  swap(a, b) {
    return this.#add({ kind: "swap", a, b });
  }
  // ── Two-qubit interaction gates ─────────────────────────────────────────
  /** XX(θ) = exp(−iθ/2 · X⊗X) — Ising-XX interaction; IonQ native. */
  xx(theta, a, b) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "xx", params: [theta], qubits: [a, b] });
    return this.#add({ kind: "two", a, b, gate: Xx(theta), meta: { name: "xx", params: [theta] } });
  }
  /** YY(θ) = exp(−iθ/2 · Y⊗Y) — Ising-YY interaction; IonQ native. */
  yy(theta, a, b) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "yy", params: [theta], qubits: [a, b] });
    return this.#add({ kind: "two", a, b, gate: Yy(theta), meta: { name: "yy", params: [theta] } });
  }
  /** ZZ(θ) = exp(−iθ/2 · Z⊗Z) — Ising-ZZ interaction; IonQ native. */
  zz(theta, a, b) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "zz", params: [theta], qubits: [a, b] });
    return this.#add({ kind: "two", a, b, gate: Zz(theta), meta: { name: "zz", params: [theta] } });
  }
  /** XY(θ) interaction gate. XY(π) = iSWAP, XY(π/2) = √iSWAP. */
  xy(theta, a, b) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "xy", params: [theta], qubits: [a, b] });
    return this.#add({ kind: "two", a, b, gate: Xy(theta), meta: { name: "xy", params: [theta] } });
  }
  /** iSWAP = XY(π): swaps qubits and multiplies each by i. */
  iswap(a, b) {
    return this.#add({ kind: "two", a, b, gate: ISwap, meta: { name: "iswap" } });
  }
  /** √iSWAP = XY(π/2): square root of iSWAP. */
  srswap(a, b) {
    return this.#add({ kind: "two", a, b, gate: SrSwap, meta: { name: "srswap" } });
  }
  // ── Controlled single-qubit gates ────────────────────────────────────────
  /** Controlled-NOT; alias for cnot. IBM/OpenQASM name. */
  cx(control, target) {
    return this.cnot(control, target);
  }
  cy(control, target) {
    return this.#ctrl(control, target, Y, { name: "cy" });
  }
  cz(control, target) {
    return this.#ctrl(control, target, Z, { name: "cz" });
  }
  ch(control, target) {
    return this.#ctrl(control, target, H, { name: "ch" });
  }
  // ── Controlled rotation gates ────────────────────────────────────────────
  crx(theta, control, target) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "crx", params: [theta], qubits: [control, target] });
    return this.#ctrl(control, target, Rx(theta), { name: "crx", params: [theta] });
  }
  cry(theta, control, target) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "cry", params: [theta], qubits: [control, target] });
    return this.#ctrl(control, target, Ry(theta), { name: "cry", params: [theta] });
  }
  crz(theta, control, target) {
    if (typeof theta === "string") return this.#add({ kind: "parametric", name: "crz", params: [theta], qubits: [control, target] });
    return this.#ctrl(control, target, Rz(theta), { name: "crz", params: [theta] });
  }
  /** Controlled-Rz(π/2) — controlled phase half-turn. */
  cr2(control, target) {
    return this.#ctrl(control, target, R2, { name: "cr2" });
  }
  /** Controlled-Rz(π/4) — controlled phase quarter-turn. */
  cr4(control, target) {
    return this.#ctrl(control, target, R4, { name: "cr4" });
  }
  /** Controlled-Rz(π/8) — controlled phase eighth-turn. */
  cr8(control, target) {
    return this.#ctrl(control, target, R8, { name: "cr8" });
  }
  // ── Controlled parameterized unitaries ───────────────────────────────────
  /** CU1(λ) — controlled phase gate; CU1(π) = CZ. */
  cu1(lambda, control, target) {
    if (typeof lambda === "string") return this.#add({ kind: "parametric", name: "cu1", params: [lambda], qubits: [control, target] });
    return this.#ctrl(control, target, U1(lambda), { name: "cu1", params: [lambda] });
  }
  /** CU2(φ,λ) = CU3(π/2,φ,λ) — controlled equatorial gate. */
  cu2(phi, lambda, control, target) {
    if (typeof phi === "string" || typeof lambda === "string")
      return this.#add({ kind: "parametric", name: "cu2", params: [phi, lambda], qubits: [control, target] });
    return this.#ctrl(control, target, U2(phi, lambda), { name: "cu2", params: [phi, lambda] });
  }
  /** CU3(θ,φ,λ) — controlled general unitary; CU3(π,0,π) = CX. */
  cu3(theta, phi, lambda, control, target) {
    if (typeof theta === "string" || typeof phi === "string" || typeof lambda === "string")
      return this.#add({ kind: "parametric", name: "cu3", params: [theta, phi, lambda], qubits: [control, target] });
    return this.#ctrl(control, target, U3(theta, phi, lambda), { name: "cu3", params: [theta, phi, lambda] });
  }
  // ── Controlled phase gates ────────────────────────────────────────────────
  cs(control, target) {
    return this.#ctrl(control, target, S, { name: "cs" });
  }
  ct(control, target) {
    return this.#ctrl(control, target, T, { name: "ct" });
  }
  csdg(control, target) {
    return this.#ctrl(control, target, Si, { name: "csdg" });
  }
  ctdg(control, target) {
    return this.#ctrl(control, target, Ti, { name: "ctdg" });
  }
  /** Controlled-√NOT (C-V); applies V = √X to target when control is |1⟩. */
  csrn(control, target) {
    return this.#ctrl(control, target, V, { name: "csrn" });
  }
  // ── Native IonQ gates ────────────────────────────────────────────────────
  /** GPI(φ) — IonQ hardware-native single-qubit gate. GPI(0) = X, GPI(π/2) = Y. */
  gpi(phi, q) {
    if (typeof phi === "string") return this.#add({ kind: "parametric", name: "gpi", params: [phi], qubits: [q] });
    return this.#add({ kind: "single", q, gate: Gpi(phi), meta: { name: "gpi", params: [phi] } });
  }
  /** GPI2(φ) — IonQ hardware-native half-rotation. GPI2(0) = Rx(π/2), GPI2(π/2) = Ry(π/2). */
  gpi2(phi, q) {
    if (typeof phi === "string") return this.#add({ kind: "parametric", name: "gpi2", params: [phi], qubits: [q] });
    return this.#add({ kind: "single", q, gate: Gpi2(phi), meta: { name: "gpi2", params: [phi] } });
  }
  /** MS(φ₀, φ₁) — Mølmer-Sørensen entangling gate; IonQ's native two-qubit operation. MS(0,0) = XX(π/2). */
  ms(phi0, phi1, a, b) {
    if (typeof phi0 === "string" || typeof phi1 === "string")
      return this.#add({ kind: "parametric", name: "ms", params: [phi0, phi1], qubits: [a, b] });
    return this.#add({ kind: "two", a, b, gate: Ms(phi0, phi1), meta: { name: "ms", params: [phi0, phi1] } });
  }
  // ── Three-qubit gates ────────────────────────────────────────────────────
  /** Toffoli (CCX): flip target if both c1 and c2 are |1⟩. Universal for reversible computation. */
  ccx(c1, c2, target) {
    return this.#add({ kind: "toffoli", c1, c2, target });
  }
  /** Fredkin (CSWAP): swap qubits a and b if control is |1⟩. */
  cswap(control, a, b) {
    return this.#add({ kind: "cswap", control, a, b });
  }
  /** C-√iSWAP: apply √iSWAP to qubits a and b if control is |1⟩. Completes the three-qubit gate set. */
  csrswap(control, a, b) {
    return this.#add({ kind: "csrswap", control, a, b });
  }
  // ── Scheduling hints ─────────────────────────────────────────────────────
  /**
   * Barrier — scheduling/grouping hint with no effect on the statevector.
   * In QASM export it emits `barrier q[a],q[b],...;`. Pass the qubit indices to barrier.
   * Calling with no arguments barriers all qubits.
   */
  barrier(...qubits) {
    const qs = qubits.length ? qubits : Array.from({ length: this.qubits }, (_, i) => i);
    return this.#add({ kind: "barrier", qubits: qs });
  }
  // ── Custom unitary gate ──────────────────────────────────────────────────
  /**
   * Apply a custom N-qubit unitary gate defined by its 2^N × 2^N matrix.
   *
   * `matrix` must be 2^N × 2^N where N = qubits.length. Entries may be
   * `Complex` objects `{ re, im }` or plain `number` (treated as real).
   * The qubit ordering matches all other multi-qubit gates: `qubits[0]` is the
   * MSB of the local state index.
   *
   * @example
   * // Real matrix
   * circuit.unitary([[1,0],[0,1]], 0)
   *
   * // Complex matrix
   * const S = [[{re:1,im:0},{re:0,im:0}],[{re:0,im:0},{re:0,im:1}]]
   * circuit.unitary(S, 0)
   */
  unitary(matrix, ...qubits) {
    const normalized = matrix.map(
      (row) => row.map((v) => typeof v === "number" ? { re: v, im: 0 } : v)
    );
    return this.#add({ kind: "unitary", qubits, matrix: normalized });
  }
  // ── Statevector inspection ────────────────────────────────────────────────
  /**
   * Simulate the circuit and return the full sparse amplitude map.
   * Only valid for pure circuits (no `measure` / `reset` / `if` ops).
   *
   * @param initialState Optional starting computational basis state as a bitstring (q0 leftmost).
   */
  statevector({ initialState } = {}) {
    if (this.#ops.some((op) => op.kind === "measure" || op.kind === "reset" || op.kind === "if")) {
      throw new TypeError("statevector() requires a pure circuit \u2014 remove measure/reset/if ops");
    }
    const unbound = collectParams(this.#ops);
    if (unbound.size) {
      throw new TypeError(`statevector() requires bound parameters. Call bind({ ${[...unbound].map((p) => `${p}: value`).join(", ")} }) first.`);
    }
    const init = initialState !== void 0 ? svFromBitstring(initialState, this.qubits) : void 0;
    return simulatePure(this.#ops, this.qubits, init);
  }
  /**
   * Return the 2^n × 2^n unitary matrix of the circuit.
   *
   * `matrix[row][col]` is the amplitude of basis state `|row⟩` after starting
   * from `|col⟩`. Row and column indices use **standard convention**: q0 is the
   * MSB, so the ordering is |00…0⟩, |00…1⟩, …, |11…1⟩ with the first qubit
   * varying slowest. This matches the convention of `unitary()`, textbooks, and
   * most quantum computing libraries.
   *
   * Note: the column/row index ordering here uses q0 as MSB of the integer index,
   * which differs from the public bitstring API where q0 is the leftmost character.
   *
   * Throws `TypeError` for circuits with mid-circuit measurement, reset, or
   * conditional ops. Throws `RangeError` for circuits wider than 12 qubits
   * (matrix would be 4096×4096 = 16M entries).
   *
   * @example
   * new Circuit(2).cnot(0, 1).circuitMatrix()
   * // [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]  — standard CNOT matrix
   */
  circuitMatrix() {
    if (this.#ops.some((op) => op.kind === "measure" || op.kind === "reset" || op.kind === "if")) {
      throw new TypeError("circuitMatrix() requires a pure circuit \u2014 remove measure/reset/if ops");
    }
    const n = this.qubits;
    const dim = 1 << n;
    if (n > 12) {
      throw new RangeError(`circuitMatrix(): circuit too large (${n} qubits = ${dim}\xD7${dim} matrix)`);
    }
    const flip = (i) => {
      let r = 0;
      for (let b = 0; b < n; b++) if (i & 1 << b) r |= 1 << n - 1 - b;
      return r;
    };
    const matrix = Array.from({ length: dim }, () => new Array(dim).fill(ZERO));
    for (let col = 0; col < dim; col++) {
      const sv = simulatePure(this.#ops, this.qubits, /* @__PURE__ */ new Map([[BigInt(flip(col)), { re: 1, im: 0 }]]));
      for (const [idx, amp] of sv) matrix[flip(Number(idx))][col] = amp;
    }
    return matrix;
  }
  /**
   * Return the complex amplitude for the basis state identified by `bitstring`.
   * Bitstring format: q0 is the leftmost character (standard convention), e.g. `'10'` = q0=1, q1=0.
   */
  amplitude(bitstring) {
    if (bitstring.length !== this.qubits || !/^[01]+$/.test(bitstring))
      throw new TypeError(`bitstring '${bitstring}' must be a ${this.qubits}-character binary string`);
    return this.statevector().get(BigInt("0b" + [...bitstring].reverse().join(""))) ?? ZERO;
  }
  /** Return the measurement probability (|amplitude|²) for the given basis state bitstring. */
  probability(bitstring) {
    const { re, im } = this.amplitude(bitstring);
    return re * re + im * im;
  }
  /**
   * Return the marginal probability P(qubit q = |1⟩) for each qubit.
   * Result[q] is the probability of measuring qubit q as 1, summed over all other qubits.
   * Only valid for pure circuits (no `measure` / `reset` / `if` ops).
   */
  marginals({ initialState } = {}) {
    const sv = this.statevector(initialState !== void 0 ? { initialState } : {});
    const out = new Array(this.qubits).fill(0);
    for (const [idx, amp] of sv) {
      const p = amp.re * amp.re + amp.im * amp.im;
      for (let q = 0; q < this.qubits; q++) {
        if (idx >> BigInt(q) & 1n) out[q] += p;
      }
    }
    return out;
  }
  /**
   * Return a human-readable representation of the statevector, e.g.:
   *   `0.7071|00⟩ + 0.7071|11⟩`
   *   `0.5|00⟩ + (0.5+0.5i)|01⟩ - 0.5i|10⟩`
   *
   * Amplitudes with magnitude² < 1e-10 are omitted.
   * Only valid for pure circuits (no `measure` / `reset` / `if` ops).
   */
  stateAsString({ initialState } = {}) {
    const sv = this.statevector(initialState !== void 0 ? { initialState } : {});
    const eps = 1e-10;
    const n = (x) => parseFloat(x.toPrecision(4)).toString();
    const fmtAmp = ({ re, im }) => {
      const rz = Math.abs(re) < eps, iz = Math.abs(im) < eps;
      if (rz && iz) return "0";
      if (iz) return n(re);
      if (rz) return `${n(im)}i`;
      const sign = im < 0 ? "" : "+";
      return `(${n(re)}${sign}${n(im)}i)`;
    };
    const entries = [...sv.entries()].filter(([, { re, im }]) => re * re + im * im > eps * eps).sort(([a], [b]) => a < b ? -1 : 1);
    if (entries.length === 0) return "0";
    const terms = entries.map(
      ([idx, amp]) => `${fmtAmp(amp)}|${idx.toString(2).padStart(this.qubits, "0").split("").reverse().join("")}\u27E9`
    );
    let result = terms[0];
    for (let i = 1; i < terms.length; i++) {
      const t = terms[i];
      result += t.startsWith("-") ? ` - ${t.slice(1)}` : ` + ${t}`;
    }
    return result;
  }
  /**
   * Return the statevector as a sorted array of basis-state entries.
   *
   * Each entry contains:
   * - `bitstring` — q0-leftmost basis label, e.g. `'10'` = q0=1, q1=0
   * - `re`, `im`  — real and imaginary parts of the amplitude
   * - `prob`      — measurement probability |amplitude|²
   * - `phase`     — argument of the amplitude in radians: `Math.atan2(im, re)`
   *
   * Entries with prob < 1e-10 are omitted. Results are sorted by prob descending.
   * Only valid for pure circuits (no `measure` / `reset` / `if` ops).
   *
   * @example
   * new Circuit(2).h(0).cnot(0, 1).stateAsArray()
   * // [
   * //   { bitstring: '00', re: 0.7071, im: 0, prob: 0.5, phase: 0 },
   * //   { bitstring: '11', re: 0.7071, im: 0, prob: 0.5, phase: 0 },
   * // ]
   */
  stateAsArray() {
    const sv = this.statevector();
    const n = this.qubits;
    const entries = [];
    for (const [idx, { re, im }] of sv) {
      const prob = re * re + im * im;
      if (prob < 1e-10) continue;
      entries.push({
        bitstring: idx.toString(2).padStart(n, "0").split("").reverse().join(""),
        re,
        im,
        prob,
        phase: Math.atan2(im, re)
      });
    }
    return entries.sort((a, b) => b.prob - a.prob);
  }
  // ── Classical registers and mid-circuit measurement ──────────────────────
  /** Declare a classical register of `size` bits. */
  creg(name, size) {
    return new _Circuit(this.qubits, this.#ops, new Map(this.#cregs).set(name, size), this.#gates);
  }
  /**
   * Measure qubit `q` in the computational basis, storing the outcome in
   * `creg[bit]`. Collapses the statevector for that shot.
   * Auto-registers the creg if not yet declared.
   */
  measure(q, creg, bit) {
    const size = Math.max(this.#cregs.get(creg) ?? 0, bit + 1);
    return new _Circuit(
      this.qubits,
      [...this.#ops, { kind: "measure", q, creg, bit }],
      new Map(this.#cregs).set(creg, size),
      this.#gates
    );
  }
  /**
   * Reset qubit `q` to the given computational basis state (default |0⟩).
   *
   * - `reset(q)` / `reset(q, 0)` — unconditionally collapses `q` to |0⟩.
   * - `reset(q, 1)` — collapses to |0⟩ then flips to |1⟩, equivalent to `reset(q).x(q)`.
   */
  reset(q, value = 0) {
    const r = this.#add({ kind: "reset", q });
    return value === 1 ? r.x(q) : r;
  }
  /**
   * Conditionally apply a gate (or sequence of gates) only when the classical
   * register `creg` equals `value`.
   *
   * `value` is compared against the register as a little-endian integer:
   * bit 0 is the LSB.  For a 1-bit register, `value` is simply 0 or 1.
   *
   * @example
   * // Apply X to q2 only if register 'c' == 1
   * circuit.if('c', 1, c => c.x(2))
   */
  if(creg, value, build) {
    const inner = build(new _Circuit(this.qubits, [], /* @__PURE__ */ new Map(), this.#gates));
    return this.#add({ kind: "if", creg, value, ops: inner.#ops });
  }
  // ── Named sub-circuit gates ──────────────────────────────────────────────
  /**
   * Register a named reusable gate defined by `sub`.
   *
   * The registered gate can be used on this circuit (and any circuit derived
   * from it) via `.gate(name, ...qubits)`.
   *
   * @example
   * const bell = new Circuit(2).h(0).cnot(0, 1)
   * const c = new Circuit(4)
   *   .defineGate('bell', bell)
   *   .gate('bell', 0, 1)   // Bell pair on qubits 0,1
   *   .gate('bell', 2, 3)   // Bell pair on qubits 2,3
   */
  defineGate(name, sub) {
    if (sub.#ops.some((op) => op.kind === "measure" || op.kind === "reset" || op.kind === "if")) {
      throw new TypeError(`Gate '${name}' contains classical ops (measure/reset/if), which are not supported inside named gates`);
    }
    return new _Circuit(this.qubits, this.#ops, this.#cregs, new Map(this.#gates).set(name, sub));
  }
  /**
   * Apply a previously registered named gate to the given parent-circuit qubits.
   *
   * The number of `qubits` must match the qubit count of the registered gate.
   * Qubit 0 of the gate maps to `qubits[0]`, qubit 1 to `qubits[1]`, etc.
   *
   * @example
   * circuit.gate('bell', 2, 3)  // apply 'bell' gate to parent qubits 2 and 3
   */
  gate(name, ...qubits) {
    const sub = this.#gates.get(name);
    if (!sub) throw new TypeError(`Unknown gate '${name}'. Register it first with .defineGate(name, subcircuit).`);
    if (qubits.length !== sub.qubits) {
      throw new TypeError(`Gate '${name}' expects ${sub.qubits} qubit(s), got ${qubits.length}`);
    }
    return this.#add({ kind: "subcircuit", name, qubits, def: sub.#ops });
  }
  /**
   * Inline all named gates, returning a new `Circuit` containing only primitive ops.
   *
   * Required before serialization (toQASM, toQiskit, etc.) when the circuit
   * contains named gates applied via `.gate()`.
   *
   * @example
   * circuit.defineGate('bell', bell).gate('bell', 0, 1).decompose()
   */
  decompose() {
    return new _Circuit(this.qubits, flattenOps(this.#ops), this.#cregs, this.#gates);
  }
  /**
   * Append all ops from `other` onto this circuit and return the combined circuit.
   *
   * Both circuits must have the same qubit count. Classical registers and named
   * gate definitions from both circuits are merged: if the same register name
   * appears in both, the larger declared size wins; if the same gate name appears
   * in both, `this` takes precedence.
   *
   * @example
   * const prep = new Circuit(2).h(0).cnot(0, 1)
   * const meas = new Circuit(2).measure(0, 'c', 0).measure(1, 'c', 1)
   * const full = prep.compose(meas)
   * full.run({ shots: 1000 })
   */
  compose(other) {
    if (other.qubits !== this.qubits)
      throw new TypeError(
        `compose: qubit count mismatch \u2014 this has ${this.qubits}, other has ${other.qubits}`
      );
    const mergedCregs = new Map(this.#cregs);
    for (const [name, size] of other.#cregs)
      mergedCregs.set(name, Math.max(mergedCregs.get(name) ?? 0, size));
    const mergedGates = new Map(other.#gates);
    for (const [name, gate] of this.#gates) mergedGates.set(name, gate);
    return new _Circuit(this.qubits, [...this.#ops, ...other.#ops], mergedCregs, mergedGates);
  }
  /**
   * Return the names of all unbound symbolic parameters in this circuit, sorted.
   *
   * @example
   * new Circuit(1).rx('theta', 0).ry('phi', 0).params  // ['phi', 'theta']
   */
  get params() {
    return [...collectParams(this.#ops)].sort();
  }
  /**
   * Substitute symbolic parameter names with concrete numeric values and return
   * a new fully-bound circuit that can be simulated.
   *
   * Throws `TypeError` if any parameter referenced in the circuit is not supplied
   * in `values`.
   *
   * @example
   * // VQE-style: build once, sweep parameters
   * const ansatz = new Circuit(1).h(0).rx('theta', 0).rz('phi', 0)
   * for (const theta of [0, Math.PI / 4, Math.PI / 2]) {
   *   const e = ansatz.bind({ theta, phi: 0.5 }).expectation('Z')
   * }
   */
  bind(values) {
    const resolve = (p) => {
      if (typeof p === "number") return p;
      if (p in values) return values[p];
      throw new TypeError(`bind: unbound parameter '${p}'. Provide a value for it in the bind() call.`);
    };
    const rebuildOps = (ops) => ops.map((op) => {
      if (op.kind === "parametric") {
        const ps = op.params.map(resolve);
        return resolveParametric(op.name, ps, op.qubits);
      }
      if (op.kind === "if") return { ...op, ops: rebuildOps(op.ops) };
      return op;
    });
    return new _Circuit(this.qubits, rebuildOps(this.#ops), this.#cregs, this.#gates);
  }
  // ── IonQ device targeting ────────────────────────────────────────────────
  /**
   * Return the published specs for any supported device (IonQ, IBM, Quantinuum).
   * Throws if the device name is not recognised.
   *
   * @example
   * Circuit.device('ibm_sherbrooke').noise  // { p1: 2.4e-4, p2: 7.4e-3, pMeas: 1.35e-2 }
   * Circuit.device('h1-1').qubits           // 20
   */
  static device(name) {
    const info = DEVICES[name];
    if (!info) throw new TypeError(`Unknown device '${name}'. Known: ${Object.keys(DEVICES).join(", ")}`);
    return info;
  }
  /**
   * Return the published specs for a named IonQ device.
   * Throws if the device name is not recognised.
   * @deprecated Use `Circuit.device(name)` instead.
   */
  static ionqDevice(name) {
    const info = IONQ_DEVICES[name];
    if (!info) throw new TypeError(`Unknown IonQ device '${name}'. Known: ${Object.keys(IONQ_DEVICES).join(", ")}`);
    return info;
  }
  /**
   * Validate that this circuit can be submitted to the named IonQ device.
   * Throws a `TypeError` listing every issue found:
   *   - qubit count exceeds device capacity
   *   - gates that have no IonQ JSON representation (use `decompose()` or replace them)
   *
   * Call this before `toIonQ()` to get a complete error report rather than a
   * first-failure throw.
   */
  checkDevice(name) {
    const info = _Circuit.ionqDevice(name);
    const issues = [];
    if (this.qubits > info.qubits)
      issues.push(`circuit uses ${this.qubits} qubits; ${name} supports at most ${info.qubits}`);
    const IONQ_SINGLE = /* @__PURE__ */ new Set(["h", "x", "y", "z", "s", "si", "t", "ti", "v", "vi", "rx", "ry", "rz", "r2", "r4", "r8", "gpi", "gpi2", "vz", "id"]);
    const IONQ_TWO = /* @__PURE__ */ new Set(["xx", "yy", "zz", "ms"]);
    const seen = /* @__PURE__ */ new Set();
    for (const op of flattenOps(this.#ops)) {
      if (op.kind === "cnot" || op.kind === "swap" || op.kind === "barrier") continue;
      if (op.kind === "single" && op.meta && IONQ_SINGLE.has(op.meta.name)) continue;
      if (op.kind === "two" && op.meta && IONQ_TWO.has(op.meta.name)) continue;
      const label = op.meta?.name ?? op.kind;
      if (!seen.has(label)) {
        seen.add(label);
        issues.push(`gate '${label}' is not supported on ${name}`);
      }
    }
    if (issues.length) throw new TypeError(`Circuit is not compatible with ${name}:
  - ${issues.join("\n  - ")}`);
  }
  // ── IonQ JSON import / export ────────────────────────────────────────────
  /**
   * Parse an `ionq.circuit.v0` JSON object into a `Circuit`.
   *
   * Angle convention: `rotation` fields are in π-radians (1.0 = π rad);
   * `phase` / `phases` fields are in turns (1.0 = 2π rad).
   */
  static fromIonQ({ qubits, circuit }) {
    let c2 = new _Circuit(qubits);
    for (const g of circuit) {
      const t = g.target ?? 0;
      const [a, b] = g.targets ?? [0, 1];
      const rot = (g.rotation ?? 0) * Math.PI;
      const ph = (g.phase ?? 0) * 2 * Math.PI;
      switch (g.gate) {
        case "h":
          c2 = c2.h(t);
          break;
        case "x":
          c2 = c2.x(t);
          break;
        case "y":
          c2 = c2.y(t);
          break;
        case "z":
          c2 = c2.z(t);
          break;
        case "s":
          c2 = c2.s(t);
          break;
        case "si":
          c2 = c2.si(t);
          break;
        case "t":
          c2 = c2.t(t);
          break;
        case "ti":
          c2 = c2.ti(t);
          break;
        case "v":
          c2 = c2.v(t);
          break;
        case "vi":
          c2 = c2.vi(t);
          break;
        case "rx":
          c2 = c2.rx(rot, t);
          break;
        case "ry":
          c2 = c2.ry(rot, t);
          break;
        case "rz":
          c2 = c2.rz(rot, t);
          break;
        case "r2":
          c2 = c2.r2(t);
          break;
        case "r4":
          c2 = c2.r4(t);
          break;
        case "r8":
          c2 = c2.r8(t);
          break;
        case "gpi":
          c2 = c2.gpi(ph, t);
          break;
        case "gpi2":
          c2 = c2.gpi2(ph, t);
          break;
        case "cnot":
          c2 = c2.cnot(g.control ?? 0, t);
          break;
        case "swap":
          c2 = c2.swap(a, b);
          break;
        case "xx":
          c2 = c2.xx(rot, a, b);
          break;
        case "yy":
          c2 = c2.yy(rot, a, b);
          break;
        case "zz":
          c2 = c2.zz(rot, a, b);
          break;
        case "ms": {
          const [p0, p1] = (g.phases ?? [0, 0]).map((p) => p * 2 * Math.PI);
          c2 = c2.ms(p0, p1, a, b);
          break;
        }
        default:
          throw new TypeError(`Unknown IonQ gate: '${g.gate}'`);
      }
    }
    return c2;
  }
  /**
   * Serialize to an `ionq.circuit.v0` JSON object ready for IonQ Cloud or qsim.
   *
   * Throws `TypeError` for any gate that has no IonQ JSON representation
   * (controlled variants, U-gates, XY/iSWAP, mid-circuit measurement, etc.).
   */
  toIonQ() {
    const IONQ_SINGLE = /* @__PURE__ */ new Set(["h", "x", "y", "z", "s", "si", "t", "ti", "v", "vi", "rx", "ry", "rz", "r2", "r4", "r8", "gpi", "gpi2"]);
    const IONQ_TWO = /* @__PURE__ */ new Set(["xx", "yy", "zz", "ms"]);
    const circuit = [];
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "cnot":
          circuit.push({ gate: "cnot", control: op.control, target: op.target });
          break;
        case "swap":
          circuit.push({ gate: "swap", targets: [op.a, op.b] });
          break;
        case "csrswap":
          throw new TypeError(`Gate 'csrswap' is not serializable to IonQ JSON`);
        case "single": {
          if (op.meta?.name === "id") {
          } else if (op.meta?.name === "vz") {
            circuit.push({ gate: "rz", target: op.q, rotation: op.meta.params[0] / Math.PI });
          } else if (op.meta && IONQ_SINGLE.has(op.meta.name)) {
            const { name, params } = op.meta;
            const g = { gate: name, target: op.q };
            if (params) {
              if (name === "gpi" || name === "gpi2") g.phase = params[0] / (2 * Math.PI);
              else g.rotation = params[0] / Math.PI;
            }
            circuit.push(g);
          } else {
            const n = op.meta?.name ?? "single";
            throw new TypeError(`Gate '${n}' is not serializable to IonQ JSON`);
          }
          break;
        }
        case "two": {
          if (op.meta && IONQ_TWO.has(op.meta.name)) {
            const { name, params } = op.meta;
            const g = { gate: name, targets: [op.a, op.b] };
            if (params) {
              if (name === "ms") g.phases = [params[0] / (2 * Math.PI), params[1] / (2 * Math.PI)];
              else g.rotation = params[0] / Math.PI;
            }
            circuit.push(g);
          } else {
            const n = op.meta?.name ?? "two";
            throw new TypeError(`Gate '${n}' is not serializable to IonQ JSON`);
          }
          break;
        }
        case "controlled": {
          const n = op.meta?.name ?? "controlled";
          throw new TypeError(`Gate '${n}' is not serializable to IonQ JSON`);
        }
        case "toffoli":
          throw new TypeError(`Gate 'toffoli' is not serializable to IonQ JSON`);
        case "cswap":
          throw new TypeError(`Gate 'cswap' is not serializable to IonQ JSON`);
        case "measure":
          throw new TypeError(`Gate 'measure' is not serializable to IonQ JSON`);
        case "reset":
          throw new TypeError(`Gate 'reset' is not serializable to IonQ JSON`);
        case "if":
          throw new TypeError(`Gate 'if' is not serializable to IonQ JSON`);
        case "unitary":
          throw new TypeError("Gate 'unitary' is not serializable to IonQ JSON");
        case "barrier":
          break;
        default: {
          const _exhaustive = op;
        }
      }
    }
    return { format: "ionq.circuit.v0", qubits: this.qubits, circuit };
  }
  // ── OpenQASM 2.0 import / export ─────────────────────────────────────────
  /**
   * Emit a valid OpenQASM 2.0 string for this circuit.
   *
   * Gate name mapping: si→sdg, ti→tdg, v→sx, vi→sxdg; r2/r4/r8→rz(π/n);
   * cs/ct/csdg/ctdg→cu1(±π/n); cr2/cr4/cr8→crz(π/n).
   * Throws `TypeError` for gates with no QASM 2.0 representation (gpi, gpi2, xx, yy, zz, ms, xy, iswap, srswap, if).
   */
  toQASM() {
    const lines = [
      "OPENQASM 2.0;",
      'include "qelib1.inc";',
      "",
      `qreg q[${this.qubits}];`
    ];
    for (const [name, size] of this.#cregs) lines.push(`creg ${name}[${size}];`);
    if (this.#cregs.size) lines.push("");
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "cnot":
          lines.push(`cx q[${op.control}],q[${op.target}];`);
          break;
        case "swap":
          lines.push(`swap q[${op.a}],q[${op.b}];`);
          break;
        case "toffoli":
          lines.push(`ccx q[${op.c1}],q[${op.c2}],q[${op.target}];`);
          break;
        case "cswap":
          lines.push(`cswap q[${op.control}],q[${op.a}],q[${op.b}];`);
          break;
        case "csrswap":
          throw new TypeError("Gate 'csrswap' has no OpenQASM 2.0 representation");
        case "measure":
          lines.push(`measure q[${op.q}] -> ${op.creg}[${op.bit}];`);
          break;
        case "reset":
          lines.push(`reset q[${op.q}];`);
          break;
        case "barrier":
          lines.push(`barrier ${op.qubits.map((q) => `q[${q}]`).join(",")};`);
          break;
        case "if":
          throw new TypeError("if ops cannot be serialized to OpenQASM 2.0");
        case "single": {
          if (!op.meta) throw new TypeError("Single-qubit op missing serialization meta");
          if (op.meta.name === "gpi" || op.meta.name === "gpi2")
            throw new TypeError(`Gate '${op.meta.name}' has no OpenQASM 2.0 representation`);
          const { qname, qparams } = qasmGateName(op.meta);
          const ps = qparams.length ? `(${qparams.map(qasmAngle).join(",")})` : "";
          lines.push(`${qname}${ps} q[${op.q}];`);
          break;
        }
        case "controlled": {
          if (!op.meta) throw new TypeError("Controlled op missing serialization meta");
          const { qname, qparams } = qasmGateName(op.meta);
          const ps = qparams.length ? `(${qparams.map(qasmAngle).join(",")})` : "";
          lines.push(`${qname}${ps} q[${op.control}],q[${op.target}];`);
          break;
        }
        case "two":
          throw new TypeError(`Gate '${op.meta?.name ?? "two"}' has no OpenQASM 2.0 representation`);
        case "unitary":
          throw new TypeError("Gate 'unitary' has no OpenQASM 2.0 representation");
        default: {
          const _exhaustive = op;
        }
      }
    }
    return lines.join("\n");
  }
  /**
   * Parse an OpenQASM 2.0 or 3.0 string into a `Circuit`. Auto-detects the version.
   *
   * **2.0 syntax supported:** `qreg`/`creg`, `measure q[i] -> c[j]`, `//` comments,
   * all qelib1.inc gates (h, x, cx, rz, u1, u2, u3, ccx, cswap, …).
   *
   * **3.0 syntax supported:** qubit[N]/bit[N] declarations, "c[j] = measure q[i]"
   * assignment form, block comments, stdgates.inc, p/sx/sdg/tdg gate names.
   *
   * Not supported: gate definitions (`gate foo …`), gate modifiers (`ctrl @`, `inv @`),
   * `if`/`else` blocks, `gphase`, multi-register qubit indexing.
   */
  static fromQASM(source) {
    const stmts = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "").split(";").map((s) => s.trim()).filter(Boolean);
    let qubits = 0;
    const cregSizes = /* @__PURE__ */ new Map();
    for (const stmt of stmts) {
      const qr2 = stmt.match(/^qreg\s+\w+\[(\d+)\]$/);
      if (qr2) {
        qubits += parseInt(qr2[1]);
        continue;
      }
      const qr3 = stmt.match(/^qubit(?:\[(\d+)\])?\s+\w+$/);
      if (qr3) {
        qubits += parseInt(qr3[1] ?? "1");
        continue;
      }
      const cr2 = stmt.match(/^creg\s+(\w+)\[(\d+)\]$/);
      if (cr2) {
        cregSizes.set(cr2[1], parseInt(cr2[2]));
        continue;
      }
      const cr3 = stmt.match(/^bit(?:\[(\d+)\])?\s+(\w+)$/);
      if (cr3) {
        cregSizes.set(cr3[2], parseInt(cr3[1] ?? "1"));
        continue;
      }
    }
    let c2 = new _Circuit(qubits);
    for (const [name, size] of cregSizes) c2 = c2.creg(name, size);
    for (const stmt of stmts) {
      if (/^(OPENQASM|include|qreg|creg|qubit|bit)\b/.test(stmt)) continue;
      const meas2 = stmt.match(/^measure\s+\w+\[(\d+)\]\s*->\s*(\w+)\[(\d+)\]$/);
      if (meas2) {
        c2 = c2.measure(parseInt(meas2[1]), meas2[2], parseInt(meas2[3]));
        continue;
      }
      const meas3 = stmt.match(/^(\w+)\[(\d+)\]\s*=\s*measure\s+\w+\[(\d+)\]$/);
      if (meas3) {
        c2 = c2.measure(parseInt(meas3[3]), meas3[1], parseInt(meas3[2]));
        continue;
      }
      const rst = stmt.match(/^reset\s+\w+\[(\d+)\]$/);
      if (rst) {
        c2 = c2.reset(parseInt(rst[1]));
        continue;
      }
      const gate = stmt.match(/^(\w+)(?:\(([^)]*)\))?\s+([\w\[\],\s]+)$/);
      if (gate) {
        const name = gate[1];
        const params = gate[2] ? gate[2].split(",").map((p) => parseAngle(p)) : [];
        const qs = [...gate[3].matchAll(/\[(\d+)\]/g)].map((m) => parseInt(m[1]));
        c2 = applyQASMGate(c2, name, params, qs);
      }
    }
    return c2;
  }
  /**
   * Parse a Quil 2.0 program and return an equivalent Circuit.
   *
   * Supported: all standard single/two/three-qubit gates, `CONTROLLED` prefix,
   * `DAGGER` prefix, `MEASURE`, `RESET`, `DECLARE BIT[]`.
   * Qubit count is inferred from the highest qubit index referenced.
   * `DEFGATE` and `PRAGMA` are silently skipped.
   *
   * @example
   * const c = Circuit.fromQuil('H 0\nCNOT 0 1\nMEASURE 0 ro[0]')
   */
  static fromQuil(source) {
    const lines = source.split("\n").map((l) => l.replace(/#.*/g, "").trim()).filter(Boolean);
    const cregs = /* @__PURE__ */ new Map();
    let maxQubit = 0;
    for (const line of lines) {
      const decl = line.match(/^DECLARE\s+(\w+)\s+BIT\[(\d+)\]/);
      if (decl) {
        cregs.set(decl[1], parseInt(decl[2]));
        continue;
      }
      const withoutParams = line.replace(/\([^)]*\)/g, "");
      for (const m of withoutParams.matchAll(/\b(\d+)\b/g)) {
        maxQubit = Math.max(maxQubit, parseInt(m[1]));
      }
    }
    let c2 = new _Circuit(maxQubit + 1);
    for (const [name, size] of cregs) c2 = c2.creg(name, size);
    for (const line of lines) {
      if (/^(DECLARE|DEFGATE|DEFCIRCUIT|PRAGMA|HALT|WAIT|NOP)\b/.test(line)) continue;
      const meas = line.match(/^MEASURE\s+(\d+)\s+(\w+)\[(\d+)\]$/);
      if (meas) {
        c2 = c2.measure(parseInt(meas[1]), meas[2], parseInt(meas[3]));
        continue;
      }
      const rst = line.match(/^RESET(?:\s+(\d+))?$/);
      if (rst) {
        if (rst[1] !== void 0) {
          c2 = c2.reset(parseInt(rst[1]));
        } else {
          for (let q = 0; q < c2.qubits; q++) c2 = c2.reset(q);
        }
        continue;
      }
      const ctrlLine = line.match(/^CONTROLLED\s+(\w+)(?:\(([^)]*)\))?\s+(\d+)\s+(\d+)$/);
      if (ctrlLine) {
        const [, gName, paramStr, ctrlStr, tgtStr] = ctrlLine;
        const p = paramStr ? paramStr.split(",").map(parseAngle) : [];
        const [con, tgt] = [parseInt(ctrlStr), parseInt(tgtStr)];
        switch (gName.toUpperCase()) {
          case "H":
            c2 = c2.ch(con, tgt);
            break;
          case "Y":
            c2 = c2.cy(con, tgt);
            break;
          case "Z":
            c2 = c2.cz(con, tgt);
            break;
          case "RX":
            c2 = c2.crx(p[0], con, tgt);
            break;
          case "RY":
            c2 = c2.cry(p[0], con, tgt);
            break;
          case "RZ":
            c2 = c2.crz(p[0], con, tgt);
            break;
          case "PHASE":
            c2 = c2.cu1(p[0], con, tgt);
            break;
          default:
            throw new TypeError(`fromQuil: unknown CONTROLLED gate '${gName}'`);
        }
        continue;
      }
      const daggerLine = line.match(/^DAGGER\s+(\w+)\s+(\d+)$/);
      if (daggerLine) {
        const [, gName, qStr] = daggerLine;
        const q = parseInt(qStr);
        switch (gName.toUpperCase()) {
          case "S":
            c2 = c2.si(q);
            break;
          case "T":
            c2 = c2.ti(q);
            break;
          default:
            throw new TypeError(`fromQuil: unknown DAGGER gate '${gName}'`);
        }
        continue;
      }
      const gate = line.match(/^(\w+)(?:\(([^)]*)\))?\s+([\d\s]+)$/);
      if (gate) {
        const [, gName, paramStr, qStr] = gate;
        const p = paramStr ? paramStr.split(",").map(parseAngle) : [];
        const qs = qStr.trim().split(/\s+/).map(Number);
        const [q0, q1, q2] = qs;
        switch (gName.toUpperCase()) {
          case "I":
            c2 = c2.id(q0);
            break;
          case "H":
            c2 = c2.h(q0);
            break;
          case "X":
            c2 = c2.x(q0);
            break;
          case "Y":
            c2 = c2.y(q0);
            break;
          case "Z":
            c2 = c2.z(q0);
            break;
          case "S":
            c2 = c2.s(q0);
            break;
          case "T":
            c2 = c2.t(q0);
            break;
          case "RX":
            c2 = c2.rx(p[0], q0);
            break;
          case "RY":
            c2 = c2.ry(p[0], q0);
            break;
          case "RZ":
            c2 = c2.rz(p[0], q0);
            break;
          case "PHASE":
            c2 = c2.u1(p[0], q0);
            break;
          case "CNOT":
            c2 = c2.cnot(q0, q1);
            break;
          case "CZ":
            c2 = c2.cz(q0, q1);
            break;
          case "SWAP":
            c2 = c2.swap(q0, q1);
            break;
          case "ISWAP":
            c2 = c2.iswap(q0, q1);
            break;
          case "CPHASE":
            c2 = c2.cu1(p[0], q0, q1);
            break;
          case "CCNOT":
            c2 = c2.ccx(q0, q1, q2);
            break;
          case "CSWAP":
            c2 = c2.cswap(q0, q1, q2);
            break;
          default:
            throw new TypeError(`fromQuil: unknown gate '${gName}'`);
        }
        continue;
      }
    }
    return c2;
  }
  /**
   * Parse Qiskit Python QuantumCircuit code back into a Circuit.
   * Accepts the output of toQiskit() — round-trips all gates supported by that method.
   */
  static fromQiskit(source) {
    const pa = (e) => parseAngle(e.replace(/math\.pi/g, "pi").trim());
    const nm = source.match(/QuantumCircuit\((\d+)\)/);
    if (!nm) throw new TypeError("fromQiskit: cannot find QuantumCircuit(N)");
    let c2 = new _Circuit(parseInt(nm[1]));
    const cregRe = /ClassicalRegister\((\d+),\s*['"](\w+)['"]\)/g;
    let cm;
    while ((cm = cregRe.exec(source)) !== null) c2 = c2.creg(cm[2], parseInt(cm[1]));
    for (const rawLine of source.split("\n")) {
      const m = rawLine.trim().match(/^qc\.(\w+)\(([^)]*(?:\[[^\]]*\][^)]*)*)\)$/);
      if (!m) continue;
      const [, method, argStr] = m;
      const args = argStr.split(",").map((s) => s.trim()).filter(Boolean);
      const qi = (s) => parseInt(s);
      switch (method) {
        case "h":
          c2 = c2.h(qi(args[0]));
          break;
        case "x":
          c2 = c2.x(qi(args[0]));
          break;
        case "y":
          c2 = c2.y(qi(args[0]));
          break;
        case "z":
          c2 = c2.z(qi(args[0]));
          break;
        case "s":
          c2 = c2.s(qi(args[0]));
          break;
        case "sdg":
          c2 = c2.si(qi(args[0]));
          break;
        case "t":
          c2 = c2.t(qi(args[0]));
          break;
        case "tdg":
          c2 = c2.ti(qi(args[0]));
          break;
        case "sx":
          c2 = c2.v(qi(args[0]));
          break;
        case "sxdg":
          c2 = c2.vi(qi(args[0]));
          break;
        case "id":
          c2 = c2.id(qi(args[0]));
          break;
        case "rx":
          c2 = c2.rx(pa(args[0]), qi(args[1]));
          break;
        case "ry":
          c2 = c2.ry(pa(args[0]), qi(args[1]));
          break;
        case "rz":
          c2 = c2.rz(pa(args[0]), qi(args[1]));
          break;
        case "u1":
          c2 = c2.u1(pa(args[0]), qi(args[1]));
          break;
        case "u2":
          c2 = c2.u2(pa(args[0]), pa(args[1]), qi(args[2]));
          break;
        case "u3":
          c2 = c2.u3(pa(args[0]), pa(args[1]), pa(args[2]), qi(args[3]));
          break;
        case "p":
          c2 = c2.p(pa(args[0]), qi(args[1]));
          break;
        case "cx":
          c2 = c2.cnot(qi(args[0]), qi(args[1]));
          break;
        case "cy":
          c2 = c2.cy(qi(args[0]), qi(args[1]));
          break;
        case "cz":
          c2 = c2.cz(qi(args[0]), qi(args[1]));
          break;
        case "ch":
          c2 = c2.ch(qi(args[0]), qi(args[1]));
          break;
        case "swap":
          c2 = c2.swap(qi(args[0]), qi(args[1]));
          break;
        case "crx":
          c2 = c2.crx(pa(args[0]), qi(args[1]), qi(args[2]));
          break;
        case "cry":
          c2 = c2.cry(pa(args[0]), qi(args[1]), qi(args[2]));
          break;
        case "crz":
          c2 = c2.crz(pa(args[0]), qi(args[1]), qi(args[2]));
          break;
        case "cu1":
          c2 = c2.cu1(pa(args[0]), qi(args[1]), qi(args[2]));
          break;
        case "cu2":
          c2 = c2.cu2(pa(args[0]), pa(args[1]), qi(args[2]), qi(args[3]));
          break;
        case "cu3":
          c2 = c2.cu3(pa(args[0]), pa(args[1]), pa(args[2]), qi(args[3]), qi(args[4]));
          break;
        case "rxx":
          c2 = c2.xx(pa(args[0]), qi(args[1]), qi(args[2]));
          break;
        case "ryy":
          c2 = c2.yy(pa(args[0]), qi(args[1]), qi(args[2]));
          break;
        case "rzz":
          c2 = c2.zz(pa(args[0]), qi(args[1]), qi(args[2]));
          break;
        case "iswap":
          c2 = c2.iswap(qi(args[0]), qi(args[1]));
          break;
        case "ccx":
          c2 = c2.ccx(qi(args[0]), qi(args[1]), qi(args[2]));
          break;
        case "cswap":
          c2 = c2.cswap(qi(args[0]), qi(args[1]), qi(args[2]));
          break;
        case "reset":
          c2 = c2.reset(qi(args[0]));
          break;
        case "barrier": {
          const qs = args.filter((a) => /^\d+$/.test(a)).map(Number);
          c2 = qs.length ? c2.barrier(...qs) : c2.barrier();
          break;
        }
        case "measure": {
          const mt = args[1].match(/(\w+)\[(\d+)\]/);
          if (!mt) throw new TypeError(`fromQiskit: invalid measure target '${args[1]}'`);
          c2 = c2.measure(qi(args[0]), mt[1], parseInt(mt[2]));
          break;
        }
        case "add_register":
          break;
        default:
          throw new TypeError(`fromQiskit: unknown method 'qc.${method}'`);
      }
    }
    return c2;
  }
  /**
   * Parse Cirq Python circuit code back into a Circuit.
   * Accepts the output of toCirq() — round-trips all gates supported by that method.
   */
  static fromCirq(source) {
    const pa = (e) => parseAngle(e.replace(/math\.pi/g, "pi").trim());
    const nm = source.match(/LineQubit\.range\((\d+)\)/);
    if (!nm) throw new TypeError("fromCirq: cannot find LineQubit.range(N)");
    let c2 = new _Circuit(parseInt(nm[1]));
    const qis = (s) => Array.from(s.matchAll(/q\[(\d+)\]/g), (m) => parseInt(m[1]));
    for (const rawLine of source.split("\n")) {
      const line = rawLine.trim().replace(/,$/, "");
      if (!line.startsWith("cirq.") && !line.startsWith("(cirq.")) continue;
      if (line.startsWith("cirq.Circuit") || line.startsWith("cirq.LineQubit")) continue;
      const zpowCtrl = line.match(/^cirq\.ZPowGate\(exponent=([^)]+)\)\.controlled\(\)\(q\[(\d+)\],\s*q\[(\d+)\]\)$/);
      if (zpowCtrl) {
        const exp = parseFloat(zpowCtrl[1]);
        const ci = parseInt(zpowCtrl[2]);
        const ti = parseInt(zpowCtrl[3]);
        if (Math.abs(exp - 0.5) < 1e-9) c2 = c2.cs(ci, ti);
        else if (Math.abs(exp - 0.25) < 1e-9) c2 = c2.ct(ci, ti);
        else if (Math.abs(exp + 0.5) < 1e-9) c2 = c2.csdg(ci, ti);
        else if (Math.abs(exp + 0.25) < 1e-9) c2 = c2.ctdg(ci, ti);
        else c2 = c2.cu1(exp * Math.PI, ci, ti);
        continue;
      }
      const zpow = line.match(/^cirq\.ZPowGate\(exponent=([^)]+)\)\(q\[(\d+)\]\)$/);
      if (zpow) {
        const exp = parseFloat(zpow[1]);
        const q = parseInt(zpow[2]);
        if (Math.abs(exp - 0.5) < 1e-9) c2 = c2.s(q);
        else if (Math.abs(exp - 0.25) < 1e-9) c2 = c2.t(q);
        else if (Math.abs(exp + 0.5) < 1e-9) c2 = c2.si(q);
        else if (Math.abs(exp + 0.25) < 1e-9) c2 = c2.ti(q);
        else c2 = c2.u1(exp * Math.PI, q);
        continue;
      }
      const radsCtrl = line.match(/^cirq\.(rx|ry|rz)\(rads=([^)]+)\)\.controlled\(\)\(q\[(\d+)\],\s*q\[(\d+)\]\)$/);
      if (radsCtrl) {
        const r = pa(radsCtrl[2]);
        const ci = parseInt(radsCtrl[3]);
        const ti = parseInt(radsCtrl[4]);
        if (radsCtrl[1] === "rx") c2 = c2.crx(r, ci, ti);
        else if (radsCtrl[1] === "ry") c2 = c2.cry(r, ci, ti);
        else c2 = c2.crz(r, ci, ti);
        continue;
      }
      const radsSingle = line.match(/^cirq\.(rx|ry|rz)\(rads=([^)]+)\)\(q\[(\d+)\]\)$/);
      if (radsSingle) {
        const r = pa(radsSingle[2]);
        const q = parseInt(radsSingle[3]);
        if (radsSingle[1] === "rx") c2 = c2.rx(r, q);
        else if (radsSingle[1] === "ry") c2 = c2.ry(r, q);
        else c2 = c2.rz(r, q);
        continue;
      }
      const xpow = line.match(/^\(?cirq\.X\*\*([^)(]+)\)?\(q\[(\d+)\]\)$/);
      if (xpow) {
        const exp = parseFloat(xpow[1].trim());
        const q = parseInt(xpow[2]);
        if (Math.abs(exp - 0.5) < 1e-9) c2 = c2.v(q);
        else if (Math.abs(exp + 0.5) < 1e-9) c2 = c2.vi(q);
        else throw new TypeError(`fromCirq: unsupported X**${exp}`);
        continue;
      }
      const namedCtrl = line.match(/^cirq\.(\w+)\.controlled\(\)\(q\[(\d+)\],\s*q\[(\d+)\]\)$/);
      if (namedCtrl) {
        const ci = parseInt(namedCtrl[2]);
        const ti = parseInt(namedCtrl[3]);
        switch (namedCtrl[1]) {
          case "Y":
            c2 = c2.cy(ci, ti);
            break;
          case "H":
            c2 = c2.ch(ci, ti);
            break;
          default:
            throw new TypeError(`fromCirq: unknown controlled gate '${namedCtrl[1]}'`);
        }
        continue;
      }
      const simple = line.match(/^cirq\.(\w+)\((.+)\)$/);
      if (simple) {
        const qs = qis(simple[2]);
        switch (simple[1]) {
          case "I":
            c2 = c2.id(qs[0]);
            break;
          case "H":
            c2 = c2.h(qs[0]);
            break;
          case "X":
            c2 = c2.x(qs[0]);
            break;
          case "Y":
            c2 = c2.y(qs[0]);
            break;
          case "Z":
            c2 = c2.z(qs[0]);
            break;
          case "S":
            c2 = c2.s(qs[0]);
            break;
          case "T":
            c2 = c2.t(qs[0]);
            break;
          case "CNOT":
            c2 = c2.cnot(qs[0], qs[1]);
            break;
          case "CX":
            c2 = c2.cnot(qs[0], qs[1]);
            break;
          case "CZ":
            c2 = c2.cz(qs[0], qs[1]);
            break;
          case "SWAP":
            c2 = c2.swap(qs[0], qs[1]);
            break;
          case "CCNOT":
            c2 = c2.ccx(qs[0], qs[1], qs[2]);
            break;
          case "CSWAP":
            c2 = c2.cswap(qs[0], qs[1], qs[2]);
            break;
          default:
            throw new TypeError(`fromCirq: unknown gate '${simple[1]}'`);
        }
        continue;
      }
    }
    return c2;
  }
  /**
   * Parse a Qiskit Qobj JSON object into a Circuit.
   * Accepts the object returned by `qc.qobj()` or IBM Quantum job results.
   * Only the first experiment is used.
   */
  static fromQobj(qobj) {
    const exp = qobj.experiments[0];
    if (!exp) throw new TypeError("fromQobj: no experiments found");
    let c2 = new _Circuit(exp.header.n_qubits);
    const slotMap = [];
    for (const [name, size] of exp.header.creg_sizes ?? []) {
      c2 = c2.creg(name, size);
      for (let b = 0; b < size; b++) slotMap.push({ creg: name, bit: b });
    }
    for (const { name, qubits: qs, params: p = [], memory: mem = [] } of exp.instructions) {
      switch (name) {
        case "id":
          c2 = c2.id(qs[0]);
          break;
        case "h":
          c2 = c2.h(qs[0]);
          break;
        case "x":
          c2 = c2.x(qs[0]);
          break;
        case "y":
          c2 = c2.y(qs[0]);
          break;
        case "z":
          c2 = c2.z(qs[0]);
          break;
        case "s":
          c2 = c2.s(qs[0]);
          break;
        case "sdg":
          c2 = c2.si(qs[0]);
          break;
        case "t":
          c2 = c2.t(qs[0]);
          break;
        case "tdg":
          c2 = c2.ti(qs[0]);
          break;
        case "sx":
          c2 = c2.v(qs[0]);
          break;
        case "sxdg":
          c2 = c2.vi(qs[0]);
          break;
        case "rx":
          c2 = c2.rx(p[0], qs[0]);
          break;
        case "ry":
          c2 = c2.ry(p[0], qs[0]);
          break;
        case "rz":
          c2 = c2.rz(p[0], qs[0]);
          break;
        case "u1":
        case "p":
          c2 = c2.u1(p[0], qs[0]);
          break;
        case "u2":
          c2 = c2.u2(p[0], p[1], qs[0]);
          break;
        case "u3":
          c2 = c2.u3(p[0], p[1], p[2], qs[0]);
          break;
        case "cx":
        case "cnot":
          c2 = c2.cnot(qs[0], qs[1]);
          break;
        case "cy":
          c2 = c2.cy(qs[0], qs[1]);
          break;
        case "cz":
          c2 = c2.cz(qs[0], qs[1]);
          break;
        case "ch":
          c2 = c2.ch(qs[0], qs[1]);
          break;
        case "swap":
          c2 = c2.swap(qs[0], qs[1]);
          break;
        case "iswap":
          c2 = c2.iswap(qs[0], qs[1]);
          break;
        case "crx":
          c2 = c2.crx(p[0], qs[0], qs[1]);
          break;
        case "cry":
          c2 = c2.cry(p[0], qs[0], qs[1]);
          break;
        case "crz":
          c2 = c2.crz(p[0], qs[0], qs[1]);
          break;
        case "cu1":
          c2 = c2.cu1(p[0], qs[0], qs[1]);
          break;
        case "cu2":
          c2 = c2.cu2(p[0], p[1], qs[0], qs[1]);
          break;
        case "cu3":
          c2 = c2.cu3(p[0], p[1], p[2], qs[0], qs[1]);
          break;
        case "rxx":
          c2 = c2.xx(p[0], qs[0], qs[1]);
          break;
        case "ryy":
          c2 = c2.yy(p[0], qs[0], qs[1]);
          break;
        case "rzz":
          c2 = c2.zz(p[0], qs[0], qs[1]);
          break;
        case "ccx":
          c2 = c2.ccx(qs[0], qs[1], qs[2]);
          break;
        case "cswap":
          c2 = c2.cswap(qs[0], qs[1], qs[2]);
          break;
        case "reset":
          c2 = c2.reset(qs[0]);
          break;
        case "barrier":
          c2 = c2.barrier(...qs);
          break;
        case "measure": {
          const slot = slotMap[mem[0]];
          if (slot) c2 = c2.measure(qs[0], slot.creg, slot.bit);
          break;
        }
        case "snapshot":
        case "save_statevector":
          break;
        default:
          throw new TypeError(`fromQobj: unknown instruction '${name}'`);
      }
    }
    return c2;
  }
  // ── Export targets ───────────────────────────────────────────────────────
  /**
   * Emit Python code for Qiskit's `QuantumCircuit` API.
   * Gate coverage: full standard gate set, rx/ry/rz, u1/u2/u3, controlled family,
   * rxx/ryy/rzz/iswap. Throws for gpi/gpi2/ms/xy/srswap/if.
   */
  toQiskit() {
    const lines = [];
    const imports = ["from qiskit import QuantumCircuit"];
    const cregLines = [];
    if (this.#cregs.size) imports.push("from qiskit.circuit import ClassicalRegister");
    imports.push("import math", "");
    lines.push(`qc = QuantumCircuit(${this.qubits})`);
    for (const [name, size] of this.#cregs) {
      cregLines.push(`${name} = ClassicalRegister(${size}, '${name}')`);
      cregLines.push(`qc.add_register(${name})`);
    }
    if (cregLines.length) lines.push(...cregLines, "");
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "cnot":
          lines.push(`qc.cx(${op.control}, ${op.target})`);
          break;
        case "swap":
          lines.push(`qc.swap(${op.a}, ${op.b})`);
          break;
        case "toffoli":
          lines.push(`qc.ccx(${op.c1}, ${op.c2}, ${op.target})`);
          break;
        case "cswap":
          lines.push(`qc.cswap(${op.control}, ${op.a}, ${op.b})`);
          break;
        case "csrswap":
          throw new TypeError("Gate 'csrswap' has no Qiskit representation");
        case "measure":
          lines.push(`qc.measure(${op.q}, ${op.creg}[${op.bit}])`);
          break;
        case "reset":
          lines.push(`qc.reset(${op.q})`);
          break;
        case "if":
          throw new TypeError("if ops cannot be serialized to Qiskit");
        case "two": {
          const n = op.meta?.name;
          if (n === "xx") {
            lines.push(`qc.rxx(${pyAngle(op.meta.params[0])}, ${op.a}, ${op.b})`);
            break;
          }
          if (n === "yy") {
            lines.push(`qc.ryy(${pyAngle(op.meta.params[0])}, ${op.a}, ${op.b})`);
            break;
          }
          if (n === "zz") {
            lines.push(`qc.rzz(${pyAngle(op.meta.params[0])}, ${op.a}, ${op.b})`);
            break;
          }
          if (n === "iswap") {
            lines.push(`qc.iswap(${op.a}, ${op.b})`);
            break;
          }
          throw new TypeError(`Gate '${n ?? "two"}' has no Qiskit representation`);
        }
        case "unitary":
          throw new TypeError("Gate 'unitary' has no Qiskit representation");
        case "single": {
          if (!op.meta) throw new TypeError("Single-qubit op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const q = op.q;
          if (n === "gpi" || n === "gpi2") throw new TypeError(`Gate '${n}' has no Qiskit representation`);
          const angle = () => pyAngle(p[0]);
          switch (n) {
            case "si":
              lines.push(`qc.sdg(${q})`);
              break;
            case "ti":
              lines.push(`qc.tdg(${q})`);
              break;
            case "v":
              lines.push(`qc.sx(${q})`);
              break;
            case "vi":
              lines.push(`qc.sxdg(${q})`);
              break;
            case "r2":
              lines.push(`qc.rz(math.pi/2, ${q})`);
              break;
            case "r4":
              lines.push(`qc.rz(math.pi/4, ${q})`);
              break;
            case "r8":
              lines.push(`qc.rz(math.pi/8, ${q})`);
              break;
            case "rx":
              lines.push(`qc.rx(${angle()}, ${q})`);
              break;
            case "ry":
              lines.push(`qc.ry(${angle()}, ${q})`);
              break;
            case "rz":
              lines.push(`qc.rz(${angle()}, ${q})`);
              break;
            case "vz":
              lines.push(`qc.rz(${angle()}, ${q})`);
              break;
            case "u1":
              lines.push(`qc.u1(${angle()}, ${q})`);
              break;
            case "u2":
              lines.push(`qc.u2(${pyAngle(p[0])}, ${pyAngle(p[1])}, ${q})`);
              break;
            case "u3":
              lines.push(`qc.u3(${pyAngle(p[0])}, ${pyAngle(p[1])}, ${pyAngle(p[2])}, ${q})`);
              break;
            default:
              lines.push(`qc.${n}(${q})`);
          }
          break;
        }
        case "controlled": {
          if (!op.meta) throw new TypeError("Controlled op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const [c2, t] = [op.control, op.target];
          const angle = () => pyAngle(p[0]);
          switch (n) {
            case "cr2":
              lines.push(`qc.crz(math.pi/2, ${c2}, ${t})`);
              break;
            case "cr4":
              lines.push(`qc.crz(math.pi/4, ${c2}, ${t})`);
              break;
            case "cr8":
              lines.push(`qc.crz(math.pi/8, ${c2}, ${t})`);
              break;
            case "cs":
              lines.push(`qc.cu1(math.pi/2, ${c2}, ${t})`);
              break;
            case "ct":
              lines.push(`qc.cu1(math.pi/4, ${c2}, ${t})`);
              break;
            case "csdg":
              lines.push(`qc.cu1(-math.pi/2, ${c2}, ${t})`);
              break;
            case "ctdg":
              lines.push(`qc.cu1(-math.pi/4, ${c2}, ${t})`);
              break;
            case "crx":
              lines.push(`qc.crx(${angle()}, ${c2}, ${t})`);
              break;
            case "cry":
              lines.push(`qc.cry(${angle()}, ${c2}, ${t})`);
              break;
            case "crz":
              lines.push(`qc.crz(${angle()}, ${c2}, ${t})`);
              break;
            case "cu1":
              lines.push(`qc.cu1(${angle()}, ${c2}, ${t})`);
              break;
            case "cu2":
              lines.push(`qc.cu2(${pyAngle(p[0])}, ${pyAngle(p[1])}, ${c2}, ${t})`);
              break;
            case "cu3":
              lines.push(`qc.cu3(${pyAngle(p[0])}, ${pyAngle(p[1])}, ${pyAngle(p[2])}, ${c2}, ${t})`);
              break;
            default:
              lines.push(`qc.${n}(${c2}, ${t})`);
          }
          break;
        }
        case "barrier": {
          const qs = op.qubits;
          lines.push(`qc.barrier(${qs.join(", ")})`);
          break;
        }
        default: {
          const _exhaustive = op;
        }
      }
    }
    return [...imports, ...lines].join("\n");
  }
  /** Build the list of `    cirq.*` op strings shared by toCirq() and toTFQ(). */
  #cirqOps() {
    const ops = [];
    const go = (gate, qs) => `    ${gate}(${qs.map((q) => `q[${q}]`).join(", ")}),`;
    const rads = (r) => `rads=${pyAngle(r)}`;
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "cnot":
          ops.push(go("cirq.CNOT", [op.control, op.target]));
          break;
        case "swap":
          ops.push(go("cirq.SWAP", [op.a, op.b]));
          break;
        case "toffoli":
          ops.push(go("cirq.CCNOT", [op.c1, op.c2, op.target]));
          break;
        case "cswap":
          ops.push(go("cirq.CSWAP", [op.control, op.a, op.b]));
          break;
        case "csrswap":
          throw new TypeError("Gate 'csrswap' has no Cirq representation");
        case "measure":
          throw new TypeError("measure ops cannot be serialized to Cirq via toCirq(); use cirq.measure() manually");
        case "reset":
          throw new TypeError("reset ops cannot be serialized to Cirq via toCirq()");
        case "if":
          throw new TypeError("if ops cannot be serialized to Cirq");
        case "two":
          throw new TypeError(`Gate '${op.meta?.name ?? "two"}' has no Cirq representation`);
        case "unitary":
          throw new TypeError("Gate 'unitary' has no Cirq representation");
        case "single": {
          if (!op.meta) throw new TypeError("Single-qubit op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const q = op.q;
          switch (n) {
            case "id":
              ops.push(go("cirq.I", [q]));
              break;
            case "h":
              ops.push(go("cirq.H", [q]));
              break;
            case "x":
              ops.push(go("cirq.X", [q]));
              break;
            case "y":
              ops.push(go("cirq.Y", [q]));
              break;
            case "z":
              ops.push(go("cirq.Z", [q]));
              break;
            case "s":
              ops.push(go("cirq.S", [q]));
              break;
            case "si":
              ops.push(`    cirq.ZPowGate(exponent=-0.5)(q[${q}]),`);
              break;
            case "t":
              ops.push(go("cirq.T", [q]));
              break;
            case "ti":
              ops.push(`    cirq.ZPowGate(exponent=-0.25)(q[${q}]),`);
              break;
            case "v":
              ops.push(`    cirq.X**0.5(q[${q}]),`);
              break;
            case "vi":
              ops.push(`    (cirq.X**-0.5)(q[${q}]),`);
              break;
            case "r2":
              ops.push(`    cirq.rz(${rads(Math.PI / 2)})(q[${q}]),`);
              break;
            case "r4":
              ops.push(`    cirq.rz(${rads(Math.PI / 4)})(q[${q}]),`);
              break;
            case "r8":
              ops.push(`    cirq.rz(${rads(Math.PI / 8)})(q[${q}]),`);
              break;
            case "rx":
              ops.push(`    cirq.rx(${rads(p[0])})(q[${q}]),`);
              break;
            case "ry":
              ops.push(`    cirq.ry(${rads(p[0])})(q[${q}]),`);
              break;
            case "rz":
              ops.push(`    cirq.rz(${rads(p[0])})(q[${q}]),`);
              break;
            case "vz":
              ops.push(`    cirq.rz(${rads(p[0])})(q[${q}]),`);
              break;
            case "u1":
              ops.push(`    cirq.ZPowGate(exponent=${pyAngle(p[0] / Math.PI)})(q[${q}]),`);
              break;
            case "u3":
              throw new TypeError("U3 has no direct Cirq equivalent; use cirq.MatrixGate(np.array([...])) with the explicit unitary");
            case "gpi":
            case "gpi2":
              throw new TypeError(`Gate '${n}' has no Cirq representation`);
            default:
              throw new TypeError(`Gate '${n}' has no Cirq representation`);
          }
          break;
        }
        case "controlled": {
          if (!op.meta) throw new TypeError("Controlled op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const [c2, t] = [op.control, op.target];
          switch (n) {
            case "cy":
              ops.push(`    cirq.Y.controlled()(q[${c2}], q[${t}]),`);
              break;
            case "cz":
              ops.push(go("cirq.CZ", [c2, t]));
              break;
            case "ch":
              ops.push(`    cirq.H.controlled()(q[${c2}], q[${t}]),`);
              break;
            case "cr2":
              ops.push(`    cirq.rz(${rads(Math.PI / 2)}).controlled()(q[${c2}], q[${t}]),`);
              break;
            case "cr4":
              ops.push(`    cirq.rz(${rads(Math.PI / 4)}).controlled()(q[${c2}], q[${t}]),`);
              break;
            case "cr8":
              ops.push(`    cirq.rz(${rads(Math.PI / 8)}).controlled()(q[${c2}], q[${t}]),`);
              break;
            case "crx":
              ops.push(`    cirq.rx(${rads(p[0])}).controlled()(q[${c2}], q[${t}]),`);
              break;
            case "cry":
              ops.push(`    cirq.ry(${rads(p[0])}).controlled()(q[${c2}], q[${t}]),`);
              break;
            case "crz":
              ops.push(`    cirq.rz(${rads(p[0])}).controlled()(q[${c2}], q[${t}]),`);
              break;
            case "cu1":
              ops.push(`    cirq.ZPowGate(exponent=${pyAngle(p[0] / Math.PI)}).controlled()(q[${c2}], q[${t}]),`);
              break;
            case "cs":
              ops.push(`    cirq.ZPowGate(exponent=0.5).controlled()(q[${c2}], q[${t}]),`);
              break;
            case "ct":
              ops.push(`    cirq.ZPowGate(exponent=0.25).controlled()(q[${c2}], q[${t}]),`);
              break;
            case "csdg":
              ops.push(`    cirq.ZPowGate(exponent=-0.5).controlled()(q[${c2}], q[${t}]),`);
              break;
            case "ctdg":
              ops.push(`    cirq.ZPowGate(exponent=-0.25).controlled()(q[${c2}], q[${t}]),`);
              break;
            default:
              throw new TypeError(`Gate '${n}' has no Cirq representation`);
          }
          break;
        }
        case "barrier":
          break;
        default: {
          const _exhaustive = op;
        }
      }
    }
    return ops;
  }
  /**
   * Emit Python code for Google Cirq.
   * Gate coverage: H/X/Y/Z/S/T, rx/ry/rz, r2/r4/r8, u1/u3,
   * CNOT/CZ/CY/CH/swap/CCNOT/CSWAP, crx/cry/crz/cu1/cu3.
   * Throws for gpi/gpi2/ms/xx/yy/zz/xy/iswap/srswap/if.
   */
  toCirq() {
    const ops = this.#cirqOps();
    const body = ops.length ? ops.join("\n") : "    # empty circuit";
    return [
      "import cirq",
      "import math",
      "",
      `q = cirq.LineQubit.range(${this.qubits})`,
      "circuit = cirq.Circuit([",
      body,
      "])"
    ].join("\n");
  }
  /**
   * Emit Python code for TensorFlow Quantum (TFQ).
   *
   * TFQ wraps Cirq circuits; qubits must be `cirq.GridQubit` instances.
   * The output includes the `tfq.convert_to_tensor` call needed to feed
   * the circuit into a TFQ layer.
   *
   * ```python
   * tensor = tfq.convert_to_tensor([circuit])
   * ```
   *
   * Same gate coverage and restrictions as `toCirq()`.
   */
  toTFQ() {
    const ops = this.#cirqOps();
    const body = ops.length ? ops.join("\n") : "    # empty circuit";
    return [
      "import cirq",
      "import math",
      "import tensorflow_quantum as tfq",
      "",
      `q = [cirq.GridQubit(0, i) for i in range(${this.qubits})]`,
      "circuit = cirq.Circuit([",
      body,
      "])",
      "tensor = tfq.convert_to_tensor([circuit])"
    ].join("\n");
  }
  /**
   * Emit a Q# operation for Microsoft Azure Quantum.
   * Gate coverage: H/X/Y/Z/S/T and adjoints, Rx/Ry/Rz, CNOT/CZ/SWAP/CCNOT,
   * controlled rotations via Controlled Rx/Ry/Rz. Throws for gpi/gpi2/ms/two-qubit interaction gates/if.
   */
  toQSharp() {
    const pi = "PI()";
    const a = (r) => fmtAngle(r, pi).replace(/(\d+)\*PI/, "$1.0*PI").replace(/PI\(?\)?\/(\d+)/, `PI()/${`$1`.padStart(1)}`);
    const qsharpAngle = (r) => {
      if (Math.abs(r) < 1e-14) return "0.0";
      const f = r / Math.PI;
      for (const d of [1, 2, 3, 4, 6, 8, 12, 16]) {
        for (let n = -16; n <= 16; n++) {
          if (n === 0) continue;
          if (Math.abs(f - n / d) < 1e-12) {
            const sign = n < 0 ? "-" : "";
            const abs = Math.abs(n);
            if (d === 1) return abs === 1 ? `${sign}PI()` : `${sign}${abs}.0*PI()`;
            return abs === 1 ? `${sign}PI()/${d}.0` : `${sign}${abs}.0*PI()/${d}.0`;
          }
        }
      }
      return r.toFixed(15);
    };
    const body = [];
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "cnot":
          body.push(`        CNOT(q[${op.control}], q[${op.target}]);`);
          break;
        case "swap":
          body.push(`        SWAP(q[${op.a}], q[${op.b}]);`);
          break;
        case "toffoli":
          body.push(`        CCNOT(q[${op.c1}], q[${op.c2}], q[${op.target}]);`);
          break;
        case "cswap":
          body.push(`        Controlled SWAP([q[${op.control}]], (q[${op.a}], q[${op.b}]));`);
          break;
        case "csrswap":
          throw new TypeError("Gate 'csrswap' has no Q# representation");
        case "measure":
          body.push(`        set ${op.creg}w${op.bit} = M(q[${op.q}]) == One;`);
          break;
        case "reset":
          body.push(`        Reset(q[${op.q}]);`);
          break;
        case "if":
          throw new TypeError("if ops cannot be serialized to Q#");
        case "two":
          throw new TypeError(`Gate '${op.meta?.name ?? "two"}' has no Q# representation`);
        case "unitary":
          throw new TypeError("Gate 'unitary' has no Q# representation");
        case "single": {
          if (!op.meta) throw new TypeError("Single-qubit op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const q = op.q;
          const ang = () => qsharpAngle(p[0]);
          switch (n) {
            case "id":
              body.push(`        I(q[${q}]);`);
              break;
            case "h":
              body.push(`        H(q[${q}]);`);
              break;
            case "x":
              body.push(`        X(q[${q}]);`);
              break;
            case "y":
              body.push(`        Y(q[${q}]);`);
              break;
            case "z":
              body.push(`        Z(q[${q}]);`);
              break;
            case "s":
              body.push(`        S(q[${q}]);`);
              break;
            case "si":
              body.push(`        Adjoint S(q[${q}]);`);
              break;
            case "t":
              body.push(`        T(q[${q}]);`);
              break;
            case "ti":
              body.push(`        Adjoint T(q[${q}]);`);
              break;
            case "v":
              body.push(`        Rx(PI()/2.0, q[${q}]);`);
              break;
            case "vi":
              body.push(`        Rx(-PI()/2.0, q[${q}]);`);
              break;
            case "r2":
              body.push(`        Rz(PI()/2.0, q[${q}]);`);
              break;
            case "r4":
              body.push(`        Rz(PI()/4.0, q[${q}]);`);
              break;
            case "r8":
              body.push(`        Rz(PI()/8.0, q[${q}]);`);
              break;
            case "rx":
              body.push(`        Rx(${ang()}, q[${q}]);`);
              break;
            case "ry":
              body.push(`        Ry(${ang()}, q[${q}]);`);
              break;
            case "rz":
              body.push(`        Rz(${ang()}, q[${q}]);`);
              break;
            case "vz":
              body.push(`        Rz(${ang()}, q[${q}]);`);
              break;
            case "u1":
              body.push(`        Rz(${ang()}, q[${q}]);`);
              break;
            case "gpi":
            case "gpi2":
              throw new TypeError(`Gate '${n}' has no Q# representation`);
            default:
              throw new TypeError(`Gate '${n}' has no Q# representation`);
          }
          break;
        }
        case "controlled": {
          if (!op.meta) throw new TypeError("Controlled op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const [c2, t] = [op.control, op.target];
          const ang = () => qsharpAngle(p[0]);
          switch (n) {
            case "cy":
              body.push(`        Controlled Y([q[${c2}]], q[${t}]);`);
              break;
            case "cz":
              body.push(`        CZ(q[${c2}], q[${t}]);`);
              break;
            case "ch":
              body.push(`        Controlled H([q[${c2}]], q[${t}]);`);
              break;
            case "crx":
              body.push(`        Controlled Rx([q[${c2}]], (${ang()}, q[${t}]));`);
              break;
            case "cry":
              body.push(`        Controlled Ry([q[${c2}]], (${ang()}, q[${t}]));`);
              break;
            case "crz":
              body.push(`        Controlled Rz([q[${c2}]], (${ang()}, q[${t}]));`);
              break;
            case "cr2":
              body.push(`        Controlled Rz([q[${c2}]], (PI()/2.0, q[${t}]));`);
              break;
            case "cr4":
              body.push(`        Controlled Rz([q[${c2}]], (PI()/4.0, q[${t}]));`);
              break;
            case "cr8":
              body.push(`        Controlled Rz([q[${c2}]], (PI()/8.0, q[${t}]));`);
              break;
            case "cs":
              body.push(`        Controlled S([q[${c2}]], q[${t}]);`);
              break;
            case "csdg":
              body.push(`        Controlled Adjoint S([q[${c2}]], q[${t}]);`);
              break;
            case "ct":
              body.push(`        Controlled T([q[${c2}]], q[${t}]);`);
              break;
            case "ctdg":
              body.push(`        Controlled Adjoint T([q[${c2}]], q[${t}]);`);
              break;
            case "cu1":
              body.push(`        Controlled Rz([q[${c2}]], (${ang()}, q[${t}]));`);
              break;
            default:
              throw new TypeError(`Gate '${n}' has no Q# representation`);
          }
          break;
        }
        case "barrier":
          break;
        default: {
          const _exhaustive = op;
        }
      }
    }
    const qregs = `        use q = Qubit[${this.qubits}];`;
    const reset = `        ResetAll(q);`;
    return [
      "namespace KetCircuit {",
      "    open Microsoft.Quantum.Intrinsic;",
      "    open Microsoft.Quantum.Canon;",
      "    open Microsoft.Quantum.Math;",
      "",
      "    operation Run() : Unit {",
      qregs,
      ...body.length ? ["", ...body, ""] : [],
      reset,
      "    }",
      "}"
    ].join("\n");
  }
  /**
   * Emit Python code for Rigetti's pyQuil.
   * Gate coverage: H/X/Y/Z/S/T (Sdg/Tdg via DAGGER), RX/RY/RZ,
   * CNOT/CZ/SWAP/CCNOT/CSWAP/ISWAP. Throws for controlled-rotation gates, U-gates, gpi/gpi2/ms/if.
   */
  toPyQuil() {
    const used = /* @__PURE__ */ new Set();
    const body = [];
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "cnot":
          used.add("CNOT");
          body.push(`p += CNOT(${op.control}, ${op.target})`);
          break;
        case "swap":
          used.add("SWAP");
          body.push(`p += SWAP(${op.a}, ${op.b})`);
          break;
        case "toffoli":
          used.add("CCNOT");
          body.push(`p += CCNOT(${op.c1}, ${op.c2}, ${op.target})`);
          break;
        case "cswap":
          used.add("CSWAP");
          body.push(`p += CSWAP(${op.control}, ${op.a}, ${op.b})`);
          break;
        case "csrswap":
          throw new TypeError("Gate 'csrswap' has no pyQuil representation");
        case "measure":
          used.add("MEASURE");
          body.push(`p += MEASURE(${op.q}, ro[${op.bit}])`);
          break;
        case "reset":
          body.push(`p += RESET(${op.q})`);
          break;
        case "if":
          throw new TypeError("if ops cannot be serialized to pyQuil");
        case "two": {
          const n = op.meta?.name;
          if (n === "iswap") {
            used.add("ISWAP");
            body.push(`p += ISWAP(${op.a}, ${op.b})`);
            break;
          }
          throw new TypeError(`Gate '${n ?? "two"}' has no pyQuil representation`);
        }
        case "unitary":
          throw new TypeError("Gate 'unitary' has no pyQuil representation");
        case "single": {
          if (!op.meta) throw new TypeError("Single-qubit op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const q = op.q;
          const ang = (i = 0) => pyAngle(p[i]);
          switch (n) {
            case "id":
              used.add("I");
              body.push(`p += I(${q})`);
              break;
            case "h":
              used.add("H");
              body.push(`p += H(${q})`);
              break;
            case "x":
              used.add("X");
              body.push(`p += X(${q})`);
              break;
            case "y":
              used.add("Y");
              body.push(`p += Y(${q})`);
              break;
            case "z":
              used.add("Z");
              body.push(`p += Z(${q})`);
              break;
            case "s":
              used.add("S");
              body.push(`p += S(${q})`);
              break;
            case "si":
              used.add("S");
              used.add("DAGGER");
              body.push(`p += DAGGER(S)(${q})`);
              break;
            case "t":
              used.add("T");
              body.push(`p += T(${q})`);
              break;
            case "ti":
              used.add("T");
              used.add("DAGGER");
              body.push(`p += DAGGER(T)(${q})`);
              break;
            case "v":
              used.add("RX");
              body.push(`p += RX(math.pi/2, ${q})`);
              break;
            case "vi":
              used.add("RX");
              body.push(`p += RX(-math.pi/2, ${q})`);
              break;
            case "r2":
              used.add("RZ");
              body.push(`p += RZ(math.pi/2, ${q})`);
              break;
            case "r4":
              used.add("RZ");
              body.push(`p += RZ(math.pi/4, ${q})`);
              break;
            case "r8":
              used.add("RZ");
              body.push(`p += RZ(math.pi/8, ${q})`);
              break;
            case "rx":
              used.add("RX");
              body.push(`p += RX(${ang()}, ${q})`);
              break;
            case "ry":
              used.add("RY");
              body.push(`p += RY(${ang()}, ${q})`);
              break;
            case "rz":
              used.add("RZ");
              body.push(`p += RZ(${ang()}, ${q})`);
              break;
            case "vz":
              used.add("RZ");
              body.push(`p += RZ(${ang()}, ${q})`);
              break;
            case "gpi":
            case "gpi2":
              throw new TypeError(`Gate '${n}' has no pyQuil representation`);
            default:
              throw new TypeError(`Gate '${n}' has no pyQuil representation`);
          }
          break;
        }
        case "controlled":
          throw new TypeError(`Gate '${op.meta?.name ?? "controlled"}' has no standard pyQuil representation`);
        case "barrier":
          break;
        default: {
          const _exhaustive = op;
        }
      }
    }
    const gateImports = [...used].filter((g) => g !== "RESET" && g !== "MEASURE").sort();
    const extras = [];
    if (used.has("MEASURE")) extras.push("from pyquil.quilbase import MemoryReference");
    if (used.has("RESET")) extras.push("# Reset: use p += RESET() or p.reset() as appropriate");
    return [
      "from pyquil import Program",
      gateImports.length ? `from pyquil.gates import ${gateImports.join(", ")}` : "",
      ...extras,
      "import math",
      "",
      "p = Program()",
      ...body
    ].filter((l, i) => l !== "" || i > 3).join("\n");
  }
  /**
   * Emit a Quil (Quantum Instruction Language) program for Rigetti hardware.
   *
   * Gate coverage: I/H/X/Y/Z/S/T and daggers, RX/RY/RZ, PHASE,
   * CNOT/CZ/SWAP/ISWAP/CCNOT/CSWAP, controlled family via CONTROLLED/CPHASE.
   * Throws for gates with no Quil representation: gpi/gpi2/ms/xx/yy/zz/xy/srswap/u2/u3/cu2/cu3/if.
   */
  toQuil() {
    const lines = [];
    const ang = (r) => fmtAngle(r, "pi");
    for (const [name, size] of this.#cregs) lines.push(`DECLARE ${name} BIT[${size}]`);
    if (this.#cregs.size) lines.push("");
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "cnot":
          lines.push(`CNOT ${op.control} ${op.target}`);
          break;
        case "swap":
          lines.push(`SWAP ${op.a} ${op.b}`);
          break;
        case "toffoli":
          lines.push(`CCNOT ${op.c1} ${op.c2} ${op.target}`);
          break;
        case "cswap":
          lines.push(`CSWAP ${op.control} ${op.a} ${op.b}`);
          break;
        case "csrswap":
          throw new TypeError("Gate 'csrswap' has no Quil representation");
        case "measure":
          lines.push(`MEASURE ${op.q} ${op.creg}[${op.bit}]`);
          break;
        case "reset":
          lines.push(`RESET ${op.q}`);
          break;
        case "if":
          throw new TypeError("if ops cannot be serialized to Quil");
        case "two": {
          const n = op.meta?.name;
          if (n === "iswap") {
            lines.push(`ISWAP ${op.a} ${op.b}`);
            break;
          }
          throw new TypeError(`Gate '${n ?? "two"}' has no Quil representation`);
        }
        case "unitary":
          throw new TypeError("Gate 'unitary' has no Quil representation");
        case "single": {
          if (!op.meta) throw new TypeError("Single-qubit op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const q = op.q;
          switch (n) {
            case "id":
              lines.push(`I ${q}`);
              break;
            case "h":
              lines.push(`H ${q}`);
              break;
            case "x":
              lines.push(`X ${q}`);
              break;
            case "y":
              lines.push(`Y ${q}`);
              break;
            case "z":
              lines.push(`Z ${q}`);
              break;
            case "s":
              lines.push(`S ${q}`);
              break;
            case "si":
              lines.push(`DAGGER S ${q}`);
              break;
            case "t":
              lines.push(`T ${q}`);
              break;
            case "ti":
              lines.push(`DAGGER T ${q}`);
              break;
            case "v":
              lines.push(`RX(pi/2) ${q}`);
              break;
            case "vi":
              lines.push(`RX(-pi/2) ${q}`);
              break;
            case "r2":
              lines.push(`RZ(pi/2) ${q}`);
              break;
            case "r4":
              lines.push(`RZ(pi/4) ${q}`);
              break;
            case "r8":
              lines.push(`RZ(pi/8) ${q}`);
              break;
            case "rx":
              lines.push(`RX(${ang(p[0])}) ${q}`);
              break;
            case "ry":
              lines.push(`RY(${ang(p[0])}) ${q}`);
              break;
            case "rz":
              lines.push(`RZ(${ang(p[0])}) ${q}`);
              break;
            case "vz":
              lines.push(`RZ(${ang(p[0])}) ${q}`);
              break;
            case "u1":
              lines.push(`PHASE(${ang(p[0])}) ${q}`);
              break;
            case "gpi":
            case "gpi2":
              throw new TypeError(`Gate '${n}' has no Quil representation`);
            default:
              throw new TypeError(`Gate '${n}' has no Quil representation`);
          }
          break;
        }
        case "controlled": {
          if (!op.meta) throw new TypeError("Controlled op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const [c2, t] = [op.control, op.target];
          switch (n) {
            case "cy":
              lines.push(`CONTROLLED Y ${c2} ${t}`);
              break;
            case "cz":
              lines.push(`CZ ${c2} ${t}`);
              break;
            case "ch":
              lines.push(`CONTROLLED H ${c2} ${t}`);
              break;
            case "crx":
              lines.push(`CONTROLLED RX(${ang(p[0])}) ${c2} ${t}`);
              break;
            case "cry":
              lines.push(`CONTROLLED RY(${ang(p[0])}) ${c2} ${t}`);
              break;
            case "crz":
              lines.push(`CONTROLLED RZ(${ang(p[0])}) ${c2} ${t}`);
              break;
            case "cu1":
              lines.push(`CPHASE(${ang(p[0])}) ${c2} ${t}`);
              break;
            case "cs":
              lines.push(`CPHASE(pi/2) ${c2} ${t}`);
              break;
            case "ct":
              lines.push(`CPHASE(pi/4) ${c2} ${t}`);
              break;
            case "csdg":
              lines.push(`CPHASE(-pi/2) ${c2} ${t}`);
              break;
            case "ctdg":
              lines.push(`CPHASE(-pi/4) ${c2} ${t}`);
              break;
            case "cr2":
              lines.push(`CPHASE(pi/2) ${c2} ${t}`);
              break;
            case "cr4":
              lines.push(`CPHASE(pi/4) ${c2} ${t}`);
              break;
            case "cr8":
              lines.push(`CPHASE(pi/8) ${c2} ${t}`);
              break;
            default:
              throw new TypeError(`Gate '${n}' has no Quil representation`);
          }
          break;
        }
        case "barrier":
          break;
        default: {
          const _exhaustive = op;
        }
      }
    }
    return lines.join("\n");
  }
  /**
   * Emit a `quantikz` LaTeX environment for the circuit.
   *
   * The output is a self-contained `\begin{quantikz}...\end{quantikz}` block
   * using the `tikz-quantikz` package.  Paste it into any LaTeX document that
   * loads `\usepackage{quantikz}` (and `\usepackage{amsmath}` for `\ket{}`).
   *
   * Gate coverage: all single-qubit gates, CNOT, SWAP, controlled family,
   * Toffoli, Fredkin, two-qubit interaction gates, measure, reset.
   * `if` ops are silently skipped (no standard quantikz representation).
   *
   * @example
   * console.log(new Circuit(2).h(0).cnot(0, 1).toLatex())
   * // \begin{quantikz}
   * // \lstick{$q_{0}$} & \gate{H} & \ctrl{1} & \qw \\
   * // \lstick{$q_{1}$} & \qw & \targ{} & \qw
   * // \end{quantikz}
   */
  toLatex() {
    const n = this.qubits;
    if (n === 0) return "\\begin{quantikz}\n\\end{quantikz}";
    const ops = flattenOps(this.#ops).filter((op) => op.kind !== "if");
    const colOf = new Array(n).fill(0);
    const placed = [];
    for (const op of ops) {
      const qs = opQubits(op);
      if (qs.length === 0) continue;
      const minQ = Math.min(...qs), maxQ = Math.max(...qs);
      let col = 0;
      for (let q = minQ; q <= maxQ; q++) col = Math.max(col, colOf[q]);
      for (let q = minQ; q <= maxQ; q++) colOf[q] = col + 1;
      placed.push({ op, col });
    }
    const numCols = Math.max(0, ...colOf);
    const cell = Array.from(
      { length: n },
      () => new Array(numCols).fill("\\qw")
    );
    for (const { op, col } of placed) {
      const qs = opQubits(op);
      const minQ = Math.min(...qs), maxQ = Math.max(...qs);
      switch (op.kind) {
        case "single":
          cell[op.q][col] = `\\gate{${latexSingleLabel(op)}}`;
          break;
        case "cnot":
          cell[op.control][col] = `\\ctrl{${op.target - op.control}}`;
          cell[op.target][col] = "\\targ{}";
          break;
        case "swap":
          cell[op.a][col] = `\\swap{${op.b - op.a}}`;
          cell[op.b][col] = `\\swap{${op.a - op.b}}`;
          break;
        case "controlled":
          cell[op.control][col] = `\\ctrl{${op.target - op.control}}`;
          cell[op.target][col] = `\\gate{${latexCtrlTargetLabel(op)}}`;
          break;
        case "toffoli":
          cell[op.c1][col] = `\\ctrl{${op.target - op.c1}}`;
          cell[op.c2][col] = `\\ctrl{${op.target - op.c2}}`;
          cell[op.target][col] = "\\targ{}";
          break;
        case "cswap":
          cell[op.control][col] = `\\ctrl{${Math.min(op.a, op.b) - op.control}}`;
          cell[Math.min(op.a, op.b)][col] = `\\swap{${Math.max(op.a, op.b) - Math.min(op.a, op.b)}}`;
          cell[Math.max(op.a, op.b)][col] = `\\swap{${Math.min(op.a, op.b) - Math.max(op.a, op.b)}}`;
          break;
        case "csrswap": {
          const [tA, tB] = [Math.min(op.a, op.b), Math.max(op.a, op.b)];
          const span = tB - tA + 1;
          cell[op.control][col] = `\\ctrl{${tA - op.control}}`;
          cell[tA][col] = `\\gate[${span}]{\\sqrt{\\text{iSWAP}}}`;
          for (let q = tA + 1; q <= tB; q++) cell[q][col] = "";
          break;
        }
        case "two": {
          const span = maxQ - minQ + 1;
          cell[minQ][col] = `\\gate[${span}]{${latexTwoLabel(op)}}`;
          for (let q = minQ + 1; q <= maxQ; q++) cell[q][col] = "";
          break;
        }
        case "unitary": {
          const span = maxQ - minQ + 1;
          cell[minQ][col] = span === 1 ? `\\gate{U}` : `\\gate[${span}]{U}`;
          for (let q = minQ + 1; q <= maxQ; q++) cell[q][col] = "";
          break;
        }
        case "measure":
          cell[op.q][col] = "\\meter{}";
          break;
        case "reset":
          cell[op.q][col] = "\\gate{\\ket{0}}";
          break;
        case "barrier":
          break;
        case "if":
          break;
        default: {
          const _exhaustive = op;
        }
      }
    }
    const rows = Array.from({ length: n }, (_, q) => {
      const cells = cell[q].filter((c2) => c2 !== "").join(" & ");
      return `\\lstick{$q_{${q}}$} & ${cells} & \\qw`;
    });
    return `\\begin{quantikz}
${rows.join(" \\\\\n")}
\\end{quantikz}`;
  }
  /**
   * Emit Python code for Amazon Braket's `Circuit` API.
   *
   * Gate coverage: full single-qubit set, rx/ry/rz, phaseshift (u1), xx/yy/zz/xy,
   * cnot/cy/cz/swap/iswap, ccnot/cswap, controlled family via control= kwarg.
   * Throws for gates with no Braket representation: gpi/gpi2/ms/srswap/u2/u3/cu2/cu3/measure/reset/if.
   */
  toBraket() {
    const lines = [];
    const a = (r) => pyAngle(r);
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "cnot":
          lines.push(`circ.cnot(${op.control}, ${op.target})`);
          break;
        case "swap":
          lines.push(`circ.swap(${op.a}, ${op.b})`);
          break;
        case "toffoli":
          lines.push(`circ.ccnot(${op.c1}, ${op.c2}, ${op.target})`);
          break;
        case "cswap":
          lines.push(`circ.cswap(${op.control}, ${op.a}, ${op.b})`);
          break;
        case "csrswap":
          throw new TypeError("Gate 'csrswap' has no Braket representation");
        case "measure":
          throw new TypeError("measure ops cannot be serialized to Braket via toBraket()");
        case "reset":
          throw new TypeError("reset ops cannot be serialized to Braket via toBraket()");
        case "if":
          throw new TypeError("if ops cannot be serialized to Braket");
        case "two": {
          const n = op.meta?.name;
          const [qa, qb] = [op.a, op.b];
          if (n === "xx") {
            lines.push(`circ.xx(${qa}, ${qb}, ${a(op.meta.params[0])})`);
            break;
          }
          if (n === "yy") {
            lines.push(`circ.yy(${qa}, ${qb}, ${a(op.meta.params[0])})`);
            break;
          }
          if (n === "zz") {
            lines.push(`circ.zz(${qa}, ${qb}, ${a(op.meta.params[0])})`);
            break;
          }
          if (n === "xy") {
            lines.push(`circ.xy(${qa}, ${qb}, ${a(op.meta.params[0])})`);
            break;
          }
          if (n === "iswap") {
            lines.push(`circ.iswap(${qa}, ${qb})`);
            break;
          }
          throw new TypeError(`Gate '${n ?? "two"}' has no Braket representation`);
        }
        case "unitary":
          throw new TypeError("Gate 'unitary' has no Braket representation");
        case "single": {
          if (!op.meta) throw new TypeError("Single-qubit op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const q = op.q;
          switch (n) {
            case "id":
              lines.push(`circ.i(${q})`);
              break;
            case "h":
              lines.push(`circ.h(${q})`);
              break;
            case "x":
              lines.push(`circ.x(${q})`);
              break;
            case "y":
              lines.push(`circ.y(${q})`);
              break;
            case "z":
              lines.push(`circ.z(${q})`);
              break;
            case "s":
              lines.push(`circ.s(${q})`);
              break;
            case "si":
              lines.push(`circ.si(${q})`);
              break;
            case "t":
              lines.push(`circ.t(${q})`);
              break;
            case "ti":
              lines.push(`circ.ti(${q})`);
              break;
            case "v":
              lines.push(`circ.v(${q})`);
              break;
            case "vi":
              lines.push(`circ.vi(${q})`);
              break;
            case "r2":
              lines.push(`circ.rz(${q}, math.pi/2)`);
              break;
            case "r4":
              lines.push(`circ.rz(${q}, math.pi/4)`);
              break;
            case "r8":
              lines.push(`circ.rz(${q}, math.pi/8)`);
              break;
            case "rx":
              lines.push(`circ.rx(${q}, ${a(p[0])})`);
              break;
            case "ry":
              lines.push(`circ.ry(${q}, ${a(p[0])})`);
              break;
            case "rz":
              lines.push(`circ.rz(${q}, ${a(p[0])})`);
              break;
            case "vz":
              lines.push(`circ.rz(${q}, ${a(p[0])})`);
              break;
            case "u1":
              lines.push(`circ.phaseshift(${q}, ${a(p[0])})`);
              break;
            case "gpi":
            case "gpi2":
              throw new TypeError(`Gate '${n}' has no Braket representation`);
            default:
              throw new TypeError(`Gate '${n}' has no Braket representation`);
          }
          break;
        }
        case "controlled": {
          if (!op.meta) throw new TypeError("Controlled op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const [c2, t] = [op.control, op.target];
          switch (n) {
            case "cy":
              lines.push(`circ.cy(${c2}, ${t})`);
              break;
            case "cz":
              lines.push(`circ.cz(${c2}, ${t})`);
              break;
            case "ch":
              lines.push(`circ.h(${t}, control=${c2})`);
              break;
            case "crx":
              lines.push(`circ.rx(${t}, ${a(p[0])}, control=${c2})`);
              break;
            case "cry":
              lines.push(`circ.ry(${t}, ${a(p[0])}, control=${c2})`);
              break;
            case "crz":
              lines.push(`circ.rz(${t}, ${a(p[0])}, control=${c2})`);
              break;
            case "cr2":
              lines.push(`circ.rz(${t}, math.pi/2, control=${c2})`);
              break;
            case "cr4":
              lines.push(`circ.rz(${t}, math.pi/4, control=${c2})`);
              break;
            case "cr8":
              lines.push(`circ.rz(${t}, math.pi/8, control=${c2})`);
              break;
            case "cu1":
              lines.push(`circ.phaseshift(${t}, ${a(p[0])}, control=${c2})`);
              break;
            case "cs":
              lines.push(`circ.phaseshift(${t}, math.pi/2, control=${c2})`);
              break;
            case "ct":
              lines.push(`circ.phaseshift(${t}, math.pi/4, control=${c2})`);
              break;
            case "csdg":
              lines.push(`circ.phaseshift(${t}, -math.pi/2, control=${c2})`);
              break;
            case "ctdg":
              lines.push(`circ.phaseshift(${t}, -math.pi/4, control=${c2})`);
              break;
            default:
              throw new TypeError(`Gate '${n}' has no Braket representation`);
          }
          break;
        }
        case "barrier":
          break;
        default: {
          const _exhaustive = op;
        }
      }
    }
    return [
      "import math",
      "from braket.circuits import Circuit",
      "",
      "circ = Circuit()",
      ...lines
    ].join("\n");
  }
  /**
   * Emit a CUDA Quantum (cudaq) Python kernel.
   *
   * ```python
   * import math
   * import cudaq
   *
   * kernel = cudaq.make_kernel()
   * q = kernel.qalloc(2)
   * kernel.h(q[0])
   * kernel.cx(q[0], q[1])
   * ```
   *
   * Limitations: classical ops (measure/reset/if), IonQ-native gates (gpi/gpi2/ms),
   * √X variants (v/vi), and interaction gates (xx/yy/zz/xy/iswap/srswap) have no
   * direct CudaQ equivalent and will throw.
   */
  toCudaQ() {
    const lines = [];
    const a = (r) => pyAngle(r);
    const qi = (i) => `q[${i}]`;
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "cnot":
          lines.push(`kernel.cx(${qi(op.control)}, ${qi(op.target)})`);
          break;
        case "swap":
          lines.push(`kernel.swap(${qi(op.a)}, ${qi(op.b)})`);
          break;
        case "toffoli":
          lines.push(`kernel.ccx(${qi(op.c1)}, ${qi(op.c2)}, ${qi(op.target)})`);
          break;
        case "cswap":
          lines.push(`kernel.cswap(${qi(op.control)}, ${qi(op.a)}, ${qi(op.b)})`);
          break;
        case "csrswap":
          throw new TypeError("Gate 'csrswap' has no CudaQ representation");
        case "measure":
          throw new TypeError("measure ops cannot be serialized to CudaQ via toCudaQ()");
        case "reset":
          throw new TypeError("reset ops cannot be serialized to CudaQ via toCudaQ()");
        case "if":
          throw new TypeError("if ops cannot be serialized to CudaQ via toCudaQ()");
        case "two":
          throw new TypeError(`Gate '${op.meta?.name ?? "two"}' has no CudaQ representation`);
        case "unitary":
          throw new TypeError("Gate 'unitary' has no CudaQ representation");
        case "single": {
          if (!op.meta) throw new TypeError("Single-qubit op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const q = qi(op.q);
          switch (n) {
            case "id":
              break;
            case "h":
              lines.push(`kernel.h(${q})`);
              break;
            case "x":
              lines.push(`kernel.x(${q})`);
              break;
            case "y":
              lines.push(`kernel.y(${q})`);
              break;
            case "z":
              lines.push(`kernel.z(${q})`);
              break;
            case "s":
              lines.push(`kernel.s(${q})`);
              break;
            case "si":
              lines.push(`kernel.sdg(${q})`);
              break;
            case "t":
              lines.push(`kernel.t(${q})`);
              break;
            case "ti":
              lines.push(`kernel.tdg(${q})`);
              break;
            case "r2":
              lines.push(`kernel.rz(math.pi/2, ${q})`);
              break;
            case "r4":
              lines.push(`kernel.rz(math.pi/4, ${q})`);
              break;
            case "r8":
              lines.push(`kernel.rz(math.pi/8, ${q})`);
              break;
            case "rx":
              lines.push(`kernel.rx(${a(p[0])}, ${q})`);
              break;
            case "ry":
              lines.push(`kernel.ry(${a(p[0])}, ${q})`);
              break;
            case "rz":
              lines.push(`kernel.rz(${a(p[0])}, ${q})`);
              break;
            case "vz":
              lines.push(`kernel.rz(${a(p[0])}, ${q})`);
              break;
            case "u1":
              lines.push(`kernel.r1(${a(p[0])}, ${q})`);
              break;
            case "u2":
              lines.push(`kernel.u3(math.pi/2, ${a(p[0])}, ${a(p[1])}, ${q})`);
              break;
            case "u3":
              lines.push(`kernel.u3(${a(p[0])}, ${a(p[1])}, ${a(p[2])}, ${q})`);
              break;
            case "v":
            case "vi":
              throw new TypeError(`Gate '${n}' has no CudaQ representation`);
            case "gpi":
            case "gpi2":
              throw new TypeError(`Gate '${n}' has no CudaQ representation`);
            default:
              throw new TypeError(`Gate '${n}' has no CudaQ representation`);
          }
          break;
        }
        case "controlled": {
          if (!op.meta) throw new TypeError("Controlled op missing serialization meta");
          const { name: n, params: p } = op.meta;
          const [c2, t] = [qi(op.control), qi(op.target)];
          switch (n) {
            case "cy":
              lines.push(`kernel.cy(${c2}, ${t})`);
              break;
            case "cz":
              lines.push(`kernel.cz(${c2}, ${t})`);
              break;
            case "ch":
              lines.push(`kernel.ch(${c2}, ${t})`);
              break;
            case "crx":
              lines.push(`kernel.crx(${a(p[0])}, ${c2}, ${t})`);
              break;
            case "cry":
              lines.push(`kernel.cry(${a(p[0])}, ${c2}, ${t})`);
              break;
            case "crz":
              lines.push(`kernel.crz(${a(p[0])}, ${c2}, ${t})`);
              break;
            case "cr2":
              lines.push(`kernel.crz(math.pi/2, ${c2}, ${t})`);
              break;
            case "cr4":
              lines.push(`kernel.crz(math.pi/4, ${c2}, ${t})`);
              break;
            case "cr8":
              lines.push(`kernel.crz(math.pi/8, ${c2}, ${t})`);
              break;
            case "cu1":
              lines.push(`kernel.cr1(${a(p[0])}, ${c2}, ${t})`);
              break;
            case "cs":
              lines.push(`kernel.cr1(math.pi/2, ${c2}, ${t})`);
              break;
            case "ct":
              lines.push(`kernel.cr1(math.pi/4, ${c2}, ${t})`);
              break;
            case "csdg":
              lines.push(`kernel.cr1(-math.pi/2, ${c2}, ${t})`);
              break;
            case "ctdg":
              lines.push(`kernel.cr1(-math.pi/4, ${c2}, ${t})`);
              break;
            default:
              throw new TypeError(`Gate '${n}' has no CudaQ representation`);
          }
          break;
        }
        case "barrier":
          break;
        default: {
          const _exhaustive = op;
        }
      }
    }
    return [
      "import math",
      "import cudaq",
      "",
      "kernel = cudaq.make_kernel()",
      `q = kernel.qalloc(${this.qubits})`,
      ...lines
    ].join("\n");
  }
  /**
   * Emit a Quirk (algassert.com/quirk) JSON circuit descriptor.
   *
   * Returns a JSON string `{"cols":[...]}` that can be pasted into Quirk's
   * "Load" dialog or appended to the URL as `#circuit=<encoded>`.
   *
   * Column structure: each column is an array indexed by qubit.
   * `1` = idle wire, `"•"` = control, named strings or `{id, arg}` = gates.
   * Angles are in half-turns: arg = θ/π (Rx(π/2) → arg=0.5).
   *
   * Limitations: U2/U3/gpi/gpi2/ms, interaction gates (xx/yy/zz/xy/iswap/srswap),
   * and if/reset ops have no Quirk equivalent and will throw.
   * Measure ops emit `"Measure"`. U1 is approximated as Rz (same unitary up to global phase).
   */
  toQuirk() {
    const cols = [];
    const n = this.qubits;
    const rot = (id, theta) => ({ id, arg: theta / Math.PI });
    const col = (entries) => {
      const c2 = new Array(n).fill(1);
      for (const [q, g] of Object.entries(entries)) c2[+q] = g;
      while (c2.length > 1 && c2[c2.length - 1] === 1) c2.pop();
      cols.push(c2);
    };
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "cnot":
          col({ [op.control]: "\u2022", [op.target]: "X" });
          break;
        case "swap":
          col({ [op.a]: "Swap", [op.b]: "Swap" });
          break;
        case "toffoli":
          col({ [op.c1]: "\u2022", [op.c2]: "\u2022", [op.target]: "X" });
          break;
        case "cswap":
          col({ [op.control]: "\u2022", [op.a]: "Swap", [op.b]: "Swap" });
          break;
        case "csrswap":
          throw new TypeError("Gate 'csrswap' has no Quirk representation");
        case "measure":
          col({ [op.q]: "Measure" });
          break;
        case "reset":
          throw new TypeError("reset ops cannot be serialized to Quirk via toQuirk()");
        case "if":
          throw new TypeError("if ops cannot be serialized to Quirk via toQuirk()");
        case "two": {
          const n2 = op.meta?.name;
          throw new TypeError(`Gate '${n2 ?? "two"}' has no Quirk representation`);
        }
        case "single": {
          if (!op.meta) throw new TypeError("Single-qubit op missing serialization meta");
          const { name: nm, params: p } = op.meta;
          const q = op.q;
          switch (nm) {
            case "id":
              break;
            case "h":
              col({ [q]: "H" });
              break;
            case "x":
              col({ [q]: "X" });
              break;
            case "y":
              col({ [q]: "Y" });
              break;
            case "z":
              col({ [q]: "Z" });
              break;
            case "s":
              col({ [q]: "S" });
              break;
            case "si":
              col({ [q]: "S\u2020" });
              break;
            case "t":
              col({ [q]: "T" });
              break;
            case "ti":
              col({ [q]: "T\u2020" });
              break;
            case "v":
              col({ [q]: "X^\xBD" });
              break;
            case "vi":
              col({ [q]: "X^-\xBD" });
              break;
            case "r2":
              col({ [q]: rot("Rz", Math.PI / 2) });
              break;
            case "r4":
              col({ [q]: rot("Rz", Math.PI / 4) });
              break;
            case "r8":
              col({ [q]: rot("Rz", Math.PI / 8) });
              break;
            case "rx":
              col({ [q]: rot("Rx", p[0]) });
              break;
            case "ry":
              col({ [q]: rot("Ry", p[0]) });
              break;
            case "rz":
              col({ [q]: rot("Rz", p[0]) });
              break;
            case "vz":
              col({ [q]: rot("Rz", p[0]) });
              break;
            case "u1":
              col({ [q]: rot("Rz", p[0]) });
              break;
            case "gpi":
            case "gpi2":
              throw new TypeError(`Gate '${nm}' has no Quirk representation`);
            default:
              throw new TypeError(`Gate '${nm}' has no Quirk representation`);
          }
          break;
        }
        case "controlled": {
          if (!op.meta) throw new TypeError("Controlled op missing serialization meta");
          const { name: nm, params: p } = op.meta;
          const [c2, t] = [op.control, op.target];
          let g;
          switch (nm) {
            case "cy":
              g = "Y";
              break;
            case "cz":
              g = "Z";
              break;
            case "ch":
              g = "H";
              break;
            case "cr2":
              g = rot("Rz", Math.PI / 2);
              break;
            case "cr4":
              g = rot("Rz", Math.PI / 4);
              break;
            case "cr8":
              g = rot("Rz", Math.PI / 8);
              break;
            case "crx":
              g = rot("Rx", p[0]);
              break;
            case "cry":
              g = rot("Ry", p[0]);
              break;
            case "crz":
              g = rot("Rz", p[0]);
              break;
            case "cu1":
              g = rot("Rz", p[0]);
              break;
            case "cs":
              g = "S";
              break;
            case "ct":
              g = "T";
              break;
            case "csdg":
              g = "S\u2020";
              break;
            case "ctdg":
              g = "T\u2020";
              break;
            default:
              throw new TypeError(`Gate '${nm}' has no Quirk representation`);
          }
          col({ [c2]: "\u2022", [t]: g });
          break;
        }
        case "unitary":
          throw new TypeError("Gate 'unitary' has no Quirk representation");
        case "barrier":
          break;
        default: {
          const _exhaustive = op;
        }
      }
    }
    return JSON.stringify({ cols });
  }
  // ── Execution ────────────────────────────────────────────────────────────
  /** Run the circuit and return a probability distribution. */
  run({ shots = 1024, seed, noise, initialState } = {}) {
    const rng = makePrng(seed);
    const init = initialState !== void 0 ? svFromBitstring(initialState, this.qubits) : void 0;
    const noiseParams = noise == null ? void 0 : typeof noise === "string" ? (() => {
      const p = DEVICE_NOISE[noise];
      if (!p) throw new TypeError(`Unknown device profile '${noise}'. Known: ${Object.keys(DEVICE_NOISE).join(", ")}`);
      return p;
    })() : noise;
    const cregCounts = new Map(
      Array.from(this.#cregs.entries(), ([name, size]) => [name, new Array(size).fill(0)])
    );
    if (!noiseParams && !this.#ops.some((op) => op.kind === "measure" || op.kind === "reset" || op.kind === "if")) {
      const sv = simulatePure(this.#ops, this.qubits, init);
      const probs = probabilities(sv);
      const sorted = Array.from(probs.entries()).toSorted(([a], [b]) => a < b ? -1 : 1);
      const cdf = [];
      let cum = 0;
      for (const [idx, p] of sorted) {
        cum += p;
        cdf.push({ idx, cumP: cum });
      }
      const last = cdf.at(-1);
      if (last) last.cumP = 1;
      const counts2 = /* @__PURE__ */ new Map();
      for (let i = 0; i < shots; i++) {
        const r = rng();
        let lo = 0;
        let hi = cdf.length - 1;
        while (lo < hi) {
          const mid = lo + hi >> 1;
          if (cdf[mid].cumP < r) lo = mid + 1;
          else hi = mid;
        }
        const idx = cdf[lo]?.idx ?? 0n;
        counts2.set(idx, (counts2.get(idx) ?? 0) + 1);
      }
      return new Distribution(this.qubits, shots, counts2, cregCounts, false, "statevector");
    }
    const counts = /* @__PURE__ */ new Map();
    const pMeas = noiseParams?.pMeas ?? 0;
    for (let i = 0; i < shots; i++) {
      const shotCregs = new Map(
        Array.from(this.#cregs.entries(), ([name, size]) => [name, new Array(size).fill(false)])
      );
      const sv = applyOps(this.#ops, init ?? zero(this.qubits), shotCregs, rng, noiseParams);
      let finalIdx = sampleSV(sv, rng());
      if (pMeas) {
        for (let q = 0; q < this.qubits; q++) {
          if (rng() < pMeas) finalIdx ^= 1n << BigInt(q);
        }
      }
      counts.set(finalIdx, (counts.get(finalIdx) ?? 0) + 1);
      for (const [name, bits] of shotCregs) {
        const acc = cregCounts.get(name);
        for (const [j, b] of bits.entries()) if (b) acc[j] += 1;
      }
    }
    return new Distribution(this.qubits, shots, counts, cregCounts, false, "statevector");
  }
  /**
   * Run the circuit using MPS (tensor-network) simulation.
   *
   * Efficient for circuits with bounded entanglement (GHZ, BV, shallow QFT, QAOA low-depth).
   * Memory: O(n · χ² · 2) vs O(2ⁿ) for full statevector.
   * Handles 50+ qubit circuits that would be intractable with the statevector backend.
   *
   * Limitations: no mid-circuit measure/reset/if; Toffoli and CSWAP
   * must be decomposed into single- and two-qubit gates first.
   *
   * With `noise`: switches to **quantum trajectory mode** — one full circuit execution per shot,
   * with random Pauli errors injected after each gate. Noise limits entanglement growth so bond
   * dimension stays tractable even at 100+ qubits. Simulates realistic NISQ hardware accurately.
   *
   * @param maxBond Initial bond dimension χ (default 64). Grows automatically — set higher to reduce reallocation overhead for high-entanglement circuits.
   */
  runMps({ shots = 1024, seed, maxBond = 64, truncErr = 0, initialState, noise: noiseRaw, workers: numWorkers = 0 } = {}) {
    const noise = noiseRaw == null ? void 0 : typeof noiseRaw === "string" ? (() => {
      const p = DEVICE_NOISE[noiseRaw];
      if (!p) throw new TypeError(`Unknown device profile '${noiseRaw}'. Known: ${Object.keys(DEVICE_NOISE).join(", ")}`);
      return p;
    })() : noiseRaw;
    const rng = makePrng(seed);
    const flat = flattenOps(this.#ops);
    const trajOps = toTrajOps(flat);
    const counts = /* @__PURE__ */ new Map();
    const cregCounts = new Map(
      Array.from(this.#cregs.entries(), ([name, size]) => [name, new Array(size).fill(0)])
    );
    const hasMidCircuit = flat.some((op) => op.kind === "measure" || op.kind === "reset" || op.kind === "if");
    const traj = new MpsTrajectory(this.qubits, maxBond, truncErr);
    if (!noise && !hasMidCircuit) {
      if (initialState !== void 0) {
        svFromBitstring(initialState, this.qubits);
        for (let q = 0; q < this.qubits; q++) {
          if (initialState[q] === "1") traj.apply1(q, X);
        }
      }
      applyTrajOps(traj, trajOps, 0, 0, rng);
      for (let i = 0; i < shots; i++) {
        const idx = traj.sample(rng);
        counts.set(idx, (counts.get(idx) ?? 0) + 1);
      }
    } else {
      const p1 = noise?.p1 ?? 0;
      const p2 = noise?.p2 ?? 0;
      const pMeas = noise?.pMeas ?? 0;
      const gamma = noise?.gamma ?? 0;
      const lambda = noise?.lambda ?? 0;
      if (noise?.kraus1 || noise?.kraus2) {
        throw new Error("kraus1/kraus2 custom channels are not supported in the MPS trajectory backend \u2014 use run() (statevector) or runDM() (density matrix) instead");
      }
      if (initialState !== void 0) svFromBitstring(initialState, this.qubits);
      const wtLocal = wt;
      const isBuilt = !import.meta.url.endsWith(".ts");
      if (numWorkers > 1 && (!isBuilt || wtLocal === null)) {
        console.warn("[ket] runMps: workers option ignored \u2014 build the bundle first (npm run build) to enable parallel trajectories");
      }
      if (numWorkers > 1 && isBuilt && wtLocal !== null && !hasMidCircuit) {
        const workerUrl = new URL("./mps.worker.js", import.meta.url);
        const baseSeed = seed !== void 0 ? seed >>> 0 : Date.now() >>> 0;
        const slices = distributeShots(shots, numWorkers);
        const flags = slices.map(() => new Int32Array(new SharedArrayBuffer(4)));
        const ws = acquirePool(numWorkers, workerUrl, wtLocal.Worker);
        const channels = slices.map(() => new MessageChannel());
        slices.forEach((sliceShots, i) => {
          const job = {
            ops: trajOps,
            n: this.qubits,
            maxBond,
            truncErr,
            p1,
            p2,
            pMeas,
            shots: sliceShots,
            seed: baseSeed * 2654435769 + i * 1818371886 >>> 0 || 1,
            initialState,
            flag: flags[i],
            port: channels[i].port2
          };
          ws[i].postMessage(job, [channels[i].port2]);
        });
        for (let i = 0; i < ws.length; i++) {
          const waitResult = Atomics.wait(flags[i], 0, 0, 3e5);
          if (waitResult === "timed-out") throw new Error(`[ket] runMps worker ${i} timed out after 5 minutes`);
          const { message } = wtLocal.receiveMessageOnPort(channels[i].port1);
          for (const [k, v] of message.counts) {
            counts.set(k, (counts.get(k) ?? 0) + v);
          }
        }
      } else {
        for (let i = 0; i < shots; i++) {
          traj.reset();
          if (initialState !== void 0) {
            for (let q = 0; q < this.qubits; q++) {
              if (initialState[q] === "1") traj.apply1(q, X);
            }
          }
          const shotCregs = hasMidCircuit ? new Map(
            Array.from(this.#cregs.entries(), ([name, size]) => [name, new Array(size).fill(false)])
          ) : void 0;
          applyTrajOps(traj, trajOps, p1, p2, rng, shotCregs, pMeas, gamma, lambda);
          let idx = traj.sample(rng);
          if (pMeas) {
            for (let q = 0; q < this.qubits; q++) {
              if (rng() < pMeas) idx ^= 1n << BigInt(q);
            }
          }
          counts.set(idx, (counts.get(idx) ?? 0) + 1);
          if (shotCregs) {
            for (const [name, bits] of shotCregs) {
              const acc = cregCounts.get(name);
              for (const [j, b] of bits.entries()) if (b) acc[j] += 1;
            }
          }
        }
      }
    }
    return new Distribution(this.qubits, shots, counts, cregCounts, traj.wasTruncated, "mps", traj.maxBondUsed());
  }
  /**
   * Compute ⟨ψ|H|ψ⟩ for a Pauli-string Hamiltonian H = Σᵢ termᵢ.coeff · termᵢ.ops.
   *
   * Drop-in replacement for `vqe()` that scales to 50+ qubits via MPS.
   * Builds the MPS state once, then sweeps each term via transfer matrix —
   * exact to floating-point precision, O(|terms| · n · χ³).
   *
   * Uses the same `PauliTerm` convention as `vqe()`:
   * `ops[0]` acts on qubit n-1 (MSB); `ops[n-1]` acts on qubit 0 (LSB).
   * **Qubit 0 is the rightmost character** — `'ZZII'` means Z on qubits 3 and 2, identity on 1 and 0.
   *
   * @example
   * // Heisenberg ZZ + XX chain on 4 qubits — same terms work with vqe() too
   * const { energy } = circuit.expectMps([
   *   { coeff: -0.5, ops: 'ZZII' },  // ZZ on qubits 3,2
   *   { coeff: -0.5, ops: 'IZZI' },  // ZZ on qubits 2,1
   *   { coeff: -0.5, ops: 'XXII' },  // XX on qubits 3,2
   * ])
   */
  expectMps(terms, { maxBond = 64, truncErr = 0, initialState } = {}) {
    if (terms.length === 0) return { energy: 0, truncated: false };
    const n = this.qubits;
    const traj = new MpsTrajectory(n, maxBond, truncErr);
    const trajOps = toTrajOps(flattenOps(this.#ops));
    const rng = () => 0;
    if (initialState !== void 0) {
      svFromBitstring(initialState, n);
      for (let q = 0; q < n; q++) {
        if (initialState[q] === "1") traj.apply1(q, X);
      }
    }
    applyTrajOps(traj, trajOps, 0, 0, rng);
    let energy = 0;
    for (const { coeff, ops } of terms) {
      if (coeff === 0) continue;
      if (ops.length !== n)
        throw new TypeError(`PauliTerm ops length ${ops.length} must equal circuit qubits (${n})`);
      const upper = ops.toUpperCase();
      if (!/^[IXYZ]+$/.test(upper))
        throw new TypeError(`PauliTerm ops '${ops}' contains characters outside {I, X, Y, Z}`);
      const gateOps = Array.from({ length: n }, (_, q) => {
        switch (upper[n - 1 - q]) {
          case "X":
            return X;
          case "Y":
            return Y;
          case "Z":
            return Z;
          default:
            return null;
        }
      });
      energy += coeff * traj.expectation(gateOps).re;
    }
    return { energy, truncated: traj.wasTruncated };
  }
  /**
   * Von Neumann entanglement entropies S_b = −Σ_k σ_k² log₂(σ_k²) at each bond.
   *
   * Builds the MPS state once from this circuit, then reads the Schmidt spectrum
   * stored at every bond in Vidal canonical form. Returns n−1 values.
   *
   * Entanglement entropy is the physicist's diagnostic for quantum correlations:
   * product states have S=0 everywhere, maximally entangled bonds have S=1,
   * and area-law states (ground states of local Hamiltonians) grow logarithmically.
   *
   * O(n · χ²) total — cheaper than any expectation value.
   *
   * @example
   * // Bell state: one bond, S = 1 (maximally entangled)
   * new Circuit(2).h(0).cnot(0, 1).bondEntropies()  // → [1.0]
   *
   * // GHZ state: all bonds saturated at S = 1
   * ghz(8).bondEntropies()  // → [1, 1, 1, 1, 1, 1, 1]
   */
  bondEntropies({ maxBond = 64, truncErr = 0, initialState } = {}) {
    const n = this.qubits;
    const traj = new MpsTrajectory(n, maxBond, truncErr);
    const trajOps = toTrajOps(flattenOps(this.#ops));
    const rng = () => 0;
    if (initialState !== void 0) {
      svFromBitstring(initialState, n);
      for (let q = 0; q < n; q++) {
        if (initialState[q] === "1") traj.apply1(q, X);
      }
    }
    applyTrajOps(traj, trajOps, 0, 0, rng);
    return traj.bondEntropies();
  }
  /**
   * Return exact floating-point probabilities from the statevector — no sampling variance.
   *
   * Keys are standard bitstrings (q0 leftmost). Only non-negligible amplitudes are included.
   * Throws for circuits containing mid-circuit measure, reset, or conditional ops.
   */
  exactProbs({ initialState } = {}) {
    if (this.#ops.some((op) => op.kind === "measure" || op.kind === "reset" || op.kind === "if")) {
      throw new TypeError("exactProbs() requires a pure circuit \u2014 no measure, reset, or if ops");
    }
    const init = initialState !== void 0 ? svFromBitstring(initialState, this.qubits) : void 0;
    const sv = simulatePure(this.#ops, this.qubits, init);
    const out = {};
    for (const [idx, p] of probabilities(sv)) {
      out[idx.toString(2).padStart(this.qubits, "0").split("").reverse().join("")] = p;
    }
    return Object.freeze(out);
  }
  /**
   * Compute the Pauli expectation value ⟨ψ|P|ψ⟩ for a tensor-product Pauli operator P.
   *
   * `pauli` is a string of length `qubits` over {I, X, Y, Z} (case-insensitive).
   * `pauli[q]` specifies the Pauli acting on qubit q (q0 leftmost, matching bitstring convention).
   *
   * Basis rotations: X → H, Y → Rx(π/2), Z/I → identity.
   * Throws `TypeError` for circuits with mid-circuit measure, reset, or if ops.
   *
   * @example
   * new Circuit(1).expectation('Z')              // 1   (|0⟩ is +1 eigenstate of Z)
   * new Circuit(1).x(0).expectation('Z')         // -1  (|1⟩ is −1 eigenstate of Z)
   * new Circuit(1).h(0).expectation('X')         // 1   (|+⟩ is +1 eigenstate of X)
   * new Circuit(2).h(0).cnot(0,1).expectation('ZZ')  // 1   (Bell state)
   */
  expectation(pauli) {
    const n = this.qubits;
    pauli = pauli.toUpperCase();
    if (pauli.length !== n) throw new TypeError(`pauli '${pauli}' length must equal qubits (${n})`);
    if (!/^[IXYZ]+$/.test(pauli)) throw new TypeError(`pauli must contain only I, X, Y, Z`);
    if (!/[XYZ]/.test(pauli)) return 1;
    let rot = this;
    for (let q = 0; q < n; q++) {
      if (pauli[q] === "X") rot = rot.h(q);
      else if (pauli[q] === "Y") rot = rot.rx(Math.PI / 2, q);
    }
    const probs = rot.exactProbs();
    let exp = 0;
    for (const [bs, prob] of Object.entries(probs)) {
      let parity = 0;
      for (let q = 0; q < n; q++) {
        if (pauli[q] !== "I" && bs[q] === "1") parity ^= 1;
      }
      exp += (parity === 0 ? 1 : -1) * prob;
    }
    return exp;
  }
  // ── JSON save / load ─────────────────────────────────────────────────────
  /**
   * Serialize the circuit to a lossless JSON object.
   *
   * Gate matrices are **not** stored — they are reconstructed from their
   * names and parameters on load, so the output is compact and stable
   * across library versions.
   *
   * @example
   * const json = circuit.toJSON()
   * fs.writeFileSync('circuit.json', JSON.stringify(json, null, 2))
   */
  toJSON() {
    const cregs = {};
    for (const [name, size] of this.#cregs) cregs[name] = size;
    const gates = {};
    for (const [name, sub] of this.#gates) {
      gates[name] = { qubits: sub.qubits, ops: opsToJSON(sub.#ops) };
    }
    return { ket: 1, qubits: this.qubits, cregs, gates, ops: opsToJSON(this.#ops) };
  }
  /**
   * Deserialize a circuit from a `CircuitJSON` object or a JSON string.
   *
   * The loaded circuit is fully functional — all gate methods, simulation,
   * serialization, and visualization APIs work identically to a hand-built circuit.
   *
   * @throws TypeError for unrecognised op kinds, unknown gate names, or wrong schema version.
   *
   * @example
   * const circuit = Circuit.fromJSON(fs.readFileSync('circuit.json', 'utf8'))
   */
  static fromJSON(json) {
    const j = typeof json === "string" ? JSON.parse(json) : json;
    if (j.ket !== 1) throw new TypeError(`fromJSON: unsupported schema version ${j.ket}`);
    const cregs = new Map(Object.entries(j.cregs));
    const gates = /* @__PURE__ */ new Map();
    for (const [name, def] of Object.entries(j.gates)) {
      gates.set(name, new _Circuit(def.qubits, opsFromJSON(def.ops), /* @__PURE__ */ new Map(), /* @__PURE__ */ new Map()));
    }
    return new _Circuit(j.qubits, opsFromJSON(j.ops), cregs, gates);
  }
  // ── Random circuit generation ─────────────────────────────────────────────
  /**
   * Generate a random circuit with `nQubits` qubits and `nGates` gates.
   *
   * Each gate is chosen uniformly at random from a standard set: H, X, Y, Z,
   * S, T, Rx, Ry, Rz (single-qubit) and CNOT, SWAP (two-qubit, 30% chance when
   * `nQubits ≥ 2`).  Rotation angles are drawn uniformly from [0, 2π).
   *
   * Useful for benchmarking, quantum-volume estimation, and noise characterisation.
   *
   * @param nQubits Number of qubits (≥ 1).
   * @param nGates  Number of gates to add.
   * @param seed    Optional PRNG seed for reproducibility.
   *
   * @example
   * const c = Circuit.random(4, 20, 42)
   * c.run({ shots: 1024 })
   */
  static random(nQubits, nGates, seed) {
    if (nQubits < 1) throw new RangeError("nQubits must be \u2265 1");
    if (nGates < 0) throw new RangeError("nGates must be \u2265 0");
    const rng = makePrng(seed);
    let c2 = new _Circuit(nQubits);
    const TAU = 2 * Math.PI;
    for (let i = 0; i < nGates; i++) {
      const twoQ = nQubits >= 2 && rng() < 0.3;
      if (twoQ) {
        const a = Math.floor(rng() * nQubits);
        let b = Math.floor(rng() * (nQubits - 1));
        if (b >= a) b++;
        c2 = rng() < 0.5 ? c2.cnot(a, b) : c2.swap(a, b);
      } else {
        const q = Math.floor(rng() * nQubits);
        const kind = Math.floor(rng() * 9);
        switch (kind) {
          case 0:
            c2 = c2.h(q);
            break;
          case 1:
            c2 = c2.x(q);
            break;
          case 2:
            c2 = c2.y(q);
            break;
          case 3:
            c2 = c2.z(q);
            break;
          case 4:
            c2 = c2.s(q);
            break;
          case 5:
            c2 = c2.t(q);
            break;
          case 6:
            c2 = c2.rx(rng() * TAU, q);
            break;
          case 7:
            c2 = c2.ry(rng() * TAU, q);
            break;
          case 8:
            c2 = c2.rz(rng() * TAU, q);
            break;
        }
      }
    }
    return c2;
  }
  // ── Visualization ────────────────────────────────────────────────────────
  /**
   * Render a minimal ASCII circuit diagram.
   *
   * @example
   * new Circuit(2).h(0).cnot(0, 1).draw()
   * // q0: ─H──●─
   * //          │
   * // q1: ─────⊕─
   */
  draw() {
    const n = this.qubits;
    if (n === 0) return "";
    const ops = flattenOps(this.#ops).filter((op) => op.kind !== "if");
    const colOf = new Array(n).fill(0);
    const placed = [];
    for (const op of ops) {
      const qs = opQubits(op);
      if (qs.length === 0) continue;
      const minQ = Math.min(...qs), maxQ = Math.max(...qs);
      let col = 0;
      for (let q = minQ; q <= maxQ; q++) col = Math.max(col, colOf[q]);
      for (let q = minQ; q <= maxQ; q++) colOf[q] = col + 1;
      placed.push({ op, col });
    }
    const numCols = Math.max(0, ...colOf);
    if (numCols === 0) {
      const pw = `q${n - 1}: `.length;
      return Array.from({ length: n }, (_, q) => `q${q}: `.padStart(pw) + "\u2500").join("\n");
    }
    const label = Array.from(
      { length: n },
      () => new Array(numCols).fill("")
    );
    const hasVert = Array.from(
      { length: numCols },
      () => new Array(n - 1).fill(false)
    );
    for (const { op, col } of placed) {
      const qs = opQubits(op);
      const minQ = Math.min(...qs), maxQ = Math.max(...qs);
      for (let gap = minQ; gap < maxQ; gap++) hasVert[col][gap] = true;
      for (const q of qs) label[q][col] = opLabel(op, q);
    }
    const colW = Array.from({ length: numCols }, (_, c2) => {
      let w = 1;
      for (let q = 0; q < n; q++) {
        const lbl = label[q][c2];
        if (lbl !== null && lbl !== "") w = Math.max(w, lbl.length);
      }
      return w;
    });
    const prefixW = `q${n - 1}: `.length;
    const lines = [];
    for (let q = 0; q < n; q++) {
      let line = `q${q}: `.padStart(prefixW);
      for (let c2 = 0; c2 < numCols; c2++) {
        const lbl = label[q][c2];
        const w = colW[c2];
        if (lbl === "" || lbl === null) {
          line += "\u2500".repeat(w + 2);
        } else {
          const pad = w - lbl.length;
          const padL = Math.floor(pad / 2);
          const padR = pad - padL;
          line += "\u2500" + "\u2500".repeat(padL) + lbl + "\u2500".repeat(padR) + "\u2500";
        }
      }
      line += "\u2500";
      lines.push(line);
      if (q < n - 1) {
        let spacer = " ".repeat(prefixW);
        for (let c2 = 0; c2 < numCols; c2++) {
          const w = colW[c2];
          if (hasVert[c2][q]) {
            const center = Math.floor((w + 2) / 2);
            spacer += " ".repeat(center) + "\u2502" + " ".repeat(w + 2 - center - 1);
          } else {
            spacer += " ".repeat(w + 2);
          }
        }
        lines.push(spacer);
      }
    }
    return lines.join("\n");
  }
  /**
   * Export the circuit as a self-contained SVG string.
   *
   * The diagram uses the same column layout as `draw()`.  No external fonts or
   * stylesheets are required — the SVG embeds a monospace `font-family` stack.
   *
   * @example
   * fs.writeFileSync('bell.svg', new Circuit(2).h(0).cnot(0, 1).toSVG())
   */
  toSVG() {
    const n = this.qubits;
    const ROW_H = 40;
    const COL_W = 52;
    const CHAR_W = 7.5;
    const BOX_H = 22;
    const R = 5;
    const ML = 52;
    const MR = 24;
    const MT = 24;
    const MB = 16;
    const ops = flattenOps(this.#ops).filter((op) => op.kind !== "if");
    const colOf = new Array(n).fill(0);
    const placed = [];
    for (const op of ops) {
      const qs = opQubits(op);
      if (qs.length === 0) continue;
      const minQ = Math.min(...qs), maxQ = Math.max(...qs);
      let col = 0;
      for (let q = minQ; q <= maxQ; q++) col = Math.max(col, colOf[q]);
      for (let q = minQ; q <= maxQ; q++) colOf[q] = col + 1;
      placed.push({ op, col });
    }
    const numCols = Math.max(0, ...colOf);
    const colPx = Array.from({ length: numCols }, (_, c2) => {
      let maxLabel = 1;
      for (const { op, col } of placed) {
        if (col !== c2) continue;
        for (const q of opQubits(op)) {
          const lbl = opLabel(op, q);
          if (lbl !== "\u25CF" && lbl !== "\u2295" && lbl !== "\u2573") maxLabel = Math.max(maxLabel, lbl.length);
        }
      }
      return Math.max(COL_W, Math.ceil(maxLabel * CHAR_W) + 20);
    });
    const colX = [];
    let cx = ML;
    for (let c2 = 0; c2 < numCols; c2++) {
      colX.push(cx + colPx[c2] / 2);
      cx += colPx[c2];
    }
    const totalW = ML + (numCols > 0 ? cx - ML : 0) + MR;
    const totalH = MT + (n - 1) * ROW_H + MB + ROW_H;
    const qy = (q) => MT + q * ROW_H + ROW_H / 2;
    const svgParts = [];
    svgParts.push(`<rect width="${totalW}" height="${totalH}" fill="#ffffff"/>`);
    for (let q = 0; q < n; q++) {
      const y = qy(q);
      svgParts.push(`<line x1="${ML - 8}" y1="${y}" x2="${totalW - MR}" y2="${y}" stroke="#334155" stroke-width="1.5"/>`);
    }
    for (let q = 0; q < n; q++) {
      svgParts.push(`<text x="${ML - 12}" y="${qy(q) + 4}" text-anchor="end" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="13" fill="#334155">q${q}:</text>`);
    }
    for (const { op, col } of placed) {
      const x = colX[col];
      const qs = opQubits(op);
      const minQ = Math.min(...qs), maxQ = Math.max(...qs);
      if (minQ !== maxQ) {
        svgParts.push(`<line x1="${x}" y1="${qy(minQ)}" x2="${x}" y2="${qy(maxQ)}" stroke="#334155" stroke-width="1.5"/>`);
      }
      for (const q of qs) {
        const lbl = opLabel(op, q);
        const y = qy(q);
        if (lbl === "\u25CF") {
          svgParts.push(`<circle cx="${x}" cy="${y}" r="5" fill="#334155"/>`);
        } else if (lbl === "\u2295") {
          svgParts.push(`<circle cx="${x}" cy="${y}" r="10" fill="none" stroke="#334155" stroke-width="1.5"/>`);
          svgParts.push(`<line x1="${x}" y1="${y - 10}" x2="${x}" y2="${y + 10}" stroke="#334155" stroke-width="1.5"/>`);
          svgParts.push(`<line x1="${x - 10}" y1="${y}" x2="${x + 10}" y2="${y}" stroke="#334155" stroke-width="1.5"/>`);
        } else if (lbl === "\u2573") {
          const d = 7;
          svgParts.push(`<line x1="${x - d}" y1="${y - d}" x2="${x + d}" y2="${y + d}" stroke="#334155" stroke-width="2"/>`);
          svgParts.push(`<line x1="${x + d}" y1="${y - d}" x2="${x - d}" y2="${y + d}" stroke="#334155" stroke-width="2"/>`);
        } else {
          const labelPx = Math.max(20, Math.ceil(lbl.length * CHAR_W) + 14);
          const bx = x - labelPx / 2, by = y - BOX_H / 2;
          svgParts.push(`<rect x="${bx}" y="${by}" width="${labelPx}" height="${BOX_H}" rx="${R}" fill="#f8fafc" stroke="#334155" stroke-width="1.5"/>`);
          svgParts.push(`<text x="${x}" y="${y + 4}" text-anchor="middle" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="13" fill="#1e293b">${lbl}</text>`);
        }
      }
    }
    const w = totalW, h = totalH;
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">
${svgParts.join("\n")}
</svg>`;
  }
  /**
   * Return the Bloch sphere angles (θ, φ) for qubit `q`.
   *
   * Computes the reduced single-qubit density matrix by tracing out all other
   * qubits, then extracts the Bloch vector (rx, ry, rz) and converts to
   * standard spherical coordinates:
   *   - θ ∈ [0, π]  — polar angle from |0⟩ (north pole)
   *   - φ ∈ (-π, π] — azimuthal angle in the equatorial plane
   *
   * For pure product states the result is exact.  For entangled qubits the
   * Bloch vector has |r| < 1 (mixed state) and the angles are still well
   * defined as long as the qubit is not maximally mixed (|r| > 0).
   *
   * @throws TypeError when called on circuits with measure/reset/if ops.
   */
  blochAngles(q, { initialState } = {}) {
    const sv = this.statevector(initialState !== void 0 ? { initialState } : {});
    const mask = 1n << BigInt(q);
    let rho00 = 0, rho11 = 0, rho01re = 0, rho01im = 0;
    for (const [idx, amp] of sv) {
      const p = amp.re * amp.re + amp.im * amp.im;
      if ((idx & mask) !== 0n) {
        rho11 += p;
        continue;
      }
      rho00 += p;
      const amp1 = sv.get(idx | mask);
      if (amp1) {
        rho01re += amp.re * amp1.re + amp.im * amp1.im;
        rho01im += amp.im * amp1.re - amp.re * amp1.im;
      }
    }
    const rz = rho00 - rho11;
    const rx = 2 * rho01re;
    const ry = -2 * rho01im;
    const theta = Math.acos(Math.max(-1, Math.min(1, rz)));
    const phi = Math.atan2(ry, rx);
    return { theta, phi };
  }
  // ── Circuit depth ─────────────────────────────────────────────────────────
  /**
   * Return the critical path length — the minimum number of time steps needed
   * to execute this circuit when gates on independent qubits run in parallel.
   *
   * Barriers are scheduling hints only and do not increment depth.
   * IfOps recurse into their inner ops.
   */
  depth() {
    const stepOf = new Array(this.qubits).fill(0);
    function processOps(ops) {
      for (const op of flattenOps(ops)) {
        if (op.kind === "barrier") continue;
        if (op.kind === "if") {
          processOps(op.ops);
          continue;
        }
        let qubits;
        switch (op.kind) {
          case "single":
            qubits = [op.q];
            break;
          case "cnot":
            qubits = [op.control, op.target];
            break;
          case "swap":
            qubits = [op.a, op.b];
            break;
          case "two":
            qubits = [op.a, op.b];
            break;
          case "controlled":
            qubits = [op.control, op.target];
            break;
          case "toffoli":
            qubits = [op.c1, op.c2, op.target];
            break;
          case "cswap":
            qubits = [op.control, op.a, op.b];
            break;
          case "csrswap":
            qubits = [op.control, op.a, op.b];
            break;
          case "measure":
            qubits = [op.q];
            break;
          case "reset":
            qubits = [op.q];
            break;
          case "unitary":
            qubits = [...op.qubits];
            break;
          default: {
            const _exhaustive = op;
            qubits = [];
            break;
          }
        }
        if (qubits.length === 0) continue;
        let maxStep = 0;
        for (const q of qubits) maxStep = Math.max(maxStep, stepOf[q] ?? 0);
        const nextStep = maxStep + 1;
        for (const q of qubits) stepOf[q] = nextStep;
      }
    }
    processOps(this.#ops);
    return Math.max(0, ...stepOf);
  }
  // ── Bloch sphere visualization ────────────────────────────────────────────
  /**
   * Return a self-contained SVG string showing the Bloch sphere for qubit `q`.
   *
   * Uses `blochAngles(q)` to get the state angles, then renders a 300×300 SVG
   * with the sphere outline, equatorial ellipse, axes, and a blue arrow for the
   * state vector using cavalier projection.
   *
   * @throws TypeError if the circuit contains measure/reset/if ops (see blochAngles).
   */
  blochSphere(q) {
    const { theta, phi } = this.blochAngles(q);
    const bx = Math.sin(theta) * Math.cos(phi);
    const by = Math.sin(theta) * Math.sin(phi);
    const bz = Math.cos(theta);
    const cx = 150, cy = 150, R = 110;
    const px = cx + R * (bx - by * 0.4);
    const py = cy - R * (bz + by * 0.1);
    const parts = [];
    parts.push(`<rect width="300" height="300" fill="white"/>`);
    parts.push(`<circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="#94a3b8" stroke-width="1.5"/>`);
    parts.push(`<ellipse cx="${cx}" cy="${cy}" rx="${R}" ry="${Math.round(R * 0.35)}" fill="none" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4,3"/>`);
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${cx}" y2="${cy - R}" stroke="#475569" stroke-width="1.2" stroke-dasharray="5,3"/>`);
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${cx}" y2="${cy + R}" stroke="#475569" stroke-width="1.2"/>`);
    parts.push(`<text x="${cx}" y="${cy - R - 8}" text-anchor="middle" font-family="serif" font-size="14" fill="#1e293b">|0\u27E9</text>`);
    parts.push(`<text x="${cx}" y="${cy + R + 18}" text-anchor="middle" font-family="serif" font-size="14" fill="#1e293b">|1\u27E9</text>`);
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${cx + R}" y2="${cy}" stroke="#475569" stroke-width="1.2"/>`);
    parts.push(`<text x="${cx + R + 8}" y="${cy + 4}" text-anchor="start" font-family="serif" font-size="13" fill="#1e293b">|+\u27E9</text>`);
    const yTipX = cx + Math.round(R * 0.4);
    const yTipY = cy + Math.round(R * 0.1) + Math.round(R * 0.35);
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${yTipX}" y2="${yTipY}" stroke="#475569" stroke-width="1.2"/>`);
    parts.push(`<text x="${yTipX + 6}" y="${yTipY + 4}" text-anchor="start" font-family="serif" font-size="13" fill="#1e293b">|i\u27E9</text>`);
    parts.push(`<line x1="${cx}" y1="${cy}" x2="${px.toFixed(1)}" y2="${py.toFixed(1)}" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round"/>`);
    parts.push(`<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="5" fill="#3b82f6"/>`);
    return `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">
${parts.join("\n")}
</svg>`;
  }
  // ── Clifford stabilizer simulation ───────────────────────────────────────
  /**
   * Simulate this circuit using the CHP (Aaronson-Gottesman 2004) Clifford
   * stabilizer algorithm — exponentially faster than the statevector for
   * Clifford circuits, with exact probabilities.
   *
   * Only Clifford gates are supported:
   *   h, x, y, z, s, si/sdg, measure, reset, barrier, cnot, swap,
   *   and controlled gates cx/cy/cz.
   *
   * Non-Clifford gates (T, Rx(θ≠kπ/2), etc.) cause a TypeError.
   *
   * @param opts.shots  Number of measurement shots (default 1024).
   * @param opts.seed   Optional PRNG seed for reproducibility.
   * @param opts.noise  Device name (any key of `DEVICES`, e.g. `'ibm_sherbrooke'`, `'h1-1'`) or
   *                    `{ p1?, p2?, pMeas? }` depolarizing + readout error rates.
   */
  runClifford({ shots = 1024, seed, noise } = {}) {
    const CLIFFORD_SINGLE = /* @__PURE__ */ new Set(["h", "x", "y", "z", "s", "si", "sdg"]);
    const CLIFFORD_CTRL = /* @__PURE__ */ new Set(["cx", "cy", "cz"]);
    const validateOps = (ops) => {
      for (const op of flattenOps(ops)) {
        switch (op.kind) {
          case "barrier":
          case "measure":
          case "reset":
            break;
          case "cnot":
          case "swap":
            break;
          case "if":
            validateOps(op.ops);
            break;
          case "single": {
            const name = op.meta?.name ?? "?";
            if (!CLIFFORD_SINGLE.has(name))
              throw new TypeError(`runClifford: gate '${name}' is not a Clifford gate`);
            break;
          }
          case "controlled": {
            const name = op.meta?.name ?? "?";
            if (!CLIFFORD_CTRL.has(name))
              throw new TypeError(`runClifford: gate '${name}' is not a Clifford gate`);
            break;
          }
          default: {
            const name = op.meta?.name ?? op.kind;
            throw new TypeError(`runClifford: gate '${name}' is not a Clifford gate`);
          }
        }
      }
    };
    validateOps(this.#ops);
    const noiseParams = noise == null ? void 0 : typeof noise === "string" ? (() => {
      const p = DEVICE_NOISE[noise];
      if (!p) throw new TypeError(`Unknown device profile '${noise}'. Known: ${Object.keys(DEVICE_NOISE).join(", ")}`);
      return p;
    })() : noise;
    const p1 = noiseParams?.p1 ?? 0;
    const p2 = noiseParams?.p2 ?? 0;
    const pMeas = noiseParams?.pMeas ?? 0;
    const rng = makePrng(seed);
    const counts = /* @__PURE__ */ new Map();
    const cregCounts = new Map(
      Array.from(this.#cregs.entries(), ([name, size]) => [name, new Array(size).fill(0)])
    );
    const flatOps = flattenOps(this.#ops);
    const cDep1 = (sim, q, p) => {
      const r = rng();
      if (r >= p) return;
      const s = r / p;
      if (s < 1 / 3) sim.x(q);
      else if (s < 2 / 3) sim.y(q);
      else sim.z(q);
    };
    const applyPauli = (sim, q, p) => {
      if (p === 1) sim.x(q);
      else if (p === 2) sim.y(q);
      else if (p === 3) sim.z(q);
    };
    const cDep2 = (sim, a, b, p) => {
      const r = rng();
      if (r >= p) return;
      const [ea, eb] = TWO_PAULI_IDX[Math.min(Math.floor(r / p * 15), 14)];
      applyPauli(sim, a, ea);
      applyPauli(sim, b, eb);
    };
    for (let shot = 0; shot < shots; shot++) {
      const sim = new CliffordSim(this.qubits);
      const shotCregs = new Map(
        Array.from(this.#cregs.entries(), ([name, size]) => [name, new Array(size).fill(false)])
      );
      const applyCliffords = (ops) => {
        for (const op of ops) {
          switch (op.kind) {
            case "barrier":
              break;
            case "single": {
              const name = op.meta?.name ?? "";
              switch (name) {
                case "h":
                  sim.h(op.q);
                  break;
                case "x":
                  sim.x(op.q);
                  break;
                case "y":
                  sim.y(op.q);
                  break;
                case "z":
                  sim.z(op.q);
                  break;
                case "s":
                  sim.s(op.q);
                  break;
                case "si":
                case "sdg":
                  sim.si(op.q);
                  break;
              }
              if (p1) cDep1(sim, op.q, p1);
              break;
            }
            case "cnot":
              sim.cnot(op.control, op.target);
              if (p2) cDep2(sim, op.control, op.target, p2);
              break;
            case "swap":
              sim.swap(op.a, op.b);
              if (p2) cDep2(sim, op.a, op.b, p2);
              break;
            case "controlled": {
              const name = op.meta?.name ?? "";
              switch (name) {
                case "cx":
                  sim.cnot(op.control, op.target);
                  break;
                case "cy":
                  sim.cy(op.control, op.target);
                  break;
                case "cz":
                  sim.cz(op.control, op.target);
                  break;
              }
              if (p2) cDep2(sim, op.control, op.target, p2);
              break;
            }
            case "measure": {
              const raw = sim.measure(op.q, rng());
              const outcome = pMeas && rng() < pMeas ? raw ^ 1 : raw;
              const reg = shotCregs.get(op.creg);
              if (reg) reg[op.bit] = outcome === 1;
              break;
            }
            case "reset": {
              const outcome = sim.measure(op.q, rng());
              if (outcome === 1) sim.x(op.q);
              break;
            }
            case "if": {
              if (cregValue(shotCregs, op.creg) === op.value) applyCliffords(flattenOps(op.ops));
              break;
            }
          }
        }
      };
      applyCliffords(flatOps);
      let idx = 0n;
      for (let q = 0; q < this.qubits; q++) {
        let bit = sim.measure(q, rng());
        if (pMeas && rng() < pMeas) bit ^= 1;
        if (bit) idx |= 1n << BigInt(q);
      }
      counts.set(idx, (counts.get(idx) ?? 0) + 1);
      for (const [name, bits] of shotCregs) {
        const acc = cregCounts.get(name);
        for (const [j, b] of bits.entries()) if (b) acc[j] += 1;
      }
    }
    return new Distribution(this.qubits, shots, counts, cregCounts, false, "clifford");
  }
  // ── Auto-routing simulation ───────────────────────────────────────────────
  /**
   * Simulate the circuit using the most efficient exact backend, chosen automatically:
   *
   * - **Clifford**: if every gate is in {H, X, Y, Z, S, S†, CNOT, CX, CY, CZ, SWAP} —
   *   O(n²) stabilizer tableau, handles 1000+ qubits.
   * - **Statevector**: if n ≤ `statevectorLimit` (default 20) — exact O(2ⁿ).
   *   Mid-circuit ops on large circuits route to MPS instead.
   * - **MPS**: otherwise — O(n·χ²) with adaptive bond dimension; exact for circuits
   *   with bounded entanglement, memory-bounded for highly entangled ones.
   *
   * The returned `Distribution` carries two extra fields:
   * - `backend` — which path was taken (`'clifford' | 'statevector' | 'mps'`)
   * - `peakChi` — peak bond dimension χ actually used (MPS only)
   *
   * ```typescript
   * const d = ghz(50).simulate({ shots: 1024 })
   * d.backend   // 'mps'
   * d.peakChi   // 2
   * ```
   */
  simulate({ shots = 1024, seed, noise, initialState, statevectorLimit = 20 } = {}) {
    const CLIFFORD_SINGLE = /* @__PURE__ */ new Set(["h", "x", "y", "z", "s", "si", "sdg"]);
    const CLIFFORD_CTRL = /* @__PURE__ */ new Set(["cx", "cy", "cz"]);
    const isClifford = (ops) => {
      for (const op of flattenOps(ops)) {
        switch (op.kind) {
          case "barrier":
          case "measure":
          case "reset":
            break;
          case "cnot":
          case "swap":
            break;
          case "if":
            if (!isClifford(op.ops)) return false;
            break;
          case "single":
            if (!CLIFFORD_SINGLE.has(op.meta?.name ?? "")) return false;
            break;
          case "controlled":
            if (!CLIFFORD_CTRL.has(op.meta?.name ?? "")) return false;
            break;
          default:
            return false;
        }
      }
      return true;
    };
    if (!initialState && isClifford(this.#ops)) {
      return this.runClifford({
        shots,
        ...seed !== void 0 && { seed },
        ...noise !== void 0 && { noise }
      });
    }
    if (this.qubits <= statevectorLimit) {
      return this.run({
        shots,
        ...seed !== void 0 && { seed },
        ...noise !== void 0 && { noise },
        ...initialState !== void 0 && { initialState }
      });
    }
    return this.runMps({
      shots,
      ...seed !== void 0 && { seed },
      ...noise !== void 0 && { noise },
      ...initialState !== void 0 && { initialState }
    });
  }
  // ── Hardware compilation ──────────────────────────────────────────────────
  /**
   * Transpile this circuit to the native gate set of the specified IonQ device.
   *
   * Supported devices: 'aria-1', 'forte-1', 'harmony' — all use {GPI, GPI2, MS, VZ}.
   *
   * Single-qubit decompositions (all exact up to global phase):
   *   h  → vz(π/2) · gpi2(0) · vz(π/2)
   *   x  → gpi(0)
   *   y  → gpi(π/2)
   *   z  → vz(π)
   *   s  → vz(π/2)
   *   si/sdg → vz(-π/2)
   *   t  → vz(π/4)
   *   ti/tdg → vz(-π/4)
   *   rz/vz → vz (pass through)
   *   gpi/gpi2 → pass through
   *
   * Two-qubit:
   *   cnot(a,b) → gpi2(π/2,a) · ms(0,0,a,b) · gpi2(3π/2,a) · vz(-π/2,a) · vz(-π/2,b)
   *   swap(a,b) → three CNOT decompositions (via above)
   *   ms → pass through
   *
   * Barriers pass through. All other gates throw TypeError.
   *
   * @param device  Target device name ('aria-1', 'forte-1', 'harmony').
   * @returns A new Circuit containing only native-gate ops.
   * @throws TypeError for unsupported gates or unknown device names.
   */
  compile(device) {
    const KNOWN_DEVICES = new Set(Object.keys(IONQ_DEVICES));
    if (!KNOWN_DEVICES.has(device)) {
      throw new TypeError(`compile: unknown device '${device}'. Known IonQ devices: ${[...KNOWN_DEVICES].join(", ")}`);
    }
    const PI = Math.PI;
    let result = new _Circuit(this.qubits, [], this.#cregs, this.#gates);
    const compileCnot = (c2, a, b) => {
      c2 = c2.gpi2(PI / 2, a);
      c2 = c2.gpi2(PI, b);
      c2 = c2.ms(0, 0, a, b);
      c2 = c2.gpi2(-PI / 2, a);
      c2 = c2.vz(-PI / 2, a);
      return c2;
    };
    for (const op of flattenOps(this.#ops)) {
      switch (op.kind) {
        case "barrier":
          result = result.barrier(...op.qubits);
          break;
        case "single": {
          const name = op.meta?.name ?? "?";
          const q = op.q;
          switch (name) {
            case "gpi":
              result = result.gpi(op.meta.params[0], q);
              break;
            case "gpi2":
              result = result.gpi2(op.meta.params[0], q);
              break;
            case "vz":
              result = result.vz(op.meta.params[0], q);
              break;
            case "rz":
              result = result.vz(op.meta.params[0], q);
              break;
            case "h":
              result = result.vz(PI / 2, q).gpi2(0, q).vz(PI / 2, q);
              break;
            case "x":
              result = result.gpi(0, q);
              break;
            case "y":
              result = result.gpi(PI / 2, q);
              break;
            case "z":
              result = result.vz(PI, q);
              break;
            case "s":
              result = result.vz(PI / 2, q);
              break;
            case "si":
            case "sdg":
              result = result.vz(-PI / 2, q);
              break;
            case "t":
              result = result.vz(PI / 4, q);
              break;
            case "ti":
            case "tdg":
              result = result.vz(-PI / 4, q);
              break;
            case "id":
              break;
            default:
              throw new TypeError(`compile: gate '${name}' cannot be compiled to ${device} native gates`);
          }
          break;
        }
        case "cnot":
          result = compileCnot(result, op.control, op.target);
          break;
        case "swap": {
          result = compileCnot(result, op.a, op.b);
          result = compileCnot(result, op.b, op.a);
          result = compileCnot(result, op.a, op.b);
          break;
        }
        case "two": {
          const name = op.meta?.name ?? "?";
          if (name === "ms") {
            result = result.ms(op.meta.params[0], op.meta.params[1], op.a, op.b);
          } else {
            throw new TypeError(`compile: gate '${name}' cannot be compiled to ${device} native gates`);
          }
          break;
        }
        case "measure":
          result = result.measure(op.q, op.creg, op.bit);
          break;
        case "reset":
          result = result.reset(op.q);
          break;
        default: {
          const name = op.meta?.name ?? op.kind;
          throw new TypeError(`compile: gate '${name}' cannot be compiled to ${device} native gates`);
        }
      }
    }
    return result;
  }
  /**
   * Simulate the circuit as an exact density matrix and return it.
   *
   * Unlike `run()` (which samples) and `statevector()` (which is pure-state only),
   * `dm()` computes the full ρ = |ψ⟩⟨ψ| evolution and applies optional per-gate
   * depolarizing noise channels exactly — no sampling, no variance.
   *
   * Only valid for pure circuits (no `measure` / `reset` / `if` ops).
   * Complexity: O(4ⁿ) — practical up to ~12 qubits.
   *
   * @param options.noise  Device name (any key of `DEVICES`, e.g. `'ibm_sherbrooke'`, `'h1-1'`) or
   *                       `{ p1?, p2? }` noise parameters.
   */
  dm(options) {
    if (this.#ops.some((op) => op.kind === "measure" || op.kind === "reset" || op.kind === "if")) {
      throw new TypeError("dm() requires a pure circuit \u2014 remove measure/reset/if ops");
    }
    const { noise } = options ?? {};
    const noiseParams = noise == null ? void 0 : typeof noise === "string" ? (() => {
      const p = DM_DEVICE_NOISE[noise];
      if (!p) throw new TypeError(`Unknown device profile '${noise}'. Known: ${Object.keys(DM_DEVICE_NOISE).join(", ")}`);
      return p;
    })() : noise;
    return runDM(flattenOps(this.#ops), this.qubits, noiseParams);
  }
};

// src/algorithms.ts
function qft(n) {
  let c2 = new Circuit(n);
  for (let j = n - 1; j >= 0; j--) {
    c2 = c2.h(j);
    for (let k = j - 1; k >= 0; k--) {
      c2 = c2.cu1(Math.PI / 2 ** (j - k), k, j);
    }
  }
  for (let i = 0; i < Math.floor(n / 2); i++) c2 = c2.swap(i, n - 1 - i);
  return c2;
}
function iqft(n) {
  let c2 = new Circuit(n);
  for (let i = 0; i < Math.floor(n / 2); i++) c2 = c2.swap(i, n - 1 - i);
  for (let j = 0; j < n; j++) {
    for (let k = j - 1; k >= 0; k--) c2 = c2.cu1(-Math.PI / 2 ** (j - k), k, j);
    c2 = c2.h(j);
  }
  return c2;
}
function groverAncilla(n) {
  return Math.max(0, n - 3);
}
function mcx(c2, controls, target, ancilla) {
  const n = controls.length;
  if (n === 0) return c2.x(target);
  if (n === 1) return c2.cnot(controls[0], target);
  if (n === 2) return c2.ccx(controls[0], controls[1], target);
  c2 = c2.ccx(controls[0], controls[1], ancilla[0]);
  for (let i = 2; i <= n - 2; i++) c2 = c2.ccx(controls[i], ancilla[i - 2], ancilla[i - 1]);
  c2 = c2.ccx(controls[n - 1], ancilla[n - 3], target);
  for (let i = n - 2; i >= 2; i--) c2 = c2.ccx(controls[i], ancilla[i - 2], ancilla[i - 1]);
  c2 = c2.ccx(controls[0], controls[1], ancilla[0]);
  return c2;
}
function groverDiffuse(c2, n, ancilla) {
  for (let q = 0; q < n; q++) c2 = c2.h(q);
  for (let q = 0; q < n; q++) c2 = c2.x(q);
  c2 = c2.h(n - 1);
  c2 = mcx(c2, Array.from({ length: n - 1 }, (_, i) => i), n - 1, ancilla);
  c2 = c2.h(n - 1);
  for (let q = 0; q < n; q++) c2 = c2.x(q);
  for (let q = 0; q < n; q++) c2 = c2.h(q);
  return c2;
}
function grover(n, oracle, iterations) {
  const anc = groverAncilla(n);
  const ancilla = Array.from({ length: anc }, (_, i) => n + i);
  const iters = iterations ?? Math.max(1, Math.round(Math.PI / 4 * Math.sqrt(2 ** n)));
  let c2 = new Circuit(n + anc);
  for (let q = 0; q < n; q++) c2 = c2.h(q);
  for (let i = 0; i < iters; i++) {
    c2 = oracle(c2);
    c2 = groverDiffuse(c2, n, ancilla);
  }
  return c2;
}
function appendIqft(c2, precision) {
  for (let i = 0; i < Math.floor(precision / 2); i++) c2 = c2.swap(i, precision - 1 - i);
  for (let j = 0; j < precision; j++) {
    for (let k = j - 1; k >= 0; k--) c2 = c2.cu1(-Math.PI / 2 ** (j - k), k, j);
    c2 = c2.h(j);
  }
  return c2;
}
function phaseEstimation(precision, unitary, targetQubits = 1) {
  const targets = Array.from({ length: targetQubits }, (_, i) => precision + i);
  let c2 = new Circuit(precision + targetQubits);
  for (let k = 0; k < precision; k++) c2 = c2.h(k);
  for (let k = 0; k < precision; k++) c2 = unitary(c2, k, 2 ** k, targets);
  return appendIqft(c2, precision);
}
function pauliEvolution(c2, n, ops, theta) {
  const active = [];
  for (let q = 0; q < n; q++) {
    const p = ops[n - 1 - q] ?? "I";
    if (p !== "I") active.push([q, p]);
  }
  if (active.length === 0) return c2;
  for (const [q, p] of active) {
    if (p === "X") c2 = c2.h(q);
    else if (p === "Y") c2 = c2.rx(Math.PI / 2, q);
  }
  for (let i = 0; i < active.length - 1; i++) c2 = c2.cnot(active[i][0], active[i + 1][0]);
  c2 = c2.rz(2 * theta, active[active.length - 1][0]);
  for (let i = active.length - 2; i >= 0; i--) c2 = c2.cnot(active[i][0], active[i + 1][0]);
  for (const [q, p] of active) {
    if (p === "X") c2 = c2.h(q);
    else if (p === "Y") c2 = c2.rx(-Math.PI / 2, q);
  }
  return c2;
}
function trotter(n, hamiltonian, t, steps = 1, order = 1) {
  for (const { ops } of hamiltonian) {
    if (ops.length !== n) throw new TypeError(`ops '${ops}' length must equal n (${n})`);
  }
  let c2 = new Circuit(n);
  const dt = t / steps;
  if (order === 1) {
    for (let s = 0; s < steps; s++)
      for (const { coeff, ops } of hamiltonian)
        c2 = pauliEvolution(c2, n, ops, coeff * dt);
  } else {
    const rev = hamiltonian.toReversed();
    for (let s = 0; s < steps; s++) {
      for (const { coeff, ops } of hamiltonian) c2 = pauliEvolution(c2, n, ops, coeff * dt / 2);
      for (const { coeff, ops } of rev) c2 = pauliEvolution(c2, n, ops, coeff * dt / 2);
    }
  }
  return c2;
}
function maxCutHamiltonian(n, edges) {
  if (edges.length === 0) return [];
  const terms = [{ coeff: edges.length / 2, ops: "I".repeat(n) }];
  for (const [u, v] of edges) {
    const arr = Array(n).fill("I");
    arr[n - 1 - u] = "Z";
    arr[n - 1 - v] = "Z";
    terms.push({ coeff: -0.5, ops: arr.join("") });
  }
  return terms;
}
function qaoa(n, edges, gamma, beta) {
  if (gamma.length !== beta.length) {
    throw new TypeError(`gamma and beta must have equal length (got ${gamma.length} vs ${beta.length})`);
  }
  let c2 = new Circuit(n);
  for (let q = 0; q < n; q++) c2 = c2.h(q);
  for (let l = 0; l < gamma.length; l++) {
    for (const [u, v] of edges) c2 = c2.cnot(u, v).rz(-gamma[l], v).cnot(u, v);
    for (let q = 0; q < n; q++) c2 = c2.rx(2 * beta[l], q);
  }
  return c2;
}
function vqe(ansatz, hamiltonian) {
  const n = ansatz.qubits;
  let energy = 0;
  for (const { coeff, ops } of hamiltonian) {
    if (ops.length !== n)
      throw new TypeError(`ops '${ops}' length must equal ansatz.qubits (${n})`);
    if (Math.abs(coeff) < 1e-15) continue;
    const upper = ops.toUpperCase();
    if (!/[XYZ]/.test(upper)) {
      energy += coeff;
      continue;
    }
    let rot = ansatz;
    for (let q = 0; q < n; q++) {
      const pauli = upper[n - 1 - q];
      if (pauli === "X") rot = rot.h(q);
      else if (pauli === "Y") rot = rot.rx(Math.PI / 2, q);
    }
    const probs = rot.exactProbs();
    let exp = 0;
    for (const [bs, prob] of Object.entries(probs)) {
      let parity = 0;
      for (let q = 0; q < n; q++) {
        const pauli = upper[n - 1 - q];
        if (pauli !== "I" && bs[q] === "1") parity ^= 1;
      }
      exp += (parity === 0 ? 1 : -1) * prob;
    }
    energy += coeff * exp;
  }
  return energy;
}
function makeAnsatz(paramCount, fn) {
  return Object.assign(fn, { paramCount });
}
function realAmplitudes(n, reps = 3) {
  const paramCount = n * (reps + 1);
  return makeAnsatz(paramCount, (params) => {
    if (params.length !== paramCount)
      throw new RangeError(`realAmplitudes(${n}, ${reps}) needs ${paramCount} params, got ${params.length}`);
    let c2 = new Circuit(n), p = 0;
    for (let r = 0; r < reps; r++) {
      for (let q = 0; q < n; q++) c2 = c2.ry(params[p++], q);
      for (let q = 0; q < n - 1; q++) c2 = c2.cnot(q, q + 1);
    }
    for (let q = 0; q < n; q++) c2 = c2.ry(params[p++], q);
    return c2;
  });
}
function efficientSU2(n, reps = 3) {
  const paramCount = 2 * n * (reps + 1);
  return makeAnsatz(paramCount, (params) => {
    if (params.length !== paramCount)
      throw new RangeError(`efficientSU2(${n}, ${reps}) needs ${paramCount} params, got ${params.length}`);
    let c2 = new Circuit(n), p = 0;
    for (let r = 0; r < reps; r++) {
      for (let q = 0; q < n; q++) {
        c2 = c2.ry(params[p++], q);
        c2 = c2.rz(params[p++], q);
      }
      for (let q = 0; q < n - 1; q++) c2 = c2.cnot(q, q + 1);
    }
    for (let q = 0; q < n; q++) {
      c2 = c2.ry(params[p++], q);
      c2 = c2.rz(params[p++], q);
    }
    return c2;
  });
}
var PAULI_IDX = { I: 0, X: 1, Y: 2, Z: 3 };
var PAULI_CHR = "IXYZ";
var PAULI_PROD = [
  [[0, 0], [1, 0], [2, 0], [3, 0]],
  // I·{I,X,Y,Z}
  [[1, 0], [0, 0], [3, 1], [2, 3]],
  // X·{I,X,Y,Z}: XX=I, XY=iZ, XZ=-iY
  [[2, 0], [3, 3], [0, 0], [1, 1]],
  // Y·{I,X,Y,Z}: YX=-iZ, YY=I, YZ=iX
  [[3, 0], [2, 1], [1, 3], [0, 0]]
  // Z·{I,X,Y,Z}: ZX=iY, ZY=-iX, ZZ=I
];
function addC(a, b) {
  return { re: a.re + b.re, im: a.im + b.im };
}
function mulC(a, b) {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}
function scaleC(c2, s) {
  return { re: c2.re * s, im: c2.im * s };
}
function phaseC(c2, exp) {
  switch (exp & 3) {
    case 1:
      return { re: -c2.im, im: c2.re };
    case 2:
      return { re: -c2.re, im: -c2.im };
    case 3:
      return { re: c2.im, im: -c2.re };
    default:
      return c2;
  }
}
var PauliOp = class _PauliOp {
  #terms;
  constructor(terms) {
    this.#terms = terms;
  }
  /** Construct from a real-coefficient Pauli Hamiltonian. */
  static from(terms) {
    return new _PauliOp(
      terms.map(({ ops, coeff }) => ({ ops: ops.toUpperCase(), coeff: { re: coeff, im: 0 } }))
    );
  }
  static #collect(terms) {
    const map = /* @__PURE__ */ new Map();
    for (const { ops, coeff } of terms) {
      const acc = map.get(ops);
      map.set(ops, acc ? addC(acc, coeff) : { ...coeff });
    }
    return new _PauliOp(
      [...map.entries()].filter(([, c2]) => Math.abs(c2.re) > 1e-15 || Math.abs(c2.im) > 1e-15).map(([ops, coeff]) => ({ ops, coeff }))
    );
  }
  /** A + B */
  add(other) {
    return _PauliOp.#collect([...this.#terms, ...other.#terms]);
  }
  /** c · A */
  scale(factor2) {
    return new _PauliOp(this.#terms.map((t) => ({ ...t, coeff: scaleC(t.coeff, factor2) })));
  }
  /** A · B  (Pauli string product with phase tracking) */
  mul(other) {
    const result = [];
    for (const a of this.#terms) {
      for (const b of other.#terms) {
        if (a.ops.length !== b.ops.length)
          throw new TypeError(`ops length mismatch: '${a.ops}' vs '${b.ops}'`);
        let phaseExp = 0, ops = "";
        for (let i = 0; i < a.ops.length; i++) {
          const [r, pe] = PAULI_PROD[PAULI_IDX[a.ops[i]]][PAULI_IDX[b.ops[i]]];
          phaseExp = phaseExp + pe & 3;
          ops += PAULI_CHR[r];
        }
        result.push({ ops, coeff: phaseC(mulC(a.coeff, b.coeff), phaseExp) });
      }
    }
    return _PauliOp.#collect(result);
  }
  /** [A, B] = AB − BA */
  commutator(other) {
    return this.mul(other).add(other.mul(this).scale(-1));
  }
  /**
   * Convert to `PauliTerm[]` for use with `vqe()`, `gradient()`, and `minimize()`.
   * Throws if any imaginary coefficients exceed `tol` (operator is not Hermitian).
   */
  toTerms(tol = 1e-10) {
    for (const { ops, coeff } of this.#terms)
      if (Math.abs(coeff.im) > tol)
        throw new TypeError(`PauliOp has imaginary coefficient on '${ops}' \u2014 operator is not Hermitian`);
    return this.#terms.map(({ ops, coeff }) => ({ ops, coeff: coeff.re }));
  }
};
function gradient(ansatz, hamiltonian, params) {
  const shift = Math.PI / 2;
  return Array.from({ length: params.length }, (_, i) => {
    const plus = params.map((p, j) => j === i ? p + shift : p);
    const minus = params.map((p, j) => j === i ? p - shift : p);
    return 0.5 * (vqe(ansatz(plus), hamiltonian) - vqe(ansatz(minus), hamiltonian));
  });
}
function minimize(ansatz, hamiltonian, initialParams, { lr = 0.1, steps = 200, tol = 1e-6 } = {}) {
  let params = [...initialParams];
  for (let step = 0; step < steps; step++) {
    const grad = gradient(ansatz, hamiltonian, params);
    const gradNorm = Math.sqrt(grad.reduce((s, g) => s + g * g, 0));
    if (gradNorm < tol) {
      return { params, energy: vqe(ansatz(params), hamiltonian), steps: step, converged: true, truncated: false };
    }
    params = params.map((p, i) => p - lr * grad[i]);
  }
  return { params, energy: vqe(ansatz(params), hamiltonian), steps, converged: false, truncated: false };
}
function gradientMps(ansatz, hamiltonian, params, { maxBond = 64, truncErr = 0 } = {}) {
  const shift = Math.PI / 2;
  const opts = { maxBond, truncErr };
  let truncated = false;
  const gradient2 = Array.from({ length: params.length }, (_, i) => {
    const plus = params.map((p, j) => j === i ? p + shift : p);
    const minus = params.map((p, j) => j === i ? p - shift : p);
    const ep = ansatz(plus).expectMps(hamiltonian, opts);
    const em = ansatz(minus).expectMps(hamiltonian, opts);
    truncated = truncated || ep.truncated || em.truncated;
    return 0.5 * (ep.energy - em.energy);
  });
  return { gradient: gradient2, truncated };
}
function minimizeMps(ansatz, hamiltonian, initialParams, { lr = 0.1, steps = 200, tol = 1e-6, maxBond = 64, truncErr = 0 } = {}) {
  let params = [...initialParams];
  const opts = { maxBond, truncErr };
  let truncated = false;
  for (let step = 0; step < steps; step++) {
    const { gradient: grad, truncated: gradTrunc } = gradientMps(ansatz, hamiltonian, params, opts);
    truncated = truncated || gradTrunc;
    const gradNorm = Math.sqrt(grad.reduce((s, g) => s + g * g, 0));
    if (gradNorm < tol) {
      const { energy: energy2, truncated: eTrunc2 } = ansatz(params).expectMps(hamiltonian, opts);
      return { params, energy: energy2, steps: step, converged: true, truncated: truncated || eTrunc2 };
    }
    params = params.map((p, i) => p - lr * grad[i]);
  }
  const { energy, truncated: eTrunc } = ansatz(params).expectMps(hamiltonian, opts);
  return { params, energy, steps, converged: false, truncated: truncated || eTrunc };
}

// src/beauregard.ts
function reduceAngle(theta) {
  const TWO_PI = 2 * Math.PI;
  let t = theta % TWO_PI;
  if (t > Math.PI) t -= TWO_PI;
  if (t <= -Math.PI) t += TWO_PI;
  return t;
}
function modPow(base, exp, m) {
  if (m === 1n) return 0n;
  let result = 1n;
  let b = base % m;
  let e = exp;
  while (e > 0n) {
    if (e & 1n) result = result * b % m;
    b = b * b % m;
    e >>= 1n;
  }
  return result;
}
function extGcd(a, b) {
  if (b === 0n) return [a, 1n, 0n];
  const [g, x, y] = extGcd(b, a % b);
  return [g, y, x - a / b * y];
}
function modInverse(a, m) {
  const [, x] = extGcd((a % m + m) % m, m);
  return (x % m + m) % m;
}
function gcd(a, b) {
  while (b) {
    [a, b] = [b, a % b];
  }
  return a;
}
function continuedFractions(measured, precision, N) {
  let num = BigInt(measured);
  let den = 1n << BigInt(precision);
  const convergents = [];
  let h0 = 0n, h1 = 1n, k0 = 1n, k1 = 0n;
  while (den > 0n) {
    const a = num / den;
    [h0, h1] = [h1, a * h1 + h0];
    [k0, k1] = [k1, a * k1 + k0];
    if (k1 > N) break;
    convergents.push([h1, k1]);
    [num, den] = [den, num - a * den];
  }
  for (let i = convergents.length - 1; i >= 0; i--) {
    const r = convergents[i][1];
    if (r > 0n && r <= N) return r;
  }
  return 1n;
}
function applyQft(c2, n, offset) {
  for (let j = n - 1; j >= 0; j--) {
    c2 = c2.h(offset + j);
    for (let k = j - 1; k >= 0; k--) {
      c2 = c2.cu1(Math.PI / 2 ** (j - k), offset + k, offset + j);
    }
  }
  for (let i = 0; i < Math.floor(n / 2); i++) c2 = c2.swap(offset + i, offset + n - 1 - i);
  return c2;
}
function applyIqft(c2, n, offset) {
  for (let i = 0; i < Math.floor(n / 2); i++) c2 = c2.swap(offset + i, offset + n - 1 - i);
  for (let j = 0; j < n; j++) {
    for (let k = j - 1; k >= 0; k--) c2 = c2.cu1(-Math.PI / 2 ** (j - k), offset + k, offset + j);
    c2 = c2.h(offset + j);
  }
  return c2;
}
function phiAdd(c2, n, a, offset = 0) {
  const mod = 1n << BigInt(n);
  const aPos = (a % mod + mod) % mod;
  for (let j = 0; j < n; j++) {
    const power = n - j;
    const div = 1n << BigInt(power);
    const aRed = Number(aPos % div);
    const angle = reduceAngle(2 * Math.PI * aRed / Number(div));
    if (Math.abs(angle) > 1e-12) c2 = c2.u1(angle, offset + j);
  }
  return c2;
}
function cPhiAdd(c2, n, a, ctrl, offset = 0) {
  const mod = 1n << BigInt(n);
  const aPos = (a % mod + mod) % mod;
  for (let j = 0; j < n; j++) {
    const power = n - j;
    const div = 1n << BigInt(power);
    const aRed = Number(aPos % div);
    const angle = reduceAngle(2 * Math.PI * aRed / Number(div));
    if (Math.abs(angle) > 1e-12) c2 = c2.cu1(angle, ctrl, offset + j);
  }
  return c2;
}
function ccPhiAdd(c2, n, a, ctrl1, ctrl2, offset = 0) {
  const mod = 1n << BigInt(n);
  const aPos = (a % mod + mod) % mod;
  for (let j = 0; j < n; j++) {
    const power = n - j;
    const div = 1n << BigInt(power);
    const aRed = Number(aPos % div);
    const theta = reduceAngle(2 * Math.PI * aRed / Number(div));
    if (Math.abs(theta) <= 1e-12) continue;
    const half = theta / 2;
    const tgt = offset + j;
    c2 = c2.cu1(half, ctrl1, tgt);
    c2 = c2.cnot(ctrl1, ctrl2);
    c2 = c2.cu1(-half, ctrl2, tgt);
    c2 = c2.cnot(ctrl1, ctrl2);
    c2 = c2.cu1(half, ctrl2, tgt);
  }
  return c2;
}
function ccPhiAddMod(c2, n, a, N, ctrl1, ctrl2, bOff, ancilla) {
  const nb = n + 1;
  const sign = bOff + nb - 1;
  c2 = ccPhiAdd(c2, nb, a, ctrl1, ctrl2, bOff);
  c2 = phiAdd(c2, nb, -N, bOff);
  c2 = applyIqft(c2, nb, bOff);
  c2 = c2.cnot(sign, ancilla);
  c2 = applyQft(c2, nb, bOff);
  c2 = cPhiAdd(c2, nb, N, ancilla, bOff);
  c2 = ccPhiAdd(c2, nb, -a, ctrl1, ctrl2, bOff);
  c2 = applyIqft(c2, nb, bOff);
  c2 = c2.x(sign);
  c2 = c2.cnot(sign, ancilla);
  c2 = c2.x(sign);
  c2 = applyQft(c2, nb, bOff);
  c2 = ccPhiAdd(c2, nb, a, ctrl1, ctrl2, bOff);
  return c2;
}
function cMultModAdd(c2, n, a, N, ctrl, xOff, bOff, ancilla) {
  c2 = applyQft(c2, n + 1, bOff);
  let aShifted = a % N;
  for (let j = 0; j < n; j++) {
    c2 = ccPhiAddMod(c2, n, aShifted, N, ctrl, xOff + j, bOff, ancilla);
    aShifted = aShifted * 2n % N;
  }
  c2 = applyIqft(c2, n + 1, bOff);
  return c2;
}
function beauregardU(c2, n, a, aInv, N, ctrl, xOff, bOff, ancilla) {
  c2 = cMultModAdd(c2, n, a, N, ctrl, xOff, bOff, ancilla);
  for (let j = 0; j < n; j++) c2 = c2.cswap(ctrl, xOff + j, bOff + j);
  c2 = cMultModAdd(c2, n, N - aInv, N, ctrl, xOff, bOff, ancilla);
  return c2;
}
function factor(N) {
  return shorBeauregard(BigInt(N)).factors;
}
function shorBeauregard(N, opts = {}) {
  const n = Math.ceil(Math.log2(Number(N)));
  const precision = opts.precision ?? 2 * n + 1;
  const shots = opts.shots ?? 1;
  const maxTries = opts.maxAttempts ?? 20;
  const truncErr = opts.truncErr ?? 0;
  const totalQ = precision + 2 * n + 2;
  const xOff = precision;
  const bOff = precision + n;
  const ancilla = precision + 2 * n + 1;
  if (N < 4n) throw new RangeError("N must be \u2265 4");
  if (N % 2n === 0n) return { factor: 2n, factors: [2n, N / 2n], a: 0n, period: void 0, attempts: 0, qubits: totalQ };
  const Nnum = Number(N);
  for (let attempt = 1; attempt <= maxTries; attempt++) {
    const aCand = opts.a ?? BigInt(2 + Math.floor(Math.random() * (Nnum - 3)));
    const g = gcd(aCand, N);
    if (g > 1n) {
      return { factor: g, factors: [g, N / g], a: aCand, period: void 0, attempts: attempt, qubits: totalQ };
    }
    const a = aCand;
    const aInv = modInverse(a, N);
    let c2 = new Circuit(totalQ);
    for (let k = 0; k < precision; k++) c2 = c2.h(k);
    c2 = c2.x(xOff);
    let ak = a % N;
    for (let k = 0; k < precision; k++) {
      const akInv = modInverse(ak, N);
      c2 = beauregardU(c2, n, ak, akInv, N, k, xOff, bOff, ancilla);
      ak = ak * ak % N;
    }
    c2 = applyIqft(c2, precision, 0);
    const dist = c2.runMps({ shots, ...opts.seed !== void 0 && { seed: opts.seed }, truncErr });
    const measured2n = 1 << precision;
    const candidates = /* @__PURE__ */ new Set();
    for (const bs of Object.keys(dist.probs)) {
      let val = 0;
      for (let i = 0; i < precision; i++) if (bs[i] === "1") val |= 1 << i;
      if (val === 0) continue;
      const r = continuedFractions(val, precision, N);
      candidates.add(r);
    }
    for (const r of candidates) {
      if (r === 0n || r > N) continue;
      if (modPow(a, r, N) !== 1n) continue;
      if (r % 2n !== 0n) continue;
      const halfPow = modPow(a, r / 2n, N);
      if (halfPow === N - 1n) continue;
      const f1 = gcd(halfPow + 1n, N);
      const f2 = gcd(halfPow - 1n, N);
      for (const f of [f1, f2]) {
        if (f > 1n && f < N) {
          return { factor: f, factors: [f, N / f], a, period: r, attempts: attempt, qubits: totalQ };
        }
      }
    }
  }
  return { factor: void 0, factors: void 0, a: opts.a ?? 0n, period: void 0, attempts: maxTries, qubits: totalQ };
}
export {
  Circuit,
  CliffordSim,
  DEVICES,
  DensityMatrix,
  Distribution,
  Gpi,
  Gpi2,
  H,
  I,
  IONQ_DEVICES,
  ISwap,
  Id,
  Ms,
  ONE,
  PauliOp,
  R2,
  R4,
  R8,
  Rx,
  Ry,
  Rz,
  S,
  Si,
  SrSwap,
  T,
  Ti,
  U1,
  U2,
  U3,
  V,
  Vi,
  X,
  Xx,
  Xy,
  Y,
  Yy,
  Z,
  ZERO,
  Zz,
  add,
  applyIqft,
  applyQft,
  c,
  conj,
  continuedFractions,
  efficientSU2,
  factor,
  gcd,
  gradient,
  gradientMps,
  grover,
  groverAncilla,
  iqft,
  maxCutHamiltonian,
  minimize,
  minimizeMps,
  modInverse,
  modPow,
  mul,
  norm2,
  phaseEstimation,
  phiAdd,
  qaoa,
  qft,
  realAmplitudes,
  scale,
  shorBeauregard,
  trotter,
  vqe
};
