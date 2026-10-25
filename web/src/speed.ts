// Speed model: a line-for-line port of src/amagi_inference_calc/speed.py
// (plus the bits of models.py and quant.py it needs), so the web page and the
// CLI give the same numbers. tests/test_web_speed_parity.py checks that.
// Pure functions only: the data files are passed in by data.ts.

export interface ModelSpec {
  id: string;
  name: string;
  params: number;
  embed_params: number;
  tied_embeddings: boolean;
  expert_params: number;
  experts_total: number;
  experts_used: number;
  layers: number;
  full_attn_layers: number;
  window_attn_layers: number;
  window: number;
  attn_heads: number;
  kv_heads: number;
  head_dim_k: number;
  head_dim_v: number;
  context_max: number;
  native_quant: string;
  source: string;
  file_quant: string;
  file_bytes: number;
  read_bytes: number;
  notes: string[];
}

export interface GPU {
  id: string;
  name: string;
  vendor: string;
  kind: string;
  vram_gb: number;
  bw_spec_gbs: number;
  bw_measured_gbs?: number | null;
  price_used_usd?: number | null;
  price_source?: string | null;
  memory_gb?: number | null;
  tflops?: number | null;
  bw_decode_gbs?: number | null;
  vram_note?: string | null;
  note?: string | null;
}

export interface Calibration {
  gpu: string;
  bw_ref_gbs: number;
  bw_spec_gbs: number;
  bw_eff_gbs: number;
  overhead_ms: number;
  moe_layer_ms: number;
  batch_ms_per_b_dense: number;
  batch_ms_per_layer_moe: number;
  tflops_dense: number;
  tflops_moe: number;
}

export interface OfficePreset {
  id: string;
  name: string;
  staff: number;
  users: number;
  context: number;
  prompt: number;
  use: string;
}

const GIB = 1024 ** 3;
export const RUNTIME_OVERHEAD_GIB = 0.5;

// -- quant.py ----------------------------------------------------------------

export const QUANT_BPW: Record<string, number> = {
  F32: 32.0,
  F16: 16.0,
  BF16: 16.0,
  Q8_0: 8.5,
  Q6_K: 6.5625,
  Q5_K_M: 5.75,
  Q5_K_S: 5.54,
  Q5_0: 5.5,
  Q4_K_M: 4.88,
  Q4_K_S: 4.58,
  Q4_0: 4.5,
  IQ4_XS: 4.25,
  MXFP4: 4.25,
  Q3_K_M: 3.91,
  Q3_K_S: 3.5,
  Q2_K: 3.0,
  AWQ: 4.25,
  GPTQ: 4.25,
  FP8: 8.0,
};

export const KV_BYTES: Record<string, number> = {
  F32: 4.0,
  F16: 2.0,
  BF16: 2.0,
  Q8_0: 34 / 32,
  Q5_1: 24 / 32,
  Q4_0: 18 / 32,
  FP8: 1.0,
};

const ALIASES: Record<string, string> = {
  Q4: 'Q4_K_M',
  Q5: 'Q5_K_M',
  Q6: 'Q6_K',
  Q8: 'Q8_0',
  Q3: 'Q3_K_M',
  Q2: 'Q2_K',
  FP16: 'F16',
  FP32: 'F32',
  INT8: 'Q8_0',
  INT4: 'Q4_K_M',
  Q4_K: 'Q4_K_M',
  Q5_K: 'Q5_K_M',
  Q3_K: 'Q3_K_M',
  MXFP4_MOE: 'MXFP4',
};

export function normalizeQuant(name: string): string {
  let key = name.trim().toUpperCase().replace(/-/g, '_');
  if (key.startsWith('UD_')) key = key.slice(3);
  key = ALIASES[key] ?? key;
  if (!(key in QUANT_BPW)) throw new Error(`unknown quant ${name}`);
  return key;
}

// -- models.py ---------------------------------------------------------------

export function activeParams(m: ModelSpec): number {
  const frac = m.experts_total ? m.experts_used / m.experts_total : 1.0;
  let dense = m.params - m.expert_params;
  if (!m.tied_embeddings) dense -= m.embed_params;
  return Math.trunc(dense + m.expert_params * frac);
}

function isExact(m: ModelSpec, quant: string): boolean {
  if (!m.file_quant) return false;
  try {
    return normalizeQuant(quant) === normalizeQuant(m.file_quant);
  } catch {
    return false;
  }
}

export function weightBytes(m: ModelSpec, quant: string): number {
  if (isExact(m, quant) && m.file_bytes) return m.file_bytes;
  return (m.params * QUANT_BPW[normalizeQuant(quant)]) / 8;
}

function expertUnion(m: ModelSpec, streams: number): number {
  if (!m.experts_total) return 1.0;
  const p = m.experts_used / m.experts_total;
  return 1.0 - (1.0 - p) ** Math.max(1, streams);
}

export function readBytesPerStep(
  m: ModelSpec,
  quant: string,
  streams = 1
): number {
  const bpw = QUANT_BPW[normalizeQuant(quant)] / 8;
  if (isExact(m, quant) && m.read_bytes) {
    if (streams > 1 && m.experts_total) {
      const one = expertUnion(m, 1);
      const expOne = m.expert_params * bpw * one;
      return m.read_bytes - expOne + (expOne / one) * expertUnion(m, streams);
    }
    return m.read_bytes;
  }
  let dense = m.params - m.expert_params;
  if (!m.tied_embeddings) dense -= m.embed_params;
  return (dense + m.expert_params * expertUnion(m, streams)) * bpw;
}

export function kvBytes(m: ModelSpec, context: number, kvType = 'F16'): number {
  const perLayerTok =
    m.kv_heads * (m.head_dim_k + m.head_dim_v) * KV_BYTES[kvType];
  const full = m.full_attn_layers * perLayerTok * context;
  const windowed =
    m.window_attn_layers * perLayerTok * Math.min(context, m.window || context);
  return full + windowed;
}

// -- speed.py ----------------------------------------------------------------

export function decodeEff(cal: Calibration): number {
  return cal.bw_eff_gbs / cal.bw_spec_gbs;
}

function fmtG(x: number): string {
  return String(Number(x.toPrecision(6)));
}

export function bandwidth(gpu: GPU, cal: Calibration): [number, string] {
  if (gpu.bw_decode_gbs)
    return [gpu.bw_decode_gbs * 1e9, 'fitted from your benchmark run'];
  if (gpu.id === cal.gpu)
    return [cal.bw_eff_gbs * 1e9, 'calibrated on this GPU'];
  if (gpu.bw_measured_gbs) {
    const scale = cal.bw_eff_gbs / cal.bw_ref_gbs;
    return [
      gpu.bw_measured_gbs * scale * 1e9,
      `memtest ${fmtG(gpu.bw_measured_gbs)} GB/s x ${scale.toFixed(2)}`,
    ];
  }
  return [
    gpu.bw_spec_gbs * decodeEff(cal) * 1e9,
    `spec x ${decodeEff(cal).toFixed(2)}`,
  ];
}

function moeLayers(m: ModelSpec): number {
  return m.experts_total ? m.layers : 0;
}

export function attentionFlops(m: ModelSpec, prompt: number): number {
  const width = (m.attn_heads * (m.head_dim_k + m.head_dim_v)) / 2;
  const full = 2.0 * prompt * prompt * width * m.full_attn_layers;
  const w = m.window ? Math.min(prompt, m.window) : prompt;
  const windowed =
    4.0 *
    prompt *
    w *
    width *
    m.window_attn_layers *
    (w === prompt ? 0.5 : 1.0);
  return full + windowed;
}

export function decodeStepS(
  m: ModelSpec,
  quant: string,
  context: number,
  streams: number,
  bw: number,
  cal: Calibration,
  kvType = 'F16'
): number {
  const mem =
    readBytesPerStep(m, quant, streams) + streams * kvBytes(m, context, kvType);
  let t =
    mem / bw + cal.overhead_ms / 1e3 + (cal.moe_layer_ms / 1e3) * moeLayers(m);
  const perStream = m.experts_total
    ? cal.batch_ms_per_layer_moe * moeLayers(m)
    : (cal.batch_ms_per_b_dense * activeParams(m)) / 1e9;
  t += (Math.max(0, streams - 1) * perStream) / 1e3;
  return t;
}

export function prefillS(
  m: ModelSpec,
  prompt: number,
  tflopsDense: number,
  tflopsMoe?: number
): number {
  const linF = (m.experts_total && tflopsMoe ? tflopsMoe : tflopsDense) * 1e12;
  return (
    (2.0 * activeParams(m) * prompt) / linF +
    attentionFlops(m, prompt) / (tflopsDense * 1e12)
  );
}

export interface Estimate {
  model: string;
  quant: string;
  gpu: string;
  gpu_count: number;
  users: number;
  context: number;
  kv_type: string;
  weights_gib: number;
  kv_gib: number;
  vram_needed_gib: number;
  vram_total_gib: number;
  fits: boolean;
  cards_needed: number;
  bandwidth_gbs: number;
  bandwidth_source: string;
  tok_s_single: number;
  tok_s_per_user: number;
  tok_s_total: number;
  prefill_tok_s: number | null;
  ttft_s: number | null;
  ttft_all_s: number | null;
  prompt: number;
  price_usd: number | null;
  usd_per_tok_s: number | null;
  warnings: string[];
}

export interface EstimateOptions {
  quant?: string;
  users?: number;
  context?: number;
  prompt?: number;
  gpuCount?: number;
  kvType?: string;
  tflops?: number;
  price?: number | null;
}

export function estimate(
  model: ModelSpec,
  gpu: GPU,
  cal: Calibration,
  o: EstimateOptions = {}
): Estimate {
  const quant = o.quant || model.native_quant || 'Q4_K_M';
  const users = Math.max(1, Math.trunc(o.users ?? 1));
  const gpuCount = Math.max(1, Math.trunc(o.gpuCount ?? 1));
  const context = o.context ?? 8192;
  const prompt =
    o.prompt == null ? Math.floor(context / 2) : Math.min(o.prompt, context);
  const kvType = o.kvType ?? 'F16';
  const warnings: string[] = [];

  const weights = weightBytes(model, quant);
  const kv = users * kvBytes(model, context, kvType);
  const need = weights + kv + RUNTIME_OVERHEAD_GIB * GIB * gpuCount;
  const vram = gpu.vram_gb * GIB;
  const total = vram * gpuCount;
  const fits = need <= total;
  const perCard = vram - RUNTIME_OVERHEAD_GIB * GIB;
  const cards =
    perCard > 0 ? Math.max(1, Math.ceil((weights + kv) / perCard)) : 0;
  if (model.context_max && context > model.context_max) {
    warnings.push(
      `context ${context} is above the model's trained ${model.context_max}`
    );
  }
  if (!fits) {
    warnings.push(
      `does not fit: needs ${(need / GIB).toFixed(1)} GiB, has ${(total / GIB).toFixed(1)} GiB (${cards} card(s) needed)`
    );
  }
  if (gpuCount > 1) {
    warnings.push(
      'multi-GPU assumes llama.cpp layer split: memory adds up, speed does not (cards take turns)'
    );
  }
  if (gpu.note) warnings.push(gpu.note);
  if (['amd', 'intel', 'apple'].includes(gpu.vendor)) {
    warnings.push(
      `${gpu.vendor.toUpperCase()} speed is scaled from an NVIDIA/CUDA calibration; real numbers often land lower`
    );
  }

  const [bw, bwSrc] = bandwidth(gpu, cal);
  const single = 1.0 / decodeStepS(model, quant, prompt, 1, bw, cal, kvType);
  const perUser =
    1.0 / decodeStepS(model, quant, prompt, users, bw, cal, kvType);

  const flops = o.tflops
    ? o.tflops
    : gpu.id === cal.gpu
      ? cal.tflops_dense
      : gpu.tflops;
  let pp: number | null = null;
  let ttft: number | null = null;
  let ttftAll: number | null = null;
  if (flops) {
    const fMoe = cal.tflops_moe * (flops / cal.tflops_dense);
    const tPp = prefillS(model, prompt, flops, fMoe);
    pp = tPp > 0 ? prompt / tPp : null;
    ttft = tPp;
    ttftAll = tPp * users;
  }

  const price =
    o.price === undefined || o.price === null
      ? (gpu.price_used_usd ?? null)
      : o.price;
  const totalPrice = price !== null ? price * gpuCount : null;
  const usd = totalPrice && fits ? totalPrice / (perUser * users) : null;
  if (users > 8)
    warnings.push(
      'more than 8 users is beyond the measured range (1-8 streams)'
    );

  return {
    model: model.id,
    quant,
    gpu: gpu.id,
    gpu_count: gpuCount,
    users,
    context,
    kv_type: kvType,
    weights_gib: weights / GIB,
    kv_gib: kv / GIB,
    vram_needed_gib: need / GIB,
    vram_total_gib: total / GIB,
    fits,
    cards_needed: cards,
    bandwidth_gbs: bw / 1e9,
    bandwidth_source: bwSrc,
    tok_s_single: single,
    tok_s_per_user: perUser,
    tok_s_total: perUser * users,
    prefill_tok_s: pp,
    ttft_s: ttft,
    ttft_all_s: ttftAll,
    prompt,
    price_usd: totalPrice,
    usd_per_tok_s: usd,
    warnings,
  };
}

// -- cli.py `office` -----------------------------------------------------------

export const OFFICE_MODELS = [
  'qwen3-8b',
  'qwen3-14b',
  'gpt-oss-20b',
  'gemma-3-27b',
  'qwen3-30b-a3b',
  'qwen3-32b',
  'llama-3.3-70b',
];

export interface OfficeOptions {
  minTokS?: number;
  maxCards?: number;
  top?: number;
  allVendors?: boolean;
  prices?: Record<string, number>;
}

export interface OfficePick extends Estimate {
  model_name: string;
  gpu_name: string;
  caveat: boolean;
}

/** Cheapest setups per model that fit and give each user at least minTokS. */
export function officePicks(
  preset: Pick<OfficePreset, 'users' | 'context' | 'prompt'>,
  models: ModelSpec[],
  gpus: GPU[],
  cal: Calibration,
  o: OfficeOptions = {}
): Record<string, OfficePick[]> {
  const minTokS = o.minTokS ?? 15;
  const maxCards = o.maxCards ?? 4;
  const top = o.top ?? 3;
  const prices = o.prices ?? {};
  let pool = gpus.filter((g) => g.price_used_usd != null || g.id in prices);
  if (!o.allVendors) pool = pool.filter((g) => g.vendor === 'nvidia');
  const rows: OfficePick[] = [];
  for (const m of models) {
    for (const g of pool) {
      for (let n = 1; n <= (g.kind === 'unified' ? 1 : maxCards); n++) {
        const e = estimate(m, g, cal, {
          users: preset.users,
          context: preset.context,
          prompt: preset.prompt,
          gpuCount: n,
          price: prices[g.id],
        });
        if (!e.fits) continue;
        if (e.tok_s_per_user >= minTokS) {
          const caveat = e.warnings.some(
            (w) => !w.startsWith('multi-GPU') && !w.startsWith('more than 8')
          );
          rows.push({ ...e, model_name: m.name, gpu_name: g.name, caveat });
        }
        break; // more cards never makes a layer split faster
      }
    }
  }
  rows.sort((a, b) => {
    const pa = a.price_usd,
      pb = b.price_usd;
    if ((pa === null) !== (pb === null)) return pa === null ? 1 : -1;
    if ((pa ?? 0) !== (pb ?? 0)) return (pa ?? 0) - (pb ?? 0);
    return b.tok_s_per_user - a.tok_s_per_user;
  });
  const best: Record<string, OfficePick[]> = {};
  for (const r of rows) (best[r.model] ??= []).push(r);
  for (const k of Object.keys(best)) best[k] = best[k].slice(0, top);
  return best;
}
