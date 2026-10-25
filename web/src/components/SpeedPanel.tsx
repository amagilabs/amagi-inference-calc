import { useMemo, useRef, useState } from 'react';
import { CAL, GPUS, MODELS, PRESETS } from '../data';
import {
  activeParams,
  estimate,
  officePicks,
  OFFICE_MODELS,
  QUANT_BPW,
  type GPU,
} from '../speed';
import { billions, fmtGiB, fmtSec, fmtTok, fmtTokens, fmtUsd } from '../format';
import { Field } from './Field';

const CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072];
const KV_TYPES = ['F16', 'Q8_0', 'Q4_0'];
const VENDORS: [string, string][] = [
  ['nvidia', 'NVIDIA'],
  ['amd', 'AMD'],
  ['intel', 'Intel'],
  ['apple', 'Apple'],
];
const OFFICE_SET = MODELS.filter((m) => OFFICE_MODELS.includes(m.id));

const clampInt = (v: string, lo: number, hi: number, fallback: number) => {
  const n = Math.trunc(Number(v));
  return Number.isFinite(n) && v !== ''
    ? Math.min(hi, Math.max(lo, n))
    : fallback;
};

const gpuLabel = (g: GPU) =>
  `${g.name}, ${g.vram_gb} GB${g.price_used_usd ? `, about ${fmtUsd(g.price_used_usd)} used` : ''}`;

export const SpeedPanel = () => {
  const start = PRESETS.find((p) => p.id === 'team-5') ?? PRESETS[0];
  const [presetId, setPresetId] = useState(start.id);
  const [users, setUsers] = useState(start.users);
  const [context, setContext] = useState(start.context);
  const [prompt, setPrompt] = useState(start.prompt);
  const [modelId, setModelId] = useState('qwen3-14b');
  const [quant, setQuant] = useState('');
  const [kvType, setKvType] = useState('F16');
  const [gpuId, setGpuId] = useState('rtx-3090');
  const [count, setCount] = useState(1);
  const [price, setPrice] = useState('');
  const [minTokS, setMinTokS] = useState(15);
  const [allVendors, setAllVendors] = useState(false);
  const [showMore, setShowMore] = useState(false);
  const resultsRef = useRef<HTMLHeadingElement>(null);

  const preset = PRESETS.find((p) => p.id === presetId);
  const model = MODELS.find((m) => m.id === modelId) ?? MODELS[0];
  const gpu = GPUS.find((g) => g.id === gpuId) ?? GPUS[0];
  const unified = gpu.kind === 'unified';
  const cards = unified ? 1 : count;
  const priceNum = price.trim() === '' ? null : Math.max(0, Number(price));

  const est = estimate(model, gpu, CAL, {
    quant: quant || undefined,
    users,
    context,
    prompt,
    gpuCount: cards,
    kvType,
    price: priceNum !== null && Number.isFinite(priceNum) ? priceNum : null,
  });
  const fillPct = Math.min(
    100,
    (est.vram_needed_gib / est.vram_total_gib) * 100
  );

  const picks = useMemo(
    () =>
      officePicks({ users, context, prompt }, OFFICE_SET, GPUS, CAL, {
        minTokS,
        allVendors,
        top: showMore ? 3 : 1,
      }),
    [users, context, prompt, minTokS, allVendors, showMore]
  );
  const pickRows = OFFICE_SET.flatMap((m) =>
    (picks[m.id] ?? []).map((p, i) => ({ ...p, first: i === 0 }))
  );
  const noPick = OFFICE_SET.filter((m) => !picks[m.id]);

  const choosePreset = (id: string) => {
    setPresetId(id);
    const p = PRESETS.find((x) => x.id === id);
    if (p) {
      setUsers(p.users);
      setContext(p.context);
      setPrompt(p.prompt);
    }
  };

  const loadPick = (m: string, g: string, n: number) => {
    setModelId(m);
    setGpuId(g);
    setCount(n);
    setQuant('');
    setKvType('F16');
    setPrice('');
    resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    resultsRef.current?.focus({ preventScroll: true });
  };

  return (
    <>
      <div className="layout">
        <form
          className="panel input-panel"
          onSubmit={(e) => e.preventDefault()}
        >
          <h2 className="section-title">Your office</h2>
          <Field
            id="sp-preset"
            label="Starting point"
            hint={
              preset
                ? `${preset.staff} staff: ${preset.use}.`
                : 'Your own numbers.'
            }
            tip="Users is how many people send a request at the same moment, not head count. The presets assume about a quarter to a third of staff hit enter together."
          >
            <select
              id="sp-preset"
              value={presetId}
              onChange={(e) => choosePreset(e.target.value)}
            >
              {PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
              <option value="custom">Custom</option>
            </select>
          </Field>

          <div className="field-row">
            <Field
              id="sp-users"
              label="People at once"
              tip="Requests running at the same moment. Each one gets its own KV cache."
            >
              <input
                id="sp-users"
                type="number"
                inputMode="numeric"
                min={1}
                max={32}
                value={users}
                onChange={(e) => {
                  setUsers(clampInt(e.target.value, 1, 32, users));
                  setPresetId('custom');
                }}
              />
            </Field>
            <Field
              id="sp-context"
              label="Context per person"
              tip="The token window each person gets. It sets how much KV cache memory to reserve."
            >
              <select
                id="sp-context"
                value={context}
                onChange={(e) => {
                  const c = Number(e.target.value);
                  setContext(c);
                  setPrompt(Math.min(prompt, c));
                  setPresetId('custom');
                }}
              >
                {CONTEXTS.map((c) => (
                  <option key={c} value={c}>
                    {fmtTokens(c)} tokens
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <Field
            id="sp-prompt"
            label="Typical request (tokens)"
            tip="How much text a request sends in: question plus pasted documents. A page of text is roughly 600 to 800 tokens. Sets time to first token."
          >
            <input
              id="sp-prompt"
              type="number"
              inputMode="numeric"
              min={1}
              max={context}
              step={100}
              value={prompt}
              onChange={(e) => {
                setPrompt(clampInt(e.target.value, 1, context, prompt));
                setPresetId('custom');
              }}
            />
          </Field>

          <h2 className="section-title">Model</h2>
          <Field id="sp-model" label="Model">
            <select
              id="sp-model"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
            >
              {MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({billions(m.params)}
                  {m.experts_total
                    ? `, ${billions(activeParams(m))} active`
                    : ''}
                  )
                </option>
              ))}
            </select>
          </Field>
          <div className="field-row">
            <Field
              id="sp-quant"
              label="Weights format"
              tip="Q4_K_M is the usual llama.cpp download. Bigger formats cost memory and speed for a little quality."
            >
              <select
                id="sp-quant"
                value={quant}
                onChange={(e) => setQuant(e.target.value)}
              >
                <option value="">
                  Default ({model.native_quant || 'Q4_K_M'})
                </option>
                {Object.keys(QUANT_BPW).map((q) => (
                  <option key={q} value={q}>
                    {q}
                  </option>
                ))}
              </select>
            </Field>
            <Field
              id="sp-kv"
              label="KV cache format"
              tip="F16 is the llama.cpp default. Q8_0 roughly halves KV memory."
            >
              <select
                id="sp-kv"
                value={kvType}
                onChange={(e) => setKvType(e.target.value)}
              >
                {KV_TYPES.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </Field>
          </div>

          <h2 className="section-title">Hardware</h2>
          <Field id="sp-gpu" label="GPU">
            <select
              id="sp-gpu"
              value={gpuId}
              onChange={(e) => setGpuId(e.target.value)}
            >
              {VENDORS.map(([v, name]) => (
                <optgroup key={v} label={name}>
                  {GPUS.filter((g) => g.vendor === v).map((g) => (
                    <option key={g.id} value={g.id}>
                      {gpuLabel(g)}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </Field>
          <div className="field-row">
            <Field
              id="sp-count"
              label="Cards"
              tip="Cards split the model by layers (llama.cpp default). Memory adds up; speed does not, because the cards take turns."
            >
              <select
                id="sp-count"
                value={cards}
                disabled={unified}
                onChange={(e) => setCount(Number(e.target.value))}
              >
                {[1, 2, 3, 4].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </Field>
            <Field
              id="sp-price"
              label="Price per card (USD)"
              tip="Leave empty to use the used asking price from gpudojo.com. Type what you would actually pay."
            >
              <input
                id="sp-price"
                type="number"
                inputMode="decimal"
                min={0}
                step={10}
                placeholder={
                  gpu.price_used_usd ? String(gpu.price_used_usd) : 'none'
                }
                value={price}
                onChange={(e) => setPrice(e.target.value)}
              />
            </Field>
          </div>
        </form>

        <section className="panel results-panel" aria-labelledby="sp-results">
          <h2
            className="section-title"
            id="sp-results"
            ref={resultsRef}
            tabIndex={-1}
          >
            {cards > 1 ? `${cards} x ` : ''}
            {gpu.name} running {model.name}
          </h2>

          <div aria-live="polite">
            <p className={`status ${est.fits ? 'status--ok' : 'status--bad'}`}>
              {est.fits
                ? `Fits: needs ${fmtGiB(est.vram_needed_gib)} of ${fmtGiB(est.vram_total_gib)}.`
                : `Does not fit: needs ${fmtGiB(est.vram_needed_gib)}, has ${fmtGiB(est.vram_total_gib)}. About ${est.cards_needed} of these cards would hold it.`}
            </p>
            <div className="meter" aria-hidden="true">
              <span
                className={`meter-fill${est.fits ? '' : ' meter-fill--over'}`}
                style={{ transform: `scaleX(${fillPct / 100})` }}
              />
            </div>
            <p className="meter-legend">
              Weights {fmtGiB(est.weights_gib)} + KV cache for {users}{' '}
              {users === 1 ? 'person' : 'people'} {fmtGiB(est.kv_gib)} + runtime{' '}
              {fmtGiB(0.5 * cards)}
            </p>

            <dl className="stat-grid">
              <div className="stat-lead">
                <dt>Speed per person</dt>
                <dd>
                  <span className="stat-big">{fmtTok(est.tok_s_per_user)}</span>{' '}
                  tok/s
                </dd>
              </div>
              <div>
                <dt>All {users} together</dt>
                <dd>{fmtTok(est.tok_s_total)} tok/s</dd>
              </div>
              <div>
                <dt>Alone on the box</dt>
                <dd>{fmtTok(est.tok_s_single)} tok/s</dd>
              </div>
              <div>
                <dt>Wait before the answer starts</dt>
                <dd>
                  {est.ttft_s === null ? (
                    <span className="muted">not measured on this card</span>
                  ) : (
                    fmtSec(est.ttft_s)
                  )}
                  {users > 1 && est.ttft_all_s !== null && (
                    <span className="sub">
                      up to {fmtSec(est.ttft_all_s)} if all {users} send at once
                    </span>
                  )}
                </dd>
              </div>
              <div>
                <dt>Reads requests at</dt>
                <dd>
                  {est.prefill_tok_s === null ? (
                    <span className="muted">not measured on this card</span>
                  ) : (
                    `${fmtTok(est.prefill_tok_s)} tok/s`
                  )}
                </dd>
              </div>
              <div>
                <dt>Hardware cost</dt>
                <dd>{fmtUsd(est.price_usd)}</dd>
              </div>
              <div>
                <dt>Cost per tok/s</dt>
                <dd>
                  {est.usd_per_tok_s === null
                    ? 'n/a'
                    : fmtUsd(est.usd_per_tok_s)}
                </dd>
              </div>
            </dl>

            <p className="fine">
              Memory bandwidth used: {est.bandwidth_gbs.toFixed(0)} GB/s (
              {est.bandwidth_source}). Weights {est.quant}, KV cache{' '}
              {est.kv_type}, request of {est.prompt.toLocaleString('en-US')}{' '}
              tokens.
            </p>
            {est.warnings.length > 0 && (
              <ul className="warnings">
                {est.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>

      <section className="panel picks" aria-labelledby="picks-title">
        <div className="picks-head">
          <div>
            <h2 className="section-title" id="picks-title">
              Cheapest setups for these settings
            </h2>
            <p className="panel-lede">
              {users} {users === 1 ? 'person' : 'people'} at once,{' '}
              {fmtTokens(context)} context, {prompt.toLocaleString('en-US')}
              -token requests. Default weights format, F16 KV cache, up to 4
              cards, used asking prices.
            </p>
          </div>
          <div className="picks-controls">
            <Field
              id="pk-min"
              label="Min tok/s per person"
              tip="People read roughly 4 words (about 5 tokens) a second, so 15 tok/s feels quick in chat. Raise it for coding."
            >
              <input
                id="pk-min"
                type="number"
                inputMode="numeric"
                min={1}
                max={200}
                value={minTokS}
                onChange={(e) =>
                  setMinTokS(clampInt(e.target.value, 1, 200, minTokS))
                }
              />
            </Field>
            <div className="check-row">
              <input
                type="checkbox"
                id="pk-vendors"
                checked={allVendors}
                onChange={(e) => setAllVendors(e.target.checked)}
              />
              <label htmlFor="pk-vendors">Include AMD and Intel</label>
            </div>
            <div className="check-row">
              <input
                type="checkbox"
                id="pk-more"
                checked={showMore}
                onChange={(e) => setShowMore(e.target.checked)}
              />
              <label htmlFor="pk-more">Three options per model</label>
            </div>
          </div>
        </div>

        <div
          className="table-wrap"
          tabIndex={0}
          role="region"
          aria-labelledby="picks-title"
        >
          <table>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Setup</th>
                <th scope="col" className="num">
                  Price
                </th>
                <th scope="col" className="num">
                  Per person
                </th>
                <th scope="col" className="num">
                  First token
                </th>
                <th scope="col" className="num">
                  $ per tok/s
                </th>
                <th scope="col">
                  <span className="visually-hidden">Action</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {pickRows.map((r) => (
                <tr
                  key={`${r.model}-${r.gpu}-${r.gpu_count}`}
                  className={r.first ? 'row-first' : undefined}
                >
                  <th scope="row">
                    {r.first ? (
                      r.model_name
                    ) : (
                      <span className="visually-hidden">{r.model_name}</span>
                    )}
                  </th>
                  <td>
                    {r.gpu_count > 1 ? `${r.gpu_count} x ` : ''}
                    {r.gpu_name}
                    {r.caveat && (
                      <span
                        className="caveat"
                        title="See the note under the table"
                      >
                        {' '}
                        *
                      </span>
                    )}
                  </td>
                  <td className="num">{fmtUsd(r.price_usd)}</td>
                  <td className="num">{fmtTok(r.tok_s_per_user)} tok/s</td>
                  <td className="num">
                    {r.ttft_s === null ? (
                      <span className="muted">not measured</span>
                    ) : (
                      fmtSec(r.ttft_s)
                    )}
                  </td>
                  <td className="num">
                    {r.usd_per_tok_s === null ? 'n/a' : fmtUsd(r.usd_per_tok_s)}
                  </td>
                  <td>
                    <button
                      type="button"
                      className="link-btn"
                      onClick={() => loadPick(r.model, r.gpu, r.gpu_count)}
                      aria-label={`Load ${r.gpu_count > 1 ? `${r.gpu_count} x ` : ''}${r.gpu_name} with ${r.model_name} into the calculator`}
                    >
                      Load
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {noPick.length > 0 && (
          <p className="fine">
            Nothing up to 4 cards reaches {minTokS} tok/s per person for:{' '}
            {noPick.map((m) => m.name).join(', ')}.
          </p>
        )}
        <p className="fine">
          First token is only estimated where prompt speed was measured (the RTX
          3090 so far). Other cards show &ldquo;not measured&rdquo;.
        </p>
        <p className="fine">
          * This setup carries a warning (an older card, speed scaled from
          NVIDIA numbers, or a context longer than the model was trained on).
          Load it to read the warning before you buy.
        </p>
      </section>
    </>
  );
};
