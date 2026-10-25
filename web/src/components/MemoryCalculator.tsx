// The original llm-inference-calculator form (memory only, one user),
// moved out of App.tsx unchanged in logic. Labels now sit outside the
// controls so the info buttons are not nested inside <label>.
import { useState } from 'react';
import { MemoryMode, ModelQuantization, KvCacheQuantization } from '../types';
import {
  calculateHardwareRecommendation,
  calculateOnDiskSize,
} from '../calculations';
import { Field } from './Field';

const num =
  (setter: (n: number) => void) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = Number(e.target.value);
    if (!isNaN(v)) setter(v);
  };

export const MemoryCalculator = () => {
  const [params, setParams] = useState<number>(65);
  const [modelQuant, setModelQuant] = useState<ModelQuantization>('Q4');
  const [useKvCache, setUseKvCache] = useState<boolean>(false);
  const [kvCacheQuant, setKvCacheQuant] = useState<KvCacheQuantization>('Q4');
  const [inferenceMode, setInferenceMode] = useState<'incremental' | 'bulk'>(
    'incremental'
  );
  const [contextLength, setContextLength] = useState<number>(4096);
  const [memoryMode, setMemoryMode] = useState<MemoryMode>('DISCRETE_GPU');
  const [systemMemory, setSystemMemory] = useState<number>(128);
  const [gpuVram, setGpuVram] = useState<number>(24);

  const recommendation = calculateHardwareRecommendation(
    params,
    modelQuant,
    contextLength,
    useKvCache,
    kvCacheQuant,
    memoryMode,
    systemMemory,
    gpuVram,
    inferenceMode
  );
  const onDiskSize = calculateOnDiskSize(params, modelQuant);

  return (
    <div className="layout">
      <form className="panel input-panel" onSubmit={(e) => e.preventDefault()}>
        <p className="panel-lede">
          The original calculator: memory for a single user (batch 1). Model and
          KV cache formats are set separately.
        </p>
        <h2 className="section-title">Model</h2>

        <Field
          id="mem-params"
          label="Parameters (billions)"
          tip="The total number of model parameters in billions. For example, '13' means a 13B model."
        >
          <div className="num-range">
            <input
              id="mem-params"
              type="number"
              min={1}
              max={1000}
              value={params}
              onChange={num(setParams)}
            />
            <input
              type="range"
              min={1}
              max={1000}
              value={params}
              aria-label="Parameters (billions) slider"
              onChange={(e) => setParams(Number(e.target.value))}
            />
          </div>
        </Field>

        <Field
          id="mem-quant"
          label="Model quantization"
          tip="The data format used to store model weights in GPU memory. For instance, F16 uses ~2GB per 1B params, Q4 ~0.5GB, etc."
        >
          <select
            id="mem-quant"
            value={modelQuant}
            onChange={(e) => setModelQuant(e.target.value as ModelQuantization)}
          >
            {[
              'F32',
              'F16',
              'Q8',
              'Q6',
              'Q5',
              'Q4',
              'Q3',
              'Q2',
              'GPTQ',
              'AWQ',
            ].map((q) => (
              <option key={q} value={q}>
                {q}
              </option>
            ))}
          </select>
        </Field>

        <Field
          id="mem-context"
          label="Context length (tokens)"
          tip="Maximum tokens (including prompt and history) available at once. Larger context = more memory usage."
        >
          <div className="num-range">
            <input
              id="mem-context"
              type="number"
              min={128}
              max={32768}
              step={128}
              value={contextLength}
              onChange={num(setContextLength)}
            />
            <input
              type="range"
              min={128}
              max={32768}
              step={128}
              value={contextLength}
              aria-label="Context length slider"
              onChange={(e) => setContextLength(Number(e.target.value))}
            />
          </div>
        </Field>

        <Field
          id="mem-mode"
          label="Inference mode"
          tip="'Incremental' is streaming token-by-token generation, 'Bulk' processes the entire context in one pass."
        >
          <select
            id="mem-mode"
            value={inferenceMode}
            onChange={(e) =>
              setInferenceMode(e.target.value as 'incremental' | 'bulk')
            }
          >
            <option value="incremental">Incremental (streaming)</option>
            <option value="bulk">Bulk (all at once)</option>
          </select>
        </Field>

        <div className="check-row">
          <input
            type="checkbox"
            id="mem-kv"
            checked={useKvCache}
            onChange={() => setUseKvCache(!useKvCache)}
          />
          <label htmlFor="mem-kv">Enable KV cache</label>
        </div>

        {useKvCache && (
          <Field
            id="mem-kvq"
            label="KV cache quantization"
            tip="Data format for KV cache memory usage. Lower precision reduces memory but may affect performance/quality."
          >
            <select
              id="mem-kvq"
              value={kvCacheQuant}
              onChange={(e) =>
                setKvCacheQuant(e.target.value as KvCacheQuantization)
              }
            >
              {['F32', 'F16', 'Q8', 'Q5', 'Q4'].map((q) => (
                <option key={q} value={q}>
                  {q}
                </option>
              ))}
            </select>
          </Field>
        )}

        <h2 className="section-title">System</h2>

        <Field id="mem-system" label="System type">
          <select
            id="mem-system"
            value={memoryMode}
            onChange={(e) => setMemoryMode(e.target.value as MemoryMode)}
          >
            <option value="DISCRETE_GPU">Discrete GPU</option>
            <option value="UNIFIED_MEMORY">
              Unified memory (Apple silicon, AMD Ryzen AI Max+ 395)
            </option>
          </select>
        </Field>

        {memoryMode === 'DISCRETE_GPU' && (
          <Field id="mem-vram" label="GPU VRAM (GB)">
            <select
              id="mem-vram"
              value={gpuVram}
              onChange={(e) => setGpuVram(Number(e.target.value))}
            >
              {[8, 12, 16, 20, 24, 32, 40, 48, 80].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </Field>
        )}

        <Field id="mem-ram" label="System memory (GB)">
          <div className="num-range">
            <input
              id="mem-ram"
              type="number"
              min={8}
              max={512}
              step={8}
              value={systemMemory}
              onChange={num(setSystemMemory)}
            />
            <input
              type="range"
              min={8}
              max={512}
              step={8}
              value={systemMemory}
              aria-label="System memory slider"
              onChange={(e) => setSystemMemory(Number(e.target.value))}
            />
          </div>
        </Field>
      </form>

      <section
        className="panel results-panel"
        aria-labelledby="mem-results"
        aria-live="polite"
      >
        <h2 className="section-title" id="mem-results">
          Hardware requirements
        </h2>
        <dl className="stat-list">
          <div>
            <dt>VRAM needed</dt>
            <dd className="stat-big">{recommendation.vramNeeded} GB</dd>
          </div>
          <div>
            <dt>On-disk size</dt>
            <dd>{onDiskSize.toFixed(2)} GB</dd>
          </div>
          <div>
            <dt>GPU config</dt>
            <dd>{recommendation.gpuType}</dd>
          </div>
          <div>
            <dt>GPUs required</dt>
            <dd>
              {recommendation.gpusRequired > 1
                ? recommendation.gpusRequired
                : `${recommendation.gpusRequired} (fits on a single GPU)`}
            </dd>
          </div>
          {memoryMode === 'DISCRETE_GPU' && (
            <div>
              <dt>System RAM</dt>
              <dd>{recommendation.systemRamNeeded.toFixed(1)} GB</dd>
            </div>
          )}
        </dl>
        {memoryMode === 'UNIFIED_MEMORY' &&
          (recommendation.fitsUnified ? (
            <p className="status status--ok">Fits in unified memory.</p>
          ) : (
            <p className="status status--bad">
              Exceeds unified memory. Increase system RAM or reduce model size.
            </p>
          ))}
        <p className="fine">
          Want speed per person and what it costs? Open the Speed and cost tab,
          which also counts several people working at once.
        </p>
      </section>
    </div>
  );
};
