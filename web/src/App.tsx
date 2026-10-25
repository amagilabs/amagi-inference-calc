import { useEffect, useRef, useState } from 'react';
import './App.css';
import { ThemeToggle } from './components/ThemeToggle';
import { SpeedPanel } from './components/SpeedPanel';
import { MemoryCalculator } from './components/MemoryCalculator';
import { useTheme } from './contexts/ThemeContext';
import { CAL, PRICE_DATE } from './data';
import lockup from './assets/amagilabs-lockup.svg';
import lockupOnDark from './assets/amagilabs-lockup-on-dark.svg';

const TABS = [
  { id: 'speed', label: 'Speed and cost', extra: '' },
  { id: 'memory', label: 'Memory only', extra: ' (original calculator)' },
] as const;
type TabId = (typeof TABS)[number]['id'];

const REPO = 'https://github.com/amagilabs/amagi-inference-calc';
const UPSTREAM = 'https://github.com/alexziskind1/llm-inference-calculator';

const fromHash = (): TabId =>
  window.location.hash === '#memory' ? 'memory' : 'speed';

function App() {
  const { theme } = useTheme();
  const [tab, setTab] = useState<TabId>(fromHash);
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  useEffect(() => {
    const onHash = () => setTab(fromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const select = (id: TabId, focus = false) => {
    setTab(id);
    history.replaceState(
      null,
      '',
      id === 'speed'
        ? window.location.pathname + window.location.search
        : '#memory'
    );
    if (focus) tabRefs.current[id]?.focus();
  };

  const onTabKey = (e: React.KeyboardEvent, i: number) => {
    const last = TABS.length - 1;
    const to =
      e.key === 'ArrowRight'
        ? i === last
          ? 0
          : i + 1
        : e.key === 'ArrowLeft'
          ? i === 0
            ? last
            : i - 1
          : e.key === 'Home'
            ? 0
            : e.key === 'End'
              ? last
              : -1;
    if (to >= 0) {
      e.preventDefault();
      select(TABS[to].id, true);
    }
  };

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to the calculator
      </a>
      <header className="site-header">
        <a
          className="brand"
          href="https://github.com/amagilabs"
          aria-label="AmagiLabs on GitHub"
        >
          <img
            src={theme === 'dark' ? lockupOnDark : lockup}
            alt="AmagiLabs"
            width="127"
            height="36"
          />
        </a>
        <ThemeToggle />
      </header>

      <main id="main" className="page">
        <div className="intro">
          <h1>What GPU does your office need for local AI?</h1>
          <p className="intro-text">
            Pick an office size, a model and a card. You get memory, speed per
            person, the wait before an answer starts, and what the hardware
            costs. Speeds come from a model fitted to benchmarks on a real RTX
            3090.
          </p>
          <p className="credit">
            Based on <a href={UPSTREAM}>llm-inference-calculator</a> by Alex
            Ziskind (MIT).
          </p>
        </div>

        <div className="tabs" role="tablist" aria-label="Calculator">
          {TABS.map((t, i) => (
            <button
              key={t.id}
              ref={(el) => {
                tabRefs.current[t.id] = el;
              }}
              type="button"
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`panel-${t.id}`}
              tabIndex={tab === t.id ? 0 : -1}
              className="tab"
              onClick={() => select(t.id)}
              onKeyDown={(e) => onTabKey(e, i)}
            >
              {t.label}
              {t.extra && <span className="tab-extra">{t.extra}</span>}
            </button>
          ))}
        </div>

        <div
          role="tabpanel"
          id="panel-speed"
          aria-labelledby="tab-speed"
          hidden={tab !== 'speed'}
        >
          {tab === 'speed' && <SpeedPanel />}
        </div>
        <div
          role="tabpanel"
          id="panel-memory"
          aria-labelledby="tab-memory"
          hidden={tab !== 'memory'}
        >
          {tab === 'memory' && <MemoryCalculator />}
        </div>
      </main>

      <footer className="site-footer">
        <p>
          Estimates, not promises. The speed model was fitted on one RTX 3090
          (memory test {CAL.bw_ref_gbs} GB/s, llama.cpp). Other cards are scaled
          by memory bandwidth, so AMD, Intel and Apple numbers are rougher.
          Prices are used asking prices from gpudojo.com, fetched {PRICE_DATE};
          they move weekly.
        </p>
        <p>
          <a href={`${REPO}/blob/main/docs/CALCULATIONS.md`}>
            How the numbers are worked out
          </a>
          <span aria-hidden="true"> · </span>
          <a href={REPO}>Source and command-line version</a>
          <span aria-hidden="true"> · </span>
          <a href="https://github.com/amagilabs">More from AmagiLabs</a>
        </p>
      </footer>
    </>
  );
}

export default App;
