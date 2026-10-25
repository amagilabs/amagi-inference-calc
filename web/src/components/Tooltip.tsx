import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { createPortal } from 'react-dom';
import './Tooltip.css';

interface TooltipProps {
  text: string;
  /** Accessible name of the trigger, e.g. "About context length". */
  label?: string;
}

const GAP = 8;
const EDGE = 8;

// Upstream issue #28: the old tooltip was an absolutely positioned child, so
// any parent with overflow (the scrolling input panel) or its own stacking
// context (the sticky results panel) clipped it or drew over it. This one is
// rendered into document.body with position: fixed, measured against the
// trigger, and kept inside the viewport.
export const Tooltip = ({ text, label = 'More info' }: TooltipProps) => {
  const id = useId();
  const trigger = useRef<HTMLButtonElement>(null);
  const tip = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [pos, setPos] = useState<{
    top: number;
    left: number;
    below: boolean;
  } | null>(null);

  const place = useCallback(() => {
    const t = trigger.current;
    const el = tip.current;
    if (!t || !el) return;
    const r = t.getBoundingClientRect();
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const left = Math.min(
      Math.max(r.left + r.width / 2 - w / 2, EDGE),
      window.innerWidth - w - EDGE
    );
    let top = r.top - h - GAP;
    let below = false;
    if (top < EDGE) {
      top = r.bottom + GAP;
      below = true;
    }
    setPos({ top, left, below });
  }, []);

  useLayoutEffect(() => {
    if (open) place();
    else setPos(null);
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
        setPinned(false);
      }
    };
    const onDown = (e: PointerEvent) => {
      if (!trigger.current?.contains(e.target as Node)) {
        setOpen(false);
        setPinned(false);
      }
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('pointerdown', onDown);
    window.addEventListener('scroll', place, true);
    window.addEventListener('resize', place);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onDown);
      window.removeEventListener('scroll', place, true);
      window.removeEventListener('resize', place);
    };
  }, [open, place]);

  return (
    <>
      <button
        type="button"
        ref={trigger}
        className="info-btn"
        aria-label={label}
        aria-describedby={id}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => !pinned && setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          setOpen(false);
          setPinned(false);
        }}
        onClick={() => {
          setPinned((p) => !p);
          setOpen((o) => !pinned || !o);
        }}
      >
        <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
          <circle
            cx="8"
            cy="8"
            r="7"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          />
          <path
            d="M8 7v4.5"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
          <circle cx="8" cy="4.6" r="1" fill="currentColor" />
        </svg>
      </button>
      <span id={id} className="visually-hidden">
        {text}
      </span>
      {open &&
        createPortal(
          <div
            ref={tip}
            className={`tooltip${pos?.below ? ' tooltip--below' : ''}`}
            aria-hidden="true"
            style={
              pos
                ? { top: pos.top, left: pos.left }
                : { top: -9999, left: -9999 }
            }
          >
            {text}
          </div>,
          document.body
        )}
    </>
  );
};
