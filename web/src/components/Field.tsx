import { ReactNode } from 'react';
import { Tooltip } from './Tooltip';

interface FieldProps {
  id: string;
  label: string;
  tip?: string;
  hint?: ReactNode;
  children: ReactNode;
}

/** Label row (with optional info tooltip), the control, and helper text. */
export const Field = ({ id, label, tip, hint, children }: FieldProps) => (
  <div className="field">
    <div className="field-head">
      <label htmlFor={id}>{label}</label>
      {tip && <Tooltip text={tip} label={`About ${label.toLowerCase()}`} />}
    </div>
    {children}
    {hint && (
      <p className="field-hint" id={`${id}-hint`}>
        {hint}
      </p>
    )}
  </div>
);
