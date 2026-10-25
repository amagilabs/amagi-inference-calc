// The same data files the Python package ships, imported at build time.
import modelsJson from '../../src/amagi_inference_calc/data/models.json';
import gpusJson from '../../src/amagi_inference_calc/data/gpus.json';
import calJson from '../../src/amagi_inference_calc/data/calibration.json';
import presetsJson from '../../src/amagi_inference_calc/data/office_presets.json';
import type { Calibration, GPU, ModelSpec, OfficePreset } from './speed';

export const MODELS = modelsJson.models as unknown as ModelSpec[];
export const GPUS = gpusJson.gpus as unknown as GPU[];
export const PRICE_DATE: string = gpusJson.price_date;
export const CAL = calJson as unknown as Calibration;
export const PRESETS = presetsJson.presets as OfficePreset[];
export const PRESETS_ABOUT: string = presetsJson._about;
