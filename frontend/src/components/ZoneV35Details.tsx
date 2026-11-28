import { useState } from "react";

/**
 * v3.5 per-zone detail: real-world chance, percentile vs ordinary days, the model's
 * fire-weather inputs and the official USFS indices for the same tile. Same visual
 * vocabulary as the map popup rows (10px uppercase section labels, label/value rows,
 * tabular numbers) so it sits inside the existing popup and research panel unchanged.
 */
export interface V35Zone {
  model?: string;
  probability?: number;
  percentile?: number;
  grid_date?: string;
  inputs?: Partial<Record<InputKey, number | null>>;
  official?: { erc?: number | null; bi?: number | null; fm100?: number | null; fm1000?: number | null };
}

type InputKey =
  | "gust_event" | "offshore_event" | "rh_min_event" | "vpd_max_event" | "rain_7d" | "fm100" | "fm1000"
  | "temp_anom" | "rh_anom" | "soil" | "slope" | "southness";

const fmt = (v: number | null | undefined, digits: number, unit = "") =>
  typeof v === "number" && Number.isFinite(v) ? `${v.toFixed(digits)}${unit}` : "—";
const signed = (v: number | null | undefined, digits: number, unit: string) =>
  typeof v === "number" && Number.isFinite(v) ? `${v > 0 ? "+" : ""}${v.toFixed(digits)}${unit}` : "—";

// [key, label, hint, render]; the first four are shown by default (highest-weight inputs)
const INPUTS: [InputKey, string, string, (v: number | null | undefined) => string][] = [
  ["gust_event", "💨 Peak gust", "Strongest gust, today and yesterday", (v) => fmt(v, 1, " m/s")],
  ["rh_min_event", "💧 Lowest humidity", "Driest hour, today and yesterday", (v) => fmt(v, 0, "%")],
  ["fm100", "🪵 100-hour fuel moisture", "Dead branches 1-3 in; lower = drier", (v) => fmt(v, 1, "%")],
  ["rain_7d", "🌧️ Rain, last 7 days", "Recent rain wets fuels", (v) => fmt(v, 1, " mm")],
  ["offshore_event", "🌬️ Offshore wind push", "Santa Ana / Diablo direction; above 0 = offshore", (v) => fmt(v, 1, " m/s")],
  ["vpd_max_event", "☀️ Peak drying power (VPD)", "How hard the air pulls moisture from fuel", (v) => fmt(v, 2, " kPa")],
  ["fm1000", "🌲 1000-hour fuel moisture", "Logs 3-8 in; the slow, seasonal dryness", (v) => fmt(v, 1, "%")],
  ["temp_anom", "🌡️ Temperature vs normal", "Compared with 1991-2020 for this month", (v) => signed(typeof v === "number" ? v * 1.8 : v, 1, "°F")],
  ["rh_anom", "💦 Humidity vs normal", "Compared with 1991-2020 for this month", (v) => signed(v, 0, " pts")],
  ["soil", "🟫 Soil moisture", "Top soil layer, volumetric", (v) => fmt(typeof v === "number" ? v * 100 : v, 0, "%")],
  ["slope", "⛰️ Slope", "Fire runs faster uphill", (v) => fmt(v, 0, "°")],
  ["southness", "🧭 Sun exposure", "South-facing slopes dry out first",
    (v) => (typeof v === "number" ? (v > 0.33 ? "South-facing" : v < -0.33 ? "North-facing" : "East/West") : "—")],
];

const labelStyle = { fontSize: 10, textTransform: "uppercase" as const, letterSpacing: 0.5, color: "#6b7280", marginBottom: 6 };
const rowStyle = { display: "flex", justifyContent: "space-between", gap: 12 };
const num = { fontVariantNumeric: "tabular-nums" as const, whiteSpace: "nowrap" as const };

function ordinal(n: number) {
  const r = Math.round(n);
  const s = r % 100 >= 11 && r % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[r % 10] || "th";
  return `${r}${s}`;
}

function chance(p: number) {
  if (!(p > 0)) return "—";
  const n = 1 / p;
  if (n > 1e6) return "under 1 in a million";
  return `1 in ${Math.round(n).toLocaleString()}`;
}

/** One-line summary for the chip row: "82nd percentile". */
export function v35Headline(z: V35Zone) {
  return typeof z.percentile === "number" ? `${ordinal(z.percentile)} percentile` : null;
}

export function ZoneV35Details({ zone }: { zone: V35Zone }) {
  const [all, setAll] = useState(false);
  if (zone.model !== "v3.5") return null;
  const shown = all ? INPUTS : INPUTS.slice(0, 4);
  const o = zone.official || {};
  return (
    <div style={{ fontSize: 12, lineHeight: 1.55 }}>
      <div style={{ color: "#374151", marginBottom: 10 }}>
        {typeof zone.percentile === "number" && (
          <div>Riskier than {Math.round(zone.percentile)}% of ordinary days in this region.</div>
        )}
        {typeof zone.probability === "number" && (
          <div>
            Chance a wildfire that grows past 1,000 acres starts here today:{" "}
            <strong style={num}>{chance(zone.probability)}</strong>
          </div>
        )}
      </div>

      <div style={labelStyle}>Fire weather and fuels</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 6 }}>
        {shown.map(([k, label, hint, render]) => (
          <div key={k} style={rowStyle}>
            <span><strong>{label}:</strong><br /><span style={{ color: "#6b7280", fontSize: 11 }}>{hint}</span></span>
            <span style={num}>{render(zone.inputs?.[k])}</span>
          </div>
        ))}
      </div>
      <button
        type="button"
        onClick={() => setAll((v) => !v)}
        aria-expanded={all}
        style={{ marginTop: 6, padding: "6px 0", minHeight: 32, background: "none", border: "none", cursor: "pointer", color: "#2563eb", fontSize: 11, fontWeight: 600 }}
      >
        {all ? "Show fewer inputs" : `Show all ${INPUTS.length} inputs`}
      </button>

      <div style={{ ...labelStyle, marginTop: 8 }}>Official indices, same spot (USFS NFDRS via gridMET)</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 12px" }}>
        <div style={rowStyle}><span>Energy Release Component</span><span style={num}>{fmt(o.erc, 0)}</span></div>
        <div style={rowStyle}><span>Burning Index</span><span style={num}>{fmt(o.bi, 0)}</span></div>
        <div style={rowStyle}><span>100-hour fuel</span><span style={num}>{fmt(o.fm100, 1, "%")}</span></div>
        <div style={rowStyle}><span>1000-hour fuel</span><span style={num}>{fmt(o.fm1000, 1, "%")}</span></div>
      </div>
      <div style={{ marginTop: 8, fontSize: 10, color: "#6b7280" }}>
        FireScope v3.5{zone.grid_date ? ` · ${zone.grid_date}` : ""} · start × growth model, each input from 2+ sources
        (RTMA, ERA5, gridMET, PRISM, SMAP, USGS 3DEP).
      </div>
    </div>
  );
}
