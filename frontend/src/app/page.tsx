"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type HistoryPoint = {
  date: string;
  close: number;
  forecast: number | null;
  actual: number | null;
};

type TickerForecast = {
  name: string;
  last_date: string;
  last_close: number;
  forecast_vol: number;
  usual_vol: number;
  percentile_vs_past_year: number;
  move_68_pct: number;
  move_90_pct: number;
  range_68: [number, number];
  range_90: [number, number];
  accuracy: {
    improvement_vs_naive: number;
    coverage_68: number;
    coverage_90: number;
    weeks_tested: number;
  };
  history: HistoryPoint[];
};

type Forecasts = {
  generated_at: string;
  horizon_days: number;
  tickers: Record<string, TickerForecast>;
};

const FUNDS = ["SPY", "QQQ", "IWM"];

const money = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD" });

const pct = (n: number, digits = 0) => `${(n * 100).toFixed(digits)}%`;

const longDate = (iso: string) =>
  new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });

const shortDate = (iso: string) =>
  new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

function describeLevel(percentile: number) {
  if (percentile >= 0.8) return { word: "rougher than usual", tone: "text-[#B4570B]" };
  if (percentile <= 0.2) return { word: "calmer than usual", tone: "text-[#1F7A5A]" };
  return { word: "about normal", tone: "text-[#14213D]" };
}

export default function Home() {
  const [data, setData] = useState<Forecasts | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ticker, setTicker] = useState("SPY");

  useEffect(() => {
    fetch("/forecasts.json", { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: Forecasts) => {
        setData(json);
        if (!json.tickers[ticker]) setTicker(Object.keys(json.tickers)[0]);
      })
      .catch(() =>
        setError(
          "Forecasts couldn't be loaded. The daily update may not have run yet; check back after the next market close."
        )
      );
    // Load once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const current = data?.tickers[ticker];
  const symbols = useMemo(() => (data ? Object.keys(data.tickers) : []), [data]);

  return (
    <main className="min-h-screen bg-[#EEF1F4] text-[#14213D]">
      <div className="mx-auto max-w-3xl px-5 py-12 sm:py-16">
        <header className="mb-10">
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
            How far could it move next week?
          </h1>
          <p className="mt-3 max-w-[62ch] text-[17px] leading-relaxed text-[#4A5A73]">
            A volatility forecast for the next five trading days, updated every weekday after
            the market closes. It estimates how much a stock is likely to move, not which way.
          </p>
        </header>

        {error && (
          <p className="rounded-md border border-[#E3B9A0] bg-white px-4 py-3 text-[#8A3B12]">
            {error}
          </p>
        )}

        {!data && !error && <p className="text-[#4A5A73]">Loading forecasts…</p>}

        {data && current && (
          <>
            <TickerPicker symbols={symbols} selected={ticker} onSelect={setTicker} data={data} />
            <RangeSection symbol={ticker} f={current} />
            <HistorySection f={current} />
            <AccuracySection f={current} />
            <MethodSection />
            <footer className="mt-14 border-t border-[#C9D2DE] pt-6 text-sm leading-relaxed text-[#5B6B82]">
              <p>
                Prices through {longDate(current.last_date)}. Forecasts generated{" "}
                {new Date(data.generated_at).toLocaleString("en-US", {
                  dateStyle: "medium",
                  timeStyle: "short",
                })}
                .
              </p>
              <p className="mt-2">
                A statistical estimate for learning and research, not investment advice. Real moves
                can exceed any forecast range, especially around earnings and market shocks.
              </p>
            </footer>
          </>
        )}
      </div>
    </main>
  );
}

function TickerPicker({
  symbols,
  selected,
  onSelect,
  data,
}: {
  symbols: string[];
  selected: string;
  onSelect: (s: string) => void;
  data: Forecasts;
}) {
  const groups = [
    { label: "Market funds", items: symbols.filter((s) => FUNDS.includes(s)) },
    { label: "Companies", items: symbols.filter((s) => !FUNDS.includes(s)) },
  ];

  return (
    <nav aria-label="Choose a ticker" className="mb-10 space-y-3">
      {groups.map((g) =>
        g.items.length ? (
          <div key={g.label} className="flex flex-wrap items-center gap-2">
            <span className="w-28 shrink-0 text-sm text-[#5B6B82]">{g.label}</span>
            {g.items.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => onSelect(s)}
                aria-pressed={s === selected}
                title={data.tickers[s].name}
                className={`rounded-full border px-3 py-1 text-sm font-medium tabular-nums transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2F6FED] ${
                  s === selected
                    ? "border-[#14213D] bg-[#14213D] text-white"
                    : "border-[#C9D2DE] bg-white text-[#14213D] hover:border-[#14213D]"
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        ) : null
      )}
    </nav>
  );
}

function RangeSection({ symbol, f }: { symbol: string; f: TickerForecast }) {
  const level = describeLevel(f.percentile_vs_past_year);

  // Map prices onto the ladder: the 90% range fills most of the width.
  const [lo90, hi90] = f.range_90;
  const pad = (hi90 - lo90) * 0.12;
  const min = lo90 - pad;
  const max = hi90 + pad;
  const x = (price: number) => `${((price - min) / (max - min)) * 100}%`;
  const width = (a: number, b: number) => `${((b - a) / (max - min)) * 100}%`;

  return (
    <section aria-labelledby="range-heading" className="mb-14">
      <h2 id="range-heading" className="text-lg font-semibold">
        {f.name} ({symbol})
      </h2>
      <p className="mt-2 text-[28px] font-semibold leading-snug tracking-tight tabular-nums sm:text-[34px]">
        Likely to stay within ±{f.move_68_pct.toFixed(1)}% over the next week
      </p>
      <p className="mt-2 max-w-[62ch] text-[17px] leading-relaxed text-[#4A5A73]">
        That&apos;s {money(f.range_68[0])} to {money(f.range_68[1])} from today&apos;s close of{" "}
        {money(f.last_close)}, with a wider ±{f.move_90_pct.toFixed(1)}% range for nine weeks in
        ten. Expected volatility is <span className={`font-semibold ${level.tone}`}>{level.word}</span>{" "}
        for {symbol}: {f.forecast_vol.toFixed(0)}% annualized, against a typical{" "}
        {f.usual_vol.toFixed(0)}% over the past year.
      </p>

      <figure className="mt-8" aria-label={`Forecast price ranges for ${symbol}`}>
        <div className="relative h-16">
          <div
            className="absolute top-5 h-6 rounded-sm bg-[#BCD0F7]"
            style={{ left: x(lo90), width: width(lo90, hi90) }}
          />
          <div
            className="absolute top-5 h-6 rounded-sm bg-[#2F6FED]"
            style={{ left: x(f.range_68[0]), width: width(f.range_68[0], f.range_68[1]) }}
          />
          <div
            className="absolute top-2 h-12 w-[3px] -translate-x-1/2 rounded bg-[#14213D]"
            style={{ left: x(f.last_close) }}
          />
        </div>
        <div className="relative h-10 text-xs tabular-nums text-[#4A5A73] sm:text-sm">
          {[lo90, f.range_68[0], f.last_close, f.range_68[1], hi90].map((p, i) => (
            <span
              key={i}
              className={`absolute -translate-x-1/2 whitespace-nowrap ${
                i === 2 ? "font-semibold text-[#14213D]" : i % 4 === 0 ? "hidden sm:inline" : ""
              }`}
              style={{ left: x(p) }}
            >
              {money(p)}
            </span>
          ))}
        </div>
        <figcaption className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-[#4A5A73]">
          <Key color="bg-[#2F6FED]" label="Likely range (68%)" />
          <Key color="bg-[#BCD0F7]" label="Wide range (90%)" />
          <Key color="bg-[#14213D]" label="Last close" thin />
        </figcaption>
      </figure>
    </section>
  );
}

function Key({ color, label, thin }: { color: string; label: string; thin?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span className={`inline-block h-3 ${thin ? "w-[3px]" : "w-5"} rounded-sm ${color}`} />
      {label}
    </span>
  );
}

function HistorySection({ f }: { f: TickerForecast }) {
  const points = f.history.filter((h) => h.forecast !== null || h.actual !== null);

  return (
    <section aria-labelledby="history-heading" className="mb-14">
      <h2 id="history-heading" className="text-xl font-semibold">
        Forecast vs. what happened
      </h2>
      <p className="mt-2 max-w-[62ch] leading-relaxed text-[#4A5A73]">
        Each day&apos;s forecast was made using only data available at the time. The amber line is
        the volatility that actually followed over the next five trading days.
      </p>
      <div className="mt-6 h-72 w-full rounded-md border border-[#C9D2DE] bg-white p-3">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="#E4E9F0" vertical={false} />
            <XAxis
              dataKey="date"
              tickFormatter={shortDate}
              minTickGap={48}
              tick={{ fontSize: 12, fill: "#5B6B82" }}
              axisLine={{ stroke: "#C9D2DE" }}
              tickLine={false}
            />
            <YAxis
              tickFormatter={(v: number) => `${v}%`}
              width={44}
              tick={{ fontSize: 12, fill: "#5B6B82" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              labelFormatter={(label: unknown) => longDate(String(label))}
              formatter={(value: unknown, name: unknown) => [
                `${Number(value).toFixed(1)}% annualized`,
                name === "forecast" ? "Forecast" : "Actual",
              ]}
              contentStyle={{ borderRadius: 6, borderColor: "#C9D2DE", fontSize: 13 }}
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#E0A100"
              strokeWidth={1.5}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="forecast"
              stroke="#2F6FED"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-3 flex gap-6 text-sm text-[#4A5A73]">
        <Key color="bg-[#2F6FED]" label="Forecast" />
        <Key color="bg-[#E0A100]" label="Actual" />
      </div>
    </section>
  );
}

function AccuracySection({ f }: { f: TickerForecast }) {
  const a = f.accuracy;
  const rows = [
    {
      value: pct(a.coverage_68),
      target: "68%",
      text: "of weeks, the real move landed inside the likely range.",
    },
    {
      value: pct(a.coverage_90),
      target: "90%",
      text: "of weeks, it landed inside the wide range.",
    },
    {
      value: pct(a.improvement_vs_naive),
      target: null,
      text: "less error than assuming next week will be as volatile as last week.",
    },
  ];

  return (
    <section aria-labelledby="accuracy-heading" className="mb-14">
      <h2 id="accuracy-heading" className="text-xl font-semibold">
        Have the ranges been honest?
      </h2>
      <p className="mt-2 max-w-[62ch] leading-relaxed text-[#4A5A73]">
        Tested on {a.weeks_tested} separate weeks, retraining once a year and never using future data.
        A well-calibrated range should contain the real move about as often as it claims to.
      </p>
      <dl className="mt-6 divide-y divide-[#C9D2DE] border-y border-[#C9D2DE]">
        {rows.map((r) => (
          <div key={r.text} className="flex items-baseline gap-4 py-4">
            <dt className="w-20 shrink-0 text-2xl font-semibold tabular-nums">{r.value}</dt>
            <dd className="leading-relaxed text-[#4A5A73]">
              {r.text}
              {r.target && <span className="text-[#5B6B82]"> The target is {r.target}.</span>}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function MethodSection() {
  return (
    <section aria-labelledby="method-heading">
      <h2 id="method-heading" className="text-xl font-semibold">
        How it works
      </h2>
      <div className="mt-3 max-w-[62ch] space-y-4 leading-relaxed text-[#4A5A73]">
        <p>
          Volatility clusters: rough weeks tend to follow rough weeks. The forecast uses a HAR
          model, which predicts next week&apos;s volatility from the past day, week and month of
          price swings.
        </p>
        <p>
          This project started as an LSTM that predicted tomorrow&apos;s price. A walk-forward
          backtest across six stocks from 2015 onward showed it had no real edge: it called the
          direction right 52.6% of the time, while the stocks rose on 53.0% of days. Volatility
          was different. Every model beat the naive forecast, and the simple HAR model beat the
          LSTM on all six stocks.
        </p>
      </div>
      <table className="mt-6 w-full max-w-md text-left text-sm tabular-nums">
        <caption className="mb-2 text-left text-[#5B6B82]">
          Average error reduction vs. the naive forecast, six stocks
        </caption>
        <tbody className="divide-y divide-[#C9D2DE] border-y border-[#C9D2DE]">
          {[
            ["HAR (used here)", "16.7%"],
            ["LSTM", "13.9%"],
            ["Exponential moving average", "11.0%"],
          ].map(([model, value]) => (
            <tr key={model}>
              <th scope="row" className="py-2 font-medium text-[#14213D]">
                {model}
              </th>
              <td className="py-2 text-right">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
