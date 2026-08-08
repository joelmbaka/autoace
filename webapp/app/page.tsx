"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

type Analysis = {
  emotional_tone: "neutral" | "satisfied" | "frustrated" | "upset" | "distressed";
  emotional_intensity: "low" | "medium" | "high";
  background_noise_present: boolean;
  background_noise_type: string;
  background_noise_severity: "none" | "low" | "medium" | "high";
  audio_quality: "clear" | "slightly_impaired" | "severely_impaired";
  speaker_overlap_present: boolean;
  long_silence_present: boolean;
  confidence: number;
};

type Usage = {
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  thinking_tokens?: number | null;
};

type Row = {
  name: string;
  status: "pending" | "completed" | "failed";
  result?: Analysis;
  error?: string | null;
  model?: string;
  request_seconds?: number;
  usage?: Usage;
  validation?: {
    field_accuracy: number;
    correct_fields: number;
    total_fields: number;
    field_matches: Record<string, boolean>;
    expected: Analysis;
  } | null;
};

type ValidationSummary = {
  labeled_files: number;
  exact_field_accuracy: number;
  per_field_accuracy: Record<string, number>;
  emotional_tone_confusion_matrix: Record<string, Record<string, number>>;
  confidence_note: string;
};

type StreamEvent = {
  type: "batch_started" | "file_completed" | "file_failed" | "batch_completed";
  total?: number;
  processable?: number;
  labeled_count?: number;
  concurrency?: number;
  warnings?: string[];
  items?: Array<{ name: string; status: "pending" | "failed"; error?: string | null }>;
  name?: string;
  result?: Analysis;
  error?: string;
  model?: string;
  request_seconds?: number;
  usage?: Usage;
  validation?: ValidationSummary | Row["validation"] | null;
  completed?: number;
  failed?: number;
};

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function downloadBlob(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function csvEscape(value: string) {
  return `"${value.replaceAll('"', '""')}"`;
}

function statusClass(status: Row["status"]) {
  if (status === "completed") return "bg-emerald-50 text-emerald-700 ring-emerald-600/20";
  if (status === "failed") return "bg-rose-50 text-rose-700 ring-rose-600/20";
  return "bg-amber-50 text-amber-700 ring-amber-600/20";
}

function boolText(value: boolean | undefined) {
  if (value === undefined) return "—";
  return value ? "Yes" : "No";
}

export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [username, setUsername] = useState("evaluator");
  const [password, setPassword] = useState("");
  const [loginBusy, setLoginBusy] = useState(false);
  const [loginError, setLoginError] = useState("");

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [validationSummary, setValidationSummary] = useState<ValidationSummary | null>(null);
  const [processing, setProcessing] = useState(false);
  const [batchError, setBatchError] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const zipInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setToken(sessionStorage.getItem("autoace_access_token"));
    folderInput.current?.setAttribute("webkitdirectory", "");
    folderInput.current?.setAttribute("directory", "");
  }, []);

  const selectedBytes = useMemo(
    () => selectedFiles.reduce((total, file) => total + file.size, 0),
    [selectedFiles],
  );

  const finished = rows.filter((row) => row.status !== "pending").length;
  const completed = rows.filter((row) => row.status === "completed").length;
  const failed = rows.filter((row) => row.status === "failed").length;
  const progress = rows.length ? Math.round((finished / rows.length) * 100) : 0;

  async function handleLogin(event: FormEvent) {
    event.preventDefault();
    setLoginBusy(true);
    setLoginError("");
    try {
      const response = await fetch(`${API_URL}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Login failed.");
      sessionStorage.setItem("autoace_access_token", body.access_token);
      setToken(body.access_token);
      setPassword("");
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : "Login failed.");
    } finally {
      setLoginBusy(false);
    }
  }

  function logout() {
    sessionStorage.removeItem("autoace_access_token");
    setToken(null);
    setRows([]);
    setSelectedFiles([]);
  }

  function chooseFiles(files: FileList | null) {
    if (!files?.length) return;
    setSelectedFiles(Array.from(files));
    setRows([]);
    setWarnings([]);
    setValidationSummary(null);
    setBatchError("");
    setExpanded(new Set());
  }

  function handleStreamEvent(event: StreamEvent) {
    if (event.type === "batch_started") {
      setWarnings(event.warnings || []);
      setRows(
        (event.items || []).map((item) => ({
          name: item.name,
          status: item.status,
          error: item.error,
        })),
      );
      return;
    }

    if (event.type === "file_completed" && event.name && event.result) {
      setRows((current) =>
        current.map((row) =>
          row.name === event.name
            ? {
                ...row,
                status: "completed",
                result: event.result,
                model: event.model,
                request_seconds: event.request_seconds,
                usage: event.usage,
                validation: (event.validation as Row["validation"]) || null,
                error: null,
              }
            : row,
        ),
      );
      return;
    }

    if (event.type === "file_failed" && event.name) {
      setRows((current) =>
        current.map((row) =>
          row.name === event.name
            ? { ...row, status: "failed", error: event.error || "Analysis failed." }
            : row,
        ),
      );
      return;
    }

    if (event.type === "batch_completed") {
      setValidationSummary((event.validation as ValidationSummary | null) || null);
    }
  }

  async function runBatch() {
    if (!token || !selectedFiles.length || processing) return;
    setProcessing(true);
    setRows([]);
    setWarnings([]);
    setValidationSummary(null);
    setBatchError("");
    setExpanded(new Set());

    const form = new FormData();
    selectedFiles.forEach((file) => form.append("files", file, file.name));

    try {
      const response = await fetch(`${API_URL}/api/v1/batches/analyze`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });

      if (response.status === 401) {
        logout();
        throw new Error("Your session expired. Please log in again.");
      }
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || `Batch request failed (${response.status}).`);
      }
      if (!response.body) throw new Error("Streaming response is unavailable in this browser.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          handleStreamEvent(JSON.parse(line) as StreamEvent);
        }
        if (done) break;
      }
      if (buffer.trim()) handleStreamEvent(JSON.parse(buffer) as StreamEvent);
    } catch (error) {
      setBatchError(error instanceof Error ? error.message : "Batch processing failed.");
    } finally {
      setProcessing(false);
    }
  }

  function exportJson() {
    const payload = rows.map((row) => ({
      name: row.name,
      result_json: row.result || null,
      ...(row.error ? { error: row.error } : {}),
    }));
    downloadBlob("autoace_predictions.json", JSON.stringify(payload, null, 2), "application/json");
  }

  function exportCsv() {
    const lines = ["name,result_json"];
    rows.forEach((row) => {
      lines.push(`${csvEscape(row.name)},${csvEscape(row.result ? JSON.stringify(row.result) : "")}`);
    });
    downloadBlob("autoace_predictions.csv", `${lines.join("\n")}\n`, "text/csv;charset=utf-8");
  }

  function toggleExpanded(name: string) {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  if (!token) {
    return (
      <main className="min-h-screen bg-slate-950 px-5 py-12 text-slate-100 sm:px-8">
        <div className="mx-auto flex min-h-[calc(100vh-6rem)] max-w-5xl items-center justify-center">
          <div className="grid w-full overflow-hidden rounded-3xl border border-white/10 bg-slate-900 shadow-2xl shadow-black/30 lg:grid-cols-[1.1fr_0.9fr]">
            <section className="hidden min-h-[570px] flex-col justify-between bg-gradient-to-br from-slate-800 to-slate-950 p-12 lg:flex">
              <div>
                <div className="mb-10 inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-300">
                  AutoAce AI
                </div>
                <h1 className="max-w-lg text-5xl font-semibold leading-[1.05] tracking-tight">
                  Production call analysis, built for batch evaluation.
                </h1>
                <p className="mt-6 max-w-lg text-lg leading-8 text-slate-300">
                  Classify customer emotion, background noise, call quality, overlap, and silence from real dealership audio.
                </p>
              </div>
              <p className="text-sm text-slate-500">Confidential technical trial · evaluator access only</p>
            </section>

            <section className="flex min-h-[570px] items-center p-7 sm:p-12">
              <form onSubmit={handleLogin} className="w-full">
                <div className="mb-8">
                  <p className="text-sm font-medium text-emerald-400">Evaluator dashboard</p>
                  <h2 className="mt-2 text-3xl font-semibold tracking-tight text-white">Sign in</h2>
                  <p className="mt-3 text-sm leading-6 text-slate-400">Use the credentials supplied with the trial submission.</p>
                </div>
                <label className="block text-sm font-medium text-slate-300">
                  Username
                  <input
                    value={username}
                    onChange={(event) => setUsername(event.target.value)}
                    autoComplete="username"
                    className="mt-2 h-12 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 text-base text-white outline-none transition focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20"
                  />
                </label>
                <label className="mt-5 block text-sm font-medium text-slate-300">
                  Password
                  <input
                    type="password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    autoComplete="current-password"
                    className="mt-2 h-12 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 text-base text-white outline-none transition focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20"
                  />
                </label>
                {loginError && <p className="mt-4 rounded-lg bg-rose-500/10 px-3 py-2 text-sm text-rose-300">{loginError}</p>}
                <button
                  type="submit"
                  disabled={loginBusy || !username || !password}
                  className="mt-7 h-12 w-full rounded-xl bg-emerald-400 font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loginBusy ? "Signing in…" : "Sign in"}
                </button>
              </form>
            </section>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-5 py-4 sm:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-950 text-sm font-bold text-emerald-300">AA</div>
            <div>
              <p className="text-sm font-semibold leading-4">AutoAce AI</p>
              <p className="text-xs text-slate-500">Call Analysis Console</p>
            </div>
          </div>
          <button onClick={logout} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-50">Sign out</button>
        </div>
      </header>

      <div className="mx-auto max-w-[1500px] px-5 py-8 sm:px-8">
        <div className="mb-8 flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <div>
            <p className="text-sm font-semibold text-emerald-700">Voice tone & background noise trial</p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight sm:text-4xl">Batch evaluation</h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
              Upload the evaluation folder or ZIP containing the audio clips and one CSV manifest. Labels are used only for post-prediction validation and are never sent to the model.
            </p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm">
            <span className="text-slate-500">API</span>
            <span className="ml-2 font-mono text-xs text-slate-800">{API_URL}</span>
          </div>
        </div>

        <section className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
              <div>
                <h2 className="text-lg font-semibold">Evaluation batch</h2>
                <p className="mt-1 text-sm text-slate-500">ZIP archive or folder selection · one CSV manifest required</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <input ref={zipInput} type="file" accept=".zip,application/zip" className="hidden" onChange={(event) => chooseFiles(event.target.files)} />
                <input ref={folderInput} type="file" multiple className="hidden" onChange={(event) => chooseFiles(event.target.files)} />
                <button onClick={() => zipInput.current?.click()} disabled={processing} className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50">Choose ZIP</button>
                <button onClick={() => folderInput.current?.click()} disabled={processing} className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50">Choose folder</button>
              </div>
            </div>

            <div
              className="mt-5 rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50 p-8 text-center transition hover:border-slate-300"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                chooseFiles(event.dataTransfer.files);
              }}
            >
              {selectedFiles.length ? (
                <>
                  <p className="font-semibold text-slate-900">{selectedFiles.length === 1 ? selectedFiles[0].name : `${selectedFiles.length} files selected`}</p>
                  <p className="mt-1 text-sm text-slate-500">{formatBytes(selectedBytes)} total</p>
                </>
              ) : (
                <>
                  <p className="font-semibold text-slate-800">Drop a ZIP here, or choose a ZIP/folder above</p>
                  <p className="mt-2 text-sm text-slate-500">Audio filenames must match the manifest name column exactly.</p>
                </>
              )}
            </div>

            {batchError && <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">{batchError}</div>}
            {warnings.length > 0 && (
              <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
                <p className="font-semibold">Batch warnings</p>
                <ul className="mt-2 list-disc space-y-1 pl-5">{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
              </div>
            )}

            <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <button
                onClick={runBatch}
                disabled={!selectedFiles.length || processing}
                className="h-11 rounded-xl bg-slate-950 px-5 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {processing ? "Analyzing batch…" : "Run analysis"}
              </button>
              {rows.length > 0 && <p className="text-sm text-slate-500">{finished}/{rows.length} finished · {completed} completed · {failed} failed</p>}
            </div>

            {rows.length > 0 && (
              <div className="mt-5">
                <div className="mb-2 flex items-center justify-between text-xs font-medium text-slate-500"><span>Batch progress</span><span>{progress}%</span></div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-emerald-500 transition-all duration-300" style={{ width: `${progress}%` }} /></div>
              </div>
            )}
          </div>

          <aside className="rounded-2xl border border-slate-200 bg-slate-950 p-5 text-white shadow-sm sm:p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-300">Run summary</p>
            <div className="mt-5 grid grid-cols-3 gap-3 xl:grid-cols-1">
              <div className="rounded-xl bg-white/5 p-3"><p className="text-2xl font-semibold">{completed}</p><p className="mt-1 text-xs text-slate-400">Completed</p></div>
              <div className="rounded-xl bg-white/5 p-3"><p className="text-2xl font-semibold">{failed}</p><p className="mt-1 text-xs text-slate-400">Failed</p></div>
              <div className="rounded-xl bg-white/5 p-3"><p className="text-2xl font-semibold">{rows.length}</p><p className="mt-1 text-xs text-slate-400">Total</p></div>
            </div>
            {validationSummary ? (
              <div className="mt-5 rounded-xl border border-emerald-400/20 bg-emerald-400/10 p-4">
                <p className="text-xs text-emerald-200">Labeled development-set validation</p>
                <p className="mt-1 text-3xl font-semibold">{Math.round(validationSummary.exact_field_accuracy * 100)}%</p>
                <p className="mt-1 text-xs text-emerald-100/70">Exact field accuracy across {validationSummary.labeled_files} labeled files</p>
              </div>
            ) : (
              <p className="mt-5 text-xs leading-5 text-slate-400">If result_json labels are present, validation is calculated only after predictions have been generated.</p>
            )}
          </aside>
        </section>

        {rows.length > 0 && (
          <section className="mt-6 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="flex flex-col justify-between gap-3 border-b border-slate-200 px-5 py-4 sm:flex-row sm:items-center sm:px-6">
              <div>
                <h2 className="font-semibold">Predictions</h2>
                <p className="mt-1 text-xs text-slate-500">One independent structured result per manifest row.</p>
              </div>
              <div className="flex gap-2">
                <button onClick={exportCsv} className="rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">Download CSV</button>
                <button onClick={exportJson} className="rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">Download JSON</button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-[1350px] w-full text-left text-sm">
                <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="px-5 py-3 font-semibold">File</th>
                    <th className="px-3 py-3 font-semibold">Status</th>
                    <th className="px-3 py-3 font-semibold">Tone</th>
                    <th className="px-3 py-3 font-semibold">Intensity</th>
                    <th className="px-3 py-3 font-semibold">Noise</th>
                    <th className="px-3 py-3 font-semibold">Severity</th>
                    <th className="px-3 py-3 font-semibold">Quality</th>
                    <th className="px-3 py-3 font-semibold">Overlap</th>
                    <th className="px-3 py-3 font-semibold">Silence</th>
                    <th className="px-3 py-3 font-semibold">Confidence</th>
                    <th className="px-3 py-3 font-semibold">Latency</th>
                    <th className="px-5 py-3 font-semibold"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {rows.map((row) => (
                    <FragmentRow key={row.name} row={row} expanded={expanded.has(row.name)} onToggle={() => toggleExpanded(row.name)} />
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

function FragmentRow({ row, expanded, onToggle }: { row: Row; expanded: boolean; onToggle: () => void }) {
  return (
    <>
      <tr className="align-top hover:bg-slate-50/60">
        <td className="px-5 py-4 font-mono text-xs font-medium text-slate-800">{row.name}</td>
        <td className="px-3 py-4"><span className={`inline-flex rounded-full px-2 py-1 text-xs font-semibold ring-1 ring-inset ${statusClass(row.status)}`}>{row.status}</span></td>
        <td className="px-3 py-4 capitalize">{row.result?.emotional_tone || "—"}</td>
        <td className="px-3 py-4 capitalize">{row.result?.emotional_intensity || "—"}</td>
        <td className="max-w-44 px-3 py-4">{row.result ? (row.result.background_noise_present ? row.result.background_noise_type || "Present" : "None") : "—"}</td>
        <td className="px-3 py-4 capitalize">{row.result?.background_noise_severity || "—"}</td>
        <td className="px-3 py-4">{row.result?.audio_quality?.replaceAll("_", " ") || "—"}</td>
        <td className="px-3 py-4">{boolText(row.result?.speaker_overlap_present)}</td>
        <td className="px-3 py-4">{boolText(row.result?.long_silence_present)}</td>
        <td className="px-3 py-4">{row.result ? row.result.confidence.toFixed(2) : "—"}</td>
        <td className="px-3 py-4">{row.request_seconds !== undefined ? `${row.request_seconds.toFixed(2)}s` : "—"}</td>
        <td className="px-5 py-4 text-right"><button onClick={onToggle} className="text-xs font-semibold text-slate-600 underline decoration-slate-300 underline-offset-4">{expanded ? "Hide" : "Details"}</button></td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={12} className="bg-slate-950 px-5 py-5 text-slate-100">
            {row.error ? <p className="text-sm text-rose-300">{row.error}</p> : <pre className="overflow-x-auto text-xs leading-5 text-slate-300">{JSON.stringify({ result: row.result, model: row.model, request_seconds: row.request_seconds, usage: row.usage, validation: row.validation }, null, 2)}</pre>}
          </td>
        </tr>
      )}
    </>
  );
}
