import React, { useEffect, useRef, useState } from "react";

const WS_URL =
  (import.meta.env.VITE_WS_URL || `ws://${window.location.hostname}:8000`) +
  "/ws/session";

const FRAME_INTERVAL_MS = 1000;
const JPEG_QUALITY = 0.6;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

function StatusPill({ status }) {
  const colors = {
    good: "bg-emerald-600",
    slouched: "bg-amber-600",
    tilted: "bg-amber-600",
    off_center: "bg-rose-600",
    too_much_angle: "bg-rose-600",
    no_person: "bg-slate-600",
  };
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
        colors[status] || "bg-slate-600"
      }`}
    >
      {status || "—"}
    </span>
  );
}

function Report({ report, metrics }) {
  return (
    <div className="space-y-6">
      <div className="rounded-2xl bg-slate-900 p-6 ring-1 ring-slate-800">
        <div className="text-xs uppercase tracking-wide text-slate-400">Global score</div>
        <div className="mt-2 text-6xl font-bold">{report.global_score}</div>
        <div className="mt-2 text-slate-300">{report.headline}</div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="rounded-2xl bg-slate-900 p-6 ring-1 ring-slate-800">
          <div className="mb-3 text-sm font-semibold text-emerald-400">Strengths</div>
          <ul className="list-disc space-y-2 pl-5 text-slate-200">
            {report.strengths?.map((s, i) => <li key={i}>{s}</li>)}
          </ul>
        </div>
        <div className="rounded-2xl bg-slate-900 p-6 ring-1 ring-slate-800">
          <div className="mb-3 text-sm font-semibold text-amber-400">Improvements</div>
          <ul className="list-disc space-y-2 pl-5 text-slate-200">
            {report.improvements?.map((s, i) => <li key={i}>{s}</li>)}
          </ul>
        </div>
      </div>

      <div className="rounded-2xl bg-indigo-900/30 p-6 ring-1 ring-indigo-800">
        <div className="mb-2 text-sm font-semibold text-indigo-300">Actionable tip</div>
        <div className="text-slate-100">{report.actionable_tip}</div>
      </div>

      <details className="rounded-2xl bg-slate-900 p-4 ring-1 ring-slate-800">
        <summary className="cursor-pointer text-sm text-slate-400">Raw metrics</summary>
        <pre className="mt-3 overflow-auto text-xs text-slate-300">
          {JSON.stringify(metrics, null, 2)}
        </pre>
      </details>
    </div>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const wsRef = useRef(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const frameTimerRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Idle. Click Start Interview to begin.");
  const [liveMetrics, setLiveMetrics] = useState(null);
  const [report, setReport] = useState(null);
  const [finalMetrics, setFinalMetrics] = useState(null);
  const [waitingForReport, setWaitingForReport] = useState(false);

  useEffect(() => {
    return () => stopEverything();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    drawOverlay(liveMetrics);
  }, [liveMetrics]);

  function stopEverything() {
    if (frameTimerRef.current) clearInterval(frameTimerRef.current);
    frameTimerRef.current = null;
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    recorderRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    wsRef.current = null;
    drawOverlay(null);
  }

  function drawOverlay(metrics) {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    if (!video || !overlay) return;

    const width = video.clientWidth;
    const height = video.clientHeight;
    if (!width || !height) return;

    overlay.width = width;
    overlay.height = height;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    const eye = metrics?.eye_contact;
    const posture = metrics?.posture;
    const faceLines = metrics?.face_lines;
    const hands = metrics?.hands?.hands || [];
    const faceMask = metrics?.face_mask;
    const objectBboxes = metrics?.objects?.bboxes || [];
    const objectFaceOverlap = metrics?.objects?.face_overlap === true;
    const faceBox = posture?.face_bbox;
    const avgBox = posture?.face_bbox_avg;
    const hasFace = !!faceBox;
    const isWarning = objectFaceOverlap || faceMask?.masked === true;

    if (hasFace) {
      const { x, y, w, h } = faceBox;
      if (isWarning) {
        ctx.fillStyle = "rgba(239, 68, 68, 0.15)";
        ctx.fillRect(x * width, y * height, w * width, h * height);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.95)";
        ctx.lineWidth = 3;
        ctx.strokeRect(x * width, y * height, w * width, h * height);
        ctx.fillStyle = "rgba(239, 68, 68, 0.95)";
        ctx.font = "bold 13px monospace";
        ctx.fillText(
          faceMask?.reason === "hand" ? "HAND OVER FACE" : "OBJECT DETECTED",
          x * width + 4,
          y * height - 6,
        );
      } else {
        ctx.strokeStyle = "rgba(16, 185, 129, 0.9)";
        ctx.lineWidth = 2;
        ctx.strokeRect(x * width, y * height, w * width, h * height);
      }
    }

    // Orange rectangles around each detected foreign object
    objectBboxes.forEach((bbox) => {
      ctx.strokeStyle = "rgba(249, 115, 22, 0.9)";
      ctx.lineWidth = 2;
      ctx.strokeRect(bbox.x * width, bbox.y * height, bbox.w * width, bbox.h * height);
      ctx.fillStyle = "rgba(249, 115, 22, 0.9)";
      ctx.font = "bold 12px monospace";
      ctx.fillText("OBJECT", bbox.x * width + 4, bbox.y * height + 14);
    });

    if (avgBox) {
      const { x: avgX, y: avgY, w: avgW, h: avgH } = avgBox;
      ctx.strokeStyle = "rgba(56, 189, 248, 0.9)";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(avgX * width, avgY * height, avgW * width, avgH * height);
      ctx.setLineDash([]);
    }

    const drawPoint = (pt, color, radius = 4) => {
      if (!pt || pt.length < 2) return;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pt[0] * width, pt[1] * height, radius, 0, Math.PI * 2);
      ctx.fill();
    };

    const drawLine = (line, color, dashed = false) => {
      if (!line?.start || !line?.end) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (dashed) ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(line.start[0] * width, line.start[1] * height);
      ctx.lineTo(line.end[0] * width, line.end[1] * height);
      ctx.stroke();
      if (dashed) ctx.setLineDash([]);
    };

    const drawHand = (hand) => {
      const landmarks = hand?.landmarks;
      if (!landmarks || landmarks.length < 21) return;
      ctx.strokeStyle = "rgba(34, 197, 94, 0.85)";
      ctx.lineWidth = 2;
      HAND_CONNECTIONS.forEach(([a, b]) => {
        const p1 = landmarks[a];
        const p2 = landmarks[b];
        if (!p1 || !p2) return;
        ctx.beginPath();
        ctx.moveTo(p1[0] * width, p1[1] * height);
        ctx.lineTo(p2[0] * width, p2[1] * height);
        ctx.stroke();
      });

      ctx.fillStyle = "rgba(34, 197, 94, 0.85)";
      landmarks.forEach((pt) => {
        if (!pt || pt.length < 2) return;
        ctx.beginPath();
        ctx.arc(pt[0] * width, pt[1] * height, 2, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    drawLine(faceLines?.upper_eye_line, "rgba(250, 204, 21, 0.9)");
    drawLine(faceLines?.lower_eye_line, "rgba(253, 186, 116, 0.9)");
    drawLine(faceLines?.midline, "rgba(168, 85, 247, 0.9)", true);

    hands.forEach(drawHand);

    if (eye?.face_landmarks?.length) {
      ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
      eye.face_landmarks.forEach((pt) => {
        if (!pt || pt.length < 2) return;
        ctx.beginPath();
        ctx.arc(pt[0] * width, pt[1] * height, 1.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    if (eye?.face_detected) {
      drawPoint(eye.eye_points?.left_iris, "rgba(59, 130, 246, 0.9)");
      drawPoint(eye.eye_points?.right_iris, "rgba(59, 130, 246, 0.9)");
    }
  }

  async function captureAndSendFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY)
    );
    if (!blob) return;
    ws.send(JSON.stringify({ type: "frame_meta" }));
    ws.send(await blob.arrayBuffer());
  }

  async function start() {
    setReport(null);
    setFinalMetrics(null);
    setLiveMetrics(null);
    setStatus("Requesting camera + microphone...");

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: true,
      });
    } catch (e) {
      setStatus(`Permission denied or no device: ${e.message}`);
      return;
    }
    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play().catch(() => {});
    }

    setStatus("Connecting to backend...");
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("Connected. Recording...");
      setRunning(true);

      frameTimerRef.current = setInterval(captureAndSendFrame, FRAME_INTERVAL_MS);

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const audioStream = new MediaStream(stream.getAudioTracks());
      const recorder = new MediaRecorder(audioStream, { mimeType });
      recorderRef.current = recorder;
      recorder.ondataavailable = async (e) => {
        if (e.data && e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "audio_meta" }));
          ws.send(await e.data.arrayBuffer());
        }
      };
      recorder.start(2000); // emit a chunk every 2s
    };

    ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === "metrics") {
        setLiveMetrics(msg);
      } else if (msg.type === "status") {
        setStatus(msg.message);
      } else if (msg.type === "report") {
        setReport(msg.report);
        setFinalMetrics(msg.metrics);
        setStatus("Report ready.");
        setWaitingForReport(false);
        stopEverything();
        setRunning(false);
      } else if (msg.type === "error") {
        setStatus(`Error: ${msg.message}`);
      }
    };

    ws.onclose = () => {
      setRunning(false);
      if (frameTimerRef.current) clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
    };

    ws.onerror = () => setStatus("WebSocket error.");
  }

  async function stop() {
    setStatus("Stopping recording...");
    setWaitingForReport(true);
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    // Give the recorder a beat to flush its last chunk.
    await new Promise((r) => setTimeout(r, 500));
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
    }
    if (frameTimerRef.current) clearInterval(frameTimerRef.current);
    frameTimerRef.current = null;
  }

  return (
    <div className="mx-auto max-w-screen-2xl px-5 py-8">
      <header className="mb-4">
        <h1 className="text-2xl font-bold tracking-tight">InterviewIQ</h1>
        <p className="text-slate-400">AI interview coach — live feedback on eye contact, posture, and pace.</p>
      </header>

      <div className="rounded-2xl bg-slate-900 p-3 ring-1 ring-slate-800">
        <div className="relative h-[70vh] w-full overflow-hidden rounded-2xl bg-black md:h-[78vh] lg:h-[82vh]">
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="h-full w-full object-cover"
          />
          <canvas
            ref={overlayRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
          <div className="absolute left-3 top-3 rounded-lg bg-slate-950/60 px-3 py-2 text-xs text-slate-100 ring-1 ring-slate-800 backdrop-blur">
            <div className="flex items-center gap-2">
              <span className="uppercase tracking-wide text-slate-400">Eye</span>
              <span className="font-semibold">
                {liveMetrics ? `${liveMetrics.eye_contact_pct}%` : "—"}
              </span>
              <span className="text-slate-400">score {liveMetrics?.eye_contact?.score ?? "—"}</span>
            </div>
            <div className="mt-1 flex items-center gap-2">
              <span className="uppercase tracking-wide text-slate-400">Posture</span>
              <StatusPill status={liveMetrics?.posture?.status} />
              <span className="text-slate-400">
                tilt {liveMetrics?.posture?.shoulder_tilt_deg ?? "—"}° · head {liveMetrics?.posture?.head_offset ?? "—"}
              </span>
            </div>
            <div className="mt-1 text-slate-400">
              head pitch {liveMetrics?.face_lines?.eye_gap_status ?? "—"}
            </div>
            <div className="mt-1 text-slate-400">
              face masked {liveMetrics?.face_mask?.masked ? "yes" : "no"} ({liveMetrics?.face_mask?.reason ?? "—"})
            </div>
            <div className="mt-1 text-slate-400">
              objects {liveMetrics?.hands?.bboxes?.length ?? 0} · motion {liveMetrics?.motion?.motion_ratio ?? "—"}
            </div>
            <div className="mt-1 text-slate-500">frames {liveMetrics?.frame_index ?? 0}</div>
          </div>
        </div>
        <canvas ref={canvasRef} className="hidden" />
        <div className="mt-3 flex gap-3">
          {!running ? (
            <button
              onClick={start}
              disabled={waitingForReport}
              className="rounded-xl bg-emerald-600 px-5 py-2 font-medium hover:bg-emerald-500 disabled:opacity-50"
            >
              Start Interview
            </button>
          ) : (
            <button
              onClick={stop}
              className="rounded-xl bg-rose-600 px-5 py-2 font-medium hover:bg-rose-500"
            >
              Stop
            </button>
          )}
          <div className="self-center text-sm text-slate-400">{status}</div>
        </div>
      </div>
      {(report || waitingForReport) && (
        <section className="mt-10">
          <h2 className="mb-4 text-2xl font-semibold">Session report</h2>
          {report ? (
            <Report report={report} metrics={finalMetrics} />
          ) : (
            <div className="rounded-2xl bg-slate-900 p-6 text-slate-400 ring-1 ring-slate-800">
              {status}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
