import { useEffect, useRef, useState, useCallback } from "react";
import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Play, Square, AlertTriangle, Activity, Eye, EyeOff, Timer, Zap, Settings2, Smile, Move3d,
} from "lucide-react";

const LEFT_EYE = { top: [159, 158, 157], bot: [145, 153, 154], l: 33, r: 133 };
const RIGHT_EYE = { top: [386, 385, 384], bot: [374, 380, 381], l: 362, r: 263 };
// Mouth landmarks (MediaPipe FaceMesh)
const MOUTH = {
  top: [13, 81, 311],   // upper inner lip points
  bot: [14, 178, 402],  // lower inner lip points
  l: 78, r: 308,        // mouth corners
};
// Nose tip used for head-nod tracking
const NOSE_TIP = 1;

type Pt = { x: number; y: number };
const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);

function ear(lm: Pt[], eye: typeof LEFT_EYE) {
  const v =
    (dist(lm[eye.top[0]], lm[eye.bot[0]]) +
      dist(lm[eye.top[1]], lm[eye.bot[1]]) +
      dist(lm[eye.top[2]], lm[eye.bot[2]])) /
    3;
  const h = dist(lm[eye.l], lm[eye.r]);
  return h > 0 ? v / h : 0;
}

function mar(lm: Pt[]) {
  const v =
    (dist(lm[MOUTH.top[0]], lm[MOUTH.bot[0]]) +
      dist(lm[MOUTH.top[1]], lm[MOUTH.bot[1]]) +
      dist(lm[MOUTH.top[2]], lm[MOUTH.bot[2]])) /
    3;
  const h = dist(lm[MOUTH.l], lm[MOUTH.r]);
  return h > 0 ? v / h : 0;
}

function makeAlarm(ctx: AudioContext, freq: number, type: OscillatorType = "square") {
  let osc: OscillatorNode | null = null;
  let gain: GainNode | null = null;
  return {
    start() {
      if (osc) return;
      osc = ctx.createOscillator();
      gain = ctx.createGain();
      osc.type = type;
      osc.frequency.value = freq;
      gain.gain.value = 0.0;
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      const t0 = ctx.currentTime;
      for (let i = 0; i < 200; i++) {
        gain.gain.setValueAtTime(0.25, t0 + i * 0.5);
        gain.gain.setValueAtTime(0.0, t0 + i * 0.5 + 0.25);
      }
    },
    stop() {
      try { osc?.stop(); } catch { /* ignore */ }
      osc?.disconnect();
      gain?.disconnect();
      osc = null;
      gain = null;
    },
  };
}

function StatCard({
  icon, label, value, sub, accent,
}: { icon: React.ReactNode; label: string; value: React.ReactNode; sub?: string; accent?: string }) {
  return (
    <div className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        {icon}<span>{label}</span>
      </div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${accent ?? ""}`}>{value}</div>
      {sub && <div className="text-[10px] text-muted-foreground mt-0.5">{sub}</div>}
    </div>
  );
}

export default function DrowsinessDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const closureAlarmRef = useRef<ReturnType<typeof makeAlarm> | null>(null);
  const blinkAlarmRef = useRef<ReturnType<typeof makeAlarm> | null>(null);
  const yawnAlarmRef = useRef<ReturnType<typeof makeAlarm> | null>(null);
  const nodAlarmRef = useRef<ReturnType<typeof makeAlarm> | null>(null);

  const stateRef = useRef({
    closedSince: null as number | null,
    closedFrameRun: 0,
    blinkTimes: [] as number[],
    blinkingSince: null as number | null,
    closureAlarmOn: false,
    blinkAlarmOn: false,
    yawnSince: null as number | null,
    yawnAlarmOn: false,
    yawnCount: 0,
    noseBaseline: null as number | null,
    noseHist: [] as { t: number; y: number }[],
    nodCount: 0,
    nodTimes: [] as number[],
    nodAlarmOn: false,
    lastNodDir: 0, // -1 up, +1 down, 0 none
  });

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [earValue, setEarValue] = useState(0);
  const [eyesClosed, setEyesClosed] = useState(false);
  const [closedFor, setClosedFor] = useState(0);
  const [blinkCount, setBlinkCount] = useState(0);
  const [closureAlarm, setClosureAlarm] = useState(false);
  const [blinkAlarm, setBlinkAlarm] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [marValue, setMarValue] = useState(0);
  const [yawning, setYawning] = useState(false);
  const [yawnFor, setYawnFor] = useState(0);
  const [yawnCount, setYawnCount] = useState(0);
  const [yawnAlarm, setYawnAlarm] = useState(false);
  const [nodCount, setNodCount] = useState(0);
  const [nodAlarm, setNodAlarm] = useState(false);

  const [earThreshold, setEarThreshold] = useState(0.22);
  const [thresholdSec, setThresholdSec] = useState(10);
  const [blinkWindowSec, setBlinkWindowSec] = useState(5);
  const [minBlinks, setMinBlinks] = useState(4);
  const [blinkAlertSec, setBlinkAlertSec] = useState(6);
  const [marThreshold, setMarThreshold] = useState(0.6);
  const [yawnSec, setYawnSec] = useState(2.5);
  const [nodWindowSec, setNodWindowSec] = useState(15);
  const [minNods, setMinNods] = useState(3);
  const [nodDelta, setNodDelta] = useState(0.025); // normalized vertical motion

  const stop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    const v = videoRef.current;
    if (v?.srcObject) {
      (v.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      v.srcObject = null;
    }
    closureAlarmRef.current?.stop();
    blinkAlarmRef.current?.stop();
    yawnAlarmRef.current?.stop();
    nodAlarmRef.current?.stop();
    closureAlarmRef.current = null;
    blinkAlarmRef.current = null;
    yawnAlarmRef.current = null;
    nodAlarmRef.current = null;
    setRunning(false);
    setClosureAlarm(false);
    setBlinkAlarm(false);
    setYawnAlarm(false);
    setNodAlarm(false);
    setFaceDetected(false);
    setStatus("Stopped");
  }, []);

  const start = useCallback(async () => {
    try {
      setStatus("Loading model…");
      if (!landmarkerRef.current) {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm",
        );
        landmarkerRef.current = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
        });
      }

      setStatus("Requesting camera…");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();

      audioCtxRef.current ??= new AudioContext();
      stateRef.current = {
        closedSince: null,
        closedFrameRun: 0,
        blinkTimes: [],
        blinkingSince: null,
        closureAlarmOn: false,
        blinkAlarmOn: false,
        yawnSince: null,
        yawnAlarmOn: false,
        yawnCount: 0,
        noseBaseline: null,
        noseHist: [],
        nodCount: 0,
        nodTimes: [],
        nodAlarmOn: false,
        lastNodDir: 0,
      };
      setRunning(true);
      setStatus("Live");

      const loop = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const lk = landmarkerRef.current;
        if (!video || !canvas || !lk) return;
        if (video.readyState < 2) {
          rafRef.current = requestAnimationFrame(loop);
          return;
        }
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d")!;
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0);
        ctx.restore();

        const now = performance.now();
        const result = lk.detectForVideo(video, now);
        const s = stateRef.current;
        const ts = now / 1000;

        const hasFace = !!result.faceLandmarks?.length;
        setFaceDetected(hasFace);

        if (hasFace) {
          const lm = result.faceLandmarks[0] as Pt[];
          const e = (ear(lm, LEFT_EYE) + ear(lm, RIGHT_EYE)) / 2;
          setEarValue(e);
          const closed = e < earThreshold;
          setEyesClosed(closed);

          if (closed) {
            if (s.closedSince === null) s.closedSince = ts;
            const elapsed = ts - s.closedSince;
            setClosedFor(elapsed);
            if (elapsed >= thresholdSec && !s.closureAlarmOn) {
              s.closureAlarmOn = true;
              setClosureAlarm(true);
              closureAlarmRef.current = makeAlarm(audioCtxRef.current!, 880, "square");
              closureAlarmRef.current.start();
            }
            s.closedFrameRun += 1;
          } else {
            if (s.closureAlarmOn) {
              s.closureAlarmOn = false;
              setClosureAlarm(false);
              closureAlarmRef.current?.stop();
              closureAlarmRef.current = null;
            }
            s.closedSince = null;
            setClosedFor(0);
            if (s.closedFrameRun >= 1 && s.closedFrameRun <= 7) {
              s.blinkTimes.push(ts);
            }
            s.closedFrameRun = 0;
          }

          while (s.blinkTimes.length && ts - s.blinkTimes[0] > blinkWindowSec) {
            s.blinkTimes.shift();
          }
          setBlinkCount(s.blinkTimes.length);

          if (s.blinkTimes.length >= minBlinks) {
            if (s.blinkingSince === null) s.blinkingSince = ts;
            if (ts - s.blinkingSince >= blinkAlertSec && !s.blinkAlarmOn) {
              s.blinkAlarmOn = true;
              setBlinkAlarm(true);
              blinkAlarmRef.current = makeAlarm(audioCtxRef.current!, 440, "sawtooth");
              blinkAlarmRef.current.start();
            }
          } else {
            if (s.blinkAlarmOn) {
              s.blinkAlarmOn = false;
              setBlinkAlarm(false);
              blinkAlarmRef.current?.stop();
              blinkAlarmRef.current = null;
            }
            s.blinkingSince = null;
          }

          // Yawning (MAR)
          const m = mar(lm);
          setMarValue(m);
          const isYawn = m > marThreshold;
          setYawning(isYawn);
          if (isYawn) {
            if (s.yawnSince === null) s.yawnSince = ts;
            const dur = ts - s.yawnSince;
            setYawnFor(dur);
            if (dur >= yawnSec && !s.yawnAlarmOn) {
              s.yawnAlarmOn = true;
              s.yawnCount += 1;
              setYawnCount(s.yawnCount);
              setYawnAlarm(true);
              yawnAlarmRef.current = makeAlarm(audioCtxRef.current!, 660, "triangle");
              yawnAlarmRef.current.start();
            }
          } else {
            if (s.yawnAlarmOn) {
              s.yawnAlarmOn = false;
              setYawnAlarm(false);
              yawnAlarmRef.current?.stop();
              yawnAlarmRef.current = null;
            }
            s.yawnSince = null;
            setYawnFor(0);
          }

          // Head nodding — track nose tip vertical (normalized) relative to face height
          const nose = lm[NOSE_TIP];
          // Normalize by inter-eye distance to be scale-invariant
          const faceScale = dist(lm[LEFT_EYE.l], lm[RIGHT_EYE.r]) || 1;
          const yNorm = nose.y / faceScale;
          s.noseHist.push({ t: ts, y: yNorm });
          while (s.noseHist.length && ts - s.noseHist[0].t > 1.5) s.noseHist.shift();
          if (s.noseHist.length > 4) {
            const ys = s.noseHist.map((p) => p.y);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            const range = maxY - minY;
            // Detect a down-up cycle exceeding nodDelta
            if (range >= nodDelta) {
              const last = s.noseHist[s.noseHist.length - 1].y;
              const dir = last > (minY + maxY) / 2 ? 1 : -1;
              if (dir !== s.lastNodDir) {
                s.lastNodDir = dir;
                if (dir === -1) {
                  // completed a down-then-up motion = one nod
                  s.nodTimes.push(ts);
                }
              }
            }
          }
          while (s.nodTimes.length && ts - s.nodTimes[0] > nodWindowSec) s.nodTimes.shift();
          setNodCount(s.nodTimes.length);
          if (s.nodTimes.length >= minNods) {
            if (!s.nodAlarmOn) {
              s.nodAlarmOn = true;
              setNodAlarm(true);
              nodAlarmRef.current = makeAlarm(audioCtxRef.current!, 520, "sine");
              nodAlarmRef.current.start();
            }
          } else if (s.nodAlarmOn) {
            s.nodAlarmOn = false;
            setNodAlarm(false);
            nodAlarmRef.current?.stop();
            nodAlarmRef.current = null;
          }

          const idxs = [
            ...LEFT_EYE.top, ...LEFT_EYE.bot, LEFT_EYE.l, LEFT_EYE.r,
            ...RIGHT_EYE.top, ...RIGHT_EYE.bot, RIGHT_EYE.l, RIGHT_EYE.r,
          ];
          ctx.fillStyle = closed ? "#ef4444" : "#4ade80";
          ctx.shadowColor = closed ? "#ef4444" : "#4ade80";
          ctx.shadowBlur = 8;
          for (const i of idxs) {
            const p = lm[i];
            const x = canvas.width - p.x * canvas.width;
            const y = p.y * canvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.shadowBlur = 0;

          // Mouth landmarks
          const mouthIdxs = [...MOUTH.top, ...MOUTH.bot, MOUTH.l, MOUTH.r];
          ctx.fillStyle = isYawn ? "#f59e0b" : "#60a5fa";
          ctx.shadowColor = isYawn ? "#f59e0b" : "#60a5fa";
          ctx.shadowBlur = 8;
          for (const i of mouthIdxs) {
            const p = lm[i];
            const x = canvas.width - p.x * canvas.width;
            const y = p.y * canvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.shadowBlur = 0;
        }

        rafRef.current = requestAnimationFrame(loop);
      };
      rafRef.current = requestAnimationFrame(loop);
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${(err as Error).message}`);
      stop();
    }
  }, [earThreshold, thresholdSec, blinkWindowSec, minBlinks, blinkAlertSec, marThreshold, yawnSec, nodWindowSec, minNods, nodDelta, stop]);

  useEffect(() => () => stop(), [stop]);

  const closureProgress = Math.min(100, (closedFor / thresholdSec) * 100);
  const blinkProgress = Math.min(100, (blinkCount / minBlinks) * 100);
  const yawnProgress = Math.min(100, (yawnFor / yawnSec) * 100);
  const nodProgress = Math.min(100, (nodCount / minNods) * 100);

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
      {/* Video panel */}
      <Card
        className="relative overflow-hidden p-0 border-border/50 bg-black/60 backdrop-blur"
        style={{ boxShadow: "var(--shadow-elegant)" }}
      >
        <div className="relative aspect-video">
          <video ref={videoRef} className="hidden" playsInline muted />
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover" />

          {!running && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-card/30">
              <div
                className="flex h-16 w-16 items-center justify-center rounded-full"
                style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
              >
                <Eye className="h-8 w-8 text-primary-foreground" />
              </div>
              <p className="text-sm text-muted-foreground">Camera off — press Start to begin</p>
            </div>
          )}

          {/* Top status bar overlay */}
          {running && (
            <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
              <Badge variant="secondary" className="bg-black/60 border-white/10 backdrop-blur text-foreground gap-1.5">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
                </span>
                LIVE
              </Badge>
              <Badge
                variant="secondary"
                className={`backdrop-blur border-white/10 ${faceDetected ? "bg-emerald-500/20 text-emerald-300" : "bg-yellow-500/20 text-yellow-300"}`}
              >
                {faceDetected ? "Face detected" : "No face"}
              </Badge>
            </div>
          )}

          {/* Alarm overlays */}
          {(closureAlarm || blinkAlarm || yawnAlarm || nodAlarm) && (
            <div className="absolute top-14 left-1/2 -translate-x-1/2 flex flex-col gap-2 items-center pointer-events-none">
              {closureAlarm && (
                <div className="flex items-center gap-2 rounded-lg bg-destructive px-4 py-2 text-sm font-bold text-destructive-foreground animate-pulse shadow-lg">
                  <AlertTriangle className="h-4 w-4" />
                  SUSTAINED CLOSURE
                </div>
              )}
              {blinkAlarm && (
                <div className="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-bold text-white animate-pulse shadow-lg"
                  style={{ background: "oklch(0.65 0.2 50)" }}>
                  <Zap className="h-4 w-4" />
                  CONTINUOUS BLINKING
                </div>
              )}
              {yawnAlarm && (
                <div className="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-bold text-white animate-pulse shadow-lg"
                  style={{ background: "oklch(0.7 0.18 70)" }}>
                  <Smile className="h-4 w-4" />
                  YAWNING DETECTED
                </div>
              )}
              {nodAlarm && (
                <div className="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-bold text-white animate-pulse shadow-lg"
                  style={{ background: "oklch(0.6 0.2 300)" }}>
                  <Move3d className="h-4 w-4" />
                  HEAD NODDING
                </div>
              )}
            </div>
          )}
        </div>
      </Card>

      {/* Right panel */}
      <div className="space-y-4">
        <Card className="p-4 space-y-4 border-border/50 bg-card/60 backdrop-blur">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold">Detection</span>
            </div>
            <Badge variant="outline" className="text-xs">{status}</Badge>
          </div>

          {!running ? (
            <Button
              onClick={start}
              className="w-full gap-2 text-primary-foreground border-0"
              style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
            >
              <Play className="h-4 w-4" /> Start detection
            </Button>
          ) : (
            <Button onClick={stop} variant="destructive" className="w-full gap-2">
              <Square className="h-4 w-4" /> Stop
            </Button>
          )}

          <div className="grid grid-cols-2 gap-2">
            <StatCard
              icon={eyesClosed ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
              label="Eyes"
              value={eyesClosed ? "CLOSED" : "OPEN"}
              accent={eyesClosed ? "text-destructive" : "text-emerald-400"}
            />
            <StatCard
              icon={<Activity className="h-3 w-3" />}
              label="EAR"
              value={earValue.toFixed(3)}
              sub={`threshold ${earThreshold.toFixed(2)}`}
            />
            <StatCard
              icon={<Smile className="h-3 w-3" />}
              label="MAR"
              value={marValue.toFixed(3)}
              sub={`yawns ${yawnCount}`}
              accent={yawning ? "text-amber-400" : undefined}
            />
            <StatCard
              icon={<Move3d className="h-3 w-3" />}
              label="Nods"
              value={`${nodCount}`}
              sub={`in ${nodWindowSec}s`}
              accent={nodAlarm ? "text-purple-400" : undefined}
            />
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="flex items-center gap-1.5 text-muted-foreground">
                  <Timer className="h-3 w-3" /> Closed for
                </span>
                <span className="tabular-nums font-medium">
                  {closedFor.toFixed(1)}s / {thresholdSec}s
                </span>
              </div>
              <Progress value={closureProgress} className="h-1.5" />
            </div>
            <div>
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="flex items-center gap-1.5 text-muted-foreground">
                  <Zap className="h-3 w-3" /> Blinks ({blinkWindowSec}s window)
                </span>
                <span className="tabular-nums font-medium">
                  {blinkCount} / {minBlinks}
                </span>
              </div>
              <Progress value={blinkProgress} className="h-1.5" />
            </div>
            <div>
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="flex items-center gap-1.5 text-muted-foreground">
                  <Smile className="h-3 w-3" /> Yawn duration
                </span>
                <span className="tabular-nums font-medium">
                  {yawnFor.toFixed(1)}s / {yawnSec}s
                </span>
              </div>
              <Progress value={yawnProgress} className="h-1.5" />
            </div>
            <div>
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="flex items-center gap-1.5 text-muted-foreground">
                  <Move3d className="h-3 w-3" /> Head nods
                </span>
                <span className="tabular-nums font-medium">
                  {nodCount} / {minNods}
                </span>
              </div>
              <Progress value={nodProgress} className="h-1.5" />
            </div>
          </div>
        </Card>

        <Card className="p-4 space-y-4 border-border/50 bg-card/60 backdrop-blur">
          <div className="flex items-center gap-2">
            <Settings2 className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold">Settings</span>
          </div>

          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">EAR threshold</Label>
            <Input
              type="number" step="0.01" value={earThreshold}
              onChange={(e) => setEarThreshold(parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Sustained closure alarm (sec)</Label>
            <Input
              type="number" value={thresholdSec}
              onChange={(e) => setThresholdSec(parseFloat(e.target.value) || 0)}
            />
          </div>

          <Separator />

          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Continuous blinking
          </p>
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Window (s)</Label>
              <Input
                type="number" value={blinkWindowSec}
                onChange={(e) => setBlinkWindowSec(parseFloat(e.target.value) || 1)}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Min blinks</Label>
              <Input
                type="number" value={minBlinks}
                onChange={(e) => setMinBlinks(parseInt(e.target.value) || 1)}
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Sustain duration (sec)</Label>
            <Input
              type="number" value={blinkAlertSec}
              onChange={(e) => setBlinkAlertSec(parseFloat(e.target.value) || 0)}
            />
          </div>

          <Separator />

          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Yawning (MAR)
          </p>
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">MAR threshold</Label>
              <Input
                type="number" step="0.05" value={marThreshold}
                onChange={(e) => setMarThreshold(parseFloat(e.target.value) || 0)}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Min duration (s)</Label>
              <Input
                type="number" step="0.5" value={yawnSec}
                onChange={(e) => setYawnSec(parseFloat(e.target.value) || 0)}
              />
            </div>
          </div>

          <Separator />

          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Head nodding
          </p>
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Window (s)</Label>
              <Input
                type="number" value={nodWindowSec}
                onChange={(e) => setNodWindowSec(parseFloat(e.target.value) || 1)}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Min nods</Label>
              <Input
                type="number" value={minNods}
                onChange={(e) => setMinNods(parseInt(e.target.value) || 1)}
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Motion sensitivity</Label>
            <Input
              type="number" step="0.005" value={nodDelta}
              onChange={(e) => setNodDelta(parseFloat(e.target.value) || 0.01)}
            />
          </div>
        </Card>
      </div>
    </div>
  );
}
