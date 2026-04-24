import { useEffect, useRef, useState, useCallback } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Eye, EyeOff, AlertTriangle, Camera, CameraOff } from "lucide-react";
import { toast } from "sonner";

// Eye landmark indices (MediaPipe FaceMesh - 468 points)
const LEFT_EYE = { top: [159, 158, 157], bottom: [145, 153, 154], left: 33, right: 133 };
const RIGHT_EYE = { top: [386, 385, 384], bottom: [374, 380, 381], left: 362, right: 263 };

// Eye Aspect Ratio (EAR) — < threshold means closed
const EAR_THRESHOLD = 0.22;

function dist(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function eyeAspectRatio(lm: any[], eye: typeof LEFT_EYE) {
  const v1 = (dist(lm[eye.top[0]], lm[eye.bottom[0]]) +
    dist(lm[eye.top[1]], lm[eye.bottom[1]]) +
    dist(lm[eye.top[2]], lm[eye.bottom[2]])) / 3;
  const h = dist(lm[eye.left], lm[eye.right]);
  return v1 / h;
}

const Index = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number>();
  const closedSinceRef = useRef<number | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const oscRef = useRef<OscillatorNode | null>(null);

  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [ear, setEar] = useState(0);
  const [closedSec, setClosedSec] = useState(0);
  const [alarm, setAlarm] = useState(false);
  const [threshold, setThreshold] = useState(10); // seconds
  const [status, setStatus] = useState<"idle" | "open" | "closed">("idle");

  const startAlarm = useCallback(() => {
    if (oscRef.current) return;
    const ctx = audioCtxRef.current ?? new AudioContext();
    audioCtxRef.current = ctx;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "square";
    osc.frequency.value = 880;
    gain.gain.value = 0.15;
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    oscRef.current = osc;
    setAlarm(true);
    toast.error("⏰ ALARM! Eyes closed too long — IoT system activated");
  }, []);

  const stopAlarm = useCallback(() => {
    oscRef.current?.stop();
    oscRef.current?.disconnect();
    oscRef.current = null;
    setAlarm(false);
  }, []);

  const loop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const lm = landmarkerRef.current;
    if (!video || !canvas || !lm || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(loop);
      return;
    }
    const ctx = canvas.getContext("2d")!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const res = lm.detectForVideo(video, performance.now());
    if (res.faceLandmarks?.[0]) {
      const pts = res.faceLandmarks[0];
      const leftEAR = eyeAspectRatio(pts, LEFT_EYE);
      const rightEAR = eyeAspectRatio(pts, RIGHT_EYE);
      const avg = (leftEAR + rightEAR) / 2;
      setEar(avg);

      const closed = avg < EAR_THRESHOLD;
      setStatus(closed ? "closed" : "open");

      if (closed) {
        if (closedSinceRef.current == null) closedSinceRef.current = performance.now();
        const sec = (performance.now() - closedSinceRef.current) / 1000;
        setClosedSec(sec);
        if (sec >= threshold && !oscRef.current) startAlarm();
      } else {
        closedSinceRef.current = null;
        setClosedSec(0);
        if (oscRef.current) stopAlarm();
      }

      // Draw eye landmarks
      ctx.fillStyle = closed ? "#ef4444" : "#22c55e";
      [...LEFT_EYE.top, ...LEFT_EYE.bottom, LEFT_EYE.left, LEFT_EYE.right,
       ...RIGHT_EYE.top, ...RIGHT_EYE.bottom, RIGHT_EYE.left, RIGHT_EYE.right].forEach(i => {
        const p = pts[i];
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    } else {
      setStatus("idle");
      closedSinceRef.current = null;
      setClosedSec(0);
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [threshold, startAlarm, stopAlarm]);

  const start = async () => {
    try {
      setLoading(true);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      videoRef.current!.srcObject = stream;
      await videoRef.current!.play();

      if (!landmarkerRef.current) {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm"
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
      setRunning(true);
      setLoading(false);
      toast.success("Camera started — monitoring eyes");
      loop();
    } catch (e: any) {
      setLoading(false);
      toast.error("Camera error: " + e.message);
    }
  };

  const stop = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach(t => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    stopAlarm();
    setRunning(false);
    setStatus("idle");
  };

  useEffect(() => () => stop(), []);

  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-8">
      <div className="max-w-5xl mx-auto space-y-6">
        <header className="text-center space-y-2">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
            Drowsiness Detection System
          </h1>
          <p className="text-muted-foreground">
            Real-time eye-closure monitoring · IoT alarm trigger after {threshold}s
          </p>
        </header>

        <Card className="overflow-hidden">
          <div className="relative bg-black aspect-video">
            <video ref={videoRef} className="hidden" playsInline muted />
            <canvas ref={canvasRef} className="w-full h-full object-contain" />
            {!running && (
              <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
                <div className="text-center space-y-2">
                  <Camera className="w-16 h-16 mx-auto opacity-40" />
                  <p>Camera off — click Start to begin</p>
                </div>
              </div>
            )}
            {alarm && (
              <div className="absolute inset-0 bg-destructive/40 animate-pulse flex items-center justify-center">
                <div className="bg-destructive text-destructive-foreground px-6 py-3 rounded-lg flex items-center gap-2 text-xl font-bold">
                  <AlertTriangle className="w-6 h-6" /> ALARM ACTIVATED
                </div>
              </div>
            )}
          </div>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="p-4 space-y-1">
            <div className="text-sm text-muted-foreground">Eye Status</div>
            <div className="flex items-center gap-2 text-2xl font-semibold">
              {status === "closed" ? (
                <><EyeOff className="text-destructive" /> Closed</>
              ) : status === "open" ? (
                <><Eye className="text-primary" /> Open</>
              ) : (
                <><Eye className="text-muted-foreground" /> —</>
              )}
            </div>
          </Card>
          <Card className="p-4 space-y-1">
            <div className="text-sm text-muted-foreground">EAR (Eye Aspect Ratio)</div>
            <div className="text-2xl font-semibold font-mono">{ear.toFixed(3)}</div>
            <div className="text-xs text-muted-foreground">Threshold: {EAR_THRESHOLD}</div>
          </Card>
          <Card className="p-4 space-y-1">
            <div className="text-sm text-muted-foreground">Closed Duration</div>
            <div className={`text-2xl font-semibold font-mono ${closedSec >= threshold ? "text-destructive" : ""}`}>
              {closedSec.toFixed(1)}s
            </div>
            <div className="text-xs text-muted-foreground">Trigger at: {threshold}s</div>
          </Card>
        </div>

        <Card className="p-4 space-y-4">
          <div>
            <div className="flex justify-between mb-2">
              <label className="text-sm font-medium">Alarm trigger threshold</label>
              <span className="text-sm font-mono">{threshold}s</span>
            </div>
            <Slider
              value={[threshold]}
              onValueChange={(v) => setThreshold(v[0])}
              min={1}
              max={30}
              step={1}
            />
          </div>
          <div className="flex gap-3">
            {!running ? (
              <Button onClick={start} disabled={loading} size="lg" className="flex-1">
                <Camera className="mr-2" /> {loading ? "Loading model..." : "Start Camera"}
              </Button>
            ) : (
              <Button onClick={stop} variant="destructive" size="lg" className="flex-1">
                <CameraOff className="mr-2" /> Stop
              </Button>
            )}
            {alarm && (
              <Button onClick={stopAlarm} variant="outline" size="lg">
                Silence Alarm
              </Button>
            )}
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="font-semibold mb-2">📦 Python + OpenCV reference code</h3>
          <p className="text-sm text-muted-foreground mb-3">
            For the original IoT/hardware version (Arduino buzzer, Raspberry Pi GPIO), download
            the Python script. Run with: <code className="bg-muted px-1 rounded">pip install opencv-python mediapipe playsound pyserial</code>
          </p>
          <Button asChild variant="outline">
            <a href="/drowsiness_detection.py" download>Download drowsiness_detection.py</a>
          </Button>
        </Card>
      </div>
    </div>
  );
};

export default Index;
