import { createFileRoute } from "@tanstack/react-router";
import DrowsinessDetector from "@/components/DrowsinessDetector";
import { Eye } from "lucide-react";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "NapDetect — Drowsiness Detection" },
      { name: "description", content: "Real-time webcam drowsiness detection with sustained-closure and continuous-blinking alarms." },
    ],
  }),
});

function Index() {
  return (
    <div className="min-h-screen text-foreground">
      <div className="mx-auto max-w-6xl px-4 py-10">
        <header className="mb-8 flex items-center gap-4">
          <div
            className="flex h-12 w-12 items-center justify-center rounded-xl shadow-[var(--shadow-glow)]"
            style={{ background: "var(--gradient-primary)" }}
          >
            <Eye className="h-6 w-6 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-4xl font-bold tracking-tight bg-clip-text text-transparent"
                style={{ backgroundImage: "var(--gradient-primary)" }}>
              NapDetect
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Real-time eye tracking · sustained closure & continuous blinking alarms
            </p>
          </div>
        </header>
        <DrowsinessDetector />
      </div>
    </div>
  );
}
