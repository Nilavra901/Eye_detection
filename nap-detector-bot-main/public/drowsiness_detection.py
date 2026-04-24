"""
Drowsiness Detection System — Python + OpenCV + MediaPipe + IoT
================================================================
Detects eye closure via webcam. If eyes remain closed for more than
THRESHOLD_SEC (default 10s), activates an alarm (sound + optional
Arduino/Raspberry Pi GPIO trigger).

Install:
    pip install opencv-python mediapipe playsound==1.2.2 pyserial

Optional hardware:
    - Arduino on /dev/ttyUSB0 (or COM3) — sends '1' to trigger buzzer
    - Raspberry Pi GPIO pin 18 — set HIGH to trigger relay/buzzer

Run:
    python drowsiness_detection.py
"""

import cv2
import time
import math
import threading
import mediapipe as mp

# ====== CONFIG ======
THRESHOLD_SEC = 10        # seconds eyes must be closed before alarm
EAR_THRESHOLD = 0.22      # below this = eye closed
ALARM_SOUND = "alarm.wav" # put a .wav file next to this script
USE_ARDUINO = False
ARDUINO_PORT = "/dev/ttyUSB0"   # or "COM3" on Windows
USE_RPI_GPIO = False
GPIO_PIN = 18
# ====================

# MediaPipe FaceMesh eye landmark indices
LEFT_EYE  = {"top": [159, 158, 157], "bot": [145, 153, 154], "l": 33,  "r": 133}
RIGHT_EYE = {"top": [386, 385, 384], "bot": [374, 380, 381], "l": 362, "r": 263}


def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def eye_aspect_ratio(landmarks, eye):
    v = sum(dist(landmarks[t], landmarks[b]) for t, b in zip(eye["top"], eye["bot"])) / 3
    h = dist(landmarks[eye["l"]], landmarks[eye["r"]])
    return v / h if h > 0 else 0


# ---------- IoT hooks ----------
arduino = None
if USE_ARDUINO:
    import serial
    arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    time.sleep(2)

if USE_RPI_GPIO:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_PIN, GPIO.OUT)


def trigger_iot(on: bool):
    if USE_ARDUINO and arduino:
        arduino.write(b'1' if on else b'0')
    if USE_RPI_GPIO:
        GPIO.output(GPIO_PIN, GPIO.HIGH if on else GPIO.LOW)


# ---------- Alarm sound ----------
_alarm_playing = False
def play_alarm():
    global _alarm_playing
    if _alarm_playing:
        return
    _alarm_playing = True
    try:
        from playsound import playsound
        while _alarm_playing:
            playsound(ALARM_SOUND)
    except Exception as e:
        print("Sound error:", e)


def stop_alarm():
    global _alarm_playing
    _alarm_playing = False


# ---------- Main ----------
def main():
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)
    cap = cv2.VideoCapture(0)

    closed_since = None
    alarm_on = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        status = "NO FACE"
        ear = 0.0
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            ear = (eye_aspect_ratio(lm, LEFT_EYE) + eye_aspect_ratio(lm, RIGHT_EYE)) / 2
            closed = ear < EAR_THRESHOLD
            status = "CLOSED" if closed else "OPEN"

            if closed:
                if closed_since is None:
                    closed_since = time.time()
                elapsed = time.time() - closed_since
                cv2.putText(frame, f"Closed: {elapsed:.1f}s",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if elapsed >= THRESHOLD_SEC and not alarm_on:
                    alarm_on = True
                    trigger_iot(True)
                    threading.Thread(target=play_alarm, daemon=True).start()
                    print("⏰ ALARM ACTIVATED")
            else:
                if alarm_on:
                    alarm_on = False
                    stop_alarm()
                    trigger_iot(False)
                    print("Alarm stopped — eyes opened")
                closed_since = None

            # draw eye points
            color = (0, 0, 255) if closed else (0, 255, 0)
            for idx in (LEFT_EYE["top"] + LEFT_EYE["bot"] + [LEFT_EYE["l"], LEFT_EYE["r"]] +
                        RIGHT_EYE["top"] + RIGHT_EYE["bot"] + [RIGHT_EYE["l"], RIGHT_EYE["r"]]):
                p = lm[idx]
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, color, -1)

        cv2.putText(frame, f"Eyes: {status}  EAR: {ear:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if alarm_on:
            cv2.putText(frame, "*** ALARM ***", (w // 2 - 120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_alarm()
    trigger_iot(False)
    cap.release()
    cv2.destroyAllWindows()
    if USE_RPI_GPIO:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
