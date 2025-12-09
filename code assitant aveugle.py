import cv2
import customtkinter as ctk
from PIL import Image
from customtkinter import CTkImage
from ultralytics import YOLO
import pyttsx3
import threading
import queue
import time
import warnings
import numpy as np

# -------------------- Config --------------------
warnings.filterwarnings("ignore", category=UserWarning, module="customtkinter")

# Voix
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)
voice_queue = queue.Queue()
processing_queue = queue.Queue()
results_queue = queue.Queue() # File d'attente pour les images traitées

# Modèle YOLO. "yolov8n.pt" sera téléchargé automatiquement s'il n'est pas trouvé.
model = YOLO("yolov8n.pt")

# Limiter répétitions vocales
last_spoken = {}
speak_delay = 3  # secondes

# Seuil de distance pour danger (en mètres)
DANGER_DISTANCE = 3.0

# Constantes
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.6

# -------------------- Thread voix --------------------
def voice_loop():
    while True:
        try:
            phrase = voice_queue.get()
            if phrase == "STOP":
                break
            engine.say(phrase)
            engine.runAndWait()
        except (queue.Empty, RuntimeError):
            # RuntimeError peut se produire si le moteur est occupé
            pass
        time.sleep(0.1)

threading.Thread(target=voice_loop, daemon=True).start()

# -------------------- Thread de traitement vidéo --------------------
def video_processing_loop(app_running_flag):
    while app_running_flag.is_set():
        try:
            frame = processing_queue.get(timeout=1)
        except queue.Empty:
            continue

        results = model(frame, verbose=False)
        annotated_frame = frame.copy()

        danger_detected = {"devant": False, "à gauche": False, "à droite": False}
        announce_list = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf < CONFIDENCE_THRESHOLD:
                continue

            w = x2 - x1
            cx = x1 + w // 2

            # Distance approximative
            distance = round(max(0.5, (7000 / (w + 1e-6))), 1)

            # Direction
            frame_width = annotated_frame.shape[1]
            if cx < frame_width / 3:
                direction = "à gauche"
            elif cx > 2 * frame_width / 3:
                direction = "à droite"
            else:
                direction = "devant"

            # Détecter danger
            if distance <= DANGER_DISTANCE:
                danger_detected[direction] = True
                key = f"{label}-{direction}"
                now = time.time()
                if key not in last_spoken or now - last_spoken[key] > speak_delay:
                    announce_list.append((distance, f"Attention, {label} {direction}, à environ {distance} mètres"))
                    last_spoken[key] = now

            # Dessiner rectangle et texte sur l'image
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} {distance}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Annonce zones libres
        now = time.time()
        for dir_name, detected in danger_detected.items():
            if not detected and (dir_name not in last_spoken or now - last_spoken[dir_name] > speak_delay):
                announce_list.append((float('inf'), f"{dir_name.capitalize()} libre")) # Mettre à la fin
                last_spoken[dir_name] = now

        # Trier par distance (les plus proches en premier)
        announce_list.sort(key=lambda x: x[0])

        if announce_list:
            phrase = " ; ".join([item[1] for item in announce_list])
            voice_queue.put(phrase)

        # Mettre l'image traitée dans la file des résultats
        if results_queue.qsize() < 2:
            results_queue.put(annotated_frame)

# -------------------- Interface --------------------
class VisualAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("👁 Assistant Visuel Intelligent")
        self.geometry("1000x700")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")

        self.cap = None
        self.running = False
        self.processing_thread = None
        self.app_running_flag = threading.Event()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Cadre gauche
        self.frame_left = ctk.CTkFrame(self, width=300, corner_radius=15)
        self.frame_left.pack(side="left", fill="y", padx=10, pady=10)
        ctk.CTkLabel(self.frame_left, text="🎯 Assistant Visuel", font=("Arial", 22, "bold")).pack(pady=20)
        self.start_btn = ctk.CTkButton(self.frame_left, text="Démarrer la caméra 🎥", command=self.start_camera)
        self.start_btn.pack(pady=10)
        self.stop_btn = ctk.CTkButton(self.frame_left, text="Arrêter la caméra ⛔", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(pady=10)

        # Cadre droit
        self.frame_right = ctk.CTkFrame(self, corner_radius=15)
        self.frame_right.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        self.video_label = ctk.CTkLabel(self.frame_right, text="")
        self.video_label.pack(expand=True)

    # -------------------- Caméra --------------------
    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                print("❌ Impossible d’ouvrir la caméra.")
                return
            self.running = True
            self.app_running_flag.set()
            self.processing_thread = threading.Thread(target=video_processing_loop, args=(self.app_running_flag,), daemon=True)
            self.processing_thread.start()

            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.update_frame()
            self.check_results_queue() # Démarrer la vérification des résultats

    def stop_camera(self):
        self.running = False
        self.app_running_flag.clear()

        if self.cap:
            self.cap.release()
            self.cap = None

        # Vider les files d'attente pour éviter les traitements résiduels
        with processing_queue.mutex:
            processing_queue.queue.clear()
        with voice_queue.mutex:
            voice_queue.queue.clear()
        with results_queue.mutex:
            results_queue.queue.clear()

        # Le message STOP est maintenant envoyé dans on_closing pour un arrêt propre
        # voice_queue.put("STOP")
        self.video_label.configure(image=None)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    # -------------------- Mise à jour vidéo --------------------
    def update_frame(self):
        if not self.running:
            return

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                # Envoyer l'image au thread de traitement
                if processing_queue.qsize() < 2: # Éviter la latence
                    processing_queue.put(frame)

        # Répéter cette fonction toutes les 30ms pour un flux vidéo fluide
        self.after(30, self.update_frame)

    def update_image(self, annotated_frame):
        # Convertir pour CustomTkinter
        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_ctk = CTkImage(light_image=img_pil, size=(900,600))
        self.video_label.configure(image=img_ctk)
        self.video_label.image = img_ctk

    def check_results_queue(self):
        if not self.running:
            return
        try:
            annotated_frame = results_queue.get_nowait()
            self.update_image(annotated_frame)
        except queue.Empty:
            pass
        self.after(30, self.check_results_queue) # Vérifier à nouveau dans 30ms

    def on_closing(self):
        self.stop_camera()
        voice_queue.put("STOP")
        self.destroy()

# -------------------- Lancer l’application --------------------
if __name__ == "__main__":
    app = VisualAssistantApp()
    app.mainloop()
