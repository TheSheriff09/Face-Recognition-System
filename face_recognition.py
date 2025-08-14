import argparse
import os
import sys
import time
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Deque
from collections import deque
import threading
import queue

import cv2
import numpy as np
import face_recognition

CONFIG = {
    'processing_interval': 0.1,
    'frame_scale': 0.5,
    'hog_model': True,
    'target_fps': 30,
    'resolution': (640, 480),
    'max_queue_size': 2,
    'tolerance': 0.45,
    'num_jitters': 1,
    'model': 'small',
    'new_person_counter': 1
}

def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise IOError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def draw_label(frame: np.ndarray, text: str, pt: Tuple[int, int], 
               color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    x, y = pt
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def build_encodings(dataset_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    all_encodings: List[np.ndarray] = []
    all_names: List[str] = []

    people = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if not people:
        raise ValueError(f"No person subfolders found in {dataset_dir}")

    print(f"Found {len(people)} identities in dataset.")

    for person_dir in people:
        name = person_dir.name
        image_paths = [p for p in person_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if not image_paths:
            print(f"[WARN] No images for {name}, skipping.")
            continue
            
        print(f"Processing {name}: {len(image_paths)} images")
        for img_path in image_paths:
            try:
                rgb = imread_rgb(str(img_path))
                boxes = face_recognition.face_locations(rgb, model="hog")
                if not boxes:
                    continue
                encs = face_recognition.face_encodings(rgb, boxes, num_jitters=CONFIG['num_jitters'], model=CONFIG['model'])
                if encs:
                    all_encodings.append(encs[0])
                    all_names.append(name)
            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")

    if not all_encodings:
        raise RuntimeError("No encodings computed. Ensure your images contain detectable faces.")
    return all_encodings, all_names

def save_encodings(encodings: List[np.ndarray], names: List[str], out_path: Path) -> None:
    data = {"encodings": encodings, "names": names}
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(encodings)} encodings to {out_path}")

def load_encodings(path: Path) -> Tuple[List[np.ndarray], List[str]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    encs = data.get("encodings", [])
    names = data.get("names", [])
    if not encs or not names:
        raise ValueError("Encodings file appears empty or corrupted.")
    print(f"Loaded {len(encs)} encodings across {len(set(names))} identities.")
    return encs, names

class FaceRecognizer:
    def __init__(self, known_encodings: List[np.ndarray], known_names: List[str]):
        self.known_encodings = np.array(known_encodings)
        self.known_names = known_names
        self.last_results = []
        self.processing_queue = queue.Queue(maxsize=CONFIG['max_queue_size'])
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._process_frames)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
    def _process_frames(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame_data = self.processing_queue.get(timeout=0.1)
                rgb_frame, callback = frame_data
                
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=CONFIG['frame_scale'], fy=CONFIG['frame_scale'])
                
                model = "hog" if CONFIG['hog_model'] else "cnn"
                boxes = face_recognition.face_locations(small_frame, model=model)
                
                boxes = [(int(top/CONFIG['frame_scale']), int(right/CONFIG['frame_scale']),
                         int(bottom/CONFIG['frame_scale']), int(left/CONFIG['frame_scale'])) 
                        for (top, right, bottom, left) in boxes]
                
                encs = face_recognition.face_encodings(rgb_frame, boxes, 
                                                     num_jitters=CONFIG['num_jitters'],
                                                     model=CONFIG['model'])
                
                results = []
                if encs:
                    for enc, box in zip(encs, boxes):
                        dists = np.linalg.norm(self.known_encodings - enc, axis=1)
                        j = np.argmin(dists)
                        best_dist = float(dists[j])
                        name = self.known_names[j] if best_dist <= CONFIG['tolerance'] else "Unknown"
                        results.append((box, name, best_dist))
                
                self.last_results = results
                if callback:
                    callback(results)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Processing error: {e}")
                
    def recognize_frame(self, rgb_frame: np.ndarray, callback: Optional[callable] = None) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        try:
            if not self.processing_queue.full():
                self.processing_queue.put((rgb_frame, callback), block=False)
        except queue.Full:
            pass
            
        return self.last_results
        
    def stop(self) -> None:
        self.stop_event.set()
        self.worker_thread.join()

class FPSCounter:
    def __init__(self, window_size: int = 10):
        self.times = deque(maxlen=window_size)
        self.fps = 0
        
    def update(self) -> float:
        self.times.append(time.time())
        if len(self.times) > 1:
            self.fps = (len(self.times) / (self.times[-1] - self.times[0]))
        return self.fps

def run_camera(encodings_path: Path, camera_index: int = 0) -> None:
    known_encodings, known_names = load_encodings(encodings_path)
    recognizer = FaceRecognizer(known_encodings, known_names)
    fps_counter = FPSCounter()
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['resolution'][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['resolution'][1])
    cap.set(cv2.CAP_PROP_FPS, CONFIG['target_fps'])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Face Recognition", CONFIG['resolution'][0], CONFIG['resolution'][1])
    
    print("Press 'q' to quit, 's' to save a snapshot.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[WARN] Failed to read frame.")
                break
                
            fps = fps_counter.update()
            
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            preds = recognizer.recognize_frame(rgb)
            
            for (top, right, bottom, left), name, dist in preds:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
                label = f"{name} ({dist:.2f})" if name != "Unknown" else name
                draw_label(frame_bgr, label, (left, max(0, top - 10)), color)
            
            status_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
            draw_label(frame_bgr, f"FPS: {fps:.1f}", (10, 30), status_color)
            
            cv2.imshow("Face Recognition", frame_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                out_name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(out_name, frame_bgr)
                print(f"Saved {out_name}")
                
    finally:
        recognizer.stop()
        cap.release()
        cv2.destroyAllWindows()

def create_new_person_folder(dataset_dir: Path) -> Path:
    while True:
        folder_name = f"new_person_{CONFIG['new_person_counter']}"
        person_dir = dataset_dir / folder_name
        if not person_dir.exists():
            person_dir.mkdir(parents=True, exist_ok=True)
            CONFIG['new_person_counter'] += 1
            return person_dir
        CONFIG['new_person_counter'] += 1

def enroll_person(dataset_dir: Path, shots: int = 20, camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    current_person_dir = None
    saved_count = 0
    capturing = False
    fps_counter = FPSCounter()
    last_face_box = None
    last_key_time = time.time()
    key_cooldown = 0.5

    print("Easy Enrollment Mode:")
    print("  Press 'n' - Create new person folder and start capturing")
    print("  Press SPACE - Capture current face")
    print("  Press 'c' - Complete current person")
    print("  Press 'q' - Quit enrollment")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame.")
                continue

            fps = fps_counter.update()
            display = frame.copy()

            status = "Press 'n' to start a new person"
            if current_person_dir:
                status = f"Capturing: {current_person_dir.name} ({saved_count}/{shots})"

            draw_label(display, status, (10, 30))
            draw_label(display, f"FPS: {fps:.1f}", (10, 60))
            draw_label(display, "SPACE: Capture | 'c': Complete | 'q': Quit", (10, 90))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            
            if boxes:
                top, right, bottom, left = boxes[0]
                last_face_box = (top, right, bottom, left)
                cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
                draw_label(display, "Face detected", (left, max(0, top - 10)), (0, 255, 0))
                
                if capturing:
                    cv2.circle(display, (left + (right-left)//2, top + (bottom-top)//2), 
                               30, (0, 0, 255), 3)
                    capturing = False

            cv2.imshow("Easy Person Enrollment", display)
            
            current_time = time.time()
            if current_time - last_key_time > key_cooldown:
                key = cv2.waitKey(1) & 0xFF
                
                if key != 255:
                    last_key_time = current_time
                    
                    if key == ord('q'):
                        break
                    elif key == ord('n'):
                        current_person_dir = create_new_person_folder(dataset_dir)
                        saved_count = 0
                        print(f"Created new person folder: {current_person_dir}")
                        capturing = True
                    elif key == 32:
                        if current_person_dir is None:
                            print("Please create a new person folder first by pressing 'n'")
                        elif last_face_box is None:
                            print("No face detected. Try again.")
                        else:
                            top, right, bottom, left = last_face_box
                            face_crop = frame[max(0, top-10):min(frame.shape[0], bottom+10), 
                                            max(0, left-10):min(frame.shape[1], right+10)]
                            
                            if face_crop.size == 0:
                                print("Invalid face area. Try repositioning.")
                                continue
                                
                            out_path = current_person_dir / f"capture_{saved_count:03d}.jpg"
                            cv2.imwrite(str(out_path), face_crop)
                            print(f"Saved {out_path}")
                            saved_count += 1
                            capturing = True
                            
                            if saved_count >= shots:
                                print(f"Finished capturing {shots} images for {current_person_dir.name}")
                                current_person_dir = None
                                saved_count = 0
                    elif key == ord('c'):
                        if current_person_dir and saved_count > 0:
                            print(f"Completed enrollment for {current_person_dir.name} with {saved_count} images")
                        current_person_dir = None
                        saved_count = 0
            else:
                cv2.waitKey(1)

    finally:
        cap.release()
        cv2.destroyAllWindows()

def cmd_prepare(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    out_path = Path(args.out)
    encs, names = build_encodings(dataset)
    save_encodings(encs, names, out_path)

def cmd_camera(args: argparse.Namespace) -> None:
    CONFIG['tolerance'] = args.tolerance
    run_camera(Path(args.encodings), camera_index=args.camera)

def cmd_enroll(args: argparse.Namespace) -> None:
    enroll_person(Path(args.dataset), shots=args.shots, camera_index=args.camera)

def cmd_list(args: argparse.Namespace) -> None:
    encs, names = load_encodings(Path(args.encodings))
    unique = sorted(set(names))
    print(f"Identities in {args.encodings} ({len(unique)}):")
    for u in unique:
        cnt = sum(1 for n in names if n == u)
        print(f" - {u}: {cnt} samples")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Face Recognition App")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--dataset", required=True)
    p_prep.add_argument("--out", default="encodings.pkl")
    p_prep.set_defaults(func=cmd_prepare)

    p_cam = sub.add_parser("camera")
    p_cam.add_argument("--encodings", required=True)
    p_cam.add_argument("--camera", type=int, default=0)
    p_cam.add_argument("--tolerance", type=float, default=0.45)
    p_cam.set_defaults(func=cmd_camera)

    p_enroll = sub.add_parser("enroll")
    p_enroll.add_argument("--dataset", required=True)
    p_enroll.add_argument("--shots", type=int, default=20)
    p_enroll.add_argument("--camera", type=int, default=0)
    p_enroll.set_defaults(func=cmd_enroll)

    p_list = sub.add_parser("list")
    p_list.add_argument("--encodings", required=True)
    p_list.set_defaults(func=cmd_list)

    return p

def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))