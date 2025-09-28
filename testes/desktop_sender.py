import argparse
import base64
import io
import sys
import time
from dataclasses import dataclass

import cv2
import requests
from PIL import Image


DEFAULT_ENDPOINT = "http://localhost:8000/detect-and-recognize"


@dataclass
class Config:
    endpoint: str
    frame_skip: int
    max_seconds: int
    timeout: float


def frame_to_data_url(frame_bgr) -> str:
    """Converte frame BGR (OpenCV) para PNG base64 data URL (sem redimensionar)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def send_frame(endpoint: str, data_url: str, timeout: float) -> None:
    payload = {"image": data_url}
    r = requests.post(endpoint, json=payload, timeout=timeout)
    r.raise_for_status()


def process_video(path: str, cfg: Config) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {path}")
        sys.exit(1)

    print(f"[INFO] Enviando frames de '{path}' para {cfg.endpoint}")
    print(f"[INFO] 1 a cada {cfg.frame_skip} frames | limite: {cfg.max_seconds}s")

    start = time.time()
    sent = 0
    total = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1

            if time.time() - start >= cfg.max_seconds:
                print("[INFO] Tempo máximo atingido; finalizando.")
                break

            if total % cfg.frame_skip != 0:
                continue

            data_url = frame_to_data_url(frame)
            try:
                send_frame(cfg.endpoint, data_url, cfg.timeout)
                sent += 1
            except requests.exceptions.RequestException as e:
                print(f"[WARN] Falha ao enviar frame {total}: {e}")

            if sent % 10 == 0:
                elapsed = time.time() - start
                print(f"[INFO] Enviados: {sent} frames | Elapsed: {elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário (Ctrl+C).")
    finally:
        cap.release()

    elapsed = time.time() - start
    print(f"[OK] Total lidos: {total} | Total enviados: {sent} | Tempo: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Envia frames de um vídeo para o backend.")
    parser.add_argument("video", help="Caminho do arquivo de vídeo (mp4/webm/avi...)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help=f"Endpoint do backend (default: {DEFAULT_ENDPOINT})")
    parser.add_argument("--skip", type=int, default=2,
                        help="Enviar 1 a cada N frames (default: 2)")
    parser.add_argument("--max-seconds", type=int, default=60,
                        help="Tempo máximo de envio (default: 60)")
    parser.add_argument("--timeout", type=float, default=8.0,
                        help="Timeout por requisição (default: 8.0s)")
    args = parser.parse_args()

    cfg = Config(endpoint=args.endpoint,
                 frame_skip=max(1, args.skip),
                 max_seconds=max(1, args.max_seconds),
                 timeout=max(1e-3, args.timeout))
    process_video(args.video, cfg)


if __name__ == "__main__":
    main()
