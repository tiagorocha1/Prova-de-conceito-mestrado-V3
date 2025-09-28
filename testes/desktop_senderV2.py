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
    min_interval_ms: int
    use_jpeg: bool
    jpeg_quality: int
    retries: int
    backoff_ms: int

def frame_to_data_url(frame_bgr, use_jpeg: bool, jpeg_quality: int) -> str:
    """Converte frame BGR -> base64 data URL (JPEG ou PNG)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    if use_jpeg:
        pil_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    else:
        pil_img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def send_frame(endpoint: str, data_url: str, timeout: float, retries: int, backoff_ms: int) -> None:
    payload = {"image": data_url}
    for attempt in range(retries + 1):
        try:
            r = requests.post(endpoint, json=payload, timeout=timeout)
            r.raise_for_status()
            return
        except requests.exceptions.RequestException as e:
            if attempt >= retries:
                raise
            sleep_s = (backoff_ms * (attempt + 1)) / 1000.0
            time.sleep(sleep_s)

def process_video(path: str, cfg: Config) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {path}")
        sys.exit(1)

    print(f"[INFO] Enviando frames de '{path}' para {cfg.endpoint}")
    print(f"[INFO] 1 a cada {cfg.frame_skip} frames | limite: {cfg.max_seconds}s | timeout: {cfg.timeout}s")
    print(f"[INFO] Formato: {'JPEG q=%d' % cfg.jpeg_quality if cfg.use_jpeg else 'PNG'} | intervalo mínimo: {cfg.min_interval_ms} ms")

    start = time.time()
    last_sent_ts = 0.0
    sent = 0
    total = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1

            # limite total de execução
            if time.time() - start >= cfg.max_seconds:
                print("[INFO] Tempo máximo atingido; finalizando.")
                break

            # pular frames
            if total % cfg.frame_skip != 0:
                continue

            # respeitar intervalo mínimo entre envios
            now = time.time() * 1000.0
            if now - last_sent_ts < cfg.min_interval_ms:
                continue

            data_url = frame_to_data_url(frame, cfg.use_jpeg, cfg.jpeg_quality)
            try:
                send_frame(cfg.endpoint, data_url, cfg.timeout, cfg.retries, cfg.backoff_ms)
                sent += 1
                last_sent_ts = now
            except requests.exceptions.RequestException as e:
                print(f"[WARN] Falha ao enviar frame {total}: {e}")

            if sent % 5 == 0:
                elapsed = time.time() - start
                print(f"[INFO] Enviados: {sent} | Lidos: {total} | Elapsed: {elapsed:.1f}s")

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
    parser.add_argument("--max-seconds", type=int, default=120,
                        help="Tempo máximo de envio (default: 60)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Timeout por requisição em segundos (default: 60.0)")
    parser.add_argument("--min-interval-ms", type=int, default=1500,
                        help="Intervalo mínimo entre envios (ms) (default: 1500)")
    parser.add_argument("--jpeg", action="store_true",
                        help="Enviar como JPEG (default: PNG)")
    parser.add_argument("--quality", type=int, default=80,
                        help="Qualidade JPEG 1-95 (default: 80)")
    parser.add_argument("--retries", type=int, default=1,
                        help="Número de tentativas extras por frame (default: 1)")
    parser.add_argument("--backoff-ms", type=int, default=500,
                        help="Backoff entre tentativas (ms) (default: 500)")
    args = parser.parse_args()

    cfg = Config(
        endpoint=args.endpoint,
        frame_skip=max(1, args.skip),
        max_seconds=max(1, args.max_seconds),
        timeout=max(1e-3, args.timeout),
        min_interval_ms=max(0, args.min_interval_ms),
        use_jpeg=bool(args.jpeg),
        jpeg_quality=max(1, min(95, args.quality)),
        retries=max(0, args.retries),
        backoff_ms=max(0, args.backoff_ms),
    )
    process_video(args.video, cfg)

if __name__ == "__main__":
    main()
