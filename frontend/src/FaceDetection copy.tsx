// FaceDetection.tsx
import React, { useEffect, useRef, useState } from 'react';
import { Camera } from '@mediapipe/camera_utils';

const CAPTURE_WIDTH = 1344;
const CAPTURE_HEIGHT = 760;
const FRAME_SKIP = 2;                 // envia 1 a cada 2 frames
const MAX_DURATION_MS = 60_000;       // 60s
const ENDPOINT = 'http://localhost:8000/detect-and-recognize';

function FaceDetectionComponent() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const cameraRef = useRef<Camera | null>(null);

  // controle de envio
  const frameCountRef = useRef<number>(0);
  const inFlightRef = useRef<boolean>(false);
  const abortRef = useRef<AbortController | null>(null);

  // timeout de 60s
  const startTsRef = useRef<number>(0);
  const stopTimerRef = useRef<number | null>(null);

  // upload de vídeo (mantido do seu original)
  const [useVideoFile, setUseVideoFile] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);

  const handleVideoFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setVideoFile(e.target.files[0]);
      setUseVideoFile(true);
    }
  };

  const sendVideoFile = async () => {
    if (!videoFile) return;
    const formData = new FormData();
    formData.append('video', videoFile);

    try {
      const response = await fetch('http://localhost:8000/process-video', {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      console.log('Resultado do processamento:', result);
    } catch (error) {
      console.error('Erro ao enviar vídeo:', error);
    }
  };

  // inicia câmera com resolução do backend e laço de captura
  useEffect(() => {
    if (!isDetecting) return;

    const start = async () => {
      // prepara abort controller (para fetchs)
      abortRef.current = new AbortController();

      // reseta contadores
      frameCountRef.current = 0;
      startTsRef.current = performance.now();

      // configura timeout de 60s (para)
      stopTimerRef.current = window.setTimeout(() => {
        setIsDetecting(false);
      }, MAX_DURATION_MS);

      if (videoRef.current) {
        cameraRef.current = new Camera(videoRef.current, {
          onFrame: async () => {
            // checa timeout por segurança (dupla garantia)
            const elapsed = performance.now() - startTsRef.current;
            if (elapsed >= MAX_DURATION_MS) {
              setIsDetecting(false);
              return;
            }

            await sendFrame(); // frame skipping e backpressure são tratados dentro
          },
          width: CAPTURE_WIDTH,
          height: CAPTURE_HEIGHT,
        });

        // força o elemento <video> a renderizar nesse tamanho também
        videoRef.current.width = CAPTURE_WIDTH;
        videoRef.current.height = CAPTURE_HEIGHT;

        await cameraRef.current.start();
      }
    };

    start();

    return () => {
      // cleanup ao sair da detecção
      if (stopTimerRef.current) {
        clearTimeout(stopTimerRef.current);
        stopTimerRef.current = null;
      }

      abortRef.current?.abort();
      abortRef.current = null;
      inFlightRef.current = false;

      try {
        cameraRef.current?.stop();
      } catch {}
      cameraRef.current = null;

      const video = videoRef.current;
      const stream = video?.srcObject as MediaStream | null;
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      if (video) video.srcObject = null;
    };
  }, [isDetecting]);

  // captura e envio (1 a cada 2 frames), tamanho fixo 1344×760, 1 requisição por vez
  const sendFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    // pula frames
    frameCountRef.current += 1;
    if (frameCountRef.current % FRAME_SKIP !== 0) return;

    if (inFlightRef.current) return; // evita fila de requisições
    inFlightRef.current = true;

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // garante o canvas no tamanho do backend
      if (canvas.width !== CAPTURE_WIDTH) canvas.width = CAPTURE_WIDTH;
      if (canvas.height !== CAPTURE_HEIGHT) canvas.height = CAPTURE_HEIGHT;

      // desenha o frame exatamente 1344×760
      ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);

      const base64Image = canvas.toDataURL('image/png');

      // payload esperado pelo backend detect-and-recognize
      const payload = { image: base64Image };

      await fetch(ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortRef.current?.signal,
      });
    } catch (err) {
      // se foi abortado, ignorar
      if ((err as any)?.name !== 'AbortError') {
        console.error('Erro ao enviar frame:', err);
      }
    } finally {
      inFlightRef.current = false;
    }
  };

  const toggleDetection = () => setIsDetecting((prev) => !prev);

  return (
    <div style={{ textAlign: 'center' }}>
      <div
        style={{
          margin: '0 auto',
          width: `${CAPTURE_WIDTH}px`,
          height: `${CAPTURE_HEIGHT}px`,
          position: 'relative',
        }}
      >
        <video
          ref={videoRef}
          width={CAPTURE_WIDTH}
          height={CAPTURE_HEIGHT}
          style={{ width: `${CAPTURE_WIDTH}px`, height: `${CAPTURE_HEIGHT}px`, background: '#000', objectFit: 'cover', display: 'block' }}
          autoPlay
          muted
          playsInline
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>

      <div style={{ marginTop: '20px', display: 'flex', gap: 12, justifyContent: 'center', alignItems: 'center' }}>
        <button
          onClick={toggleDetection}
          style={{
            padding: '10px 20px',
            backgroundColor: isDetecting ? '#d93025' : '#4285F4',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '16px',
          }}
        >
          {isDetecting ? 'Parar (auto para em 60s)' : 'Iniciar Detecção (60s)'}
        </button>

        <label style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
          <input
            type="checkbox"
            checked={useVideoFile}
            onChange={() => setUseVideoFile((prev) => !prev)}
          />
          Usar vídeo
        </label>

        {useVideoFile && (
          <input
            type="file"
            accept="video/mp4,video/webm,video/ogg"
            onChange={handleVideoFileChange}
          />
        )}

        <button
          onClick={sendVideoFile}
          disabled={!videoFile}
          style={{
            padding: '10px 20px',
            backgroundColor: videoFile ? '#34A853' : '#9aa0a6',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: videoFile ? 'pointer' : 'not-allowed',
            fontSize: '16px',
          }}
        >
          Enviar vídeo para processamento
        </button>
      </div>
    </div>
  );
}

export default FaceDetectionComponent;
