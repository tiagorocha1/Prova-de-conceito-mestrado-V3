
import datetime
from bson import ObjectId
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from deepface import DeepFace
import uuid
import os
import base64
import io
from PIL import Image
from pymongo import MongoClient
import shutil
from typing import List
import asyncio
from datetime import datetime
from fastapi import UploadFile, File
# ----------------------------
# Global Setup and Model Loading
# ----------------------------

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["reconhecimento-facial-v3"]
pessoas = db["pessoas"]
presencas = db["presencas"]

# Directories for storing images and temporary files
IMAGES_DIR = "faces_images"
os.makedirs(IMAGES_DIR, exist_ok=True)
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Load the DeepFace model once at startup (global variable)
model_facenet512 = DeepFace.build_model("Facenet512")
print("DeepFace model loaded.")

quantidade_fotos_relacionadas = 1000

# ----------------------------
# Inicialização do detector dlib
# ----------------------------
import dlib
import cv2  # Necessário para conversão para escala de cinza
detector = dlib.get_frontal_face_detector()

# ----------------------------
# FastAPI App and Middleware
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the images directory as static files
app.mount("/static", StaticFiles(directory=IMAGES_DIR), name="static")

# ----------------------------
# Pydantic Models
# ----------------------------
class ImagePayload(BaseModel):
    image: str  # Base64-encoded image

class TagPayload(BaseModel):
    tag: str

class FaceItem(BaseModel):
    image: str  # Base64 da imagem
    timestamp: int  # Timestamp enviado pelo frontend (em milissegundos)

class BatchImagePayload(BaseModel):
    images: List[FaceItem]

# ----------------------------
# Função interna de reconhecimento
# ----------------------------
def process_face(image: Image.Image, start_time: datetime = None) -> dict:
    """
    Processa uma face (imagem PIL) realizando o reconhecimento e o registro de presença.
    Registra os campos: inicio, fim e tempo_processamento (ms).
    Retorna um dicionário com o resultado (uuid, tags, primary_photo).
    """
    if start_time is None:
        start_time = datetime.now()

    # Salva a imagem em um arquivo temporário
    temp_file = os.path.join(TEMP_DIR, "temp_input.png")
    image.save(temp_file)

    known_people = list(pessoas.find({}))
    match_found = False
    matched_uuid = None
    captured_photo_path = None

    for pessoa in known_people:
        person_uuid = pessoa["uuid"]
        image_paths = pessoa.get("image_paths", [])
        for stored_image_path in image_paths:
            try:
                result = DeepFace.verify(
                    img1_path=temp_file,
                    img2_path=stored_image_path,
                    enforce_detection=False,
                    model_name="Facenet512"
                )
                if result.get("verified") is True:
                    match_found = True
                    matched_uuid = person_uuid
                    person_folder = os.path.join(IMAGES_DIR, matched_uuid)
                    new_filename = f"{uuid.uuid4()}.png"
                    captured_photo_path = os.path.join(person_folder, new_filename)
                    image.save(captured_photo_path)
                    pessoas.update_one(
                        {"uuid": matched_uuid},
                        {"$push": {"image_paths": captured_photo_path}}
                    )
                    break
            except Exception as e:
                print(f"Erro ao verificar com {stored_image_path}: {e}")
        if match_found:
            break

    if not match_found:
        new_uuid_str = str(uuid.uuid4())
        person_folder = os.path.join(IMAGES_DIR, new_uuid_str)
        os.makedirs(person_folder, exist_ok=True)
        new_filename = f"{new_uuid_str}.png"
        captured_photo_path = os.path.join(person_folder, new_filename)
        image.save(captured_photo_path)
        new_face_doc = {
            "uuid": new_uuid_str,
            "image_paths": [captured_photo_path],
            "tags": []
        }
        pessoas.insert_one(new_face_doc)
        matched_uuid = new_uuid_str

    if os.path.exists(temp_file):
        os.remove(temp_file)

    pessoa = pessoas.find_one({"uuid": matched_uuid})
    if not pessoa:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada")
    primary_photo = None
    if pessoa.get("image_paths"):
        primary_photo = f"http://localhost:8000/static/{os.path.relpath(pessoa['image_paths'][0], IMAGES_DIR).replace(os.path.sep, '/')}"

    finish_time = datetime.now()
    processing_time_ms = int((finish_time - start_time).total_seconds() * 1000)

    # Registra a presença com os tempos de início, fim e o tempo de processamento (ms)
    presence_doc = {
        "data": start_time.strftime("%Y-%m-%d"),
        "hora": start_time.strftime("%H:%M:%S"),
        "inicio": start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "fim": finish_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "tempo_processamento": processing_time_ms,
        "pessoa": matched_uuid,
        "foto_captura": captured_photo_path,
        "tags": pessoa.get("tags", [])
    }
    presencas.insert_one(presence_doc)

    return {
        "uuid": matched_uuid,
        "tags": pessoa.get("tags", []),
        "primary_photo": primary_photo
    }

# ----------------------------
# Endpoints
# --------------------------


@app.post("/process-video")
async def process_video(video: UploadFile = File(...)):
    """
    Recebe um vídeo, extrai 1 frame a cada 2 e processa cada frame como imagem.
    """
    import cv2
    import tempfile
    import numpy as np
    from PIL import Image

    # Salva o vídeo temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    frame_results = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 2 == 0:
            # Converte frame para PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            # Chama a função de processamento facial já existente
            result = process_face(pil_image)
            frame_results.append(result)
        frame_idx += 1

    cap.release()
    os.remove(temp_video_path)

    return {"frames": frame_results}

@app.post("/recognize")
async def recognize_face(payload: ImagePayload):
    """
    Rota para reconhecimento de face a partir de uma imagem única.
    """
    # Registra o início do processamento
    start_time = datetime.now()
    image_data = payload.image.split("base64,")[1]
    result = process_face(Image.open(io.BytesIO(base64.b64decode(image_data))), start_time=start_time)
    return JSONResponse(result, status_code=200)


@app.post("/detect-and-recognize")
async def detect_and_recognize(payload: ImagePayload):
    """
    Rota que recebe um frame (imagem em Base64), realiza a detecção das faces utilizando dlib,
    recorta cada face detectada e, para cada uma delas, realiza o reconhecimento e o registro de presença,
    medindo os tempos de início, fim e tempo de processamento.
    Retorna um array com os resultados para cada face processada.
    """
    base64_image = payload.image
    try:
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image = image.resize((1344, 760))

        # Converte a imagem para array para o dlib
        import numpy as np
        image_np = np.array(image)

        # Converte a imagem para escala de cinza (melhora a detecção com dlib)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            return JSONResponse({"faces": []}, status_code=200)

        faces_results = []
        for rect in faces:
            # Registra o início do processamento para esta face
            start_time = datetime.now()
            x_min = rect.left()
            y_min = rect.top()
            x_max = rect.right()
            y_max = rect.bottom()
            # Recorta a face a partir das coordenadas obtidas
            face_image = image.crop((x_min, y_min, x_max, y_max))
            # Processa o reconhecimento para a face recortada, passando o start_time
            result_face = process_face(face_image, start_time=start_time)
            faces_results.append(result_face)

        return JSONResponse({"faces": faces_results}, status_code=200)
    except Exception as e:
        import traceback
        print("Erro no detect-and-recognize:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/pessoas")
async def list_pessoas(page: int = 1, limit: int = 10):
    """
    Retorna uma lista paginada de pessoas com seus UUIDs e tags (sem fotos).
    """
    try:
        total = pessoas.count_documents({})
        skip = (page - 1) * limit
        cursor = pessoas.find({}).skip(skip).limit(limit)
        result = []
        for p in cursor:
            result.append({
                "uuid": p["uuid"],
                "tags": p.get("tags", [])
            })
        return JSONResponse({"pessoas": result, "total": total}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/pessoas/{uuid}")
async def get_pessoa(uuid: str):
    """
    Retorna os detalhes de uma pessoa, incluindo UUID, tags e a URL da foto principal.
    """
    try:
        pessoa = pessoas.find_one({"uuid": uuid})
        if not pessoa:
            raise HTTPException(status_code=404, detail="Pessoa não encontrada")
        primary_photo = None
        if pessoa.get("image_paths"):
            primary_photo = f"http://localhost:8000/static/{os.path.relpath(pessoa['image_paths'][0], IMAGES_DIR).replace(os.path.sep, '/')}"
        return JSONResponse({
            "uuid": pessoa["uuid"],
            "tags": pessoa.get("tags", []),
            "primary_photo": primary_photo
        }, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/pessoas/{uuid}/photos")
async def list_photos(uuid: str):
    """
    Retorna as URLs de todas as fotos de uma pessoa.
    """
    try:
        pessoa = pessoas.find_one({"uuid": uuid})
        if not pessoa:
            raise HTTPException(status_code=404, detail="Pessoa não encontrada")
        image_paths = pessoa.get("image_paths", [])
        image_urls = [
            f"http://localhost:8000/static/{os.path.relpath(path, IMAGES_DIR).replace(os.path.sep, '/')}"
            for path in image_paths
        ]
        return JSONResponse({"uuid": uuid, "image_urls": image_urls}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/pessoas/{uuid}/photo")
async def get_primary_photo(uuid: str):
    """
    Retorna a URL da foto principal (primeira foto) de uma pessoa.
    """
    try:
        pessoa = pessoas.find_one({"uuid": uuid})
        if not pessoa:
            raise HTTPException(status_code=404, detail="Pessoa não encontrada")
        image_paths = pessoa.get("image_paths", [])
        if not image_paths:
            raise HTTPException(status_code=404, detail="Nenhuma foto encontrada")
        primary_photo = image_paths[0]
        url = f"http://localhost:8000/static/{os.path.relpath(primary_photo, IMAGES_DIR).replace(os.path.sep, '/')}"
        return JSONResponse({"uuid": uuid, "primary_photo": url}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/pessoas/{uuid}")
async def delete_pessoa(uuid: str):
    """
    Exclui uma pessoa com o UUID fornecido e remove sua pasta de imagens.
    """
    try:
        result = pessoas.delete_one({"uuid": uuid})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Pessoa não encontrada")
        person_folder = os.path.join(IMAGES_DIR, uuid)
        if os.path.exists(person_folder):
            shutil.rmtree(person_folder)
        return JSONResponse({"message": "Pessoa deletada com sucesso"}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/pessoas/{uuid}/tags")
async def add_tag(uuid: str, payload: TagPayload):
    """
    Adiciona uma tag à pessoa com o UUID fornecido.
    """
    try:
        tag = payload.tag.strip()
        if not tag:
            raise HTTPException(status_code=400, detail="Tag inválida")
        result = pessoas.update_one(
            {"uuid": uuid},
            {"$push": {"tags": tag}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Pessoa não encontrada")
        pessoa = pessoas.find_one({"uuid": uuid})
        primary_photo = None
        if pessoa.get("image_paths"):
            primary_photo = f"http://localhost:8000/static/{os.path.relpath(pessoa['image_paths'][0], IMAGES_DIR).replace(os.path.sep, '/')}"
        return JSONResponse({
            "message": "Tag adicionada com sucesso",
            "uuid": pessoa["uuid"],
            "tags": pessoa.get("tags", []),
            "primary_photo": primary_photo
        }, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/pessoas/{uuid}/tags")
async def remove_tag(uuid: str, payload: TagPayload):
    """
    Remove uma tag da pessoa com o UUID fornecido.
    """
    try:
        tag = payload.tag.strip()
        if not tag:
            raise HTTPException(status_code=400, detail="Tag inválida")
        result = pessoas.update_one(
            {"uuid": uuid},
            {"$pull": {"tags": tag}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Pessoa não encontrada")
        pessoa = pessoas.find_one({"uuid": uuid})
        return JSONResponse({
            "message": "Tag removida com sucesso",
            "uuid": pessoa["uuid"],
            "tags": pessoa.get("tags", [])
        }, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/pessoas/{uuid}/photos/count")
async def count_photos(uuid: str):
    try:
        pessoa = pessoas.find_one({"uuid": uuid})
        if not pessoa:
            raise HTTPException(status_code=404, detail="Pessoa não encontrada")
        count = len(pessoa.get("image_paths", []))
        return JSONResponse({"uuid": uuid, "photo_count": count}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/presencas/{id}")
async def delete_presenca(id: str):
    """
    Exclui o registro de presença com o _id fornecido.
    """
    try:
        result = presencas.delete_one({"_id": ObjectId(id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Presença não encontrada")
        return JSONResponse({"message": "Presença deletada com sucesso"}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Para executar:
# python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000

    
@app.get("/presencas")
async def list_presencas(date: str = None, page: int = 1, limit: int = 10):
    """
    Retorna uma lista paginada de registros de presença filtrados pela data (formato YYYY-MM-DD),
    ordenados dos registros mais recentes para os mais antigos.
    Se a data não for fornecida, utiliza a data atual.
    """
    try:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        skip = (page - 1) * limit

        cursor = presencas.find({"data": date}).sort([("data", -1), ("hora", -1)]).skip(skip).limit(limit)
        results = []
        for p in cursor:
            # Converte o caminho da foto para URL
            foto_captura = p.get("foto_captura")
            foto_url = None
            if foto_captura:
                foto_url = f"http://localhost:8000/static/{os.path.relpath(foto_captura, IMAGES_DIR).replace(os.path.sep, '/')}"
            results.append({
                "id": str(p["_id"]),  # Inclui o _id convertido para string
                "uuid": p.get("pessoa"),
                "data": p.get("data"),
                "hora": p.get("hora"),
                "foto_captura": foto_url,
                "tags": p.get("tags", []),
                "inicio": p.get("inicio"),
                "fim": p.get("fim"),
                "tempo_processamento": p.get("tempo_processamento")
            })
        total = presencas.count_documents({"data": date})
        return JSONResponse({"presencas": results, "total": total, "date": date}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# To run:
# python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
