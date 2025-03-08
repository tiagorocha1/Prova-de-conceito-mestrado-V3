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
from pydantic import BaseModel
import asyncio
from datetime import datetime


# ----------------------------
# Global Setup and Model Loading
# ----------------------------

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["reconhecimento-facial-testes"]
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
# Endpoints
# ----------------------------

@app.post("/recognize")
async def recognize_face(payload: ImagePayload):
    """
    Receives a Base64 image and compares it with registered images.
    If a match is found, adds the new face to the person and returns the person's UUID, tags, and primary photo.
    Otherwise, creates a new person.
    """
    base64_image = payload.image
    try:
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]

        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))

        temp_file = os.path.join(TEMP_DIR, "temp_input.png")
        image.save(temp_file)

        known_people = list(pessoas.find({}))
        match_found = False
        matched_uuid = None

        for pessoa in known_people:
            person_uuid = pessoa["uuid"]
            image_paths = pessoa.get("image_paths", [])
            for stored_image_path in image_paths:
                try:
                    # Try using the global model:
                    result = DeepFace.verify(
                        img1_path=temp_file,
                        img2_path=stored_image_path,
                        enforce_detection=False,
                        #odel=model_facenet512  # global model
                        # Alternatively, for testing, try:
                        model_name="Facenet512"
                    )
                    #print(f"Verification result between {temp_file} and {stored_image_path}: {result}")
                    if result.get("verified") is True:
                        match_found = True
                        matched_uuid = person_uuid
                        person_folder = os.path.join(IMAGES_DIR, matched_uuid)
                        new_filename = f"{uuid.uuid4()}.png"
                        new_image_path = os.path.join(person_folder, new_filename)
                        image.save(new_image_path)
                        pessoas.update_one(
                            {"uuid": matched_uuid},
                            {"$push": {"image_paths": new_image_path}}
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
            new_image_path = os.path.join(person_folder, f"{new_uuid_str}.png")
            image.save(new_image_path)
            new_face_doc = {
                "uuid": new_uuid_str,
                "image_paths": [new_image_path],
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
        return JSONResponse({
            "uuid": matched_uuid,
            "tags": pessoa.get("tags", []),
            "primary_photo": primary_photo
        }, status_code=200)
    except Exception as e:
        import traceback
        print("Erro completo:", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/pessoas")
async def list_pessoas(page: int = 1, limit: int = 10):
    """
    Returns a paginated list of people with their UUID and tags (no photos).
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
    Returns details for a person, including UUID, tags, and primary photo URL.
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
    Returns URLs of all photos for a person.
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
    Returns the URL of the primary photo (first photo) of a person.
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
    Deletes a person with the given UUID and removes their images folder.
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
    Adds a tag to the person with the given UUID.
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
    Removes a tag from the person with the given UUID.
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

"""
@app.post("/recognize-batch")
async def recognize_faces(payload: BatchImagePayload):
   
    #Recebe um array de imagens em Base64, realiza o reconhecimento em cada uma delas e retorna
    #um array com os resultados para cada face.

    results = []
    for face_item in payload.images:
        base64_image = face_item.image
        try:
            if "base64," in base64_image:
                base64_image = base64_image.split("base64,")[1]

            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes))

            temp_file = os.path.join(TEMP_DIR, "temp_input.png")
            image.save(temp_file)

            known_people = list(pessoas.find({}))
            match_found = False
            matched_uuid = None

            for pessoa in known_people:
                person_uuid = pessoa["uuid"]
                image_paths = pessoa.get("image_paths", [])
                for stored_image_path in image_paths:
                    try:
                        result_verify = DeepFace.verify(
                            img1_path=temp_file,
                            img2_path=stored_image_path,
                            enforce_detection=False,
                            model_name="Facenet512"
                        )
                        #print(f"Resultado da verificação entre {temp_file} e {stored_image_path}: {result_verify}")
                        if result_verify.get("verified") is True:
                            match_found = True
                            matched_uuid = person_uuid
                            person_folder = os.path.join(IMAGES_DIR, matched_uuid)
                            new_filename = f"{uuid.uuid4()}.png"
                            new_image_path = os.path.join(person_folder, new_filename)
                            image.save(new_image_path)
                            pessoas.update_one(
                                {"uuid": matched_uuid},
                                {"$push": {"image_paths": new_image_path}}
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
                new_image_path = os.path.join(person_folder, f"{new_uuid_str}.png")
                image.save(new_image_path)
                new_face_doc = {
                    "uuid": new_uuid_str,
                    "image_paths": [new_image_path],
                    "tags": []
                }
                pessoas.insert_one(new_face_doc)
                matched_uuid = new_uuid_str

            if os.path.exists(temp_file):
                os.remove(temp_file)

            pessoa = pessoas.find_one({"uuid": matched_uuid})
            if not pessoa:
                results.append({"error": "Pessoa não encontrada"})
                continue

            primary_photo = None
            if pessoa.get("image_paths"):
                primary_photo = f"http://localhost:8000/static/{os.path.relpath(pessoa['image_paths'][0], IMAGES_DIR).replace(os.path.sep, '/')}"
            results.append({
                "uuid": matched_uuid,
                "tags": pessoa.get("tags", []),
                "primary_photo": primary_photo
            })
        except Exception as e:
            import traceback
            print("Erro completo:", traceback.format_exc())
            results.append({"error": str(e)})

    return JSONResponse({"faces": results}, status_code=200)
 """

#async def recognize_faces(payload: BatchImagePayload):
@app.post("/recognize-batch")
async def recognize_faces(payload: BatchImagePayload):
    """
    Recebe um array de imagens em Base64, realiza o reconhecimento em cada uma delas e retorna
    um array com os resultados para cada face. Para cada face processada (reconhecida ou nova),
    registra uma presença com data, hora, pessoa, foto da captura e tags.
    """
    results = []
    for face_item in payload.images:
        base64_image = face_item.image
        captured_photo_path = None  # Caminho da foto capturada para uso no registro de presença
        try:
            if "base64," in base64_image:
                base64_image = base64_image.split("base64,")[1]

            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes))

            temp_file = os.path.join(TEMP_DIR, "temp_input.png")
            image.save(temp_file)

            known_people = list(pessoas.find({}))
            match_found = False
            matched_uuid = None

            # Tenta encontrar correspondência entre as faces cadastradas
            for pessoa in known_people:
                person_uuid = pessoa["uuid"]
                image_paths = pessoa.get("image_paths", [])
                for stored_image_path in image_paths:
                    try:
                        result_verify = DeepFace.verify(
                            img1_path=temp_file,
                            img2_path=stored_image_path,
                            enforce_detection=False,
                            model_name="Facenet512"
                        )
                        if result_verify.get("verified") is True:
                            match_found = True
                            matched_uuid = person_uuid
                            # Define a pasta da pessoa e o nome do novo arquivo
                            person_folder = os.path.join(IMAGES_DIR, matched_uuid)
                            new_filename = f"{uuid.uuid4()}.png"
                            captured_photo_path = os.path.join(person_folder, new_filename)
                            image.save(captured_photo_path)
                            if len(image_paths) < quantidade_fotos_relacionadas:
                                pessoas.update_one(
                                    {"uuid": matched_uuid},
                                    {"$push": {"image_paths": captured_photo_path}}
                                )
                            else:
                                print(f"Limite de {quantidade_fotos_relacionadas} fotos atingido para a pessoa {matched_uuid}.")
                            break
                    except Exception as e:
                        print(f"Erro ao verificar com {stored_image_path}: {e}")
                if match_found:
                    break

            # Se não houver correspondência, cria uma nova pessoa
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
                results.append({"error": "Pessoa não encontrada"})
                continue

            primary_photo = None
            if pessoa.get("image_paths"):
                primary_photo = f"http://localhost:8000/static/{os.path.relpath(pessoa['image_paths'][0], IMAGES_DIR).replace(os.path.sep, '/')}"

            # Registra a presença
            # Recebe o timestamp enviado para o início do processamento
            time_inicio = face_item.timestamp
            # Obtém o timestamp atual (fim do processamento) em milissegundos
            time_fim = int(datetime.now().timestamp() * 1000)
            # Calcula a diferença
            tempo_processamento = time_fim - time_inicio
            
            presence_doc = {
                "data": datetime.now().strftime("%Y-%m-%d"),
                "hora": datetime.now().strftime("%H:%M:%S"),
                "inicio": time_inicio,
                "fim": time_fim,
                "tempo_processamento": tempo_processamento,
                "pessoa": matched_uuid,
                "foto_captura": captured_photo_path,
                "tags": pessoa.get("tags", [])
            }
            presencas.insert_one(presence_doc)

            results.append({
                "uuid": matched_uuid,
                "tags": pessoa.get("tags", []),
                "primary_photo": primary_photo
            })
        except Exception as e:
            import traceback
            print("Erro completo:", traceback.format_exc())
            results.append({"error": str(e)})

    return JSONResponse({"faces": results}, status_code=200)

@app.delete("/pessoas/{uuid}/photos")
async def remove_photo(uuid: str, payload: dict):
    """
    Remove uma foto (passada na propriedade "photo" do JSON) do registro da pessoa e do sistema de arquivos.
    O payload deve conter: { "photo": "<URL da foto>" }
    """
    try:
        photo_url = payload.get("photo", "").strip()
        if not photo_url:
            raise HTTPException(status_code=400, detail="Foto inválida")
        
        # Converte a URL para o caminho físico, removendo o prefixo e trocando as barras
        prefix = "http://localhost:8000/static/"
        if photo_url.startswith(prefix):
            photo_path_relative = photo_url[len(prefix):] 
            # Converte as barras ("/") para o separador do sistema (por exemplo, "\" no Windows)
            photo_path = os.path.join(IMAGES_DIR, photo_path_relative.replace("/", os.path.sep))
        else:
            # Se não estiver no formato esperado, utiliza o valor recebido (ou lança erro)
            photo_path = photo_url

        # Remove a foto do array image_paths
        result = pessoas.update_one(
            {"uuid": uuid},
            {"$pull": {"image_paths": photo_path}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Pessoa não encontrada")

        # Busca o registro atualizado da pessoa
        pessoa = pessoas.find_one({"uuid": uuid})
        primary_photo = None
        if pessoa.get("image_paths") and len(pessoa["image_paths"]) > 0:
            primary_photo = f"http://localhost:8000/static/{os.path.relpath(pessoa['image_paths'][0], IMAGES_DIR).replace(os.path.sep, '/')}"
        
        return JSONResponse({
            "message": "Foto removida com sucesso",
            "uuid": pessoa["uuid"],
            "primary_photo": primary_photo,
            "image_paths": pessoa.get("image_paths", [])
        }, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

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


@app.delete("/presencas/{id}")
async def delete_presenca(id: str):
    """
    Deleta o registro de presença com o _id fornecido.
    """
    try:
        result = presencas.delete_one({"_id": ObjectId(id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Presença não encontrada")
        return JSONResponse({"message": "Presença deletada com sucesso"}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
# To run:
# python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000