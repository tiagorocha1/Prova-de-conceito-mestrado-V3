# FacePresenceTracker - Projeto Full-Stack

FacePresenceTracker é uma aplicação de reconhecimento facial que utiliza a webcam para capturar imagens, detectar faces e registrar presenças no backend. O projeto é composto por dois diretórios principais:

- **frontend/**: Contém o aplicativo React que gerencia a captura, a detecção facial em tempo real e a visualização dos registros.
- **backend/**: Contém o serviço FastAPI que processa as imagens, realiza o reconhecimento utilizando a biblioteca DeepFace e gerencia o armazenamento dos registros de pessoas e presenças em um banco de dados MongoDB.

---

## Funcionalidades Gerais

- **Detecção Facial em Tempo Real:**  
  O frontend utiliza a biblioteca [MediaPipe Face Detection](https://github.com/google/mediapipe) para capturar e detectar faces em tempo real a partir da webcam.  
  - Possui um botão para iniciar/parar a detecção.

- **Reconhecimento e Cadastro:**  
  Ao detectar uma face, o backend utiliza o [DeepFace](https://github.com/serengil/deepface) para comparar a face capturada com as já cadastradas.  
  - Se a face for reconhecida, a nova imagem é adicionada ao registro da pessoa; caso contrário, uma nova pessoa é criada.
  - O frontend envia também um timestamp para registrar o tempo de processamento.

- **Registro de Presença:**  
  Para cada face processada (reconhecida ou nova), o backend registra uma presença contendo:  
  - Data e hora da captura;
  - Timestamp de início e fim do processamento (e tempo total de processamento);
  - UUID da pessoa;
  - Caminho da foto capturada;
  - Tags associadas à pessoa.

- **Gestão de Pessoas e Presenças:**  
  O backend disponibiliza rotas para:
  - Listar pessoas cadastradas (com paginação);
  - Visualizar detalhes de uma pessoa (tags, foto primária);
  - Adicionar e remover tags de uma pessoa;
  - Listar as fotos de uma pessoa (com opção de remoção individual);
  - Listar e deletar registros de presença filtrados pela data atual (tabela paginada);
  - Remover fotos específicas de um registro de pessoa.

- **Atualização Assíncrona:**  
  O envio das imagens para o backend é feito de forma assíncrona, sem bloquear o fluxo principal da detecção.

---

## Screenshots e Exemplo de Telas

- **Tela de Detecção Facial**  
  ![Tela de Detecção Facial](./screenshots/deteccao_facial.png)  
  *Exibe o feed da webcam com retângulos desenhados ao redor das faces detectadas e um botão para iniciar ou parar a detecção.*

- **Listagem de Pessoas**  
  ![Listagem de Pessoas](./screenshots/listagem_pessoas.png)  
  *Exibe cartões com a foto primária, UUID e tags. Também há botões para listar fotos (com badge de contagem) e para deletar a pessoa.*

- **Registro de Presença**  
  ![Registro de Presença](./screenshots/registros_presenca.png)  
  *Tabela paginada com os registros de presença filtrados pela data atual, exibindo ID, UUID, data, hora, foto de captura e tags, com opção de exclusão de cada registro.*

- **Modal de Fotos**  
  ![Modal de Fotos](./screenshots/modal_fotos.png)  
  *Ao clicar no botão "Listar Fotos" de um cartão de pessoa, um modal é aberto exibindo todas as fotos daquela pessoa, cada uma com um botão de remoção no canto superior direito.*

---

## Bibliotecas e Tecnologias Utilizadas

### Frontend
- **React**: Biblioteca para construção de interfaces dinâmicas.
- **MediaPipe Face Detection**: Para detecção de faces em tempo real.
- **Camera Utils (MediaPipe)**: Facilita a integração com a webcam.
- **React Router**: Para gerenciamento de rotas na aplicação.
- **React Modal**: Para exibição de modais (ex.: listagem de fotos).
- **Fetch API**: Para comunicação com o backend.

### Backend
- **FastAPI**: Framework para criação de APIs REST de alta performance.
- **DeepFace**: Biblioteca para reconhecimento facial (usando modelos como Facenet512).
- **MongoDB**: Banco de dados NoSQL para armazenamento dos registros de pessoas e presenças.
- **Pillow (PIL)**: Para processamento e manipulação de imagens.
- **Uvicorn**: Servidor ASGI para rodar o FastAPI.
- **Shutil**: Para operações com o sistema de arquivos (exclusão de pastas).
- **Datetime**: Para manipulação de datas e horários.

---

## Instruções para Rodar o Projeto

### Pré-requisitos

- [Node.js](https://nodejs.org/) instalado (para o frontend).
- [Python 3.8+](https://www.python.org/) instalado (para o backend).
- [MongoDB](https://www.mongodb.com/) instalado e rodando.
- (Opcional) Ambiente virtual Python configurado.

### Frontend

1. Navegue até a pasta `frontend`:
   ```
   bash
   ```
   ```
   cd frontend
   ```
2. Instale as dependências:
   ```
    npm install
	npm install react-router-dom react-modal
   ```
   
3. Inicie o servidor de desenvolvimento:
	```
	npm start

	```
	O aplicativo React estará disponível geralmente em http://localhost:3000.
	
### Backend

1. Navegue até a pasta backend:
 ```
   bash
   ```
   ```
   cd backend
   ```
2. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```
   python -m venv .venv
   ```
   ```
   source .venv/bin/activate    # No Linux/Mac
   ```
   ```
   .\.venv\Scripts\activate     # No Windows
   ```
3. Instale as dependências (verifique o arquivo requirements.txt):
   ```
   pip install -r requirements.txt
   ```
4. Inicie o servidor FastAPI com Uvicorn:
   ```
   python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```
   A API estará disponível em http://localhost:8000.

---
<!-- 
## Melhorias Futuras

1. Autenticação e Autorização:
Implementar uma camada de autenticação (por exemplo, com JWT) para proteger as rotas de API e gerenciar usuários.

2. Armazenamento de Imagens em MinIO:
Em vez de salvar imagens no sistema de arquivos local, integrar com MinIO para armazenamento de objetos, melhorando escalabilidade e gerenciamento.

3. Integração com Kafka:
Utilizar Kafka para criar um pipeline de processamento de eventos, o que pode ser útil para processamento em tempo real, auditoria e escalabilidade.

4. Otimizações de Processamento:
Melhorar a lógica de detecção e reconhecimento para reduzir falsos positivos.
Cache de modelos e resultados para diminuir a latência nas comparações.
Interface do Usuário e Experiência (UX):

5. Melhorar a UI do frontend, utilizando frameworks de design (como Material-UI ou Tailwind CSS).
Adicionar gráficos e estatísticas sobre a presença e reconhecimento ao longo do tempo.

7. Monitoramento e Log:
Implementar uma camada de monitoramento (por exemplo, com Prometheus/Grafana) e log centralizado para facilitar a manutenção e a detecção de falhas.

8. Testes Automatizados:
Desenvolver uma suíte de testes (unitários e de integração) para ambas as camadas (frontend e backend) garantindo a estabilidade do sistema.

-->


