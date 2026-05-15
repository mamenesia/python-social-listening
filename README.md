# Social Listening AI Analysis API

Python FastAPI backend that provides LLM-powered insights and recommendations for the Social Listening dashboard. Uses Google Gemini to analyse Kalventis vs GSK competitive data.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/v1/analysis` | Generate AI analysis from dashboard snapshot |

## Prerequisites

- Python 3.12+
- A Google Gemini API key

## Setup

**1. Clone / navigate to this directory**

```bash
cd D:\fastapi_all\python-social-listening
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Copy `.env.example` to `.env` and fill in your Gemini API key:

```bash
cp .env.example .env
```

`.env` contents:

```
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

## Running

**Development (with auto-reload)**

```bash
uvicorn main:app --port 8000 --reload
```

**Production**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

The API will be available at `http://localhost:8000`.

Check it's running:

```bash
curl http://localhost:8000/api/health
# {"status":"ok"}
```

Interactive API docs: `http://localhost:8000/docs`

## Running with Docker

**Build the image**

```bash
docker build -t social-listening-api .
```

**Run the container**

```bash
docker run -p 8000:8000 --env-file .env social-listening-api
```

## Connecting to the Frontend

The Next.js frontend (at `social-listening-monitoring/`) calls this backend via the `/api/analysis` proxy route. Make sure this backend is running on port 8000 before clicking **Generate Analysis** on the dashboard overview page.

If you need to run it on a different port, set `ANALYSIS_API_URL` in the Next.js `.env`:

```
ANALYSIS_API_URL=http://localhost:9000
```
