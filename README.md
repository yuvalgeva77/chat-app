# Chat Application with RAG

A full-stack chat application featuring a React/TypeScript frontend and FastAPI backend with Retrieval-Augmented Generation (RAG) capabilities. The application allows for natural language conversations with AI, enhanced by local document retrieval.

## Technologies Used

### Frontend
- **React 18** - Frontend framework
- **TypeScript** - Type safety and better development experience
- **Vite** - Fast development and build tool
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icons

### Backend
- **Python FastAPI** - High-performance web framework
- **PyTorch** - Deep learning framework
- **HuggingFace Transformers** - For natural language processing
- **Sentence-Transformers** - For text embeddings
- **Uvicorn** - ASGI server

## Features

- 🚀 **Real-time Chat Interface** - Instant message display with auto-scroll
- 🧠 **RAG Integration** - Enhanced responses using document retrieval
- 💬 **Session Management** - Maintains conversation context across sessions
- 📚 **Chat History** - Load and display previous conversations
- 🗑️ **History Management** - Clear chat history or start new sessions
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile
- 🐳 **Docker Support** - Easy containerized deployment
- 🎨 **Modern UI** - Clean, professional interface with loading states
- ❌ **Error Handling** - Graceful error messages and recovery

## Prerequisites

- Docker and Docker Compose
- Node.js (v18 or higher) for frontend development
- Python 3.11+ for backend development

## Quick Start with Docker

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd chat-app
   ```

2. **Build and start the application:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8002
   - API Documentation: http://localhost:8002/docs

## Development Setup

### Frontend Development

```bash
cd chat-frontend
npm install
npm run dev
```

### Backend Development

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the development server:
   ```bash
   cd app
   uvicorn main:app --reload --port 8002
   ```

## Project Structure

```
chat-app/
├── app/                    # Backend FastAPI application
│   ├── api/                # API routes
│   ├── core/               # Core functionality
│   ├── models/             # ML models and embeddings
│   └── main.py             # FastAPI application entry point
├── chat-frontend/          # React frontend
│   ├── public/             # Static files
│   └── src/                # Source code
├── data/                   # Data files for RAG
│   ├── facts.json          # Structured facts data
│   └── cv.pdf              # CV document (optional)
├── docker/                 # Docker configuration
├── .env                    # Environment variables
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile.backend      # Backend Dockerfile
├── Dockerfile.frontend     # Frontend Dockerfile
└── requirements.txt        # Python dependencies
```

## Configuration

The application can be configured using environment variables or by modifying the configuration files:

- `DATA_DIR`: Directory containing RAG data files (default: `/app/data` in Docker)
- `INDEX_DIR`: Directory for storing vector indices (default: `/app/var/index` in Docker)
- `MODEL_NAME`: HuggingFace model name (default: "Qwen/Qwen2.5-0.5B-Instruct")

## License

[Your License Here]
