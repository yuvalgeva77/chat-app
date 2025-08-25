# Youva's Resume Assistant
### Chat Application with RAG

A full-stack chat application featuring a React/TypeScript frontend and FastAPI backend with Retrieval-Augmented Generation (RAG) capabilities. The application allows for natural language conversations with AI, enhanced by local document retrieval.
The chat's purpose is to Provide a memorable experience for potential employers and collaborators.

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

- **Real-time Chat Interface** - Instant message display with auto-scroll
- **RAG Integration** - Enhanced responses using document retrieval
- **Session Management** - Maintains conversation context across sessions
- **Chat History** - Load and display previous conversations
- **History Management** - Clear chat history or start new sessions
- **Responsive Design** - Works seamlessly on desktop and mobile
- **Docker Support** - Easy containerized deployment
- **Modern UI** - Clean, professional interface with loading states
- **Error Handling** - Graceful error messages and recovery

## Key Design Decisions

### 1. Response Generation Architecture

The system uses a two-layer response generation approach to ensure accuracy and relevance:

1. **Structured Data Responses**
   - For specific queries about skills, experience, or education, the system retrieves and formats information directly from structured data files
   - Ensures factual accuracy for key professional information
   - Maintains consistency in how experience and skills are presented

2. **LLM-Powered Responses**
   - For open-ended questions, the system uses a fine-tuned Qwen2.5-0.5B-Instruct model
   - The model is provided with relevant context from the resume and projects
   - Responses are grounded in the provided data to maintain accuracy

### 2. Response Strategy

The system employs a hybrid response strategy to handle different types of queries:

- **Deterministic Responses**: For specific queries (e.g., "What's your experience with Python?"), the system retrieves and formats information directly from the structured data
- **RAG-Enhanced Responses**: For more complex queries, the system retrieves relevant document chunks and uses them to generate informed responses
- **Fallback to LLM**: When no relevant information is found in the local data, the system can generate a response using the base LLM knowledge

### 3. Model Selection

- **Qwen2.5-0.5B-Instruct** was chosen for its balance between performance and resource efficiency
- The model is fine-tuned to maintain a professional yet approachable tone
- Response length is limited to ensure concise, relevant answers

## Data Flow

1. User submits a query through the chat interface
2. The backend processes the query to determine the intent
3. For fact-based queries, the system retrieves information from the structured data
4. For open-ended questions, the RAG system retrieves relevant document chunks
5. The appropriate response template is selected based on the query type
6. The response is formatted and returned to the user

## Chat History Management

The application maintains conversation context using an in-memory history system, though the UI for viewing past conversations is planned for a future update. The current implementation includes:

- **API Access**: Full history management is available through the backend API
- **Context Window**: Keeps the last 2 conversation turns (configurable via `HISTORY_MAX_TURNS` in `config.py`)
- **Session-based**: Each chat session is identified by a unique `session_id`
- **Automatic Trimming**: History is automatically managed to maintain optimal context length

### API Endpoints

- `GET /history?session_id=<id>` - Retrieve chat history for a specific session
- `POST /history/reset?session_id=<id>` - Reset history for a specific session (or all sessions if no ID provided)

### Configuration

- `HISTORY_MAX_TURNS`: Controls how many conversation turns to keep in context (default: 2)
- `HISTORY_TRIM_ON_EACH_REPLY`: When `True`, trims history after each message (default: `True`)

## Future Enhancements

- Add UI for viewing and managing chat history
- Implement persistent storage for chat history
- Add search functionality for past conversations
- Enable exporting chat history
- Integration with professional social media profiles
- Interactive project demos and code samples
- Multi-language support
- Analytics dashboard to track common queries and user engagement

## Customization

The application can be easily customized for different professional profiles by:

1. Updating the files in the `data/` directory
2. Modifying the response templates in the backend
3. Adjusting the RAG parameters in `app/core/config.py`
4. Updating the frontend styling to match personal branding

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
   .venv\Scripts\activate  # On Windows: 
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the development server:
   ```bash
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
