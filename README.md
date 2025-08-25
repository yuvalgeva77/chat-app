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

## Purpose

This application was designed to:
- Present a professional profile in an engaging, interactive format
- Demonstrate technical expertise in full-stack development and AI integration
- Showcase the ability to build production-ready AI applications
- Provide a memorable experience for potential employers and collaborators

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

### 2. Data Organization

The application uses a structured approach to manage resume data:

```
data/
├── facts.json          # Structured professional information
│   ├── basics          # Contact information and summary
│   ├── experience      # Work history and roles
│   ├── education       # Academic background
│   ├── skills          # Technical and professional skills
│   └── projects        # Key projects and contributions
├── cv.pdf              # Traditional PDF resume
└── portfolio/          # Detailed project documentation
    ├── project1.md
    └── project2.md
```

### 3. Response Strategy

The system employs a hybrid response strategy to handle different types of queries:

- **Deterministic Responses**: For specific queries (e.g., "What's your experience with Python?"), the system retrieves and formats information directly from the structured data
- **RAG-Enhanced Responses**: For more complex queries, the system retrieves relevant document chunks and uses them to generate informed responses
- **Fallback to LLM**: When no relevant information is found in the local data, the system can generate a response using the base LLM knowledge

### 4. Model Selection

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

## Customization

The application can be easily customized for different professional profiles by:

1. Updating the files in the `data/` directory
2. Modifying the response templates in the backend
3. Adjusting the RAG parameters in `app/core/config.py`
4. Updating the frontend styling to match personal branding

## Future Enhancements

- Integration with professional social media profiles
- Interactive project demos and code samples
- Multi-language support
- Analytics dashboard to track common queries and user engagement

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
