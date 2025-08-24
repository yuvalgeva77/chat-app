# Chat App
A modern, responsive chat application built with React, TypeScript, and Tailwind CSS. This frontend connects to a FastAPI backend to provide real-time conversations with AI models.

## Technologies Used
- **React 18** - Frontend framework
- **TypeScript** - Type safety and better development experience
- **Vite** - Fast development and build tool
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icons
- **Python FastAPI** - HTTP client for backend communication

## Features
- 🚀 **Real-time Chat Interface** - Instant message display with auto-scroll
- 💬 **Session Management** - Maintains conversation context across sessions
- 📚 **Chat History** - Load and display previous conversations
- 🗑️ **History Management** - Clear chat history or start new sessions
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile
- ⌨️ **Keyboard Shortcuts** - Press Enter to send messages
- 🎨 **Modern UI** - Clean, professional interface with loading states
- ❌ **Error Handling** - Graceful error messages and recovery

## Prerequisites
- Node.js (v18 or higher)
- npm or yarn
- Running FastAPI backend on `http://localhost:8002`
- Make sure your FastAPI backend has CORS enabled

## Installation
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd chat-frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

## Backend Integration

This frontend is designed to work with a FastAPI backend that provides these endpoints:

- `POST /chat` - Send messages to the AI
- `GET /history?session_id=<id>` - Retrieve conversation history
- `POST /history/reset?session_id=<id>` - Clear chat history

## Project Structure

```
src/
├── App.tsx              # Main chat application component
├── main.tsx            # React application entry point
├── index.css           # Global styles with Tailwind CSS
└── vite-env.d.ts       # Vite type definitions
```

## Key Components

### ChatAPI Class
Handles all communication with the FastAPI backend:
- Message sending with session management
- History retrieval and management
- Error handling with proper HTTP status codes

### ChatMessage Component
Renders individual chat messages with:
- User/bot differentiation
- Timestamps
- Responsive design

### ChatInput Component
Message input interface with:
- Enter key submission
- Input validation
- Loading state handling

### Styling
The application uses Tailwind CSS for styling.

## Logging
- console logging (timestamp, level, file, line).
