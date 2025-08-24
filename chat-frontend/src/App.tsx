import React, { useState, useEffect, useRef } from 'react';
import { Send, Trash2, MessageCircle, Bot, User, Loader2 } from 'lucide-react';

/**
 * Type definitions for the chat application
 */
interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ChatResponse {
  reply: string;
  session_id: string;
}

interface HistoryResponse {
  session_id: string;
  history: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

/**
 * API Service class for handling communication with the FastAPI backend
 */
class ChatAPI {
  private baseUrl = 'http://localhost:8002'; // Updated to match your backend port

  /**
   * Send a message to the chat API (non-streaming)
   * @param message - The user's message
   * @param sessionId - Optional session ID for conversation continuity
   * @returns Promise with the bot's response and session ID
   */
  async sendMessage(message: string, sessionId?: string): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to send message`);
    }

    return response.json();
  }

  /**
   * Send a message with streaming response using Server-Sent Events
   * @param message - The user's message
   * @param sessionId - Optional session ID for conversation continuity
   * @param onToken - Callback function called for each token received
   * @param onComplete - Callback function called when streaming is complete
   * @param onError - Callback function called on error
   */
  async sendMessageStream(
    message: string,
    sessionId: string | undefined,
    onToken: (token: string, sessionId: string) => void,
    onComplete: (sessionId: string) => void,
    onError: (error: string) => void
  ): Promise<void> {
    const params = new URLSearchParams({
      message,
      ...(sessionId && { session_id: sessionId }),
    });

    const eventSource = new EventSource(`${this.baseUrl}/chat/stream?${params}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.done) {
          eventSource.close();
          onComplete(data.session_id);
        } else if (data.token) {
          onToken(data.token, data.session_id);
        }
      } catch (err) {
        console.error('Error parsing SSE data:', err);
        eventSource.close();
        onError('Failed to parse streaming response');
      }
    };

    eventSource.onerror = (event) => {
      console.error('SSE error:', event);
      eventSource.close();
      onError('Connection error during streaming');
    };

    // Handle custom error events from backend
    eventSource.addEventListener('error', (event) => {
      try {
        const data = JSON.parse((event as any).data);
        if (data.error) {
          eventSource.close();
          onError(data.error);
        }
      } catch (err) {
        eventSource.close();
        onError('Unknown streaming error');
      }
    });
  }

  /**
   * Retrieve conversation history for a session
   * @param sessionId - The session ID to get history for
   * @returns Promise with conversation history
   */
  async getHistory(sessionId: string): Promise<HistoryResponse> {
    const response = await fetch(`${this.baseUrl}/history?session_id=${sessionId}`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to get history`);
    }

    return response.json();
  }

  /**
   * Reset conversation history for a session or all sessions
   * @param sessionId - Optional session ID to reset. If not provided, resets all sessions
   */
  async resetHistory(sessionId?: string): Promise<void> {
    const url = sessionId
      ? `${this.baseUrl}/history/reset?session_id=${sessionId}`
      : `${this.baseUrl}/history/reset`;

    const response = await fetch(url, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to reset history`);
    }
  }
}

/**
 * Loading spinner component displayed when AI is processing
 */
const LoadingSpinner: React.FC = () => (
  <div className="flex items-center space-x-2 text-blue-500">
    <Loader2 className="w-4 h-4 animate-spin" />
    <span className="text-sm">AI is thinking...</span>
  </div>
);

/**
 * Individual chat message component
 * @param message - Message object containing text, sender, and timestamp
 */
const ChatMessage: React.FC<{ message: Message }> = ({ message }) => {
  const isUser = message.sender === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
        isUser 
          ? 'bg-blue-500 text-white' 
          : 'bg-gray-200 text-gray-800'
      }`}>
        <div className="flex items-center space-x-2 mb-1">
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
          <span className="text-xs opacity-70">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
        <p className="text-sm whitespace-pre-wrap">{message.text}</p>
      </div>
    </div>
  );
};

/**
 * Chat input component for sending messages
 * @param onSendMessage - Callback function when user sends a message
 * @param disabled - Whether the input should be disabled (e.g., during loading)
 */
const ChatInput: React.FC<{
  onSendMessage: (message: string) => void;
  disabled: boolean;
}> = ({ onSendMessage, disabled }) => {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim() && !disabled) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t bg-white p-4">
      <div className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={disabled}
          className="flex-1 border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        />
        <button
          onClick={handleSend}
          disabled={disabled || !input.trim()}
          className="bg-blue-500 hover:bg-blue-600 disabled:opacity-50 text-white px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
        >
          <Send className="w-4 h-4" />
          <span className="hidden sm:inline">Send</span>
        </button>
      </div>
    </div>
  );
};

/**
 * Main Chat Application Component
 */
const ChatApp: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatAPI = new ChatAPI();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load history when session ID changes
  useEffect(() => {
    if (sessionId) {
      loadHistory();
    }
  }, [sessionId]);

  const loadHistory = async () => {
    try {
      const historyResponse = await chatAPI.getHistory(sessionId);
      const historyMessages: Message[] = historyResponse.history.map((item, index) => ({
        id: `history-${index}`,
        text: item.content,
        sender: item.role === 'user' ? 'user' : 'bot',
        timestamp: new Date(),
      }));
      setMessages(historyMessages);
    } catch (err) {
      console.warn('Could not load history:', err);
    }
  };

  const sendMessage = async (messageText: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      text: messageText,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsStreaming(true);
    setError('');

    // Create placeholder bot message for streaming
    const botMessageId = Date.now().toString() + '-bot';
    const botMessage: Message = {
      id: botMessageId,
      text: '',
      sender: 'bot',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, botMessage]);

    try {
      await chatAPI.sendMessageStream(
        messageText,
        sessionId,
        // onToken: Update the bot message with new tokens
        (token: string, newSessionId: string) => {
          if (!sessionId) {
            setSessionId(newSessionId);
          }

          setMessages(prev =>
            prev.map(msg =>
              msg.id === botMessageId
                ? { ...msg, text: msg.text + token }
                : msg
            )
          );
        },
        // onComplete: Streaming finished
        (newSessionId: string) => {
          if (!sessionId) {
            setSessionId(newSessionId);
          }
          setIsStreaming(false);
        },
        // onError: Handle streaming errors
        (errorMessage: string) => {
          setError(`Streaming error: ${errorMessage}`);
          setIsStreaming(false);
          // Remove the placeholder bot message on error
          setMessages(prev => prev.filter(msg => msg.id !== botMessageId));
        }
      );
    } catch (err) {
      setError('Failed to send message. Please try again.');
      setIsStreaming(false);
      // Remove the placeholder bot message on error
      setMessages(prev => prev.filter(msg => msg.id !== botMessageId));
      console.error('Error sending message:', err);
    }
  };

  const clearHistory = async () => {
    try {
      if (sessionId) {
        await chatAPI.resetHistory(sessionId);
      }
      setMessages([]);
      setError('');
    } catch (err) {
      setError('Failed to clear history. Please try again.');
      console.error('Error clearing history:', err);
    }
  };

  const startNewSession = () => {
    setSessionId('');
    setMessages([]);
    setError('');
  };

  return (
    <div className="h-screen bg-gray-100 flex flex-col">
      <div className="flex-1 bg-white shadow-lg flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <MessageCircle className="w-6 h-6" />
              <div>
                <h1 className="text-xl font-bold">AI Chat Assistant</h1>
                <p className="text-sm opacity-80">
                  {sessionId ? `Session: ${sessionId}` : 'New Session'}
                </p>
              </div>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={startNewSession}
                className="bg-white bg-opacity-20 hover:bg-opacity-30 px-3 py-1 rounded text-sm transition-colors"
              >
                New Chat
              </button>
              {messages.length > 0 && (
                <button
                  onClick={clearHistory}
                  className="bg-red-500 bg-opacity-80 hover:bg-opacity-100 px-3 py-1 rounded text-sm transition-colors flex items-center space-x-1"
                >
                  <Trash2 className="w-4 h-4" />
                  <span>Clear</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 mx-4 mt-4 rounded">
            {error}
          </div>
        )}

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p className="text-lg mb-2">Welcome to AI Chat!</p>
                <p className="text-sm">Send a message to start the conversation.</p>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {isStreaming && (
                <div className="flex justify-start mb-4">
                  <div className="max-w-xs lg:max-w-md px-4 py-2 rounded-lg bg-gray-200">
                    <LoadingSpinner />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input */}
        <ChatInput onSendMessage={sendMessage} disabled={isStreaming} />
      </div>
    </div>
  );
};

export default ChatApp;