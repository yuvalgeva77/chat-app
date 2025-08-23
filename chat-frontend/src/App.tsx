import React, { useState, useEffect, useRef } from 'react';
import { Send, Trash2, MessageCircle, Bot, User, Loader2 } from 'lucide-react';

// Types
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

// API Service
class ChatAPI {
  private baseUrl = 'http://localhost:8000'; // Adjust to your backend URL

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
      throw new Error('Failed to send message');
    }

    return response.json();
  }

  async getHistory(sessionId: string): Promise<HistoryResponse> {
    const response = await fetch(`${this.baseUrl}/history?session_id=${sessionId}`);

    if (!response.ok) {
      throw new Error('Failed to get history');
    }

    return response.json();
  }

  async resetHistory(sessionId?: string): Promise<void> {
    const url = sessionId
      ? `${this.baseUrl}/history/reset?session_id=${sessionId}`
      : `${this.baseUrl}/history/reset`;

    const response = await fetch(url, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error('Failed to reset history');
    }
  }
}

// Components
const LoadingSpinner: React.FC = () => (
  <div className="flex items-center space-x-2 text-blue-500">
    <Loader2 className="w-4 h-4 animate-spin" />
    <span className="text-sm">AI is thinking...</span>
  </div>
);

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

// Main Chat Application
const ChatApp: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
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
    setIsLoading(true);
    setError('');

    try {
      const response = await chatAPI.sendMessage(messageText, sessionId);

      // Update session ID if it's new
      if (!sessionId) {
        setSessionId(response.session_id);
      }

      const botMessage: Message = {
        id: Date.now().toString() + '-bot',
        text: response.reply,
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      setError('Failed to send message. Please try again.');
      console.error('Error sending message:', err);
    } finally {
      setIsLoading(false);
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
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-4xl mx-auto bg-white shadow-lg">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <MessageCircle className="w-6 h-6" />
              <div>
                <h1 className="text-xl font-bold">AI Chat Assistant</h1>
                <p className="text-sm opacity-80">
                  {sessionId ? `Session: ${sessionId.slice(0, 8)}...` : 'New Session'}
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
        <div className="h-96 overflow-y-auto p-4 bg-gray-50">
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
              {isLoading && (
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
        <ChatInput onSendMessage={sendMessage} disabled={isLoading} />
      </div>
    </div>
  );
};

export default ChatApp;