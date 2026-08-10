import { useState, useEffect, useRef } from 'react';
import { Send, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- Config ---
// Set VITE_API_URL at build time to point the UI at a non-local backend; see
// .env.example. Vite inlines it, so it must be a public value -- never a secret.
// The trailing slash is stripped so `${API_BASE}/api/...` cannot produce a
// double slash.
const API_BASE = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/+$/, '');

// --- Utils ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Types ---
interface Persona {
  id: string;
  name: string;
  description: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

// --- Main Component ---
export default function App() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [selectedPersona, setSelectedPersona] = useState<string>('VADER');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/personas`)
      .then(res => res.json())
      .then(data => setPersonas(data))
      .catch(err => console.error("Error fetching personas:", err));
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;

    const userMsg: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, persona: selectedPersona })
      });

      if (!response.body) throw new Error('No body');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMsg = '';

      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (dataStr === '[DONE]') break;
            try {
              const data = JSON.parse(dataStr);
              if (data.content) {
                assistantMsg += data.content;
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1].content = assistantMsg;
                  return updated;
                });
              }
            } catch (e) { /* ignore parse errors */ }
          }
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex h-screen bg-sith-obsidian text-sith-steel relative overflow-hidden font-mono">
      {/* CRT Scanline Overlay */}
      <div className="crt-overlay" />

      {/* Sidebar: Persona Selector */}
      <div className="w-64 border-r border-sith-red bg-sith-obsidian/80 z-10 flex flex-col p-4">
        <div className="flex items-center gap-2 mb-8 text-sith-red glow-text">
          <Zap size={24} />
          <h1 className="text-xl font-bold tracking-widest uppercase">Holocron</h1>
        </div>

        <div className="space-y-4">
          <p className="text-xs uppercase text-sith-red/60 font-bold mb-2">Commune with:</p>
          {personas.map(p => (
            <button
              key={p.id}
              onClick={() => setSelectedPersona(p.id)}
              className={cn(
                "w-full text-left p-3 angular-box transition-all duration-300 border",
                selectedPersona === p.id 
                  ? "bg-sith-red/20 border-sith-red text-sith-red glow-text" 
                  : "border-transparent text-sith-steel/40 hover:text-sith-steel/80"
              )}
            >
              <div className="text-sm font-bold uppercase">{p.name}</div>
              <div className="text-[10px] leading-tight opacity-60 mt-1 line-clamp-2">{p.description}</div>
            </button>
          ))}
        </div>

        <div className="mt-auto pt-8 border-t border-sith-red/20 text-[10px] text-sith-red/40 italic">
          Power resides in the Dark Side.
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative">
        {/* Background Atmosphere */}
        <div className="absolute inset-0 flex items-center justify-center opacity-5 pointer-events-none">
          <div className="w-[500px] h-[500px] rounded-full bg-sith-red blur-[120px]" />
        </div>

        {/* Chat Messages */}
        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-8 space-y-6 scrollbar-hide z-10"
        >
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-4">
              <div className="w-24 h-24 holocron-core border-2 border-sith-red rotate-45 flex items-center justify-center">
                <div className="w-16 h-16 border border-sith-red/50 animate-spin-slow" />
              </div>
              <div className="text-sith-red glow-text uppercase tracking-widest text-lg animate-pulse">
                Awaiting your command...
              </div>
            </div>
          )}

          <AnimatePresence initial={false}>
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: msg.role === 'user' ? 20 : -20 }}
                animate={{ opacity: 1, x: 0 }}
                className={cn(
                  "flex flex-col max-w-[80%]",
                  msg.role === 'user' ? "ml-auto items-end" : "items-start"
                )}
              >
                <div className={cn(
                  "text-[10px] uppercase mb-1 font-bold",
                  msg.role === 'user' ? "text-sith-steel/60" : "text-sith-red/60"
                )}>
                  {msg.role === 'user' ? 'Acolyte' : selectedPersona}
                </div>
                <div className={cn(
                  "p-4 angular-box border text-sm relative",
                  msg.role === 'user' 
                    ? "bg-sith-charcoal/50 border-sith-steel/20" 
                    : "bg-sith-red/5 border-sith-red text-sith-red/90 glow-text"
                )}>
                  {msg.content}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {isTyping && (
             <div className="text-sith-red/40 animate-pulse text-xs uppercase italic ml-2">
               Channeling the Dark Side...
             </div>
          )}
        </div>

        {/* Input Field */}
        <div className="p-8 z-10">
          <div className="relative group">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Submit to the Force..."
              className="w-full bg-sith-obsidian border-b-2 border-sith-red/30 p-4 focus:outline-none focus:border-sith-red text-sith-steel placeholder-sith-steel/20 transition-all duration-500 pr-16"
            />
            <button
              onClick={handleSend}
              disabled={isTyping}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-sith-red/50 hover:text-sith-red transition-colors duration-300"
            >
              <Send size={20} className={isTyping ? "animate-pulse" : ""} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
