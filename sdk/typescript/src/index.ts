// infergo TypeScript SDK. OPT-84.

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface Detection {
  class_id: number;
  confidence: number;
  x1: number; y1: number; x2: number; y2: number;
}

export interface InfergoConfig {
  baseUrl?: string;
  apiKey?: string;
}

export class InfergoClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(config: InfergoConfig = {}) {
    this.baseUrl = (config.baseUrl || "http://localhost:9090").replace(/\/$/, "");
    this.apiKey = config.apiKey || "";
  }

  private async request(method: string, path: string, body?: any): Promise<any> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

    const resp = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
    return resp.json();
  }

  async chat(messages: ChatMessage[], model = "llm", maxTokens = 256, temperature = 0.7): Promise<string> {
    const resp = await this.request("POST", "/v1/chat/completions", {
      model, messages, max_tokens: maxTokens, temperature,
    });
    return resp.choices[0].message.content;
  }

  async *chatStream(messages: ChatMessage[], model = "llm", maxTokens = 256, temperature = 0.7): AsyncIterableIterator<string> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

    const resp = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature, stream: true }),
    });

    const reader = resp.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = decoder.decode(value);
      for (const line of text.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") return;
        const chunk = JSON.parse(data);
        const content = chunk.choices?.[0]?.delta?.content;
        if (content) yield content;
      }
    }
  }

  async embed(text: string, model = "embed"): Promise<number[]> {
    const resp = await this.request("POST", "/v1/embeddings", { model, input: text });
    return resp.data[0].embedding;
  }

  async detect(imageB64: string, model = "detect", conf = 0.25): Promise<Detection[]> {
    const resp = await this.request("POST", "/v1/detect", {
      model, image_b64: imageB64, conf_thresh: conf,
    });
    return resp.objects || [];
  }

  async search(query: string, model = "embed", k = 5, mode = "hybrid"): Promise<any[]> {
    const resp = await this.request("POST", "/v1/search", { model, query, k, mode });
    return resp.results || [];
  }

  async models(): Promise<any[]> {
    const resp = await this.request("GET", "/v1/models");
    return resp.data || [];
  }

  async health(): Promise<any> {
    return this.request("GET", "/health/live");
  }
}

export default InfergoClient;
