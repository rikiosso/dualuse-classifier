// Thin Anthropic Messages API client (raw fetch — no SDK dependency keeps the
// Worker bundle tiny) plus a canned stub for tests. Same discipline as the
// watcher's AnalysisModel/CannedAnalysisModel: tests never touch the network.

export interface ClaudeRequest {
  model: string;
  max_tokens: number;
  system: unknown;
  messages: unknown[];
  tools?: unknown[];
  tool_choice?: unknown;
}

export interface ClaudeUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

export interface ClaudeResponse {
  content: { type: string; [k: string]: unknown }[];
  stop_reason: string;
  usage: ClaudeUsage;
}

export interface ClaudeClient {
  complete(req: ClaudeRequest): Promise<ClaudeResponse>;
}

export class AnthropicClient implements ClaudeClient {
  constructor(
    private apiKey: string,
    private fetcher: typeof fetch = fetch,
  ) {}

  async complete(req: ClaudeRequest): Promise<ClaudeResponse> {
    const resp = await this.fetcher("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": this.apiKey,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify(req),
    });
    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`anthropic ${resp.status}: ${body.slice(0, 300)}`);
    }
    return (await resp.json()) as ClaudeResponse;
  }
}

// Replays scripted responses in order — for tests and `wrangler dev` without a key.
export class CannedClaudeClient implements ClaudeClient {
  public requests: ClaudeRequest[] = [];
  private i = 0;

  constructor(private responses: ClaudeResponse[]) {}

  async complete(req: ClaudeRequest): Promise<ClaudeResponse> {
    this.requests.push(req);
    if (this.i >= this.responses.length) throw new Error("CannedClaudeClient exhausted");
    return this.responses[this.i++];
  }
}
