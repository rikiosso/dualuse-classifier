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
  thinking?: unknown;
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
    // bound wrapper: a bare `fetch` reference invoked as `this.fetcher(...)`
    // throws "Illegal invocation" in workerd (wrong `this`)
    private fetcher: typeof fetch = (...args: Parameters<typeof fetch>) => fetch(...args),
  ) {}

  async complete(req: ClaudeRequest): Promise<ClaudeResponse> {
    // one retry on transient failures (rate limits, overload, 5xx, network
    // errors) — a live turn surfaced "upstream_error" to the user for a blip
    // that a 1.5s backoff absorbs. Non-retryable statuses (400s) still throw
    // immediately. Every call carries a hard 100s deadline: without one, a
    // stalled upstream keeps the request (and a streaming response) pending
    // indefinitely — nothing else in the pipeline bounds an in-flight call.
    for (let attempt = 0; ; attempt++) {
      const ctl = new AbortController();
      const deadline = setTimeout(() => ctl.abort(), 100_000);
      let resp: Response;
      try {
        resp = await this.fetcher("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: {
            "x-api-key": this.apiKey,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
          },
          body: JSON.stringify(req),
          signal: ctl.signal,
        });
      } catch (err) {
        clearTimeout(deadline);
        if (ctl.signal.aborted) throw new Error("anthropic timeout: no response within 100s");
        if (attempt === 0) {
          await new Promise((r) => setTimeout(r, 1500));
          continue;
        }
        throw err;
      }
      clearTimeout(deadline);
      if (resp.ok) return (await resp.json()) as ClaudeResponse;
      const body = await resp.text();
      const retryable = resp.status === 429 || resp.status >= 500;
      if (retryable && attempt === 0) {
        await new Promise((r) => setTimeout(r, 1500));
        continue;
      }
      throw new Error(`anthropic ${resp.status}: ${body.slice(0, 300)}`);
    }
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
