export interface ParsedSseEvent {
  event: string;
  data: unknown;
}

/**
 * Incremental Server-Sent Events (SSE) parser safe across arbitrary UTF-8 chunk boundaries.
 */
export class SseStreamParser {
  private buffer = "";
  private currentEvent = "message";
  private dataLines: string[] = [];

  push(chunk: string): ParsedSseEvent[] {
    this.buffer += chunk;
    const events: ParsedSseEvent[] = [];

    let newlineIndex = this.buffer.indexOf("\n");
    while (newlineIndex !== -1) {
      const rawLine = this.buffer.slice(0, newlineIndex).replace(/\r$/, "");
      this.buffer = this.buffer.slice(newlineIndex + 1);
      const emitted = this.processLine(rawLine);
      if (emitted) {
        events.push(emitted);
      }
      newlineIndex = this.buffer.indexOf("\n");
    }
    return events;
  }

  flush(): ParsedSseEvent[] {
    const events: ParsedSseEvent[] = [];
    if (this.buffer.length > 0) {
      const emitted = this.processLine(this.buffer.replace(/\r$/, ""));
      this.buffer = "";
      if (emitted) {
        events.push(emitted);
      }
    }
    const trailing = this.dispatchCurrent();
    if (trailing) {
      events.push(trailing);
    }
    return events;
  }

  private processLine(line: string): ParsedSseEvent | null {
    if (line === "") {
      return this.dispatchCurrent();
    }
    if (line.startsWith(":")) {
      return null;
    }
    if (line.startsWith("event:")) {
      this.currentEvent = line.slice("event:".length).trim() || "message";
      return null;
    }
    if (line.startsWith("data:")) {
      const value = line.slice("data:".length);
      this.dataLines.push(value.startsWith(" ") ? value.slice(1) : value);
      return null;
    }
    return null;
  }

  private dispatchCurrent(): ParsedSseEvent | null {
    if (this.dataLines.length === 0) {
      this.currentEvent = "message";
      return null;
    }
    const rawData = this.dataLines.join("\n");
    const eventName = this.currentEvent;
    this.currentEvent = "message";
    this.dataLines = [];

    let parsed: unknown = rawData;
    try {
      parsed = JSON.parse(rawData);
    } catch {
      parsed = { raw: rawData };
    }
    return { event: eventName, data: parsed };
  }
}
