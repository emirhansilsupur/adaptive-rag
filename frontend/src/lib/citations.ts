export type TextCitationSegment =
  | { type: "text"; value: string }
  | { type: "citation"; ids: string[]; raw: string };

// Matches bracketed citation tokens like [C1], [W2], [C1, C2], [C1, W1]
const CITATION_BRACKET_RE = /\[((?:[CW]\d+)(?:\s*,\s*[CW]\d+)*)\]/g;

/**
 * Split a plain text span into regular text and structured citation pills.
 * Supports both single ([C1]) and grouped ([C1, W2]) bracket citations.
 */
export function splitTextWithCitations(text: string): TextCitationSegment[] {
  if (!text) return [];
  const segments: TextCitationSegment[] = [];
  let lastIndex = 0;

  CITATION_BRACKET_RE.lastIndex = 0;
  let match = CITATION_BRACKET_RE.exec(text);
  while (match !== null) {
    if (match.index > lastIndex) {
      segments.push({
        type: "text",
        value: text.slice(lastIndex, match.index),
      });
    }
    const ids = match[1]
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    segments.push({
      type: "citation",
      ids,
      raw: match[0],
    });
    lastIndex = match.index + match[0].length;
    match = CITATION_BRACKET_RE.exec(text);
  }

  if (lastIndex < text.length) {
    segments.push({
      type: "text",
      value: text.slice(lastIndex),
    });
  }

  return segments;
}
