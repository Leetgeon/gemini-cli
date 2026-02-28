/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Pure functions for bidirectional conversion between @google/genai types
 * and Claude Messages API format for Claude models on Vertex AI.
 */

import type {
  Content,
  ContentListUnion,
  ContentUnion,
  GenerateContentParameters,
  GenerateContentResponse as GeminiResponse,
  Part,
  PartUnion,
  FunctionDeclaration,
} from '@google/genai';
import { GenerateContentResponse, FinishReason } from '@google/genai';

// --- Claude Types ---

interface ClaudeMessage {
  role: 'user' | 'assistant';
  content: ClaudeContentBlock[];
}

interface ClaudeTextBlock {
  type: 'text';
  text: string;
}

interface ClaudeThinkingBlock {
  type: 'thinking';
  thinking: string;
}

interface ClaudeToolUseBlock {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
}

interface ClaudeToolResultBlock {
  type: 'tool_result';
  tool_use_id: string;
  content: string;
}

type ClaudeContentBlock =
  | ClaudeTextBlock
  | ClaudeThinkingBlock
  | ClaudeToolUseBlock
  | ClaudeToolResultBlock;

interface ClaudeToolInputSchema {
  type: 'object';
  properties?: Record<string, unknown>;
  required?: string[];
  [key: string]: unknown;
}

interface ClaudeTool {
  name: string;
  description?: string;
  input_schema: ClaudeToolInputSchema;
}

export interface ClaudeRequest {
  anthropic_version: string;
  messages: ClaudeMessage[];
  max_tokens: number;
  system?: string;
  tools?: ClaudeTool[];
  stream?: boolean;
  thinking?: {
    type: 'enabled';
    budget_tokens: number;
  };
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop_sequences?: string[];
}

interface ClaudeResponseUsage {
  input_tokens: number;
  output_tokens: number;
}

export interface ClaudeResponse {
  id: string;
  type: 'message';
  role: 'assistant';
  content: ClaudeContentBlock[];
  model: string;
  stop_reason: string | null;
  usage: ClaudeResponseUsage;
}

// --- Streaming Event Types ---

export interface ClaudeStreamEvent {
  type: string;
  index?: number;
  message?: ClaudeResponse;
  content_block?: ClaudeContentBlock;
  delta?: {
    type?: string;
    text?: string;
    thinking?: string;
    partial_json?: string;
    stop_reason?: string;
  };
  usage?: ClaudeResponseUsage;
}

// --- JSON Parsing Helpers ---

/** Type guard to narrow unknown to a record for safe property access. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

/**
 * Validates an unknown value as a ClaudeResponse with runtime checks.
 * Uses eslint-disable for the final assertion since there is no way to
 * satisfy the type-assertion rule without a schema validation library.
 * This matches the pattern used throughout the codebase (see
 * oauth-provider.ts, oauth-utils.ts, integrity.ts).
 */
export function validateClaudeResponse(value: unknown): ClaudeResponse {
  if (
    !isRecord(value) ||
    typeof value['id'] !== 'string' ||
    !Array.isArray(value['content']) ||
    !isRecord(value['usage']) ||
    typeof value['usage']['input_tokens'] !== 'number' ||
    typeof value['usage']['output_tokens'] !== 'number'
  ) {
    throw new Error('Invalid Claude response structure');
  }

  // Validate content block types
  for (const block of value['content']) {
    if (!isRecord(block) || typeof block['type'] !== 'string') {
      throw new Error('Invalid Claude content block structure');
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion -- validated above
  return value as unknown as ClaudeResponse;
}

/**
 * Parses a JSON string into a ClaudeStreamEvent with runtime validation.
 */
function parseClaudeStreamEvent(text: string): ClaudeStreamEvent | null {
  const value: unknown = JSON.parse(text);
  if (!isRecord(value)) {
    return null;
  }
  // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion -- validated above
  return value as unknown as ClaudeStreamEvent;
}

// --- Conversion Functions ---

/**
 * Converts a Gemini GenerateContentParameters request to a Claude Messages API request.
 */
export function convertGeminiRequestToClaude(
  req: GenerateContentParameters,
  stream: boolean,
): ClaudeRequest {
  const contents: Content[] = normalizeContents(req.contents);
  const messages = convertContentsToMessages(contents);
  const config = req.config;

  const claudeReq: ClaudeRequest = {
    anthropic_version: 'vertex-2023-10-16',
    messages,
    max_tokens: config?.maxOutputTokens ?? 8192,
    stream,
  };

  if (config?.temperature !== undefined) {
    claudeReq.temperature = config.temperature;
  }
  // Claude on Vertex AI forbids specifying both temperature and top_p.
  // Prioritize temperature, and only set top_p if temperature is not set
  // or if top_p is not the default value of 1.0.
  if (config?.topP !== undefined && config.topP !== 1.0) {
    if (claudeReq.temperature === undefined) {
      claudeReq.top_p = config.topP;
    }
  }
  if (config?.topK !== undefined) {
    claudeReq.top_k = config.topK;
  }
  if (config?.stopSequences !== undefined) {
    claudeReq.stop_sequences = config.stopSequences;
  }

  // System instruction
  if (config?.systemInstruction) {
    const systemContent = extractTextFromContentUnion(config.systemInstruction);
    if (systemContent) {
      claudeReq.system = systemContent;
    }
  }

  // Tools
  if (config?.tools) {
    const claudeTools: ClaudeTool[] = [];
    for (const tool of config.tools) {
      if (
        tool &&
        typeof tool === 'object' &&
        'functionDeclarations' in tool &&
        tool.functionDeclarations
      ) {
        for (const func of tool.functionDeclarations) {
          claudeTools.push(convertFunctionDeclaration(func));
        }
      }
    }
    if (claudeTools.length > 0) {
      claudeReq.tools = claudeTools;
    }
  }

  // Thinking config
  if (config?.thinkingConfig?.thinkingBudget) {
    claudeReq.thinking = {
      type: 'enabled',
      budget_tokens: config.thinkingConfig.thinkingBudget,
    };
  }

  return claudeReq;
}

function extractTextFromContentUnion(content: ContentUnion): string {
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((p) => (typeof p === 'string' ? p : 'text' in p ? p.text : ''))
      .filter(Boolean)
      .join('\n');
  }
  if ('parts' in content && content.parts) {
    return content.parts
      .map((p) => (typeof p === 'string' ? p : 'text' in p ? p.text : ''))
      .filter(Boolean)
      .join('\n');
  }
  if ('text' in content) {
    return content.text || '';
  }
  return '';
}

/**
 * Normalizes ContentListUnion to Content[] without mangling thought parts.
 * Unlike toContents from the code_assist converter, this preserves the
 * `thought` property on parts for proper Claude thinking block conversion.
 */
function normalizeContents(contents: ContentListUnion): Content[] {
  if (Array.isArray(contents)) {
    return contents.map(normalizeContent);
  }
  return [normalizeContent(contents)];
}

function normalizeContent(content: ContentUnion): Content {
  if (typeof content === 'string') {
    return { role: 'user', parts: [{ text: content }] };
  }
  if (Array.isArray(content)) {
    return { role: 'user', parts: normalizeParts(content) };
  }
  if ('parts' in content && content.parts) {
    return {
      ...content,
      parts: normalizeParts(content.parts.filter((p) => p != null)),
    };
  }
  if ('role' in content) {
    return { ...content, parts: content.parts || [] };
  }
  // It's a single Part (text, functionCall, etc.) — wrap it as a user message
  if (
    'text' in content ||
    'functionCall' in content ||
    'functionResponse' in content
  ) {
    return { role: 'user', parts: [content] };
  }
  return { role: 'user', parts: [] };
}

function normalizeParts(parts: PartUnion[]): Part[] {
  return parts.map(normalizePart);
}

function normalizePart(part: PartUnion): Part {
  if (typeof part === 'string') {
    return { text: part };
  }
  return part;
}

/**
 * Converts Gemini Content[] to Claude messages, merging consecutive same-role messages.
 */
function convertContentsToMessages(contents: Content[]): ClaudeMessage[] {
  const messages: ClaudeMessage[] = [];

  for (const content of contents) {
    const role: 'user' | 'assistant' =
      content.role === 'model' ? 'assistant' : 'user';
    const blocks = convertPartsToBlocks(content.parts || [], role);

    if (blocks.length === 0) continue;

    // Merge consecutive same-role messages
    const lastMsg = messages[messages.length - 1];
    if (lastMsg && lastMsg.role === role) {
      lastMsg.content.push(...blocks);
    } else {
      messages.push({ role, content: blocks });
    }
  }

  return messages;
}

/**
 * Sanitizes a tool name to match Claude's allowed pattern: ^[a-zA-Z0-9_-]{1,128}$
 */
function sanitizeToolName(name: string): string {
  // Replace any characters that are not alphanumeric, underscores, or hyphens with underscores
  const sanitized = name.replace(/[^a-zA-Z0-9_-]/g, '_');
  // Truncate to 128 characters if necessary
  const result = sanitized.slice(0, 128);
  // Ensure it's not empty
  return result || 'unnamed_tool';
}

let toolCallCounter = 0;

function convertPartsToBlocks(
  parts: Part[],
  role: 'user' | 'assistant',
): ClaudeContentBlock[] {
  const blocks: ClaudeContentBlock[] = [];

  for (const part of parts) {
    if (part.functionCall) {
      blocks.push({
        type: 'tool_use',
        id: part.functionCall.id || `call_${Date.now()}_${toolCallCounter++}`,
        name: sanitizeToolName(part.functionCall.name || ''),
        input: part.functionCall.args ?? {},
      });
    } else if (part.functionResponse) {
      blocks.push({
        type: 'tool_result',
        tool_use_id: part.functionResponse.id || '',
        content: JSON.stringify(part.functionResponse.response || {}),
      });
    } else if ('thought' in part && part.thought && role === 'assistant') {
      blocks.push({
        type: 'thinking',
        thinking: part.text || '',
      });
    } else if (part.text !== undefined && part.text !== null) {
      blocks.push({
        type: 'text',
        text: part.text,
      });
    }
  }

  return blocks;
}

function convertFunctionDeclaration(func: FunctionDeclaration): ClaudeTool {
  const schema: ClaudeToolInputSchema = { type: 'object' };

  if (func.parameters) {
    if (func.parameters.properties) {
      schema.properties = func.parameters.properties as Record<string, unknown>;
    }
    if (func.parameters.required) {
      schema.required = func.parameters.required;
    }
  }

  return {
    name: sanitizeToolName(func.name || ''),
    description: func.description,
    input_schema: schema,
  };
}

/**
 * Converts a Claude response to a Gemini GenerateContentResponse.
 */
export function convertClaudeResponseToGemini(
  response: ClaudeResponse,
): GeminiResponse {
  const parts = convertClaudeBlocksToParts(response.content);
  const finishReason = mapStopReason(response.stop_reason);

  const out = new GenerateContentResponse();
  out.candidates = [
    {
      content: { role: 'model', parts },
      finishReason,
      index: 0,
    },
  ];
  out.modelVersion = response.model;
  out.responseId = response.id;
  out.usageMetadata = {
    promptTokenCount: response.usage.input_tokens,
    candidatesTokenCount: response.usage.output_tokens,
    totalTokenCount: response.usage.input_tokens + response.usage.output_tokens,
  };

  return out;
}

function convertClaudeBlocksToParts(blocks: ClaudeContentBlock[]): Part[] {
  const parts: Part[] = [];

  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        parts.push({ text: block.text });
        break;
      case 'thinking':
        parts.push({ text: block.thinking, thought: true });
        break;
      case 'tool_use':
        parts.push({
          functionCall: {
            id: block.id,
            name: block.name,
            args: block.input,
          },
        });
        break;
      default:
        // tool_result blocks are not converted back to parts
        break;
    }
  }

  return parts;
}

function mapStopReason(stopReason: string | null): FinishReason | undefined {
  switch (stopReason) {
    case 'end_turn':
      return FinishReason.STOP;
    case 'max_tokens':
      return FinishReason.MAX_TOKENS;
    case 'stop_sequence':
      return FinishReason.STOP;
    case 'tool_use':
      return FinishReason.STOP;
    default:
      return stopReason ? FinishReason.OTHER : undefined;
  }
}

// --- Streaming Conversion ---

/**
 * Accumulator for partial tool call JSON during streaming.
 */
export interface ToolJsonAccumulator {
  [index: number]: {
    id: string;
    name: string;
    json: string;
  };
}

/**
 * Converts an individual SSE event from Claude streaming to a Gemini response chunk.
 * Returns null for events that don't produce output (e.g., ping, intermediate deltas for tool JSON).
 */
export function convertClaudeStreamEventToGemini(
  event: ClaudeStreamEvent,
  toolJsonAccumulator: ToolJsonAccumulator,
): GeminiResponse | null {
  switch (event.type) {
    case 'message_start': {
      // Return an empty response with usage metadata from the message start
      if (event.message?.usage) {
        const out = new GenerateContentResponse();
        out.candidates = [
          {
            content: { role: 'model', parts: [] },
            index: 0,
          },
        ];
        out.modelVersion = event.message.model;
        out.responseId = event.message.id;
        out.usageMetadata = {
          promptTokenCount: event.message.usage.input_tokens,
          candidatesTokenCount: 0,
          totalTokenCount: event.message.usage.input_tokens,
        };
        return out;
      }
      return null;
    }

    case 'content_block_start': {
      const block = event.content_block;
      if (!block) return null;

      if (block.type === 'tool_use' && event.index !== undefined) {
        toolJsonAccumulator[event.index] = {
          id: block.id,
          name: block.name,
          json: '',
        };
        return null;
      }

      // For text/thinking block starts, nothing to emit yet
      return null;
    }

    case 'content_block_delta': {
      const delta = event.delta;
      if (!delta) return null;

      if (delta.type === 'text_delta' && delta.text) {
        const out = new GenerateContentResponse();
        out.candidates = [
          {
            content: {
              role: 'model',
              parts: [{ text: delta.text }],
            },
            index: 0,
          },
        ];
        return out;
      }

      if (delta.type === 'thinking_delta' && delta.thinking) {
        const out = new GenerateContentResponse();
        out.candidates = [
          {
            content: {
              role: 'model',
              parts: [{ text: delta.thinking, thought: true }],
            },
            index: 0,
          },
        ];
        return out;
      }

      if (
        delta.type === 'input_json_delta' &&
        delta.partial_json !== undefined &&
        event.index !== undefined
      ) {
        const acc = toolJsonAccumulator[event.index];
        if (acc) {
          acc.json += delta.partial_json;
        }
        return null; // Accumulate, emit on content_block_stop
      }

      return null;
    }

    case 'content_block_stop': {
      if (event.index !== undefined) {
        const acc = toolJsonAccumulator[event.index];
        if (acc) {
          let args: Record<string, unknown> = {};
          try {
            const raw: unknown = JSON.parse(acc.json || '{}');
            if (isRecord(raw)) {
              args = raw;
            }
          } catch {
            // If JSON is malformed, pass empty args
          }

          const out = new GenerateContentResponse();
          out.candidates = [
            {
              content: {
                role: 'model',
                parts: [
                  {
                    functionCall: {
                      id: acc.id,
                      name: acc.name,
                      args,
                    },
                  },
                ],
              },
              index: 0,
            },
          ];

          delete toolJsonAccumulator[event.index];
          return out;
        }
      }
      return null;
    }

    case 'message_delta': {
      if (event.delta?.stop_reason || event.usage) {
        const out = new GenerateContentResponse();
        out.candidates = [
          {
            content: { role: 'model', parts: [] },
            finishReason: mapStopReason(event.delta?.stop_reason ?? null),
            index: 0,
          },
        ];
        if (event.usage) {
          out.usageMetadata = {
            candidatesTokenCount: event.usage.output_tokens,
            totalTokenCount: event.usage.output_tokens,
          };
        }
        return out;
      }
      return null;
    }

    default:
      return null;
  }
}

/**
 * Parses a ReadableStream of SSE data into individual events.
 */
export async function* parseSSEStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<ClaudeStreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete events (separated by double newlines)
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        const event = parseSSEEvent(part);
        if (event) {
          yield event;
        }
      }
    }

    // Process remaining buffer
    if (buffer.trim()) {
      const event = parseSSEEvent(buffer);
      if (event) {
        yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseSSEEvent(raw: string): ClaudeStreamEvent | null {
  let eventType = '';
  let data = '';

  for (const line of raw.split('\n')) {
    if (line.startsWith('event: ')) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      // Per the SSE spec, data can span multiple lines.
      // Append if it's not the first data line.
      if (data) {
        data += '\n' + line.slice(6);
      } else {
        data = line.slice(6);
      }
    }
  }

  if (!data || eventType === 'ping') {
    return null;
  }

  try {
    const parsed = parseClaudeStreamEvent(data);
    if (!parsed) return null;
    // Ensure the type field is set from the SSE event type if not in the data
    if (!parsed.type && eventType) {
      parsed.type = eventType;
    }
    return parsed;
  } catch {
    return null;
  }
}
