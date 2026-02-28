/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import {
  convertGeminiRequestToClaude,
  convertClaudeResponseToGemini,
  convertClaudeStreamEventToGemini,
  parseSSEStream,
} from './claudeVertexConverter.js';
import type {
  ClaudeResponse,
  ClaudeStreamEvent,
  ToolJsonAccumulator,
} from './claudeVertexConverter.js';
import type { GenerateContentParameters } from '@google/genai';
import { Type } from '@google/genai';

describe('convertGeminiRequestToClaude', () => {
  it('should convert a simple text request', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [
        {
          role: 'user',
          parts: [{ text: 'Hello, Claude!' }],
        },
      ],
      config: {
        maxOutputTokens: 4096,
      },
    };

    const result = convertGeminiRequestToClaude(req, false);

    expect(result.anthropic_version).toBe('vertex-2023-10-16');
    expect(result.max_tokens).toBe(4096);
    expect(result.stream).toBe(false);
    expect(result.messages).toEqual([
      {
        role: 'user',
        content: [{ type: 'text', text: 'Hello, Claude!' }],
      },
    ]);
  });

  it('should convert generation config parameters', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
      config: {
        temperature: 0.7,
        topP: 0.9,
        topK: 50,
        stopSequences: ['STOP'],
      },
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.temperature).toBe(0.7);
    // top_p should be omitted because temperature is provided
    expect(result.top_p).toBeUndefined();
    expect(result.top_k).toBe(50);
    expect(result.stop_sequences).toEqual(['STOP']);
  });

  it('should set top_p if temperature is missing and top_p is not 1.0', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
      config: {
        topP: 0.5,
      },
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.temperature).toBeUndefined();
    expect(result.top_p).toBe(0.5);
  });

  it('should omit top_p if it is the default 1.0', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
      config: {
        topP: 1.0,
      },
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.top_p).toBeUndefined();
  });

  it('should use default max_tokens when not specified', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
    };

    const result = convertGeminiRequestToClaude(req, true);
    expect(result.max_tokens).toBe(8192);
    expect(result.stream).toBe(true);
  });

  it('should convert system instruction as string', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
      config: {
        systemInstruction: 'You are a helpful assistant.',
      },
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.system).toBe('You are a helpful assistant.');
  });

  it('should convert system instruction as Content object', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
      config: {
        systemInstruction: {
          role: 'user',
          parts: [{ text: 'System prompt part 1' }, { text: 'Part 2' }],
        },
      },
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.system).toBe('System prompt part 1\nPart 2');
  });

  it('should convert model role to assistant', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi there!' }] },
        { role: 'user', parts: [{ text: 'How are you?' }] },
      ],
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.messages).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Hi there!' }] },
      { role: 'user', content: [{ type: 'text', text: 'How are you?' }] },
    ]);
  });

  it('should merge consecutive same-role messages', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [
        { role: 'user', parts: [{ text: 'Part 1' }] },
        { role: 'user', parts: [{ text: 'Part 2' }] },
      ],
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].content).toEqual([
      { type: 'text', text: 'Part 1' },
      { type: 'text', text: 'Part 2' },
    ]);
  });

  it('should convert function calls', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [
        { role: 'user', parts: [{ text: 'Read the file' }] },
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                id: 'call_123',
                name: 'readFile',
                args: { path: '/tmp/test.txt' },
              },
            },
          ],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                id: 'call_123',
                name: 'readFile',
                response: { content: 'file contents' },
              },
            },
          ],
        },
      ],
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.messages[1]).toEqual({
      role: 'assistant',
      content: [
        {
          type: 'tool_use',
          id: 'call_123',
          name: 'readFile',
          input: { path: '/tmp/test.txt' },
        },
      ],
    });
    expect(result.messages[2]).toEqual({
      role: 'user',
      content: [
        {
          type: 'tool_result',
          tool_use_id: 'call_123',
          content: JSON.stringify({ content: 'file contents' }),
        },
      ],
    });
  });

  it('should convert tool declarations', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
      config: {
        tools: [
          {
            functionDeclarations: [
              {
                name: 'readFile',
                description: 'Reads a file',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    path: { type: Type.STRING, description: 'File path' },
                  },
                  required: ['path'],
                },
              },
            ],
          },
        ],
      },
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.tools).toEqual([
      {
        name: 'readFile',
        description: 'Reads a file',
        input_schema: {
          type: 'object',
          properties: {
            path: { type: Type.STRING, description: 'File path' },
          },
          required: ['path'],
        },
      },
    ]);
  });

  it('should convert thinking config', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [{ role: 'user', parts: [{ text: 'Think about this' }] }],
      config: {
        thinkingConfig: {
          thinkingBudget: 4096,
        },
      },
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.thinking).toEqual({
      type: 'enabled',
      budget_tokens: 4096,
    });
  });

  it('should normalize string contents', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: 'Just a string prompt',
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.messages).toEqual([
      {
        role: 'user',
        content: [{ type: 'text', text: 'Just a string prompt' }],
      },
    ]);
  });

  it('should skip empty content blocks', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [
        { role: 'user', parts: [] },
        { role: 'user', parts: [{ text: 'Real message' }] },
      ],
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].content).toEqual([
      { type: 'text', text: 'Real message' },
    ]);
  });

  it('should not set tools when no functionDeclarations are provided', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
      config: {
        tools: [{}],
      },
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.tools).toBeUndefined();
  });

  it('should not set thinking when thinkingBudget is not provided', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
      config: {
        thinkingConfig: {},
      },
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.thinking).toBeUndefined();
  });

  it('should convert thought parts in assistant messages', () => {
    const req: GenerateContentParameters = {
      model: 'claude-opus-4-6',
      contents: [
        { role: 'user', parts: [{ text: 'Hello' }] },
        {
          role: 'model',
          parts: [
            { text: 'Let me think...', thought: true },
            { text: 'Here is my answer.' },
          ],
        },
        { role: 'user', parts: [{ text: 'Thanks' }] },
      ],
    };

    const result = convertGeminiRequestToClaude(req, false);
    expect(result.messages[1].content).toEqual([
      { type: 'thinking', thinking: 'Let me think...' },
      { type: 'text', text: 'Here is my answer.' },
    ]);
  });

  describe('tool name sanitization', () => {
    it('should sanitize tool names with invalid characters in declarations', () => {
      const req: GenerateContentParameters = {
        model: 'claude-opus-4-6',
        contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
        config: {
          tools: [
            {
              functionDeclarations: [
                {
                  name: 'my.tool@name!',
                  description: 'A tool with a bad name',
                  parameters: { type: Type.OBJECT, properties: {} },
                },
              ],
            },
          ],
        },
      };

      const result = convertGeminiRequestToClaude(req, false);
      expect(result.tools![0].name).toBe('my_tool_name_');
    });

    it('should sanitize tool names with invalid characters in function calls', () => {
      const req: GenerateContentParameters = {
        model: 'claude-opus-4-6',
        contents: [
          {
            role: 'assistant',
            parts: [
              {
                functionCall: {
                  id: 'call_1',
                  name: 'invalid:tool.name',
                  args: {},
                },
              },
            ],
          },
        ],
      };

      const result = convertGeminiRequestToClaude(req, false);
      expect(result.messages[0].content[0].type).toBe('tool_use');
      const toolUse = result.messages[0].content[0] as {
        type: 'tool_use';
        name: string;
      };
      expect(toolUse.name).toBe('invalid_tool_name');
    });

    it('should truncate tool names longer than 128 characters', () => {
      const longName = 'a'.repeat(150);
      const req: GenerateContentParameters = {
        model: 'claude-opus-4-6',
        contents: [{ role: 'user', parts: [{ text: 'Hi' }] }],
        config: {
          tools: [
            {
              functionDeclarations: [
                {
                  name: longName,
                  parameters: { type: Type.OBJECT, properties: {} },
                },
              ],
            },
          ],
        },
      };

      const result = convertGeminiRequestToClaude(req, false);
      expect(result.tools![0].name).toHaveLength(128);
      expect(result.tools![0].name).toBe('a'.repeat(128));
    });
  });
});

describe('convertClaudeResponseToGemini', () => {
  it('should convert a text response', () => {
    const response: ClaudeResponse = {
      id: 'msg_123',
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: 'Hello from Claude!' }],
      model: 'claude-opus-4-6',
      stop_reason: 'end_turn',
      usage: { input_tokens: 10, output_tokens: 5 },
    };

    const result = convertClaudeResponseToGemini(response);

    expect(result.responseId).toBe('msg_123');
    expect(result.modelVersion).toBe('claude-opus-4-6');
    expect(result.candidates).toHaveLength(1);
    expect(result.candidates![0].content?.parts).toEqual([
      { text: 'Hello from Claude!' },
    ]);
    expect(result.candidates![0].finishReason).toBe('STOP');
    expect(result.usageMetadata?.promptTokenCount).toBe(10);
    expect(result.usageMetadata?.candidatesTokenCount).toBe(5);
    expect(result.usageMetadata?.totalTokenCount).toBe(15);
  });

  it('should convert thinking blocks', () => {
    const response: ClaudeResponse = {
      id: 'msg_456',
      type: 'message',
      role: 'assistant',
      content: [
        { type: 'thinking', thinking: 'Analyzing the problem...' },
        { type: 'text', text: 'The answer is 42.' },
      ],
      model: 'claude-opus-4-6',
      stop_reason: 'end_turn',
      usage: { input_tokens: 20, output_tokens: 15 },
    };

    const result = convertClaudeResponseToGemini(response);
    expect(result.candidates![0].content?.parts).toEqual([
      { text: 'Analyzing the problem...', thought: true },
      { text: 'The answer is 42.' },
    ]);
  });

  it('should convert tool use blocks', () => {
    const response: ClaudeResponse = {
      id: 'msg_789',
      type: 'message',
      role: 'assistant',
      content: [
        {
          type: 'tool_use',
          id: 'toolu_123',
          name: 'readFile',
          input: { path: '/tmp/test.txt' },
        },
      ],
      model: 'claude-opus-4-6',
      stop_reason: 'tool_use',
      usage: { input_tokens: 30, output_tokens: 10 },
    };

    const result = convertClaudeResponseToGemini(response);
    expect(result.candidates![0].content?.parts).toEqual([
      {
        functionCall: {
          id: 'toolu_123',
          name: 'readFile',
          args: { path: '/tmp/test.txt' },
        },
      },
    ]);
    expect(result.candidates![0].finishReason).toBe('STOP');
  });

  it('should map max_tokens stop reason', () => {
    const response: ClaudeResponse = {
      id: 'msg_max',
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: 'Truncated...' }],
      model: 'claude-opus-4-6',
      stop_reason: 'max_tokens',
      usage: { input_tokens: 10, output_tokens: 1000 },
    };

    const result = convertClaudeResponseToGemini(response);
    expect(result.candidates![0].finishReason).toBe('MAX_TOKENS');
  });

  it('should map stop_sequence stop reason to STOP', () => {
    const response: ClaudeResponse = {
      id: 'msg_ss',
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: 'stopped' }],
      model: 'claude-opus-4-6',
      stop_reason: 'stop_sequence',
      usage: { input_tokens: 5, output_tokens: 3 },
    };

    const result = convertClaudeResponseToGemini(response);
    expect(result.candidates![0].finishReason).toBe('STOP');
  });

  it('should map null stop reason to undefined', () => {
    const response: ClaudeResponse = {
      id: 'msg_null',
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: 'partial' }],
      model: 'claude-opus-4-6',
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 3 },
    };

    const result = convertClaudeResponseToGemini(response);
    expect(result.candidates![0].finishReason).toBeUndefined();
  });

  it('should map unknown stop reason to OTHER', () => {
    const response: ClaudeResponse = {
      id: 'msg_other',
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: 'other' }],
      model: 'claude-opus-4-6',
      stop_reason: 'some_unknown_reason',
      usage: { input_tokens: 5, output_tokens: 3 },
    };

    const result = convertClaudeResponseToGemini(response);
    expect(result.candidates![0].finishReason).toBe('OTHER');
  });
});

describe('convertClaudeStreamEventToGemini', () => {
  it('should handle message_start event', () => {
    const event: ClaudeStreamEvent = {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        content: [],
        model: 'claude-opus-4-6',
        stop_reason: null,
        usage: { input_tokens: 50, output_tokens: 0 },
      },
    };

    const acc: ToolJsonAccumulator = {};
    const result = convertClaudeStreamEventToGemini(event, acc);

    expect(result).not.toBeNull();
    expect(result!.responseId).toBe('msg_stream');
    expect(result!.usageMetadata?.promptTokenCount).toBe(50);
  });

  it('should handle text_delta events', () => {
    const event: ClaudeStreamEvent = {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: 'Hello' },
    };

    const acc: ToolJsonAccumulator = {};
    const result = convertClaudeStreamEventToGemini(event, acc);

    expect(result).not.toBeNull();
    expect(result!.candidates![0].content?.parts).toEqual([{ text: 'Hello' }]);
  });

  it('should handle thinking_delta events', () => {
    const event: ClaudeStreamEvent = {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'thinking_delta', thinking: 'Hmm...' },
    };

    const acc: ToolJsonAccumulator = {};
    const result = convertClaudeStreamEventToGemini(event, acc);

    expect(result).not.toBeNull();
    expect(result!.candidates![0].content?.parts).toEqual([
      { text: 'Hmm...', thought: true },
    ]);
  });

  it('should accumulate tool JSON and emit on content_block_stop', () => {
    const acc: ToolJsonAccumulator = {};

    // content_block_start for tool_use
    const startEvent: ClaudeStreamEvent = {
      type: 'content_block_start',
      index: 0,
      content_block: {
        type: 'tool_use',
        id: 'toolu_abc',
        name: 'readFile',
        input: {},
      },
    };
    expect(convertClaudeStreamEventToGemini(startEvent, acc)).toBeNull();
    expect(acc[0]).toEqual({ id: 'toolu_abc', name: 'readFile', json: '' });

    // input_json_delta
    const delta1: ClaudeStreamEvent = {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '{"path":' },
    };
    expect(convertClaudeStreamEventToGemini(delta1, acc)).toBeNull();

    const delta2: ClaudeStreamEvent = {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '"/tmp/f.txt"}' },
    };
    expect(convertClaudeStreamEventToGemini(delta2, acc)).toBeNull();

    // content_block_stop
    const stopEvent: ClaudeStreamEvent = {
      type: 'content_block_stop',
      index: 0,
    };
    const result = convertClaudeStreamEventToGemini(stopEvent, acc);
    expect(result).not.toBeNull();
    expect(result!.candidates![0].content?.parts).toEqual([
      {
        functionCall: {
          id: 'toolu_abc',
          name: 'readFile',
          args: { path: '/tmp/f.txt' },
        },
      },
    ]);
    expect(acc[0]).toBeUndefined();
  });

  it('should handle message_delta with stop_reason', () => {
    const event: ClaudeStreamEvent = {
      type: 'message_delta',
      delta: { type: 'message_delta', stop_reason: 'end_turn' },
      usage: { input_tokens: 0, output_tokens: 42 },
    };

    const acc: ToolJsonAccumulator = {};
    const result = convertClaudeStreamEventToGemini(event, acc);

    expect(result).not.toBeNull();
    expect(result!.candidates![0].finishReason).toBe('STOP');
    expect(result!.usageMetadata?.candidatesTokenCount).toBe(42);
  });

  it('should return null for ping events', () => {
    const event: ClaudeStreamEvent = { type: 'ping' };
    const acc: ToolJsonAccumulator = {};
    expect(convertClaudeStreamEventToGemini(event, acc)).toBeNull();
  });

  it('should return null for content_block_start of text', () => {
    const event: ClaudeStreamEvent = {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    };
    const acc: ToolJsonAccumulator = {};
    expect(convertClaudeStreamEventToGemini(event, acc)).toBeNull();
  });

  it('should return null for message_start without usage', () => {
    const event: ClaudeStreamEvent = {
      type: 'message_start',
      message: {
        id: 'msg_no_usage',
        type: 'message',
        role: 'assistant',
        content: [],
        model: 'claude-opus-4-6',
        stop_reason: null,
        usage: undefined as unknown as {
          input_tokens: number;
          output_tokens: number;
        },
      },
    };
    const acc: ToolJsonAccumulator = {};
    expect(convertClaudeStreamEventToGemini(event, acc)).toBeNull();
  });

  it('should return null for content_block_start without content_block', () => {
    const event: ClaudeStreamEvent = {
      type: 'content_block_start',
      index: 0,
    };
    const acc: ToolJsonAccumulator = {};
    expect(convertClaudeStreamEventToGemini(event, acc)).toBeNull();
  });

  it('should return null for content_block_delta without delta', () => {
    const event: ClaudeStreamEvent = {
      type: 'content_block_delta',
      index: 0,
    };
    const acc: ToolJsonAccumulator = {};
    expect(convertClaudeStreamEventToGemini(event, acc)).toBeNull();
  });

  it('should return null for content_block_stop without matching accumulator', () => {
    const event: ClaudeStreamEvent = {
      type: 'content_block_stop',
      index: 5,
    };
    const acc: ToolJsonAccumulator = {};
    expect(convertClaudeStreamEventToGemini(event, acc)).toBeNull();
  });

  it('should handle malformed tool JSON gracefully', () => {
    const acc: ToolJsonAccumulator = {};

    // Start tool use
    const startEvent: ClaudeStreamEvent = {
      type: 'content_block_start',
      index: 0,
      content_block: {
        type: 'tool_use',
        id: 'toolu_bad',
        name: 'myTool',
        input: {},
      },
    };
    convertClaudeStreamEventToGemini(startEvent, acc);

    // Add malformed JSON
    const deltaEvent: ClaudeStreamEvent = {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '{not valid json' },
    };
    convertClaudeStreamEventToGemini(deltaEvent, acc);

    // Stop should emit with empty args
    const stopEvent: ClaudeStreamEvent = {
      type: 'content_block_stop',
      index: 0,
    };
    const result = convertClaudeStreamEventToGemini(stopEvent, acc);
    expect(result).not.toBeNull();
    expect(
      result!.candidates![0].content?.parts?.[0].functionCall?.args,
    ).toEqual({});
  });

  it('should return null for message_delta without stop_reason or usage', () => {
    const event: ClaudeStreamEvent = {
      type: 'message_delta',
      delta: { type: 'message_delta' },
    };
    const acc: ToolJsonAccumulator = {};
    expect(convertClaudeStreamEventToGemini(event, acc)).toBeNull();
  });

  it('should return null for content_block_stop without index', () => {
    const event: ClaudeStreamEvent = {
      type: 'content_block_stop',
    };
    const acc: ToolJsonAccumulator = {};
    expect(convertClaudeStreamEventToGemini(event, acc)).toBeNull();
  });
});

describe('parseSSEStream', () => {
  function createReadableStream(chunks: string[]): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder();
    let index = 0;
    return new ReadableStream({
      pull(controller) {
        if (index < chunks.length) {
          controller.enqueue(encoder.encode(chunks[index]));
          index++;
        } else {
          controller.close();
        }
      },
    });
  }

  it('should parse SSE events from a stream', async () => {
    const sseData = [
      'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-opus-4-6","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n',
      'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ];

    const stream = createReadableStream(sseData);
    const events: ClaudeStreamEvent[] = [];
    for await (const event of parseSSEStream(stream)) {
      events.push(event);
    }

    expect(events).toHaveLength(3);
    expect(events[0].type).toBe('message_start');
    expect(events[1].type).toBe('content_block_delta');
    expect(events[2].type).toBe('message_stop');
  });

  it('should skip ping events', async () => {
    const sseData = [
      'event: ping\ndata: {}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n',
    ];

    const stream = createReadableStream(sseData);
    const events: ClaudeStreamEvent[] = [];
    for await (const event of parseSSEStream(stream)) {
      events.push(event);
    }

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe('content_block_delta');
  });

  it('should handle chunked data across boundaries', async () => {
    const sseData = [
      'event: content_block_delta\ndata: {"type":"content_block_del',
      'ta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
    ];

    const stream = createReadableStream(sseData);
    const events: ClaudeStreamEvent[] = [];
    for await (const event of parseSSEStream(stream)) {
      events.push(event);
    }

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe('content_block_delta');
  });

  it('should handle empty stream', async () => {
    const stream = createReadableStream([]);
    const events: ClaudeStreamEvent[] = [];
    for await (const event of parseSSEStream(stream)) {
      events.push(event);
    }
    expect(events).toHaveLength(0);
  });

  it('should skip events with malformed JSON data', async () => {
    const sseData = [
      'event: content_block_delta\ndata: {not valid json}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"OK"}}\n\n',
    ];

    const stream = createReadableStream(sseData);
    const events: ClaudeStreamEvent[] = [];
    for await (const event of parseSSEStream(stream)) {
      events.push(event);
    }

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe('content_block_delta');
  });

  it('should handle remaining buffer after stream ends', async () => {
    // Data without trailing double newline
    const sseData = [
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Last"}}',
    ];

    const stream = createReadableStream(sseData);
    const events: ClaudeStreamEvent[] = [];
    for await (const event of parseSSEStream(stream)) {
      events.push(event);
    }

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe('content_block_delta');
  });

  it('should set type from SSE event line when not in parsed data', async () => {
    const sseData = ['event: message_stop\ndata: {"extra":"field"}\n\n'];

    const stream = createReadableStream(sseData);
    const events: ClaudeStreamEvent[] = [];
    for await (const event of parseSSEStream(stream)) {
      events.push(event);
    }

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe('message_stop');
  });
});
