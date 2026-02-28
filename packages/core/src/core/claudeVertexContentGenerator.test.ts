/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ClaudeVertexContentGenerator } from './claudeVertexContentGenerator.js';
import type { GenerateContentParameters } from '@google/genai';
import { LlmRole } from '../telemetry/llmRole.js';

const mockGetAccessToken = vi.fn().mockResolvedValue({ token: 'mock-token' });
const mockGetClient = vi.fn().mockResolvedValue({
  getAccessToken: mockGetAccessToken,
});

vi.mock('google-auth-library', () => ({
  GoogleAuth: vi.fn().mockImplementation(() => ({
    getClient: mockGetClient,
  })),
}));

describe('ClaudeVertexContentGenerator', () => {
  let generator: ClaudeVertexContentGenerator;

  beforeEach(() => {
    vi.clearAllMocks();
    generator = new ClaudeVertexContentGenerator('test-project', 'us-east5');
  });

  describe('generateContent', () => {
    it('should make a POST request to rawPredict endpoint', async () => {
      const mockResponse = {
        id: 'msg_123',
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: 'Hello!' }],
        model: 'claude-opus-4-6',
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 5 },
      };

      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const req: GenerateContentParameters = {
        model: 'claude-opus-4-6',
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };

      const result = await generator.generateContent(
        req,
        'prompt-1',
        LlmRole.MAIN,
      );

      expect(fetchSpy).toHaveBeenCalledWith(
        'https://us-east5-aiplatform.googleapis.com/v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-opus-4-6:rawPredict',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            Authorization: 'Bearer mock-token',
          }),
        }),
      );

      expect(result.responseId).toBe('msg_123');
      expect(result.candidates![0].content?.parts).toEqual([
        { text: 'Hello!' },
      ]);

      fetchSpy.mockRestore();
    });

    it('should throw on API error', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
        ok: false,
        status: 429,
        text: async () => 'Rate limit exceeded',
      } as Response);

      const req: GenerateContentParameters = {
        model: 'claude-opus-4-6',
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };

      await expect(
        generator.generateContent(req, 'prompt-1', LlmRole.MAIN),
      ).rejects.toThrow(
        'Claude Vertex AI API error (429): Rate limit exceeded',
      );

      fetchSpy.mockRestore();
    });
  });

  describe('auth token failure', () => {
    it('should throw when access token cannot be obtained', async () => {
      mockGetAccessToken.mockResolvedValueOnce({
        token: null,
        res: { data: {} },
      });

      const req: GenerateContentParameters = {
        model: 'claude-opus-4-6',
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };

      await expect(
        generator.generateContent(req, 'prompt-1', LlmRole.MAIN),
      ).rejects.toThrow('Failed to obtain access token');
    });
  });

  describe('generateContentStream', () => {
    it('should make a POST request to streamRawPredict endpoint', async () => {
      const encoder = new TextEncoder();
      const sseData =
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_s1","type":"message","role":"assistant","content":[],"model":"claude-opus-4-6","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n' +
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n' +
        'event: message_delta\ndata: {"type":"message_delta","delta":{"type":"message_delta","stop_reason":"end_turn"},"usage":{"input_tokens":0,"output_tokens":5}}\n\n';

      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode(sseData));
          controller.close();
        },
      });

      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
        ok: true,
        body: stream,
      } as Response);

      const req: GenerateContentParameters = {
        model: 'claude-opus-4-6',
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };

      const gen = await generator.generateContentStream(
        req,
        'prompt-1',
        LlmRole.MAIN,
      );

      const chunks = [];
      for await (const chunk of gen) {
        chunks.push(chunk);
      }

      expect(fetchSpy).toHaveBeenCalledWith(
        'https://us-east5-aiplatform.googleapis.com/v1/projects/test-project/locations/us-east5/publishers/anthropic/models/claude-opus-4-6:streamRawPredict',
        expect.objectContaining({
          method: 'POST',
        }),
      );

      expect(chunks.length).toBeGreaterThan(0);

      // Find the text chunk
      const textChunk = chunks.find(
        (c) => c.candidates?.[0]?.content?.parts?.[0]?.text === 'Hi',
      );
      expect(textChunk).toBeDefined();

      fetchSpy.mockRestore();
    });

    it('should use the base hostname for global location', async () => {
      const globalGenerator = new ClaudeVertexContentGenerator(
        'test-project',
        'global',
      );
      const mockResponse = {
        id: 'msg_123',
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: 'Hello!' }],
        model: 'claude-opus-4-6',
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 5 },
      };

      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const req: GenerateContentParameters = {
        model: 'claude-opus-4-6',
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };

      await globalGenerator.generateContent(req, 'prompt-1', LlmRole.MAIN);

      expect(fetchSpy).toHaveBeenCalledWith(
        'https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/anthropic/models/claude-opus-4-6:rawPredict',
        expect.any(Object),
      );

      fetchSpy.mockRestore();
    });

    it('should throw on stream API error', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
        ok: false,
        status: 500,
        text: async () => 'Internal Server Error',
      } as Response);

      const req: GenerateContentParameters = {
        model: 'claude-opus-4-6',
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };

      await expect(
        generator.generateContentStream(req, 'prompt-1', LlmRole.MAIN),
      ).rejects.toThrow(
        'Claude Vertex AI API error (500): Internal Server Error',
      );

      fetchSpy.mockRestore();
    });

    it('should throw when response body is missing', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
        ok: true,
        body: null,
      } as Response);

      const req: GenerateContentParameters = {
        model: 'claude-opus-4-6',
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      };

      await expect(
        generator.generateContentStream(req, 'prompt-1', LlmRole.MAIN),
      ).rejects.toThrow('No response body received from Claude Vertex AI');

      fetchSpy.mockRestore();
    });
  });

  describe('countTokens', () => {
    it('should return zero tokens', async () => {
      const result = await generator.countTokens({
        model: 'claude-opus-4-6',
        contents: 'test',
      });
      expect(result.totalTokens).toBe(0);
    });
  });

  describe('embedContent', () => {
    it('should throw an error', async () => {
      await expect(
        generator.embedContent({
          model: 'claude-opus-4-6',
          contents: 'test',
        }),
      ).rejects.toThrow('Embedding is not supported for Claude models');
    });
  });
});
