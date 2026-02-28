/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * ContentGenerator implementation for Claude models hosted on Vertex AI.
 * Uses google-auth-library for ADC tokens and fetch for HTTP calls.
 */

import type {
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  GenerateContentParameters,
  GenerateContentResponse,
} from '@google/genai';
import { GoogleAuth } from 'google-auth-library';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';
import {
  convertGeminiRequestToClaude,
  convertClaudeResponseToGemini,
  convertClaudeStreamEventToGemini,
  parseSSEStream,
  validateClaudeResponse,
} from './claudeVertexConverter.js';
import type { ToolJsonAccumulator } from './claudeVertexConverter.js';

export class ClaudeVertexContentGenerator implements ContentGenerator {
  private readonly auth: GoogleAuth;
  private readonly project: string;
  private readonly location: string;

  constructor(project: string, location: string) {
    this.project = project;
    this.location = location;
    this.auth = new GoogleAuth({
      scopes: ['https://www.googleapis.com/auth/cloud-platform'],
    });
  }

  private getEndpoint(
    model: string,
    method: 'rawPredict' | 'streamRawPredict',
  ): string {
    const host =
      this.location === 'global'
        ? 'aiplatform.googleapis.com'
        : `${this.location}-aiplatform.googleapis.com`;
    return `https://${host}/v1/projects/${this.project}/locations/${this.location}/publishers/anthropic/models/${model}:${method}`;
  }

  private async getAccessToken(): Promise<string> {
    const client = await this.auth.getClient();
    const tokenResponse = await client.getAccessToken();
    if (!tokenResponse.token) {
      throw new Error(
        'Failed to obtain access token. Ensure Application Default Credentials are configured.',
      );
    }
    return tokenResponse.token;
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId?: string,
    _role?: LlmRole,
  ): Promise<GenerateContentResponse> {
    const claudeReq = convertGeminiRequestToClaude(request, false);
    const endpoint = this.getEndpoint(request.model, 'rawPredict');
    const token = await this.getAccessToken();

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(claudeReq),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Claude Vertex AI API error (${response.status}): ${errorBody}`,
      );
    }

    const responseBody: unknown = await response.json();
    const claudeResponse = validateClaudeResponse(responseBody);
    return convertClaudeResponseToGemini(claudeResponse);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId?: string,
    _role?: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const claudeReq = convertGeminiRequestToClaude(request, true);
    const endpoint = this.getEndpoint(request.model, 'streamRawPredict');
    const token = await this.getAccessToken();

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(claudeReq),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Claude Vertex AI API error (${response.status}): ${errorBody}`,
      );
    }

    if (!response.body) {
      throw new Error('No response body received from Claude Vertex AI');
    }

    return this.streamFromResponse(response.body);
  }

  private async *streamFromResponse(
    body: ReadableStream<Uint8Array>,
  ): AsyncGenerator<GenerateContentResponse> {
    const toolJsonAccumulator: ToolJsonAccumulator = {};

    for await (const event of parseSSEStream(body)) {
      const chunk = convertClaudeStreamEventToGemini(
        event,
        toolJsonAccumulator,
      );
      if (chunk) {
        yield chunk;
      }
    }
  }

  async countTokens(
    _request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Claude does not have a count tokens endpoint on Vertex AI.
    return { totalTokens: 0 };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error(
      'Embedding is not supported for Claude models. Use a Gemini embedding model instead.',
    );
  }
}
