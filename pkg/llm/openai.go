package llm

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

type openAIClient struct {
	api            *openai.Client
	model          string
	embeddingModel string
}

func NewOpenAIClient(cfg Config) Client {
	clientConfig := openai.DefaultConfig(cfg.APIKey)

	if cfg.BaseURL != "" {
		clientConfig.BaseURL = cfg.BaseURL
	}

	return &openAIClient{
		api:            openai.NewClientWithConfig(clientConfig),
		model:          cfg.Model,
		embeddingModel: cfg.EmbeddingModel,
	}
}

func (c *openAIClient) ListModels(ctx context.Context) ([]string, error) {
	models, err := c.api.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	res := make([]string, 0, len(models.Models))
	for _, m := range models.Models {
		res = append(res, m.ID)
	}
	return res, nil
}

func (c *openAIClient) Chat(ctx context.Context, prompt string) (string, error) {
	if c.model == "" {
		return "", fmt.Errorf("chat model is not configured")
	}

	resp, err := c.api.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model: c.model,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
		},
	)

	if err != nil {
		return "", fmt.Errorf("chat completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response choices returned from API")
	}

	return resp.Choices[0].Message.Content, nil
}

func (c *openAIClient) CreateEmbedding(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := c.CreateEmbeddings(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embedding vector returned")
	}
	return embeddings[0], nil
}

func (c *openAIClient) CreateEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	model := c.embeddingModel
	if model == "" {
		model = string(openai.AdaEmbeddingV2)
	}

	resp, err := c.api.CreateEmbeddings(
		ctx,
		openai.EmbeddingRequest{
			Input: texts,
			Model: openai.EmbeddingModel(model),
		},
	)

	if err != nil {
		return nil, fmt.Errorf("embedding creation failed: %w", err)
	}

	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("API returned %d embeddings, but %d were requested", len(resp.Data), len(texts))
	}

	res := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		res[i] = d.Embedding
	}

	return res, nil
}
