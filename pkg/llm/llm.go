package llm

import "context"

type Client interface {
	Chat(ctx context.Context, prompt string) (string, error)
	CreateEmbedding(ctx context.Context, text string) ([]float32, error)
	CreateEmbeddings(ctx context.Context, texts []string) ([][]float32, error)
	ListModels(ctx context.Context) ([]string, error)
}

type Config struct {
	APIKey         string
	BaseURL        string
	Model          string
	EmbeddingModel string
}
