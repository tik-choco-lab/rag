package llm

import (
	"context"

	"github.com/sashabaranov/go-openai"
)

type Client interface {
	Chat(ctx context.Context, prompt string) (string, error)
	ChatMessages(ctx context.Context, messages []openai.ChatCompletionMessage) (string, error)
	ChatMessagesWithFormat(ctx context.Context, messages []openai.ChatCompletionMessage, format *openai.ChatCompletionResponseFormat) (string, error)
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
