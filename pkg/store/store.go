package store

import (
	"context"

	"github.com/tik-choco-lab/rag/pkg/content"
)

type SearchOptions struct {
	TopK      int
	Threshold float32
	MMRLambda float32
	Metadata  map[string]string
}

type Store interface {
	AddDocument(ctx context.Context, docID string, text string, metadata map[string]string, chunkSize, overlap int, embeddingsFunc func(ctx context.Context, chunks []string) ([][]float32, error)) error
	Search(ctx context.Context, queryEmbedding []float32, options SearchOptions) ([]content.SearchResult, error)
	DeleteDocument(ctx context.Context, docID string) error
}
