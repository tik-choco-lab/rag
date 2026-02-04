package store

import (
	"context"

	"github.com/tik-choco-lab/rag/pkg/content"
)

type Store interface {
	Add(ctx context.Context, chunks []string, embeddings [][]float32) error
	Search(ctx context.Context, queryEmbedding []float32, k int, threshold float32, mmrLambda float32) ([]content.SearchResult, error)
}
