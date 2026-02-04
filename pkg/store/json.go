package store

import (
	"context"
	"encoding/json"
	"os"

	"github.com/tik-choco-lab/rag/pkg/content"
)

type record struct {
	Text      string    `json:"text"`
	Embedding []float32 `json:"embedding"`
}

type jsonStore struct {
	path    string
	records []record
}

func NewJSONStore(path string) Store {
	s := &jsonStore{path: path}
	s.load()
	return s
}

func (s *jsonStore) Add(ctx context.Context, chunks []string, embeddings [][]float32) error {
	for i, chunk := range chunks {
		s.records = append(s.records, record{
			Text:      chunk,
			Embedding: embeddings[i],
		})
	}
	return s.save()
}

func (s *jsonStore) Search(ctx context.Context, queryEmbedding []float32, k int, threshold float32, mmrLambda float32) ([]content.SearchResult, error) {
	chunks := make([]string, len(s.records))
	embeddings := make([][]float32, len(s.records))

	for i, r := range s.records {
		chunks[i] = r.Text
		embeddings[i] = r.Embedding
	}

	return content.SearchTopK(queryEmbedding, chunks, embeddings, k, threshold, mmrLambda), nil
}

func (s *jsonStore) save() error {
	data, err := json.MarshalIndent(s.records, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, data, 0644)
}

func (s *jsonStore) load() error {
	if _, err := os.Stat(s.path); os.IsNotExist(err) {
		return nil
	}
	data, err := os.ReadFile(s.path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, &s.records)
}
