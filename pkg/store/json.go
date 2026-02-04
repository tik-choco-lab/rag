package store

import (
	"context"
	"encoding/json"
	"os"

	"github.com/tik-choco-lab/rag/pkg/content"
)

type record struct {
	DocID     string            `json:"doc_id"`
	Hash      string            `json:"hash"`
	Text      string            `json:"text"`
	Embedding []float32         `json:"embedding"`
	Metadata  map[string]string `json:"metadata"`
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

func (s *jsonStore) AddDocument(ctx context.Context, docID string, text string, metadata map[string]string, chunkSize, overlap int, embeddingsFunc func(ctx context.Context, chunks []string) ([][]float32, error)) error {
	cleanText := content.CleanText(text)
	newHash := content.CalculateHash(cleanText)

	duplicate := false
	for _, r := range s.records {
		if r.DocID == docID && r.Hash == newHash {
			duplicate = true
			break
		}
	}
	if duplicate {
		return nil
	}

	if err := s.DeleteDocument(ctx, docID); err != nil {
		return err
	}

	chunks := content.SplitText(cleanText, chunkSize, overlap)
	embeddings, err := embeddingsFunc(ctx, chunks)
	if err != nil {
		return err
	}

	for i, chunk := range chunks {
		s.records = append(s.records, record{
			DocID:     docID,
			Hash:      newHash,
			Text:      chunk,
			Embedding: embeddings[i],
			Metadata:  metadata,
		})
	}

	return s.save()
}

func (s *jsonStore) Search(ctx context.Context, queryEmbedding []float32, options SearchOptions) ([]content.SearchResult, error) {
	var filteredChunks []string
	var filteredEmbeddings [][]float32

	for _, r := range s.records {
		match := true
		for k, v := range options.Metadata {
			if r.Metadata[k] != v {
				match = false
				break
			}
		}
		if match {
			filteredChunks = append(filteredChunks, r.Text)
			filteredEmbeddings = append(filteredEmbeddings, r.Embedding)
		}
	}

	if len(filteredChunks) == 0 {
		return nil, nil
	}

	return content.SearchTopK(queryEmbedding, filteredChunks, filteredEmbeddings, options.TopK, options.Threshold, options.MMRLambda), nil
}

func (s *jsonStore) DeleteDocument(ctx context.Context, docID string) error {
	var newRecords []record
	for _, r := range s.records {
		if r.DocID != docID {
			newRecords = append(newRecords, r)
		}
	}
	s.records = newRecords
	return s.save()
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
