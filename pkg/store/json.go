package store

import (
	"context"
	"encoding/json"
	"os"
	"slices"
	"time"

	"github.com/tik-choco-lab/rag/pkg/content"
)

var jst = time.FixedZone("Asia/Tokyo", 9*60*60)

type record struct {
	DocID     string            `json:"doc_id"`
	Hash      string            `json:"hash"`
	Text      string            `json:"text"`
	Embedding []float32         `json:"embedding"`
	Metadata  map[string]string `json:"metadata"`
	CreatedAt int64             `json:"created_at"`
	Date      string            `json:"date"`
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

	now := time.Now().In(jst)
	timestamp := now.Unix()
	isoDate := now.Format(time.RFC3339)

	for i, chunk := range chunks {
		s.records = append(s.records, record{
			DocID:     docID,
			Hash:      newHash,
			Text:      chunk,
			Embedding: embeddings[i],
			Metadata:  metadata,
			CreatedAt: timestamp,
			Date:      isoDate,
		})
	}

	return s.save()
}

func (s *jsonStore) Search(ctx context.Context, queryEmbedding []float32, options SearchOptions) ([]content.SearchResult, error) {
	var filteredChunks []string
	var filteredEmbeddings [][]float32

	for _, r := range s.records {
		if s.matchMetadata(r.Metadata, options.Metadata) {
			filteredChunks = append(filteredChunks, r.Text)
			filteredEmbeddings = append(filteredEmbeddings, r.Embedding)
		}
	}

	if len(filteredChunks) == 0 {
		return nil, nil
	}

	return content.SearchTopK(queryEmbedding, filteredChunks, filteredEmbeddings, options.TopK, options.Threshold, options.MMRLambda), nil
}

func (s *jsonStore) RecencySearch(ctx context.Context, queryEmbedding []float32, options SearchOptions) ([]content.SearchResult, error) {
	var filtered []record
	var maxTS, minTS int64

	for _, r := range s.records {
		if s.matchMetadata(r.Metadata, options.Metadata) {
			filtered = append(filtered, r)
			if r.CreatedAt > maxTS {
				maxTS = r.CreatedAt
			}
			if minTS == 0 || r.CreatedAt < minTS {
				minTS = r.CreatedAt
			}
		}
	}

	if len(filtered) == 0 {
		return nil, nil
	}

	chunks := make([]string, len(filtered))
	embeddings := make([][]float32, len(filtered))
	for i, r := range filtered {
		chunks[i] = r.Text
		embeddings[i] = r.Embedding
	}

	results := content.SearchTopK(queryEmbedding, chunks, embeddings, len(filtered), options.Threshold, 1.0)

	for i := range results {
		var ts int64
		for _, r := range filtered {
			if r.Text == results[i].Text {
				ts = r.CreatedAt
				break
			}
		}

		timeScore := float32(0)
		if maxTS != minTS {
			timeScore = float32(ts-minTS) / float32(maxTS-minTS)
		}

		results[i].Score = (1.0-options.RecencyWeight)*results[i].Score + options.RecencyWeight*timeScore
	}

	slices.SortFunc(results, func(a, b content.SearchResult) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		return 0
	})

	if len(results) > options.TopK {
		return results[:options.TopK], nil
	}
	return results, nil
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

func (s *jsonStore) matchMetadata(recordMeta, searchMeta map[string]string) bool {
	for k, v := range searchMeta {
		if recordMeta[k] != v {
			return false
		}
	}
	return true
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
