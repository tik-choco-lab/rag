package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	_ "github.com/lib/pq"
	"github.com/pgvector/pgvector-go"
	"github.com/tik-choco-lab/rag/pkg/content"
)

const (
	recencySampleMultiplier = 2
	sqlParamStartIndex      = 2
)

type pgStore struct {
	db        *sql.DB
	tableName string
}

func NewPostgresStore(connStr string, tableName string) (Store, error) {
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, err
	}

	if err := db.Ping(); err != nil {
		return nil, err
	}

	s := &pgStore{
		db:        db,
		tableName: tableName,
	}
	if err := s.init(); err != nil {
		return nil, err
	}

	return s, nil
}

func (s *pgStore) init() error {
	_, err := s.db.Exec(`CREATE EXTENSION IF NOT EXISTS vector`)
	if err != nil {
		return err
	}

	query := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			id SERIAL PRIMARY KEY,
			doc_id TEXT,
			hash TEXT,
			content TEXT,
			embedding vector,
			metadata JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
		)
	`, s.tableName)
	_, err = s.db.Exec(query)
	return err
}

func (s *pgStore) AddDocument(ctx context.Context, docID string, text string, metadata map[string]string, chunkSize, overlap int, embeddingsFunc func(ctx context.Context, chunks []string) ([][]float32, error)) error {
	cleanText := content.CleanText(text)
	newHash := content.CalculateHash(cleanText)

	var existingID int
	query := fmt.Sprintf("SELECT id FROM %s WHERE doc_id = $1 AND hash = $2 LIMIT 1", s.tableName)
	err := s.db.QueryRowContext(ctx, query, docID, newHash).Scan(&existingID)
	if err == nil {
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

	metaJSON, err := json.Marshal(metadata)
	if err != nil {
		return err
	}

	txn, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer txn.Rollback()

	for i, chunk := range chunks {
		query := fmt.Sprintf("INSERT INTO %s (doc_id, hash, content, embedding, metadata, created_at) VALUES ($1, $2, $3, $4, $5, $6)", s.tableName)
		_, err = txn.ExecContext(ctx, query,
			docID, newHash, chunk, pgvector.NewVector(embeddings[i]), metaJSON, time.Now().In(jst),
		)
		if err != nil {
			return err
		}
	}

	return txn.Commit()
}

func (s *pgStore) Search(ctx context.Context, queryEmbedding []float32, options SearchOptions) ([]content.SearchResult, error) {
	where, args := s.buildWhere(options.Metadata)
	query := fmt.Sprintf(`
		SELECT content, 1 - (embedding <=> $1) as score
		FROM %s
		%%s
		ORDER BY embedding <=> $1
		LIMIT %%d
	`, s.tableName)
	query = fmt.Sprintf(query, where, options.TopK)

	fullArgs := append([]interface{}{pgvector.NewVector(queryEmbedding)}, args...)

	rows, err := s.db.QueryContext(ctx, query, fullArgs...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []content.SearchResult
	for rows.Next() {
		var res content.SearchResult
		if err := rows.Scan(&res.Text, &res.Score); err != nil {
			return nil, err
		}
		if res.Score >= options.Threshold {
			results = append(results, res)
		}
	}

	return results, nil
}

func (s *pgStore) RecencySearch(ctx context.Context, queryEmbedding []float32, options SearchOptions) ([]content.SearchResult, error) {
	where, args := s.buildWhere(options.Metadata)
	query := fmt.Sprintf(`
		SELECT content, 1 - (embedding <=> $1) as score, created_at
		FROM %s
		%%s
		ORDER BY embedding <=> $1
		LIMIT %%d
	`, s.tableName)
	query = fmt.Sprintf(query, where, options.TopK*recencySampleMultiplier)

	fullArgs := append([]interface{}{pgvector.NewVector(queryEmbedding)}, args...)
	rows, err := s.db.QueryContext(ctx, query, fullArgs...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type intermediate struct {
		res content.SearchResult
		ts  time.Time
	}

	var samples []intermediate
	var maxTS, minTS int64

	for rows.Next() {
		var item intermediate
		if err := rows.Scan(&item.res.Text, &item.res.Score, &item.ts); err != nil {
			return nil, err
		}
		if item.res.Score < options.Threshold {
			continue
		}
		samples = append(samples, item)
		ts := item.ts.Unix()
		if ts > maxTS {
			maxTS = ts
		}
		if minTS == 0 || ts < minTS {
			minTS = ts
		}
	}

	for i := range samples {
		timeScore := float32(0)
		if maxTS != minTS {
			timeScore = float32(samples[i].ts.Unix()-minTS) / float32(maxTS-minTS)
		}
		samples[i].res.Score = (1.0-options.RecencyWeight)*samples[i].res.Score + options.RecencyWeight*timeScore
	}

	var finalResults []content.SearchResult
	for _, s := range samples {
		finalResults = append(finalResults, s.res)
	}

	slices.SortFunc(finalResults, func(a, b content.SearchResult) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		return 0
	})

	if len(finalResults) > options.TopK {
		return finalResults[:options.TopK], nil
	}
	return finalResults, nil
}

func (s *pgStore) DeleteDocument(ctx context.Context, docID string) error {
	query := fmt.Sprintf("DELETE FROM %s WHERE doc_id = $1", s.tableName)
	_, err := s.db.ExecContext(ctx, query, docID)
	return err
}

func (s *pgStore) buildWhere(metadata map[string]string) (string, []interface{}) {
	if len(metadata) == 0 {
		return "", nil
	}

	var conditions []string
	var args []interface{}
	i := sqlParamStartIndex
	for k, v := range metadata {
		conditions = append(conditions, fmt.Sprintf("metadata->>'%s' = $%d", k, i))
		args = append(args, v)
		i++
	}

	return "WHERE " + strings.Join(conditions, " AND "), args
}
