package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/tik-choco-lab/rag/internal/config"
	"github.com/tik-choco-lab/rag/pkg/content"
	"github.com/tik-choco-lab/rag/pkg/llm"
	"github.com/tik-choco-lab/rag/pkg/store"
)

const (
	configPath     = "config.json"
	samplePath     = "sample.txt"
	dbPath         = "store.json"
	tableName      = "documents"
	displayLen     = 20
	searchSliceLen = 50
)

func main() {
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	if cfg.API.APIKey == "" {
		fmt.Println("Error: OPENAI_API_KEY is not set.")
		os.Exit(1)
	}

	client := llm.NewOpenAIClient(llm.Config{
		APIKey:         cfg.API.APIKey,
		BaseURL:        cfg.API.BaseURL,
		Model:          cfg.API.Model,
		EmbeddingModel: cfg.API.EmbeddingModel,
	})

	var dataStore store.Store
	if cfg.StoreType == "postgres" {
		connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
			cfg.Postgres.Host, cfg.Postgres.Port, cfg.Postgres.User, cfg.Postgres.Password, cfg.Postgres.DBName, cfg.Postgres.SSLMode)
		var err error
		dataStore, err = store.NewPostgresStore(connStr, tableName)
		if err != nil {
			log.Fatalf("Failed to initialize postgres store: %v", err)
		}
		fmt.Println("Using Postgres Store")
	} else {
		dataStore = store.NewJSONStore(dbPath)
		fmt.Println("Using JSON Store")
	}
	ctx := context.Background()

	rawText, err := content.ReadTextFile(samplePath)
	if err != nil {
		log.Fatalf("Failed to read file: %v", err)
	}

	metadata := map[string]string{
		"version": "v1.0",
		"date":    "2024-02-04",
	}

	err = dataStore.AddDocument(ctx, samplePath, rawText, metadata, cfg.Chunk.Size, cfg.Chunk.Overlap, client.CreateEmbeddings)
	if err != nil {
		log.Fatalf("Failed to add document: %v", err)
	}

	query := "RAGのメリットは何ありますか？"
	fmt.Printf("\n--- Query: %s ---\n", query)

	queryEmbedding, err := client.CreateEmbedding(ctx, query)
	if err != nil {
		log.Fatalf("Query embedding failed: %v", err)
	}

	searchOpts := store.SearchOptions{
		TopK:          cfg.Retrieval.TopK,
		Threshold:     cfg.Retrieval.Threshold,
		MMRLambda:     cfg.Retrieval.MMRLambda,
		RecencyWeight: cfg.Retrieval.RecencyWeight,
		Metadata: map[string]string{
			"version": "v1.0",
		},
	}

	results, err := dataStore.RecencySearch(ctx, queryEmbedding, searchOpts)

	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Println("\n--- Search Results ---")
	for _, res := range results {
		r := []rune(res.Text)
		l := searchSliceLen
		if len(r) < l {
			l = len(r)
		}
		fmt.Printf("[Score: %.4f] %s...\n", res.Score, string(r[:l]))
	}

	answer, err := client.Chat(ctx, buildPrompt(results, query))
	if err != nil {
		log.Fatalf("Chat failed: %v", err)
	}

	fmt.Printf("\nAnswer:\n%s\n", answer)
}

func buildPrompt(results []content.SearchResult, query string) string {
	if len(results) == 0 {
		return fmt.Sprintf("資料が見つかりませんでした。以下の質問にあなたの知識で答えてください。\n\n# 質問\n%s", query)
	}

	var contextText string
	for _, res := range results {
		contextText += res.Text + "\n---\n"
	}

	return fmt.Sprintf("以下の資料を参考に、質問に答えてください。\n\n# 資料\n%s\n\n# 質問\n%s", contextText, query)
}
