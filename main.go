package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/tik-choco-lab/rag/internal/config"
	"github.com/tik-choco-lab/rag/pkg/content"
	"github.com/tik-choco-lab/rag/pkg/llm"
)

func main() {
	cfg, err := config.LoadConfig("config.json")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	if cfg.API.APIKey == "" {
		fmt.Println("Error: OPENAI_API_KEY is not set.")
		fmt.Println("Please set it in .env file or config.json")
		os.Exit(1)
	}

	client := llm.NewOpenAIClient(llm.Config{
		APIKey:         cfg.API.APIKey,
		BaseURL:        cfg.API.BaseURL,
		Model:          cfg.API.Model,
		EmbeddingModel: cfg.API.EmbeddingModel,
	})

	ctx := context.Background()

	fmt.Println("--- Text Input ---")
	text, err := content.ReadTextFile("sample.txt")
	if err != nil {
		log.Fatalf("Failed to read file: %v", err)
	}
	fmt.Printf("Read %d characters from sample.txt\n", len(text))

	fmt.Println("\n--- Chunking ---")
	chunks := content.SplitText(text, cfg.Chunk.Size, cfg.Chunk.Overlap)
	fmt.Printf("Split into %d chunks (Size: %d, Overlap: %d)\n", len(chunks), cfg.Chunk.Size, cfg.Chunk.Overlap)
	for i, c := range chunks {
		r := []rune(c)
		displayLen := 20
		if len(r) < displayLen {
			displayLen = len(r)
		}
		fmt.Printf("Chunk %d (%d chars): %s...\n", i, len(r), string(r[:displayLen]))
	}

	fmt.Println("\n--- Vectorization ---")
	embeddings, err := client.CreateEmbeddings(ctx, chunks)
	if err != nil {
		log.Fatalf("Vectorization failed: %v", err)
	}
	fmt.Printf("Successfully generated %d embeddings\n", len(embeddings))
	if len(embeddings) > 0 {
		fmt.Printf("Vector dimension: %d\n", len(embeddings[0]))
	}
}
