package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/tik-choco-lab/rag/internal/config"
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

	fmt.Println("--- Chat Test ---")
	prompt := "Go言語のインターフェースのメリットを簡潔に教えてください。"
	response, err := client.Chat(ctx, prompt)
	if err != nil {
		log.Printf("Chat error: %v", err)
	} else {
		fmt.Printf("Prompt: %s\nResponse: %s\n", prompt, response)
	}

	fmt.Println("\n--- Embedding Test ---")
	text := "これは埋め込みテスト用のテキストです。"
	embedding, err := client.CreateEmbedding(ctx, text)
	if err != nil {
		log.Printf("Embedding error: %v", err)
	} else {
		fmt.Printf("Text: %s\nEmbedding vector length: %d\n", text, len(embedding))
		if len(embedding) > 0 {
			fmt.Printf("First 3 dimensions: %v\n", embedding[:3])
		}
	}
}
