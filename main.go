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

const (
	configPath     = "config.json"
	samplePath     = "sample.txt"
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
	text, err := content.ReadTextFile(samplePath)
	if err != nil {
		log.Fatalf("Failed to read file: %v", err)
	}
	fmt.Printf("Read %d characters from %s\n", len(text), samplePath)

	fmt.Println("\n--- Chunking ---")
	chunks := content.SplitText(text, cfg.Chunk.Size, cfg.Chunk.Overlap)
	fmt.Printf("Split into %d chunks (Size: %d, Overlap: %d)\n", len(chunks), cfg.Chunk.Size, cfg.Chunk.Overlap)
	for i, c := range chunks {
		r := []rune(c)
		l := displayLen
		if len(r) < l {
			l = len(r)
		}
		fmt.Printf("Chunk %d (%d chars): %s...\n", i, len(r), string(r[:l]))
	}

	fmt.Println("\n--- Vectorization ---")
	embeddings, err := client.CreateEmbeddings(ctx, chunks)
	if err != nil {
		log.Fatalf("Vectorization failed: %v", err)
	}
	fmt.Printf("Successfully generated %d embeddings\n", len(embeddings))

	query := "RAGのメリットは何ですか？"
	fmt.Printf("\n--- Query: %s ---\n", query)

	queryEmbedding, err := client.CreateEmbedding(ctx, query)
	if err != nil {
		log.Fatalf("Query embedding failed: %v", err)
	}

	topK := content.SearchTopK(queryEmbedding, chunks, embeddings, cfg.Retrieval.TopK, cfg.Retrieval.Threshold, cfg.Retrieval.MMRLambda)
	fmt.Println("\n--- Search Results ---")
	if len(topK) == 0 {
		fmt.Println("No relevant chunks found above threshold.")
	}
	for _, res := range topK {
		r := []rune(res.Text)
		l := searchSliceLen
		if len(r) < l {
			l = len(r)
		}
		fmt.Printf("[Score: %.4f] %s...\n", res.Score, string(r[:l]))
	}

	contextText := ""
	for _, res := range topK {
		contextText += res.Text + "\n---\n"
	}

	var prompt string
	if contextText == "" {
		prompt = fmt.Sprintf("資料が見つかりませんでした。以下の質問にあなたの知識で答えてください。\n\n# 質問\n%s", query)
	} else {
		prompt = fmt.Sprintf("以下の資料を参考に、質問に答えてください。\n\n# 資料\n%s\n\n# 質問\n%s", contextText, query)
	}

	fmt.Println("\n--- Generating Response ---")

	answer, err := client.Chat(ctx, prompt)
	if err != nil {
		log.Fatalf("Chat failed: %v", err)
	}

	fmt.Printf("\nAnswer:\n%s\n", answer)
}
