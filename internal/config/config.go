package config

import (
	"encoding/json"
	"os"

	"github.com/joho/godotenv"
)

type APIConfig struct {
	APIKey         string `json:"api_key"`
	BaseURL        string `json:"base_url"`
	Model          string `json:"model"`
	EmbeddingModel string `json:"embedding_model"`
}

type ChunkConfig struct {
	Size    int `json:"size"`
	Overlap int `json:"overlap"`
}

type Config struct {
	API   APIConfig   `json:"api"`
	Chunk ChunkConfig `json:"chunk"`
}

func LoadConfig(path string) (*Config, error) {
	_ = godotenv.Load()

	cfg := &Config{
		Chunk: ChunkConfig{
			Size:    500,
			Overlap: 50,
		},
	}

	if path != "" {
		if _, err := os.Stat(path); err == nil {
			f, err := os.Open(path)
			if err == nil {
				defer f.Close()
				if err := json.NewDecoder(f).Decode(cfg); err != nil {
					return nil, err
				}
			}
		}
	}

	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		cfg.API.APIKey = apiKey
	}
	if baseURL := os.Getenv("OPENAI_API_BASE_URL"); baseURL != "" {
		cfg.API.BaseURL = baseURL
	}
	if model := os.Getenv("OPENAI_MODEL"); model != "" {
		cfg.API.Model = model
	}
	if embModel := os.Getenv("OPENAI_EMBEDDING_MODEL"); embModel != "" {
		cfg.API.EmbeddingModel = embModel
	}

	return cfg, nil
}
