package config

import (
	"encoding/json"
	"os"
	"strconv"

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

type RetrievalConfig struct {
	TopK          int     `json:"top_k"`
	Threshold     float32 `json:"threshold"`
	MMRLambda     float32 `json:"mmr_lambda"`
	RecencyWeight float32 `json:"recency_weight"`
}

type PostgresConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	User     string `json:"user"`
	Password string `json:"password"`
	DBName   string `json:"dbname"`
	SSLMode  string `json:"sslmode"`
}

type Config struct {
	API       APIConfig       `json:"api"`
	Chunk     ChunkConfig     `json:"chunk"`
	Retrieval RetrievalConfig `json:"retrieval"`
	Postgres  PostgresConfig  `json:"postgres"`
	StoreType string          `json:"store_type"`
}

const (
	defaultChunkSize     = 500
	defaultChunkOverlap  = 50
	defaultTopK          = 5
	defaultThreshold     = 0.1
	defaultMMRLambda     = 0.5
	defaultRecencyWeight = 0.2
	defaultPostgresPort  = 5432
	defaultStoreType     = "json"
	defaultSSLMode       = "disable"
)

func LoadConfig(path string) (*Config, error) {
	_ = godotenv.Load()

	cfg := &Config{
		Chunk: ChunkConfig{
			Size:    defaultChunkSize,
			Overlap: defaultChunkOverlap,
		},
		Retrieval: RetrievalConfig{
			TopK:          defaultTopK,
			Threshold:     defaultThreshold,
			MMRLambda:     defaultMMRLambda,
			RecencyWeight: defaultRecencyWeight,
		},
		StoreType: defaultStoreType,
		Postgres: PostgresConfig{
			Port:    defaultPostgresPort,
			SSLMode: defaultSSLMode,
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
	if v := os.Getenv("POSTGRES_HOST"); v != "" {
		cfg.Postgres.Host = v
	}
	if v := os.Getenv("POSTGRES_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			cfg.Postgres.Port = p
		}
	}
	if v := os.Getenv("POSTGRES_USER"); v != "" {
		cfg.Postgres.User = v
	}
	if v := os.Getenv("POSTGRES_PASSWORD"); v != "" {
		cfg.Postgres.Password = v
	}
	if v := os.Getenv("POSTGRES_DBNAME"); v != "" {
		cfg.Postgres.DBName = v
	}
	if v := os.Getenv("POSTGRES_SSLMODE"); v != "" {
		cfg.Postgres.SSLMode = v
	}

	return cfg, nil
}
