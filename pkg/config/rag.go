package config

import (
	"github.com/spf13/viper"
	"github.com/yusufcanb/tlm/pkg/rag"
)

// RAGConfig holds RAG-specific configuration
type RAGConfig struct {
	DefaultMode         string `mapstructure:"default_mode"`
	ChunkSize          int    `mapstructure:"chunk_size"`
	ChunkOverlap       int    `mapstructure:"chunk_overlap"`
	MaxRetrievedDocs   int    `mapstructure:"max_retrieved_docs"`
	MaxContextTokens   int    `mapstructure:"max_context_tokens"`
	SimilarityThreshold float64 `mapstructure:"similarity_threshold"`
	EmbeddingModel     string `mapstructure:"embedding_model"`
}

// DefaultRAGConfig returns default RAG configuration
func DefaultRAGConfig() *RAGConfig {
	return &RAGConfig{
		DefaultMode:         "basic",
		ChunkSize:          1000,
		ChunkOverlap:       200,
		MaxRetrievedDocs:   5,
		MaxContextTokens:   8192,
		SimilarityThreshold: 0.3,
		EmbeddingModel:     "nomic-embed-text",
	}
}

// LoadRAGConfig loads RAG configuration from viper
func LoadRAGConfig() *RAGConfig {
	config := DefaultRAGConfig()
	
	if viper.IsSet("rag.default_mode") {
		config.DefaultMode = viper.GetString("rag.default_mode")
	}
	if viper.IsSet("rag.chunk_size") {
		config.ChunkSize = viper.GetInt("rag.chunk_size")
	}
	if viper.IsSet("rag.chunk_overlap") {
		config.ChunkOverlap = viper.GetInt("rag.chunk_overlap")
	}
	if viper.IsSet("rag.max_retrieved_docs") {
		config.MaxRetrievedDocs = viper.GetInt("rag.max_retrieved_docs")
	}
	if viper.IsSet("rag.max_context_tokens") {
		config.MaxContextTokens = viper.GetInt("rag.max_context_tokens")
	}
	if viper.IsSet("rag.similarity_threshold") {
		config.SimilarityThreshold = viper.GetFloat64("rag.similarity_threshold")
	}
	if viper.IsSet("rag.embedding_model") {
		config.EmbeddingModel = viper.GetString("rag.embedding_model")
	}
	
	return config
}

// ToEnhancedRAGConfig converts to enhanced RAG config
func (rc *RAGConfig) ToEnhancedRAGConfig(model string) *rag.EnhancedRAGConfig {
	return &rag.EnhancedRAGConfig{
		Model:               model,
		MaxContextTokens:    rc.MaxContextTokens,
		MaxRetrievedDocs:   rc.MaxRetrievedDocs,
		ChunkSize:          rc.ChunkSize,
		ChunkOverlap:       rc.ChunkOverlap,
		SimilarityThreshold: rc.SimilarityThreshold,
		EmbeddingModel:      rc.EmbeddingModel,
	}
}

// SetRAGDefaults sets default RAG configuration in viper
func SetRAGDefaults() {
	config := DefaultRAGConfig()
	
	viper.SetDefault("rag.default_mode", config.DefaultMode)
	viper.SetDefault("rag.chunk_size", config.ChunkSize)
	viper.SetDefault("rag.chunk_overlap", config.ChunkOverlap)
	viper.SetDefault("rag.max_retrieved_docs", config.MaxRetrievedDocs)
	viper.SetDefault("rag.max_context_tokens", config.MaxContextTokens)
	viper.SetDefault("rag.similarity_threshold", config.SimilarityThreshold)
	viper.SetDefault("rag.embedding_model", config.EmbeddingModel)
}