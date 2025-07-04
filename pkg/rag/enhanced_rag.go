package rag

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	ollama "github.com/jmorganca/ollama/api"
	"github.com/yusufcanb/tlm/pkg/packer"
)

// EnhancedRAGChat provides enhanced RAG capabilities with vector search
type EnhancedRAGChat struct {
	api         *ollama.Client
	vectorStore VectorStore
	chunker     *DocumentChunker
	model       string
	history     []ollama.Message
	
	// Configuration
	maxContextTokens    int
	maxRetrievedDocs   int
	chunkSize          int
	chunkOverlap       int
	similarityThreshold float64
}

// EnhancedRAGConfig holds configuration for the enhanced RAG system
type EnhancedRAGConfig struct {
	Model               string
	MaxContextTokens    int
	MaxRetrievedDocs   int
	ChunkSize          int
	ChunkOverlap       int
	SimilarityThreshold float64
	EmbeddingModel      string
}

// DefaultEnhancedRAGConfig returns a default configuration
func DefaultEnhancedRAGConfig() *EnhancedRAGConfig {
	return &EnhancedRAGConfig{
		Model:               "qwen2.5-coder:3b",
		MaxContextTokens:    8192,
		MaxRetrievedDocs:   5,
		ChunkSize:          1000,
		ChunkOverlap:       200,
		SimilarityThreshold: 0.3,
		EmbeddingModel:      "nomic-embed-text",
	}
}

// NewEnhancedRAGChat creates a new enhanced RAG chat instance
func NewEnhancedRAGChat(api *ollama.Client, config *EnhancedRAGConfig) *EnhancedRAGChat {
	if config == nil {
		config = DefaultEnhancedRAGConfig()
	}

	vectorStore := NewInMemoryVectorStore(api)
	chunker := NewDocumentChunker(config.ChunkSize, config.ChunkOverlap)

	return &EnhancedRAGChat{
		api:                 api,
		vectorStore:         vectorStore,
		chunker:             chunker,
		model:               config.Model,
		history:             make([]ollama.Message, 0),
		maxContextTokens:    config.MaxContextTokens,
		maxRetrievedDocs:   config.MaxRetrievedDocs,
		chunkSize:          config.ChunkSize,
		chunkOverlap:       config.ChunkOverlap,
		similarityThreshold: config.SimilarityThreshold,
	}
}

// IndexDocuments processes and indexes documents from a directory
func (erag *EnhancedRAGChat) IndexDocuments(ctx context.Context, contextPath string, includePatterns, excludePatterns []string) error {
	fmt.Println("🔍 Indexing documents for enhanced RAG...")
	
	// Get file paths using public packer function
	filePaths, err := packer.GetContextFilePaths(contextPath, includePatterns, excludePatterns)
	if err != nil {
		return fmt.Errorf("failed to get context files: %w", err)
	}

	var totalChunks int
	var totalFiles int

	for _, fp := range filePaths {
		content, _, _, err := packer.GetFileContent(contextPath, fp)
		if err != nil {
			fmt.Printf("⚠️  Skipping %s: %v\n", fp, err)
			continue
		}

		fullPath := filepath.Join(contextPath, fp)
		
		// Chunk the document
		chunks := erag.chunker.ChunkDocument(content, fullPath)
		if len(chunks) == 0 {
			continue
		}

		// Add chunks to vector store
		if err := erag.vectorStore.AddDocuments(ctx, chunks); err != nil {
			fmt.Printf("⚠️  Failed to index %s: %v\n", fp, err)
			continue
		}

		totalChunks += len(chunks)
		totalFiles++
	}

	fmt.Printf("✅ Successfully indexed %d files (%d chunks)\n", totalFiles, totalChunks)
	return nil
}

// RetrieveRelevantContext retrieves relevant context for a query using vector search
func (erag *EnhancedRAGChat) RetrieveRelevantContext(ctx context.Context, query string) (string, error) {
	// Search for relevant documents
	docs, err := erag.vectorStore.Search(ctx, query, erag.maxRetrievedDocs)
	if err != nil {
		return "", fmt.Errorf("failed to search documents: %w", err)
	}

	if len(docs) == 0 {
		return "", nil
	}

	// Build context from retrieved documents
	var contextBuilder strings.Builder
	contextBuilder.WriteString("# Relevant Context\n\n")

	for i, doc := range docs {
		contextBuilder.WriteString(fmt.Sprintf("## Document %d: %s\n", i+1, doc.Metadata.Source))
		if doc.Metadata.Language != "" {
			contextBuilder.WriteString(fmt.Sprintf("Language: %s\n", doc.Metadata.Language))
		}
		if len(doc.Metadata.Tags) > 0 {
			contextBuilder.WriteString(fmt.Sprintf("Tags: %s\n", strings.Join(doc.Metadata.Tags, ", ")))
		}
		contextBuilder.WriteString("```\n")
		contextBuilder.WriteString(doc.Content)
		contextBuilder.WriteString("\n```\n\n")
	}

	return contextBuilder.String(), nil
}

// SendWithVectorSearch sends a message using enhanced RAG with vector search
func (erag *EnhancedRAGChat) SendWithVectorSearch(ctx context.Context, message string) (string, error) {
	// Retrieve relevant context
	context, err := erag.RetrieveRelevantContext(ctx, message)
	if err != nil {
		return "", fmt.Errorf("failed to retrieve context: %w", err)
	}

	// Build the prompt with context
	var prompt string
	if context != "" {
		prompt = fmt.Sprintf(`Based on the following context, please answer the question:

%s

Question: %s

Please provide a helpful and accurate answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please mention what additional information might be needed.`, context, message)
	} else {
		prompt = message
	}

	return erag.sendMessage(ctx, prompt)
}

// Send sends a message using the existing simple RAG approach (for compatibility)
func (erag *EnhancedRAGChat) Send(ctx context.Context, message string, contextString string) (string, error) {
	var prompt string
	if contextString != "" {
		if len(erag.history) == 0 {
			prompt = contextString + "\n\n" + message
		} else {
			prompt = message
		}
	} else {
		prompt = message
	}

	return erag.sendMessage(ctx, prompt)
}

// sendMessage handles the actual message sending
func (erag *EnhancedRAGChat) sendMessage(ctx context.Context, message string) (string, error) {
	// Add system message if this is the first message
	if len(erag.history) == 0 {
		erag.history = append(erag.history, ollama.Message{
			Role:    "system",
			Content: "You are a helpful software engineering assistant. Provide accurate, concise, and practical answers based on the context provided. When answering questions about code, include relevant examples and explanations.",
		})
	}

	// Add user message
	erag.history = append(erag.history, ollama.Message{
		Role:    "user",
		Content: message,
	})

	// Send to Ollama
	var responseBuilder strings.Builder
	
	err := erag.api.Chat(ctx, &ollama.ChatRequest{
		Model:    erag.model,
		Messages: erag.history,
		Options: map[string]interface{}{
			"temperature": 0.5,
			"num_ctx":     erag.maxContextTokens,
		},
	}, func(res ollama.ChatResponse) error {
		fmt.Print(res.Message.Content)
		responseBuilder.WriteString(res.Message.Content)
		return nil
	})

	if err != nil {
		return "", fmt.Errorf("error sending message: %w", err)
	}

	response := responseBuilder.String()

	// Add assistant response to history
	erag.history = append(erag.history, ollama.Message{
		Role:    "assistant",
		Content: response,
	})

	return response, nil
}

// GetDocumentStats returns statistics about indexed documents
func (erag *EnhancedRAGChat) GetDocumentStats(ctx context.Context) (map[string]interface{}, error) {
	// This is a simple implementation for in-memory store
	// In a real implementation, you'd query the vector store for stats
	stats := map[string]interface{}{
		"total_documents": len(erag.vectorStore.(*InMemoryVectorStore).documents),
		"vector_store_type": "in_memory",
		"chunk_size": erag.chunkSize,
		"chunk_overlap": erag.chunkOverlap,
		"max_retrieved_docs": erag.maxRetrievedDocs,
	}
	
	return stats, nil
}

// ClearHistory clears the conversation history
func (erag *EnhancedRAGChat) ClearHistory() {
	erag.history = make([]ollama.Message, 0)
}

// ClearVectorStore clears all indexed documents
func (erag *EnhancedRAGChat) ClearVectorStore(ctx context.Context) error {
	return erag.vectorStore.Clear(ctx)
}

// Close closes the RAG chat and cleans up resources
func (erag *EnhancedRAGChat) Close() error {
	return erag.vectorStore.Close()
}

// RAGMode represents different RAG operation modes
type RAGMode string

const (
	RAGModeBasic    RAGMode = "basic"    // Uses existing file packing approach
	RAGModeVector   RAGMode = "vector"   // Uses vector-based semantic search
	RAGModeHybrid   RAGMode = "hybrid"   // Combines both approaches
)

// SmartRAGSystem provides a unified interface for different RAG modes
type SmartRAGSystem struct {
	EnhancedRAG *EnhancedRAGChat
	BasicRAG    *RAGChat
	Mode        RAGMode
}

// NewSmartRAGSystem creates a new smart RAG system
func NewSmartRAGSystem(api *ollama.Client, mode RAGMode, config *EnhancedRAGConfig) *SmartRAGSystem {
	if config == nil {
		config = DefaultEnhancedRAGConfig()
	}

	system := &SmartRAGSystem{
		Mode: mode,
	}

	if mode == RAGModeVector || mode == RAGModeHybrid {
		system.EnhancedRAG = NewEnhancedRAGChat(api, config)
	}

	// Note: BasicRAG will be initialized later with context in the CLI
	return system
}

// SetBasicRAGContext sets up the basic RAG with context
func (srs *SmartRAGSystem) SetBasicRAGContext(api *ollama.Client, context, model string) {
	if srs.Mode == RAGModeBasic || srs.Mode == RAGModeHybrid {
		srs.BasicRAG = NewRAGChat(api, context, model)
	}
}

// ProcessContext processes context based on the selected mode
func (srs *SmartRAGSystem) ProcessContext(ctx context.Context, contextPath string, includePatterns, excludePatterns []string) (string, error) {
	switch srs.Mode {
	case RAGModeVector:
		err := srs.EnhancedRAG.IndexDocuments(ctx, contextPath, includePatterns, excludePatterns)
		return "", err // Vector mode doesn't return a context string
		
	case RAGModeBasic:
		// Use existing packer logic (return context string for basic RAG)
		return "", fmt.Errorf("basic mode context processing should use existing packer")
		
	case RAGModeHybrid:
		// Index for vector search and return context string for basic RAG
		err := srs.EnhancedRAG.IndexDocuments(ctx, contextPath, includePatterns, excludePatterns)
		return "", err
		
	default:
		return "", fmt.Errorf("unknown RAG mode: %s", srs.Mode)
	}
}

// SendMessage sends a message using the appropriate RAG mode
func (srs *SmartRAGSystem) SendMessage(ctx context.Context, message string, basicContext string) (string, error) {
	switch srs.Mode {
	case RAGModeVector:
		if srs.EnhancedRAG == nil {
			return "", fmt.Errorf("enhanced RAG not initialized")
		}
		return srs.EnhancedRAG.SendWithVectorSearch(ctx, message)
		
	case RAGModeBasic:
		if srs.BasicRAG == nil {
			return "", fmt.Errorf("basic RAG not initialized")
		}
		_, err := srs.BasicRAG.Send(message, 8192)
		return "", err // Basic RAG prints directly
		
	case RAGModeHybrid:
		// Try vector search first, fall back to basic if no good results
		if srs.EnhancedRAG != nil {
			docs, err := srs.EnhancedRAG.vectorStore.Search(ctx, message, 3)
			if err == nil && len(docs) > 0 {
				// Use vector search if we have good results
				return srs.EnhancedRAG.SendWithVectorSearch(ctx, message)
			}
		}
		
		// Fall back to basic RAG
		if srs.BasicRAG != nil {
			_, err := srs.BasicRAG.Send(message, 8192)
			return "", err
		}
		
		return "", fmt.Errorf("no RAG systems available")
		
	default:
		return "", fmt.Errorf("unknown RAG mode: %s", srs.Mode)
	}
}

// Close closes the smart RAG system
func (srs *SmartRAGSystem) Close() error {
	if srs.EnhancedRAG != nil {
		if err := srs.EnhancedRAG.Close(); err != nil {
			return err
		}
	}
	return nil
}