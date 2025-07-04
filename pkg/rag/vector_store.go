package rag

import (
	"context"
	"crypto/sha256"
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	ollama "github.com/jmorganca/ollama/api"
	"gonum.org/v1/gonum/floats"
)

// Document represents a document chunk in the vector store
type Document struct {
	ID          string    `json:"id"`
	Content     string    `json:"content"`
	Metadata    Metadata  `json:"metadata"`
	Embedding   []float64 `json:"embedding"`
	CreatedAt   time.Time `json:"created_at"`
}

// Metadata contains document metadata
type Metadata struct {
	Source     string            `json:"source"`
	ChunkIndex int               `json:"chunk_index"`
	ChunkSize  int               `json:"chunk_size"`
	Language   string            `json:"language"`
	FileType   string            `json:"file_type"`
	Tags       []string          `json:"tags"`
	Extra      map[string]string `json:"extra"`
}

// VectorStore interface defines the operations for vector storage and retrieval
type VectorStore interface {
	AddDocument(ctx context.Context, doc *Document) error
	AddDocuments(ctx context.Context, docs []*Document) error
	Search(ctx context.Context, query string, limit int) ([]*Document, error)
	SimilaritySearch(ctx context.Context, embedding []float64, limit int) ([]*Document, error)
	GetDocument(ctx context.Context, id string) (*Document, error)
	DeleteDocument(ctx context.Context, id string) error
	Clear(ctx context.Context) error
	Close() error
}

// InMemoryVectorStore implements VectorStore in memory
type InMemoryVectorStore struct {
	documents map[string]*Document
	ollama    *ollama.Client
}

// NewInMemoryVectorStore creates a new in-memory vector store
func NewInMemoryVectorStore(ollama *ollama.Client) *InMemoryVectorStore {
	return &InMemoryVectorStore{
		documents: make(map[string]*Document),
		ollama:    ollama,
	}
}

// generateEmbedding generates embeddings using Ollama
func (vs *InMemoryVectorStore) generateEmbedding(ctx context.Context, text string) ([]float64, error) {
	req := &ollama.EmbeddingRequest{
		Model:  "nomic-embed-text", // Default embedding model
		Prompt: text,
	}

	resp, err := vs.ollama.Embeddings(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	return resp.Embedding, nil
}

func (vs *InMemoryVectorStore) AddDocument(ctx context.Context, doc *Document) error {
	if doc.ID == "" {
		doc.ID = generateDocumentID(doc.Content, doc.Metadata.Source)
	}

	if len(doc.Embedding) == 0 {
		embedding, err := vs.generateEmbedding(ctx, doc.Content)
		if err != nil {
			return fmt.Errorf("failed to generate embedding: %w", err)
		}
		doc.Embedding = embedding
	}

	if doc.CreatedAt.IsZero() {
		doc.CreatedAt = time.Now()
	}

	vs.documents[doc.ID] = doc
	return nil
}

func (vs *InMemoryVectorStore) AddDocuments(ctx context.Context, docs []*Document) error {
	for _, doc := range docs {
		if err := vs.AddDocument(ctx, doc); err != nil {
			return err
		}
	}
	return nil
}

func (vs *InMemoryVectorStore) Search(ctx context.Context, query string, limit int) ([]*Document, error) {
	queryEmbedding, err := vs.generateEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	return vs.SimilaritySearch(ctx, queryEmbedding, limit)
}

func (vs *InMemoryVectorStore) SimilaritySearch(ctx context.Context, embedding []float64, limit int) ([]*Document, error) {
	type docWithScore struct {
		doc   *Document
		score float64
	}

	var candidates []docWithScore

	for _, doc := range vs.documents {
		if len(doc.Embedding) == 0 {
			continue
		}

		score := cosineSimilarity(embedding, doc.Embedding)
		candidates = append(candidates, docWithScore{doc: doc, score: score})
	}

	// Sort by similarity score (highest first)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	// Return top results
	var results []*Document
	for i, candidate := range candidates {
		if i >= limit {
			break
		}
		results = append(results, candidate.doc)
	}

	return results, nil
}

func (vs *InMemoryVectorStore) GetDocument(ctx context.Context, id string) (*Document, error) {
	doc, exists := vs.documents[id]
	if !exists {
		return nil, fmt.Errorf("document not found: %s", id)
	}
	return doc, nil
}

func (vs *InMemoryVectorStore) DeleteDocument(ctx context.Context, id string) error {
	delete(vs.documents, id)
	return nil
}

func (vs *InMemoryVectorStore) Clear(ctx context.Context) error {
	vs.documents = make(map[string]*Document)
	return nil
}

func (vs *InMemoryVectorStore) Close() error {
	return nil
}

// Utility functions

func generateDocumentID(content, source string) string {
	hash := sha256.Sum256([]byte(content + source))
	return fmt.Sprintf("%x", hash[:8]) // Use first 8 bytes for shorter ID
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	dotProduct := floats.Dot(a, b)
	normA := floats.Norm(a, 2)
	normB := floats.Norm(b, 2)

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (normA * normB)
}

// DocumentChunker handles intelligent document chunking
type DocumentChunker struct {
	ChunkSize    int
	ChunkOverlap int
}

// NewDocumentChunker creates a new document chunker
func NewDocumentChunker(chunkSize, chunkOverlap int) *DocumentChunker {
	return &DocumentChunker{
		ChunkSize:    chunkSize,
		ChunkOverlap: chunkOverlap,
	}
}

// ChunkDocument splits a document into chunks
func (dc *DocumentChunker) ChunkDocument(content, source string) []*Document {
	var chunks []*Document

	// Detect file type and language
	fileType := detectFileType(source)
	language := detectLanguage(source, content)

	// Use different chunking strategies based on file type
	var textChunks []string
	switch fileType {
	case "code":
		textChunks = dc.chunkCode(content, language)
	case "markdown":
		textChunks = dc.chunkMarkdown(content)
	default:
		textChunks = dc.chunkText(content)
	}

	// Create document chunks
	for i, chunk := range textChunks {
		if strings.TrimSpace(chunk) == "" {
			continue
		}

		doc := &Document{
			Content: chunk,
			Metadata: Metadata{
				Source:     source,
				ChunkIndex: i,
				ChunkSize:  len(chunk),
				Language:   language,
				FileType:   fileType,
				Tags:       extractTags(chunk, fileType),
				Extra:      make(map[string]string),
			},
		}
		chunks = append(chunks, doc)
	}

	return chunks
}

func (dc *DocumentChunker) chunkText(content string) []string {
	// Split by sentences first, then group into chunks
	sentences := splitIntoSentences(content)
	return dc.groupSentencesIntoChunks(sentences)
}

func (dc *DocumentChunker) chunkCode(content, language string) []string {
	// For code, try to keep functions/classes together
	switch language {
	case "go":
		return dc.chunkGoCode(content)
	case "python":
		return dc.chunkPythonCode(content)
	case "javascript", "typescript":
		return dc.chunkJSCode(content)
	default:
		return dc.chunkText(content)
	}
}

func (dc *DocumentChunker) chunkMarkdown(content string) []string {
	// Split by headers while preserving structure
	lines := strings.Split(content, "\n")
	var chunks []string
	var currentChunk strings.Builder
	var currentSize int

	for _, line := range lines {
		// Check if this is a header
		if strings.HasPrefix(line, "#") {
			// Start new chunk if current one is not empty and size limit reached
			if currentSize > 0 && currentSize+len(line) > dc.ChunkSize {
				chunks = append(chunks, currentChunk.String())
				currentChunk.Reset()
				currentSize = 0
			}
		}

		currentChunk.WriteString(line + "\n")
		currentSize += len(line) + 1

		if currentSize >= dc.ChunkSize {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
			currentSize = 0
		}
	}

	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}

	return chunks
}

func (dc *DocumentChunker) chunkGoCode(content string) []string {
	// Simple function-based chunking for Go
	lines := strings.Split(content, "\n")
	var chunks []string
	var currentChunk strings.Builder
	var braceCount int
	var inFunction bool

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		
		// Check for function start
		if strings.HasPrefix(trimmed, "func ") {
			if currentChunk.Len() > 0 {
				chunks = append(chunks, currentChunk.String())
				currentChunk.Reset()
			}
			inFunction = true
			braceCount = 0
		}

		currentChunk.WriteString(line + "\n")

		// Count braces to track function end
		if inFunction {
			braceCount += strings.Count(line, "{") - strings.Count(line, "}")
			if braceCount <= 0 && strings.Contains(line, "}") {
				inFunction = false
				chunks = append(chunks, currentChunk.String())
				currentChunk.Reset()
			}
		}

		// Fallback to size-based chunking
		if currentChunk.Len() > dc.ChunkSize && !inFunction {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
		}
	}

	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}

	return chunks
}

func (dc *DocumentChunker) chunkPythonCode(content string) []string {
	// Function and class-based chunking for Python
	lines := strings.Split(content, "\n")
	var chunks []string
	var currentChunk strings.Builder
	var indentLevel int = -1

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		currentIndent := len(line) - len(strings.TrimLeft(line, " \t"))

		// Check for function or class definition
		if strings.HasPrefix(trimmed, "def ") || strings.HasPrefix(trimmed, "class ") {
			if currentChunk.Len() > 0 && indentLevel >= 0 && currentIndent <= indentLevel {
				chunks = append(chunks, currentChunk.String())
				currentChunk.Reset()
			}
			indentLevel = currentIndent
		}

		currentChunk.WriteString(line + "\n")

		// Fallback to size-based chunking
		if currentChunk.Len() > dc.ChunkSize {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
			indentLevel = -1
		}
	}

	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}

	return chunks
}

func (dc *DocumentChunker) chunkJSCode(content string) []string {
	// Function-based chunking for JavaScript/TypeScript
	lines := strings.Split(content, "\n")
	var chunks []string
	var currentChunk strings.Builder
	var braceCount int
	var inFunction bool

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		
		// Check for function start
		if strings.Contains(trimmed, "function ") || 
		   regexp.MustCompile(`\w+\s*\(`).MatchString(trimmed) ||
		   strings.Contains(trimmed, "=>") {
			if currentChunk.Len() > 0 && !inFunction {
				chunks = append(chunks, currentChunk.String())
				currentChunk.Reset()
			}
			inFunction = true
			braceCount = 0
		}

		currentChunk.WriteString(line + "\n")

		// Count braces to track function end
		if inFunction {
			braceCount += strings.Count(line, "{") - strings.Count(line, "}")
			if braceCount <= 0 && strings.Contains(line, "}") {
				inFunction = false
				chunks = append(chunks, currentChunk.String())
				currentChunk.Reset()
			}
		}

		// Fallback to size-based chunking
		if currentChunk.Len() > dc.ChunkSize && !inFunction {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
		}
	}

	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}

	return chunks
}

func (dc *DocumentChunker) groupSentencesIntoChunks(sentences []string) []string {
	var chunks []string
	var currentChunk strings.Builder
	var currentSize int

	for _, sentence := range sentences {
		sentenceLen := len(sentence)
		
		if currentSize+sentenceLen > dc.ChunkSize && currentChunk.Len() > 0 {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
			currentSize = 0
			
			// Add overlap
			if len(chunks) > 0 && dc.ChunkOverlap > 0 {
				overlapText := chunks[len(chunks)-1]
				if len(overlapText) > dc.ChunkOverlap {
					overlapText = overlapText[len(overlapText)-dc.ChunkOverlap:]
				}
				currentChunk.WriteString(overlapText)
				currentSize = len(overlapText)
			}
		}
		
		currentChunk.WriteString(sentence + " ")
		currentSize += sentenceLen + 1
	}

	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}

	return chunks
}

// Utility functions for document processing

func detectFileType(filename string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	
	codeExtensions := map[string]bool{
		".go": true, ".py": true, ".js": true, ".ts": true, ".java": true,
		".c": true, ".cpp": true, ".h": true, ".hpp": true, ".rs": true,
		".php": true, ".rb": true, ".swift": true, ".kt": true, ".scala": true,
	}
	
	if codeExtensions[ext] {
		return "code"
	}
	
	if ext == ".md" || ext == ".markdown" {
		return "markdown"
	}
	
	if ext == ".txt" || ext == ".text" {
		return "text"
	}
	
	return "unknown"
}

func detectLanguage(filename, content string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	
	langMap := map[string]string{
		".go":   "go",
		".py":   "python",
		".js":   "javascript",
		".ts":   "typescript",
		".java": "java",
		".c":    "c",
		".cpp":  "cpp",
		".h":    "c",
		".hpp":  "cpp",
		".rs":   "rust",
		".php":  "php",
		".rb":   "ruby",
		".swift": "swift",
		".kt":   "kotlin",
		".scala": "scala",
		".md":   "markdown",
	}
	
	if lang, exists := langMap[ext]; exists {
		return lang
	}
	
	return "text"
}

func extractTags(content, fileType string) []string {
	var tags []string
	
	switch fileType {
	case "code":
		// Extract function names, class names, etc.
		if strings.Contains(content, "func ") {
			tags = append(tags, "function")
		}
		if strings.Contains(content, "class ") {
			tags = append(tags, "class")
		}
		if strings.Contains(content, "interface ") {
			tags = append(tags, "interface")
		}
		if strings.Contains(content, "struct ") {
			tags = append(tags, "struct")
		}
	case "markdown":
		// Extract header levels
		if strings.Contains(content, "# ") {
			tags = append(tags, "header")
		}
		if strings.Contains(content, "```") {
			tags = append(tags, "code-block")
		}
	}
	
	return tags
}

func splitIntoSentences(text string) []string {
	// Simple sentence splitting by periods, exclamation marks, and question marks
	re := regexp.MustCompile(`[.!?]+\s+`)
	sentences := re.Split(text, -1)
	
	var result []string
	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if sentence != "" {
			result = append(result, sentence)
		}
	}
	
	return result
}