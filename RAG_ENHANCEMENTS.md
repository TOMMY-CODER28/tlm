# Enhanced RAG Support for TLM

This document describes the enhanced Retrieval-Augmented Generation (RAG) capabilities added to tlm, providing vector-based semantic search, intelligent document chunking, and multiple RAG modes.

## Overview

The enhanced RAG system extends tlm's existing context-aware functionality with:

- **Vector-based semantic search** using embeddings
- **Intelligent document chunking** optimized for different file types
- **Multiple RAG modes** (basic, vector, hybrid)
- **Configurable parameters** for fine-tuning
- **Language-aware processing** for code and markdown files

## RAG Modes

### 1. Basic Mode (Default)
Uses the existing file packing approach - all matching files are included as context.
```bash
tlm ask --rag-mode basic --context . --include "*.md" "How does the build system work?"
```

### 2. Vector Mode
Uses vector embeddings for semantic search and retrieval.
```bash
tlm ask --rag-mode vector --context . --include "*.go" "How is the RAG system implemented?"
```

### 3. Hybrid Mode
Combines vector search with fallback to basic mode when no relevant vectors are found.
```bash
tlm ask --rag-mode hybrid --context . "Explain the overall architecture"
```

## Usage Examples

### Basic RAG with Context
```bash
# Traditional approach - includes all files matching patterns
tlm ask --context . --include "*.go" --exclude "*_test.go" "How does authentication work?"
```

### Vector-based RAG
```bash
# Semantic search approach - only includes relevant chunks
tlm ask --rag-mode vector --context . --chunk-size 800 --max-docs 3 "Show me error handling patterns"
```

### Interactive Mode with Enhanced RAG
```bash
# Start an interactive session with vector-based RAG
tlm ask --rag-mode vector --context . --interactive "Let's discuss the codebase"
```

### Custom Configuration
```bash
# Fine-tune chunking and retrieval parameters
tlm ask --rag-mode vector \
  --context . \
  --chunk-size 1200 \
  --chunk-overlap 300 \
  --max-docs 7 \
  "Analyze the performance bottlenecks"
```

## Configuration

Enhanced RAG settings can be configured in your `~/.tlm.yml` file:

```yaml
rag:
  default_mode: "vector"           # Default RAG mode: basic, vector, hybrid
  chunk_size: 1000                 # Size of document chunks (characters)
  chunk_overlap: 200               # Overlap between chunks (characters)
  max_retrieved_docs: 5            # Maximum documents to retrieve
  max_context_tokens: 8192         # Maximum context window size
  similarity_threshold: 0.3        # Minimum similarity score for retrieval
  embedding_model: "nomic-embed-text"  # Model used for embeddings
```

## Intelligent Chunking

The system uses different chunking strategies based on file type:

### Code Files (.go, .py, .js, etc.)
- **Function-based chunking**: Keeps functions and methods together
- **Class-based chunking**: Preserves class structures
- **Language-aware**: Understands syntax for better boundaries

### Markdown Files (.md)
- **Header-based chunking**: Splits at header boundaries
- **Structure preservation**: Maintains document hierarchy
- **Code block awareness**: Keeps code examples intact

### Text Files (.txt)
- **Sentence-based chunking**: Splits at natural sentence boundaries
- **Paragraph awareness**: Respects paragraph structure
- **Overlap handling**: Ensures context continuity

## Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--rag-mode` | RAG mode (basic/vector/hybrid) | `basic` |
| `--chunk-size` | Document chunk size in characters | `1000` |
| `--chunk-overlap` | Overlap between chunks | `200` |
| `--max-docs` | Maximum documents to retrieve | `5` |
| `--context` | Directory path for context | - |
| `--include` | File patterns to include | - |
| `--exclude` | File patterns to exclude | - |
| `--interactive` | Enable interactive chat mode | `false` |

## Interactive Commands

When in interactive mode, you can use these commands:

- `exit` - Exit the interactive session
- `clear` - Clear conversation history
- Regular queries - Continue the conversation

## Embedding Requirements

Vector mode requires an embedding model. The default is `nomic-embed-text`, which you can install with:

```bash
ollama pull nomic-embed-text
```

Other compatible embedding models:
- `mxbai-embed-large`
- `snowflake-arctic-embed`
- `bge-large-en`

## Performance Considerations

### Vector Mode
- **Faster retrieval** for large codebases
- **More relevant results** based on semantic similarity
- **Higher memory usage** due to embeddings
- **Initial indexing time** required

### Basic Mode
- **Faster startup** (no indexing required)
- **Lower memory usage**
- **May include irrelevant content** in large contexts
- **Token limit constraints** with many files

### Hybrid Mode
- **Best of both worlds** - semantic search with fallback
- **Slightly higher overhead** due to dual systems
- **Recommended for most use cases**

## Troubleshooting

### Error: "enhanced RAG not initialized"
Ensure you're using vector or hybrid mode when this error occurs.

### Error: "failed to generate embedding"
1. Verify the embedding model is installed: `ollama list`
2. Pull the model if needed: `ollama pull nomic-embed-text`
3. Check if Ollama service is running

### Poor Retrieval Results
1. Adjust `chunk_size` for your content type
2. Increase `max_docs` for more comprehensive results
3. Lower `similarity_threshold` in configuration
4. Try hybrid mode for better coverage

### Performance Issues
1. Reduce `chunk_size` for faster processing
2. Decrease `max_docs` for quicker responses
3. Use basic mode for small projects
4. Adjust `max_context_tokens` based on your model

## Advanced Usage

### Custom Embedding Models
Configure a different embedding model in your config:

```yaml
rag:
  embedding_model: "mxbai-embed-large"
```

### File Type Optimization
Use include/exclude patterns to optimize for specific file types:

```bash
# Focus on Go source files only
tlm ask --rag-mode vector --context . --include "*.go" --exclude "*_test.go,*_mock.go"

# Include documentation and code
tlm ask --rag-mode vector --context . --include "*.md,*.go,*.py"

# Exclude generated files
tlm ask --rag-mode hybrid --context . --exclude "vendor/**,node_modules/**,*.pb.go"
```

### Large Codebase Optimization
For very large codebases, consider:

1. **Smaller chunk sizes** (500-800 characters)
2. **Higher chunk overlap** (300-400 characters)
3. **Selective file inclusion** using patterns
4. **Vector mode** for better relevance filtering

## Future Enhancements

Planned improvements include:

- **Persistent vector storage** for faster reindexing
- **Multi-language embedding models**
- **Custom chunking strategies**
- **Web content integration**
- **Database query support**
- **Cross-reference awareness**

## Examples Repository

See the `examples/` directory for more usage examples and configuration templates.