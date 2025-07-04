package ask

import (
	"context"
	"fmt"
	"os/user"

	"github.com/spf13/viper"
	"github.com/urfave/cli/v2"
	"github.com/yusufcanb/tlm/pkg/config"
	"github.com/yusufcanb/tlm/pkg/packer"
	"github.com/yusufcanb/tlm/pkg/rag"
)

var usageText string = `tlm ask "<prompt>" # ask a question
tlm ask --context . --include *.md "<prompt>" # ask a question with a context
tlm ask --rag-mode vector --context . "<prompt>" # use enhanced vector-based RAG
tlm ask --rag-mode hybrid --context . --include *.go "<prompt>" # use hybrid RAG mode`

func (a *Ask) beforeAction(c *cli.Context) error {
	prompt := c.Args().First()
	if prompt == "" {
		cli.ShowSubcommandHelp(c)
		return cli.Exit("", -1)
	}

	overrideModel := c.String("model")
	if overrideModel != "" {
		a.model = overrideModel
	}

	user, err := user.Current()
	if err != nil {
		a.user = "User"
	} else {
		a.user = user.Username
	}

	return nil
}

func (a *Ask) action(c *cli.Context) error {
	ctx := context.Background()
	isInteractive := c.Bool("interactive")
	contextDir := c.Path("context")
	ragMode := c.String("rag-mode")
	chunkSize := c.Int("chunk-size")
	maxRetrievedDocs := c.Int("max-docs")

	// Validate RAG mode
	var mode rag.RAGMode
	switch ragMode {
	case "basic":
		mode = rag.RAGModeBasic
	case "vector":
		mode = rag.RAGModeVector
	case "hybrid":
		mode = rag.RAGModeHybrid
	default:
		mode = rag.RAGModeBasic // Default to basic for backward compatibility
	}

	var chatContext string    // chat context for basic RAG
	var numCtx int = 1024 * 8 // num_ctx in Ollama API

	fmt.Printf("🤖 %s (RAG Mode: %s)\n───────────────────\n", a.model, ragMode)

	// Configure enhanced RAG if using vector or hybrid mode
	var smartRAG *rag.SmartRAGSystem
	if mode == rag.RAGModeVector || mode == rag.RAGModeHybrid {
		ragConfig := config.LoadRAGConfig()
		enhancedConfig := ragConfig.ToEnhancedRAGConfig(a.model)
		
		// Override with command line flags if provided
		if chunkSize > 0 {
			enhancedConfig.ChunkSize = chunkSize
		}
		if maxRetrievedDocs > 0 {
			enhancedConfig.MaxRetrievedDocs = maxRetrievedDocs
		}
		
		smartRAG = rag.NewSmartRAGSystem(a.api, mode, enhancedConfig)
		defer smartRAG.Close()
	}

	// Process context if provided
	if contextDir != "" {
		includePatterns := c.StringSlice("include")
		excludePatterns := c.StringSlice("exclude")

		if mode == rag.RAGModeVector || mode == rag.RAGModeHybrid {
			// Use enhanced RAG indexing
			fmt.Println("🚀 Using Enhanced RAG with Vector Search")
			_, err := smartRAG.ProcessContext(ctx, contextDir, includePatterns, excludePatterns)
			if err != nil {
				return fmt.Errorf("failed to process context with enhanced RAG: %w", err)
			}
		}

		if mode == rag.RAGModeBasic || mode == rag.RAGModeHybrid {
			// Use existing packer for basic RAG (needed for hybrid mode fallback)
			fmt.Println("📁 Processing context with traditional file packing...")
			packer := packer.New()
			res, err := packer.Pack(contextDir, includePatterns, excludePatterns)
			if err != nil {
				return err
			}

			// Sort the files by the number of tokens
			packer.PrintTopFiles(res, 5)

			// Print the context summary
			packer.PrintContextSummary(res)

			// Render the packer result
			chatContext, err = packer.Render(res)
			if err != nil {
				return err
			}

			// Set up basic RAG with context if using smart RAG system
			if smartRAG != nil {
				smartRAG.SetBasicRAGContext(a.api, chatContext, a.model)
			}
		}
	}

	// Send the first message
	prompt := c.Args().First()
	
	if mode == rag.RAGModeVector || (mode == rag.RAGModeHybrid && smartRAG != nil) {
		// Use enhanced RAG
		_, err := smartRAG.SendMessage(ctx, prompt, chatContext)
		if err != nil {
			return fmt.Errorf("failed to send message with enhanced RAG: %w", err)
		}
	} else {
		// Use basic RAG
		ragChat := rag.NewRAGChat(a.api, chatContext, a.model)
		_, err := ragChat.Send(prompt, numCtx)
		if err != nil {
			return err
		}
	}

	// Interactive mode
	if isInteractive {
		fmt.Printf("\n\n💬 Interactive Mode Enabled (type 'exit' to quit, 'clear' to clear history)\n")
		
		var ragChat *rag.RAGChat
		if mode == rag.RAGModeBasic {
			ragChat = rag.NewRAGChat(a.api, chatContext, a.model)
		}

		for {
			fmt.Printf("\n\n👤 %s\n───────────────────\n", a.user)
			var input string
			fmt.Print("> ")
			fmt.Scanln(&input)

			if input == "exit" {
				break
			}
			
			if input == "clear" {
				if mode == rag.RAGModeVector || mode == rag.RAGModeHybrid {
					if smartRAG != nil && smartRAG.EnhancedRAG != nil {
						smartRAG.EnhancedRAG.ClearHistory()
					}
				} else if ragChat != nil {
					ragChat = rag.NewRAGChat(a.api, chatContext, a.model)
				}
				fmt.Println("🧹 Conversation history cleared!")
				continue
			}

			if input == "" {
				continue
			}

			// Send follow-up message
			if mode == rag.RAGModeVector || (mode == rag.RAGModeHybrid && smartRAG != nil) {
				_, err := smartRAG.SendMessage(ctx, input, chatContext)
				if err != nil {
					fmt.Printf("❌ Error: %v\n", err)
				}
			} else if ragChat != nil {
				_, err := ragChat.Send(input, numCtx)
				if err != nil {
					fmt.Printf("❌ Error: %v\n", err)
				}
			}
		}
	}

	return nil
}

func (a *Ask) afterAction(c *cli.Context) error {
	return nil
}

func (a *Ask) Command() *cli.Command {
	model := viper.GetString("llm.model")
	ragConfig := config.LoadRAGConfig()

	return &cli.Command{
		Name:      "ask",
		Usage:     "Asks a question with enhanced RAG support",
		UsageText: usageText,
		Aliases:   []string{"a"},
		Action:    a.action,
		Before:    a.beforeAction,
		After:     a.afterAction,
		Flags: []cli.Flag{
			&cli.PathFlag{
				Name:    "context",
				Aliases: []string{"c"},
				Usage:   "context directory path",
			},
			&cli.StringSliceFlag{
				Name:    "include",
				Aliases: []string{"i"},
				Usage:   "include patterns. e.g. --include=*.txt or --include=*.txt,*.md",
			},
			&cli.StringSliceFlag{
				Name:    "exclude",
				Aliases: []string{"e"},
				Usage:   "exclude patterns. e.g. --exclude=**/*_test.go or --exclude=*.pyc,*.pyd",
			},
			&cli.BoolFlag{
				Name:    "interactive",
				Aliases: []string{"it"},
				Usage:   "enable interactive chat mode",
			},
			&cli.StringFlag{
				Name:        "model",
				Aliases:     []string{"m"},
				Usage:       "override the model for command suggestion.",
				DefaultText: model,
			},
			&cli.StringFlag{
				Name:    "rag-mode",
				Aliases: []string{"r"},
				Usage:   "RAG mode: basic, vector, or hybrid",
				Value:   ragConfig.DefaultMode,
			},
			&cli.IntFlag{
				Name:    "chunk-size",
				Usage:   "chunk size for document splitting (vector mode)",
				Value:   ragConfig.ChunkSize,
			},
			&cli.IntFlag{
				Name:    "max-docs",
				Usage:   "maximum number of documents to retrieve (vector mode)",
				Value:   ragConfig.MaxRetrievedDocs,
			},
			&cli.IntFlag{
				Name:    "chunk-overlap",
				Usage:   "chunk overlap for document splitting (vector mode)",
				Value:   ragConfig.ChunkOverlap,
			},
		},
	}
}
