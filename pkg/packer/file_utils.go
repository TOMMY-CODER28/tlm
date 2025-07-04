package packer

import "github.com/yusufcanb/tlm/pkg/packer/internal"

// GetContextFilePaths is a public wrapper for internal.GetContextFilePaths
func GetContextFilePaths(path string, includePatterns []string, excludePatterns []string) ([]string, error) {
	return internal.GetContextFilePaths(path, includePatterns, excludePatterns)
}

// GetFileContent is a public wrapper for internal.GetFileContent
func GetFileContent(baseDir string, filePath string) (string, int, int, error) {
	return internal.GetFileContent(baseDir, filePath)
}