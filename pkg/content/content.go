package content

import (
	"os"
)

func ReadTextFile(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func SplitText(text string, maxChunkSize int, overlap int) []string {
	runes := []rune(text)
	if len(runes) <= maxChunkSize {
		return []string{text}
	}

	var chunks []string
	for i := 0; i < len(runes); {
		end := i + maxChunkSize
		if end > len(runes) {
			end = len(runes)
		}

		chunks = append(chunks, string(runes[i:end]))

		if end == len(runes) {
			break
		}

		step := maxChunkSize - overlap
		if step <= 0 {
			step = 1
		}
		i += step
	}

	return chunks
}
