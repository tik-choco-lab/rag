package content

import (
	"math"
	"os"
	"slices"
)

const (
	minSimilarity = -1e9
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

func CosineSimilarity(v1, v2 []float32) float32 {
	if len(v1) != len(v2) || len(v1) == 0 {
		return 0
	}
	var dotProduct, norm1, norm2 float64
	for i := range v1 {
		dotProduct += float64(v1[i]) * float64(v2[i])
		norm1 += float64(v1[i]) * float64(v1[i])
		norm2 += float64(v2[i]) * float64(v2[i])
	}
	if norm1 == 0 || norm2 == 0 {
		return 0
	}
	return float32(dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2)))
}

type SearchResult struct {
	Text  string
	Score float32
}

func SearchTopK(queryEmbedding []float32, chunks []string, embeddings [][]float32, k int, threshold float32, mmrLambda float32) []SearchResult {
	var candidates []int
	for i, emb := range embeddings {
		score := CosineSimilarity(queryEmbedding, emb)
		if score >= threshold {
			candidates = append(candidates, i)
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	if mmrLambda >= 1.0 {
		var results []SearchResult
		for _, idx := range candidates {
			score := CosineSimilarity(queryEmbedding, embeddings[idx])
			results = append(results, SearchResult{Text: chunks[idx], Score: score})
		}

		slices.SortFunc(results, func(a, b SearchResult) int {
			if a.Score > b.Score {
				return -1
			}
			if a.Score < b.Score {
				return 1
			}
			return 0
		})

		if len(results) > k {
			return results[:k]
		}
		return results
	}

	selectedIndices := make([]int, 0, k)
	for len(selectedIndices) < k && len(selectedIndices) < len(candidates) {
		bestIdx := -1
		var maxMMR float32 = minSimilarity

		for _, candIdx := range candidates {
			if contains(selectedIndices, candIdx) {
				continue
			}

			simQuery := CosineSimilarity(queryEmbedding, embeddings[candIdx])
			var maxSimSelected float32 = minSimilarity
			if len(selectedIndices) == 0 {
				maxSimSelected = 0
			} else {
				for _, selIdx := range selectedIndices {
					simDoc := CosineSimilarity(embeddings[candIdx], embeddings[selIdx])
					if simDoc > maxSimSelected {
						maxSimSelected = simDoc
					}
				}
			}

			mmrScore := mmrLambda*simQuery - (1-mmrLambda)*maxSimSelected
			if mmrScore > maxMMR {
				maxMMR = mmrScore
				bestIdx = candIdx
			}
		}

		if bestIdx == -1 {
			break
		}
		selectedIndices = append(selectedIndices, bestIdx)
	}

	var finalResults []SearchResult
	for _, idx := range selectedIndices {
		finalResults = append(finalResults, SearchResult{
			Text:  chunks[idx],
			Score: CosineSimilarity(queryEmbedding, embeddings[idx]),
		})
	}
	return finalResults
}

func contains(slice []int, val int) bool {
	for _, item := range slice {
		if item == val {
			return true
		}
	}
	return false
}
