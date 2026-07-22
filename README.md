# rag

チャンク分割・埋め込み・検索・LLM 応答生成までを一通り実装した、Go 製の RAG(Retrieval-Augmented Generation)ライブラリ兼サンプル実装。`agent-action` / `agent-talk` から `pkg/llm` が共通クライアントとして利用されている。

## Features

- `pkg/content`: テキストクリーニング、チャンク分割、コサイン類似度、MMR(Maximal Marginal Relevance)によるトップK検索
- `pkg/store`: ドキュメントストアの抽象化(`Store` インターフェース)。JSON ファイル実装(`JSONStore`)と PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) 実装(`PostgresStore`)の2種類
- `pkg/llm`: OpenAI 互換 API を使った埋め込み生成・チャット補完クライアント
- `RecencySearch` で類似度に加えて新しさ(recency)を考慮した検索が可能
- `config.json` でチャンクサイズ・オーバーラップ・TopK・閾値・MMR パラメータ・ストア種別を設定

## Usage

```bash
cp .env.example .env   # OPENAI_API_KEY, OPENAI_API_BASE_URL, (Postgres利用時)POSTGRES_* を設定
go build .

./rag                   # sample.txt を取り込み、サンプルクエリで検索・応答を実行
```

ライブラリとして利用する場合は `go get github.com/tik-choco-lab/rag` し、`pkg/llm` / `pkg/store` / `pkg/content` を import する。

## Requirements

- Go 1.23 以上
- OpenAI 互換 API(embedding / chat)
- `store_type: "postgres"` を使う場合は PostgreSQL + pgvector 拡張
