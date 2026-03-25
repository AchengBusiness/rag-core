# @career-hub/rag-core

![version](https://img.shields.io/badge/version-1.0.0-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![no dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

Career-Hub 生态系统的**共享 RAG 知识库模块**。
Shared **RAG Knowledge Base Module** for the Career-Hub ecosystem.

轻量级、自包含、内存 TF-IDF 检索增强生成（RAG）引擎，无外部依赖，支持中英文。  
Lightweight, self-contained, in-memory TF-IDF RAG engine — no external dependencies, supports Chinese and English.

## 生态系统 | Ecosystem

本模块被以下仓库引用 | Referenced by these repos:

| 仓库 | 用途 |
|------|------|
| [media-ops](https://github.com/AchengBusiness/media-ops) | 内容运营 RAG |
| [ops-flow](https://github.com/AchengBusiness/ops-flow) | 运营工作流 RAG |
| [sales-crm](https://github.com/AchengBusiness/sales-crm) | 销售知识库 RAG |
| [pm-toolkit](https://github.com/AchengBusiness/pm-toolkit) | 项目管理 RAG |

## 安装 | Installation

```bash
npm install github:AchengBusiness/rag-core
```

或在 `package.json` 中指定 | Or in `package.json`:

```json
{
  "dependencies": {
    "@career-hub/rag-core": "github:AchengBusiness/rag-core"
  }
}
```

## 快速开始 | Quick Start

```js
import RAGKnowledgeBase from '@career-hub/rag-core';

const kb = new RAGKnowledgeBase({ chunkSize: 300, topK: 3 });

// 添加文档 | Add documents
kb.addDocument('doc1', '产品路线图是规划产品发展方向的工具', { source: 'pm-guide' });
kb.addDocuments([
  { id: 'doc2', content: 'Follow-up best practices for sales teams...', metadata: { category: 'sales' } },
  { id: 'doc3', content: '运营数据分析方法与指标体系建设', metadata: { category: 'ops' } },
]);

// 查询 | Query
const results = kb.query('路线图规划', { topK: 2 });
results.forEach(r => console.log(r.text, r.score));

// 统计 | Stats
console.log(kb.getStats());
// { documentCount: 3, chunkCount: 3, vocabularySize: 42, isIndexed: true }

// 持久化 | Persist
const snapshot = kb.exportKnowledge();
const kb2 = new RAGKnowledgeBase();
kb2.importKnowledge(snapshot);
```

## API 参考 | API Reference

### `new RAGKnowledgeBase(config?)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config.chunkStrategy` | `'paragraph'│'fixed'│'heading'` | `'paragraph'` | 分块策略 |
| `config.chunkSize` | `number` | `500` | 每块最大字符数 |
| `config.chunkOverlap` | `number` | `50` | 块间重叠字符数（fixed 策略） |
| `config.topK` | `number` | `5` | 默认返回结果数 |

### `addDocument(id, content, metadata?)`

添加单个文档。返回产生的块数量。  
Add a single document. Returns the number of chunks produced.

### `addDocuments(docs)`

批量添加文档数组 `[{ id, content, metadata? }]`。  
Batch add an array of documents.

### `removeDocument(id)`

删除文档及其所有块。返回 `boolean`。  
Remove a document and all its chunks. Returns `boolean`.

### `query(query, options?)`

查询知识库，返回最相关的块列表。  
Query the knowledge base; returns sorted relevant chunks.

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `options.topK` | `number` | config.topK | 返回结果数 |
| `options.rerank` | `boolean` | `false` | 启用关键词重排序 |
| `options.minScore` | `number` | `0` | 最低相似度过滤 |

返回 | Returns:
```ts
Array<{
  documentId: string,
  chunkId: string,
  chunkIndex: number,
  text: string,
  metadata: Object,
  score: number,       // TF-IDF 余弦相似度 [0, 1]
}>
```

### `getStats()`

返回统计信息 `{ documentCount, chunkCount, vocabularySize, isIndexed, config }`。

### `getDocument(id)`

获取指定文档元数据，不存在时返回 `null`。

### `listDocuments()`

返回所有文档 ID 的数组。

### `clear()`

清空知识库所有数据。

### `exportKnowledge()`

导出为可 JSON 序列化的快照对象（用于持久化）。

### `importKnowledge(data)`

从 `exportKnowledge()` 导出的数据恢复知识库。返回 `this`（链式调用）。

## 子模块导出 | Sub-module Exports

```js
import {
  RAGKnowledgeBase,
  ChunkStrategy,
  chunkText,
  chunkByParagraph,
  chunkByHeading,
  TFIDFVectorizer,
  tokenize,
  computeTF,
  Retriever,
} from '@career-hub/rag-core';
```

## 架构 | Architecture

```
src/
├── index.js       # RAGKnowledgeBase 主类 + 统一导出
├── chunker.js     # 文本分块器（fixed / paragraph / heading）
├── vectorizer.js  # TF-IDF 向量化器（支持中英文）
└── retriever.js   # 检索引擎（TF-IDF + 余弦相似度 + 重排序）
```

## License

MIT © AchengBusiness
