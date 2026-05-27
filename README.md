# neuron-core/php-vector

PHPVector adapter for the [Neuron AI](https://neuron-ai.dev) framework. Implements
`NeuronAI\RAG\VectorStore\VectorStoreInterface` on top of `ezimuel/phpvector`.

## Installation

```bash
composer require neuron-core/php-vector
```

## Usage

```php
use NeuronAI\PHPVector\PHPVector;
use PHPVector\VectorDatabase;

// Persistent database: pass a path to enable on-disk storage.
$store = new PHPVector(
    database: new VectorDatabase(path: '/var/data/mydb'),
    topK: 5,
);
```

Inside a Neuron RAG class:

```php
protected function vectorStore(): VectorStoreInterface
{
    return new PHPVector(
        database: new VectorDatabase(path: '/var/data/mydb'),
        topK: 5,
    );
}
```

## Persistence

PHPVector separates document storage from index storage:

- `new VectorDatabase(path: '...')` creates (or targets) a database directory.
- `VectorDatabase::open('...')` loads an existing database from disk.
- `addDocument()` writes the document file to disk on each call (asynchronously via `pcntl_fork` when available, otherwise synchronously).
- `save()` persists the HNSW + BM25 index and finalizes deletions.

By default this adapter auto-saves after every mutation (`addDocument`, `addDocuments`,
`deleteBy`), batched to a single `save()` per call, so persistence "just works". Disable it
to manage `save()` yourself:

```php
$store = new PHPVector(database: $db, autoSave: false);
// ... many addDocument() calls ...
$db->save();
```

Auto-save is skipped for in-memory databases (no path), so it never throws.

## Deletion

`deleteBy()` removes documents by Neuron's `sourceType` / `sourceName`, which this adapter
stores as PHPVector metadata:

```php
$store->deleteBy('pdf');               // all documents from sourceType "pdf"
$store->deleteBy('pdf', 'manual.pdf'); // only that exact source
```

## Requirements

- PHP 8.1+
- `ezimuel/phpvector` ^0.3.0
- `neuron-core/neuron-ai` ^3.0
