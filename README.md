# Neuron AI Adapter for php-vector

PHPVector adapter for the [Neuron AI](https://neuron-ai.dev) framework. Implements
`NeuronAI\RAG\VectorStore\VectorStoreInterface` on top of `ezimuel/phpvector`.

## Dependencies

- PHP ^8.2
- [ezimuel/phpvector](https://github.com/ezimuel/PHPVector) ^0.3.0
- [Neuron AI](https://github.com/neuron-core/neuron-ai) ^3.0

## Installation

```bash
composer require neuron-core/php-vector
```

## Usage

Inside a Neuron RAG class:

```php
class MyRAG extends RAG
{
    ...

    protected function vectorStore(): VectorStoreInterface
    {
        return new PHPVector(
            path: '/var/data/mydb',
            topK: 5,
        );
    }
}
```

Use it as a standalone component:

```php
use NeuronAI\PHPVector\PHPVector;
use PHPVector\VectorDatabase;

// Persistent database: pass a path to enable on-disk storage.
$store = new PHPVector(
    path: '/var/data/mydb',
    topK: 5,
);
```

## Persistence

By default, this adapter auto-saves after every mutation (`addDocument`, `addDocuments`,
`deleteBy`), batched to a single `save()` per call, so persistence "just works".

## Deletion

`deleteBy()` removes documents by Neuron's `sourceType` / `sourceName`, which this adapter
stores as PHPVector metadata:

```php
$store->deleteBy('pdf');               // all documents from sourceType "pdf"
$store->deleteBy('pdf', 'manual.pdf'); // only that exact source
```
