# PHPVector adapter for Neuron AI framework

This is a PHPVector adapter for the [Neuron AI framework](https://neuron-ai.dev/).

## Install

```
composer require neuron-core/php-vector
```

## Use in RAG or retrieval components

```php
use NeuronAI\PHPVector\PHPVector;

class MyRAG extends RAG
{
    ...

    protected function vectorStore(): VectorStoreInterface {
        return new PHPVector(
            database: new VectorDatabase(path: '/var/data/mydb'),
            topK: 5
        );
    }
}
```

## Official documentation

**[Go to the official documentation](https://neuron.inspector.dev/)**
