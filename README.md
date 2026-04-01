# PHPVector adapter for Neuron AI framework

This is a [PHPVector](https://github.com/ezimuel/PHPVector) adapter for the [Neuron AI framework](https://neuron-ai.dev/).

## Install

```
composer require neuron-core/php-vector
```

## Use in RAG

```php
use NeuronAI\PHPVector\PHPVector;

class MyRAG extends RAG
{
    ...

    protected function vectorStore(): VectorStoreInterface
    {
        return new PHPVector(
            database: new VectorDatabase(path: '/var/data/mydb'),
            topK: 5
        );
    }
}
```

## Use in Retrieval components

```php
use NeuronAI\PHPVector\PHPVector;

class MyAgent extends Agent
{
    ...

    protected function tools(): array
    {
        return [
            RetrievalTool::make(
                new SimilarityRetrieval(
                    $this->vectorStore(),
                    $this->embeddings()
                )
            ),
        ];
    }

    protected function vectorStore(): VectorStoreInterface
    {
        return new PHPVector(
            database: new VectorDatabase(path: '/var/data/mydb'),
            topK: 5
        );
    }

    protected function embeddings(): EmbeddingsProviderInterface
    {
        return new OllamaEmbeddingsProvider(
            model: 'OLLAMA_EMBEDDINGS_MODEL'
        );
    }
}
```

## Official documentation

**[Go to the official documentation](https://neuron.inspector.dev/)**
