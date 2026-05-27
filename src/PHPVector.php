<?php

declare(strict_types=1);

namespace NeuronAI\PHPVector;

use NeuronAI\Exceptions\VectorStoreException;
use NeuronAI\RAG\Document as NeuronDocument;
use NeuronAI\RAG\VectorStore\VectorStoreInterface;
use NeuronAI\StaticConstructor;
use PHPVector\Document;
use PHPVector\Metadata\MetadataFilter;
use PHPVector\SearchResult;
use PHPVector\VectorDatabase;

use function array_map;

class PHPVector implements VectorStoreInterface
{
    use StaticConstructor;

    private const SOURCE_TYPE_KEY = 'sourceType';
    private const SOURCE_NAME_KEY = 'sourceName';

    public function __construct(
        protected VectorDatabase $database,
        protected int $topK = 5,
    ) {
    }

    public function addDocument(NeuronDocument $document): VectorStoreInterface
    {
        $this->write($document);

        return $this;
    }

    /**
     * @param NeuronDocument[] $documents
     */
    public function addDocuments(array $documents): VectorStoreInterface
    {
        foreach ($documents as $document) {
            $this->write($document);
        }

        return $this;
    }

    /**
     * Persist a Neuron document into PHPVector.
     *
     * Neuron's `sourceType`/`sourceName` are top-level Document properties, but
     * PHPVector only stores `metadata`. They are folded into metadata under the
     * reserved keys so `deleteBy()` can filter on them; `similaritySearch()`
     * restores them and strips the reserved keys back out.
     */
    private function write(NeuronDocument $document): void
    {
        $this->database->addDocument(
            new Document(
                id: $document->id,
                vector: $document->embedding,
                text: $document->content,
                metadata: [
                    ...$document->metadata,
                    self::SOURCE_TYPE_KEY => $document->sourceType,
                    self::SOURCE_NAME_KEY => $document->sourceName,
                ],
            )
        );
    }

    /**
     * @throws VectorStoreException
     */
    public function deleteBy(string $sourceType, ?string $sourceName = null): VectorStoreInterface
    {
        throw new VectorStoreException('Deletion not supported.');
    }

    /**
     * @throws VectorStoreException
     */
    public function deleteBySource(string $sourceType, string $sourceName): VectorStoreInterface
    {
        $this->deleteBy($sourceType, $sourceName);
        return $this;
    }

    /**
     * @param array<float> $embedding
     * @return iterable<NeuronDocument>
     */
    public function similaritySearch(array $embedding): iterable
    {
        $results = $this->database->vectorSearch(
            vector: $embedding,
            k: $this->topK,
        );

        return array_map(function (SearchResult $result): NeuronDocument {
            $phpDoc = $result->document;

            $metadata = $phpDoc->metadata;
            $sourceType = $metadata[self::SOURCE_TYPE_KEY] ?? 'manual';
            $sourceName = $metadata[self::SOURCE_NAME_KEY] ?? 'manual';
            unset($metadata[self::SOURCE_TYPE_KEY], $metadata[self::SOURCE_NAME_KEY]);

            $document = new NeuronDocument($phpDoc->text);
            $document->id = $phpDoc->id;
            $document->embedding = $phpDoc->vector;
            $document->sourceType = $sourceType;
            $document->sourceName = $sourceName;
            $document->metadata = $metadata;
            $document->score = $result->score;

            return $document;
        }, $results);
    }
}
