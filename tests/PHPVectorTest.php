<?php

declare(strict_types=1);

namespace NeuronAI\PHPVector\Tests;

use NeuronAI\PHPVector\PHPVector;
use NeuronAI\RAG\Document as NeuronDocument;
use PHPVector\VectorDatabase;
use PHPUnit\Framework\TestCase;

class PHPVectorTest extends TestCase
{
    private string $tempDir;

    protected function setUp(): void
    {
        parent::setUp();
        $this->tempDir = sys_get_temp_dir() . '/phpvector_test_' . uniqid();
    }

    protected function tearDown(): void
    {
        parent::tearDown();
        $this->removeDirectory($this->tempDir);
    }

    private function removeDirectory(string $dir): void
    {
        if (!is_dir($dir)) {
            return;
        }

        $files = array_diff(scandir($dir), ['.', '..']);
        foreach ($files as $file) {
            $path = $dir . '/' . $file;
            is_dir($path) ? $this->removeDirectory($path) : unlink($path);
        }
        rmdir($dir);
    }

    private function createTestEmbedding(int $dimensions = 128): array
    {
        $embedding = [];
        for ($i = 0; $i < $dimensions; $i++) {
            $embedding[] = mt_rand() / mt_getrandmax();
        }
        return $embedding;
    }

    public function testAddDocumentIncreasesCount(): void
    {
        $database = new VectorDatabase();
        $adapter = new PHPVector($database);

        $document = new NeuronDocument('Test content');
        $document->embedding = $this->createTestEmbedding();

        $this->assertEquals(0, $database->count());

        $adapter->addDocument($document);

        $this->assertEquals(1, $database->count());
    }

    public function testAddDocumentsAddsMultipleDocuments(): void
    {
        $database = new VectorDatabase();
        $adapter = new PHPVector($database);

        $documents = [
            $this->createDocumentWithEmbedding('Document 1'),
            $this->createDocumentWithEmbedding('Document 2'),
            $this->createDocumentWithEmbedding('Document 3'),
        ];

        $this->assertEquals(0, $database->count());

        $adapter->addDocuments($documents);

        $this->assertEquals(3, $database->count());
    }

    public function testPersistDocumentsAcrossInstances(): void
    {
        // Create and persist documents with first instance
        $database = new VectorDatabase(path: $this->tempDir);
        $adapter = new PHPVector($database);

        $documents = [
            $this->createDocumentWithEmbedding('Persisted document 1'),
            $this->createDocumentWithEmbedding('Persisted document 2'),
        ];

        $adapter->addDocuments($documents);
        $database->save();

        $this->assertEquals(2, $database->count());

        // Load a new instance and verify documents persist
        $newDatabase = VectorDatabase::open($this->tempDir);
        $this->assertEquals(2, $newDatabase->count());
    }

    public function testSimilaritySearchReturnsResults(): void
    {
        $database = new VectorDatabase();
        $adapter = new PHPVector($database);

        // Create documents with known embeddings for predictable search
        $embedding1 = array_fill(0, 128, 0.0);
        $embedding1[0] = 1.0; // First vector points in direction [1, 0, 0, ...]

        $embedding2 = array_fill(0, 128, 0.0);
        $embedding2[1] = 1.0; // Second vector points in direction [0, 1, 0, ...]

        $embedding3 = array_fill(0, 128, 0.0);
        $embedding3[0] = 0.9; // Third vector is similar to first
        $embedding3[1] = 0.1;

        $doc1 = new NeuronDocument('Document about cats');
        $doc1->id = 'doc1';
        $doc1->embedding = $embedding1;

        $doc2 = new NeuronDocument('Document about dogs');
        $doc2->id = 'doc2';
        $doc2->embedding = $embedding2;

        $doc3 = new NeuronDocument('Document about pets');
        $doc3->id = 'doc3';
        $doc3->embedding = $embedding3;

        $adapter->addDocuments([$doc1, $doc2, $doc3]);

        // Search with a vector similar to doc1
        $queryEmbedding = array_fill(0, 128, 0.0);
        $queryEmbedding[0] = 1.0;

        $results = $adapter->similaritySearch($queryEmbedding);

        $this->assertNotEmpty($results);
        $this->assertIsIterable($results);

        $resultsArray = is_array($results) ? $results : iterator_to_array($results);
        $this->assertCount(3, $resultsArray);

        // First result should be doc1 (most similar)
        $firstResult = $resultsArray[0];
        $this->assertInstanceOf(NeuronDocument::class, $firstResult);
        $this->assertEquals('doc1', $firstResult->id);
        $this->assertGreaterThan(0, $firstResult->score);
    }

    public function testSimilaritySearchRespectsTopK(): void
    {
        $database = new VectorDatabase();
        $adapter = new PHPVector($database, topK: 2);

        $documents = [
            $this->createDocumentWithEmbedding('Document 1'),
            $this->createDocumentWithEmbedding('Document 2'),
            $this->createDocumentWithEmbedding('Document 3'),
            $this->createDocumentWithEmbedding('Document 4'),
        ];

        $adapter->addDocuments($documents);

        $queryEmbedding = $this->createTestEmbedding();
        $results = $adapter->similaritySearch($queryEmbedding);

        $resultsArray = is_array($results) ? $results : iterator_to_array($results);
        $this->assertCount(2, $resultsArray);
    }

    public function testDocumentMetadataIsPreserved(): void
    {
        $database = new VectorDatabase();
        $adapter = new PHPVector($database);

        $document = new NeuronDocument('Test content');
        $document->embedding = $this->createTestEmbedding();
        $document->metadata = ['key' => 'value', 'number' => 42];

        $adapter->addDocument($document);

        $queryEmbedding = $document->embedding;
        $results = $adapter->similaritySearch($queryEmbedding);

        $resultsArray = is_array($results) ? $results : iterator_to_array($results);
        $firstResult = $resultsArray[0];

        $this->assertEquals(['key' => 'value', 'number' => 42], $firstResult->metadata);
    }

    public function testDocumentContentIsPreserved(): void
    {
        $database = new VectorDatabase();
        $adapter = new PHPVector($database);

        $expectedContent = 'This is the document content to preserve';

        $document = new NeuronDocument($expectedContent);
        $document->embedding = $this->createTestEmbedding();

        $adapter->addDocument($document);

        $results = $adapter->similaritySearch($document->embedding);
        $resultsArray = is_array($results) ? $results : iterator_to_array($results);

        $this->assertEquals($expectedContent, $resultsArray[0]->content);
    }

    public function testAddDocumentReturnsAdapterInstance(): void
    {
        $database = new VectorDatabase();
        $adapter = new PHPVector($database);

        $document = new NeuronDocument('Test');
        $document->embedding = $this->createTestEmbedding();

        $result = $adapter->addDocument($document);

        $this->assertSame($adapter, $result);
    }

    public function testAddDocumentsReturnsAdapterInstance(): void
    {
        $database = new VectorDatabase();
        $adapter = new PHPVector($database);

        $documents = [
            $this->createDocumentWithEmbedding('Doc 1'),
            $this->createDocumentWithEmbedding('Doc 2'),
        ];

        $result = $adapter->addDocuments($documents);

        $this->assertSame($adapter, $result);
    }

    private function createDocumentWithEmbedding(string $content): NeuronDocument
    {
        $document = new NeuronDocument($content);
        $document->embedding = $this->createTestEmbedding();
        return $document;
    }
}
