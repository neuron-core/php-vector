<?php

declare(strict_types=1);

namespace NeuronAI\PHPVector\Tests;

use NeuronAI\PHPVector\PHPVector;
use NeuronAI\RAG\Document as NeuronDocument;
use PHPVector\VectorDatabase;
use PHPUnit\Framework\TestCase;
use ReflectionProperty;

use function array_diff;
use function array_fill;
use function is_array;
use function is_dir;
use function iterator_to_array;
use function mt_getrandmax;
use function mt_rand;
use function rmdir;
use function scandir;
use function sys_get_temp_dir;
use function uniqid;
use function unlink;

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

    /**
     * Extract the internal VectorDatabase from a PHPVector instance.
     *
     * The new PHPVector constructor accepts a directory path and manages its own
     * VectorDatabase internally (protected property). Tests that need to assert
     * document counts or call save() use this helper to reach the inner instance.
     */
    private function getDatabase(PHPVector $store): VectorDatabase
    {
        $ref = new ReflectionProperty(PHPVector::class, 'database');
        return $ref->getValue($store);
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
        $store = new PHPVector($this->tempDir);

        $document = new NeuronDocument('Test content');
        $document->embedding = $this->createTestEmbedding();

        $store->addDocument($document);

        $this->assertEquals(1, $this->getDatabase($store)->count());
    }

    public function testAddDocumentsAddsMultipleDocuments(): void
    {
        $store = new PHPVector($this->tempDir);

        $documents = [
            $this->createDocumentWithEmbedding('Document 1'),
            $this->createDocumentWithEmbedding('Document 2'),
            $this->createDocumentWithEmbedding('Document 3'),
        ];

        $store->addDocuments($documents);

        $this->assertEquals(3, $this->getDatabase($store)->count());
    }

    public function testPersistDocumentsAcrossInstances(): void
    {
        // Create and persist documents with first instance.
        $store = new PHPVector($this->tempDir);

        $documents = [
            $this->createDocumentWithEmbedding('Persisted document 1'),
            $this->createDocumentWithEmbedding('Persisted document 2'),
        ];

        $store->addDocuments($documents);

        $this->assertEquals(2, $this->getDatabase($store)->count());

        // Open a fresh VectorDatabase from the same path and verify documents persist.
        $reopened = VectorDatabase::open($this->tempDir);
        $this->assertEquals(2, $reopened->count());
    }

    public function testSimilaritySearchReturnsResults(): void
    {
        $store = new PHPVector($this->tempDir);

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

        $store->addDocuments([$doc1, $doc2, $doc3]);

        // Search with a vector similar to doc1
        $queryEmbedding = array_fill(0, 128, 0.0);
        $queryEmbedding[0] = 1.0;

        $results = $store->similaritySearch($queryEmbedding);

        $this->assertNotEmpty($results);

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
        $store = new PHPVector($this->tempDir, topK: 2);

        $documents = [
            $this->createDocumentWithEmbedding('Document 1'),
            $this->createDocumentWithEmbedding('Document 2'),
            $this->createDocumentWithEmbedding('Document 3'),
            $this->createDocumentWithEmbedding('Document 4'),
        ];

        $store->addDocuments($documents);

        $queryEmbedding = $this->createTestEmbedding();
        $results = $store->similaritySearch($queryEmbedding);

        $resultsArray = is_array($results) ? $results : iterator_to_array($results);
        $this->assertCount(2, $resultsArray);
    }

    public function testDocumentMetadataIsPreserved(): void
    {
        $store = new PHPVector($this->tempDir);

        $document = new NeuronDocument('Test content');
        $document->embedding = $this->createTestEmbedding();
        $document->metadata = ['key' => 'value', 'number' => 42];

        $store->addDocument($document);

        $queryEmbedding = $document->embedding;
        $results = $store->similaritySearch($queryEmbedding);

        $resultsArray = is_array($results) ? $results : iterator_to_array($results);
        $firstResult = $resultsArray[0];

        $this->assertEquals(['key' => 'value', 'number' => 42], $firstResult->metadata);
    }

    public function testDocumentContentIsPreserved(): void
    {
        $store = new PHPVector($this->tempDir);

        $expectedContent = 'This is the document content to preserve';

        $document = new NeuronDocument($expectedContent);
        $document->embedding = $this->createTestEmbedding();

        $store->addDocument($document);

        $results = $store->similaritySearch($document->embedding);
        $resultsArray = is_array($results) ? $results : iterator_to_array($results);

        $this->assertEquals($expectedContent, $resultsArray[0]->content);
    }

    public function testAddDocumentReturnsAdapterInstance(): void
    {
        $store = new PHPVector($this->tempDir);

        $document = new NeuronDocument('Test');
        $document->embedding = $this->createTestEmbedding();

        $result = $store->addDocument($document);

        $this->assertSame($store, $result);
    }

    public function testAddDocumentsReturnsAdapterInstance(): void
    {
        $store = new PHPVector($this->tempDir);

        $documents = [
            $this->createDocumentWithEmbedding('Doc 1'),
            $this->createDocumentWithEmbedding('Doc 2'),
        ];

        $result = $store->addDocuments($documents);

        $this->assertSame($store, $result);
    }

    public function testSourceTypeAndNameRoundTripWithoutLeakingIntoMetadata(): void
    {
        $store = new PHPVector($this->tempDir);

        $document = new NeuronDocument('Round trip content');
        $document->id = 'rt1';
        $document->embedding = $this->createTestEmbedding();
        $document->sourceType = 'pdf';
        $document->sourceName = 'manual.pdf';
        $document->metadata = ['author' => 'jane', 'pages' => 12, 'published' => true];

        $store->addDocument($document);

        $results = $store->similaritySearch($document->embedding);
        $resultsArray = is_array($results) ? $results : iterator_to_array($results);
        $first = $resultsArray[0];

        self::assertSame('pdf', $first->sourceType);
        self::assertSame('manual.pdf', $first->sourceName);
        self::assertSame(['author' => 'jane', 'pages' => 12, 'published' => true], $first->metadata);
    }

    public function testMutationsPersistWhenAutoSaveEnabled(): void
    {
        $store = new PHPVector($this->tempDir);

        $store->addDocuments([
            $this->createDocumentWithEmbedding('Auto 1'),
            $this->createDocumentWithEmbedding('Auto 2'),
        ]);

        // No explicit save(): auto-save should have persisted the index.
        $reopened = VectorDatabase::open($this->tempDir);
        self::assertSame(2, $reopened->count());
    }

    public function testAutoSaveDisabledDoesNotPersistUntilManualSave(): void
    {
        $store = new PHPVector($this->tempDir, autoSave: false);

        $store->addDocuments([
            $this->createDocumentWithEmbedding('Manual 1'),
            $this->createDocumentWithEmbedding('Manual 2'),
        ]);

        // Index not yet persisted: meta.json must not exist on disk.
        self::assertFileDoesNotExist($this->tempDir . '/meta.json');

        $this->getDatabase($store)->save();
        $afterSave = VectorDatabase::open($this->tempDir);
        self::assertSame(2, $afterSave->count());
    }

    public function testDeleteByRemovesMatchingSourceType(): void
    {
        $store = new PHPVector($this->tempDir);

        $store->addDocuments([
            $this->makeSourcedDocument('a', 'pdf', 'one.pdf'),
            $this->makeSourcedDocument('b', 'pdf', 'two.pdf'),
            $this->makeSourcedDocument('c', 'web', 'site'),
        ]);
        self::assertSame(3, $this->getDatabase($store)->count());

        $store->deleteBy('pdf');

        self::assertSame(1, $this->getDatabase($store)->count());
    }

    public function testDeleteByRemovesOnlyExactTypeAndName(): void
    {
        $store = new PHPVector($this->tempDir);

        $store->addDocuments([
            $this->makeSourcedDocument('a', 'pdf', 'one.pdf'),
            $this->makeSourcedDocument('b', 'pdf', 'two.pdf'),
        ]);

        $store->deleteBy('pdf', 'one.pdf');

        self::assertSame(1, $this->getDatabase($store)->count());
    }

    public function testDeleteByWithNoMatchIsNoop(): void
    {
        $store = new PHPVector($this->tempDir);

        $store->addDocument($this->makeSourcedDocument('a', 'pdf', 'one.pdf'));

        $result = $store->deleteBy('missing');

        self::assertSame(1, $this->getDatabase($store)->count());
        self::assertSame($store, $result);
    }

    public function testDeleteBySourceDelegatesToDeleteBy(): void
    {
        $store = new PHPVector($this->tempDir);

        $store->addDocuments([
            $this->makeSourcedDocument('a', 'pdf', 'one.pdf'),
            $this->makeSourcedDocument('b', 'web', 'site'),
        ]);

        $store->deleteBySource('pdf', 'one.pdf');

        self::assertSame(1, $this->getDatabase($store)->count());
    }

    public function testDeleteByPersistsWhenAutoSaveEnabled(): void
    {
        $store = new PHPVector($this->tempDir);

        $store->addDocuments([
            $this->makeSourcedDocument('a', 'pdf', 'one.pdf'),
            $this->makeSourcedDocument('b', 'web', 'site'),
        ]);

        $store->deleteBy('pdf');

        $reopened = VectorDatabase::open($this->tempDir);
        self::assertSame(1, $reopened->count());
    }

    private function createDocumentWithEmbedding(string $content): NeuronDocument
    {
        $document = new NeuronDocument($content);
        $document->embedding = $this->createTestEmbedding();
        return $document;
    }

    private function makeSourcedDocument(string $id, string $sourceType, string $sourceName): NeuronDocument
    {
        $document = new NeuronDocument('content ' . $id);
        $document->id = $id;
        $document->embedding = $this->createTestEmbedding();
        $document->sourceType = $sourceType;
        $document->sourceName = $sourceName;
        return $document;
    }
}
