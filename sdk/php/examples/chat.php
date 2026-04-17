<?php
require_once __DIR__ . '/../src/Infergo.php';

$llm = new Infergo\LLM($argv[1] ?? "model.gguf");
echo $llm->generate("What is PHP?", maxTokens: 64) . "\n";
$llm->close();
