#!/usr/bin/env node
/* examples/chat.js — Interactive chat with a GGUF model using infergo */

'use strict';

const readline = require('readline');
const { LLM } = require('../lib/index');

const MODEL_PATH = process.argv[2] || process.env.INFERGO_MODEL || 'models/llama3-8b-q4.gguf';

const GPU_LAYERS = parseInt(process.env.INFERGO_GPU_LAYERS || '99', 10);
const CTX_SIZE   = parseInt(process.env.INFERGO_CTX_SIZE || '4096', 10);
const MAX_TOKENS = parseInt(process.env.INFERGO_MAX_TOKENS || '512', 10);
const TEMP       = parseFloat(process.env.INFERGO_TEMPERATURE || '0.7');

console.log(`Loading model: ${MODEL_PATH}`);
console.log(`  GPU layers: ${GPU_LAYERS}, context: ${CTX_SIZE}, max tokens: ${MAX_TOKENS}, temp: ${TEMP}`);
console.log();

const llm = new LLM(MODEL_PATH, {
  gpuLayers: GPU_LAYERS,
  ctxSize: CTX_SIZE,
});

const messages = [
  { role: 'system', content: 'You are a helpful assistant. Be concise.' }
];

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: 'You> ',
});

console.log('Chat started. Type your message and press Enter. Ctrl+C to quit.\n');
rl.prompt();

rl.on('line', (line) => {
  const input = line.trim();
  if (!input) {
    rl.prompt();
    return;
  }

  messages.push({ role: 'user', content: input });

  try {
    const reply = llm.chat(messages, {
      maxTokens: MAX_TOKENS,
      temperature: TEMP,
      topP: 0.9,
    });

    messages.push({ role: 'assistant', content: reply });
    console.log(`\nAssistant> ${reply}\n`);
  } catch (err) {
    console.error(`\nError: ${err.message}\n`);
  }

  rl.prompt();
});

rl.on('close', () => {
  console.log('\nGoodbye!');
  llm.destroy();
  process.exit(0);
});

/* Clean up on signals */
process.on('SIGINT', () => {
  console.log('\nGoodbye!');
  llm.destroy();
  process.exit(0);
});
