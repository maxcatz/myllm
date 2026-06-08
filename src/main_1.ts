import { SimpleTokenizer } from './tokenizer';
import { EmbeddingLayer, addPositionalEncoding, AttentionLayer, NeuralLayer } from './layers';

// 1. Settings for our "micro-model"
const D_MODEL = 4; // Dimension of the semantic vector (in real LLMs this is 4096)
const textData = "dog eats meat cat drinks milk";
const inputPhrase = "cat drinks milk";

console.log("=== 1. Initialization ===");
const tokenizer = new SimpleTokenizer();
tokenizer.train(textData);
console.log(`Vocabulary trained. Size: ${tokenizer.getVocabSize()} tokens`);

const embeddings = new EmbeddingLayer(tokenizer.getVocabSize(), D_MODEL);
const attention = new AttentionLayer(D_MODEL);
const ffn = new NeuralLayer(D_MODEL, D_MODEL); // Feed-Forward Network layer

// 2. Tokenization
console.log("\n=== 2. Data Processing ===");
const indices = tokenizer.encode(inputPhrase);
console.log(`Phrase "${inputPhrase}" converted to indices:`, indices);

// 3. Forward Pass
console.log("\n=== 3. Pass through model ===");

// 3.1. Embeddings and Positioning
let hiddenState: number[][] = [];
for (let i = 0; i < indices.length; i++) {
    // Get raw word vector
    const rawVector = embeddings.forward(indices[i]!);
    // Add position information
    const posVector = addPositionalEncoding(rawVector, i, D_MODEL);
    hiddenState.push(posVector);
}

console.log("Vectors after embeddings and positioning (N x D):");
console.table(hiddenState);

// 3.2. Attention Layer (Self-Attention)
hiddenState = attention.forward(hiddenState);

console.log("\nVectors after Attention layer (words 'mixed' meanings):");
console.table(hiddenState);

// 3.3. Neural layer (FFN with ReLU) - applied to each word separately
const outputState = hiddenState.map(vector => ffn.forward(vector));

console.log("\nVectors after Fully Connected layer (Final output):");
console.table(outputState);
console.log("\nMagic happened: text turned into numbers, enriched with context and passed through non-linear logic!");
