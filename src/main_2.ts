import { SimpleTokenizer } from './tokenizer';
import { EmbeddingLayer, addPositionalEncoding, AttentionLayer, NeuralLayer, LMHead, backwardLinear, backwardInput } from './layers';
import { computeLossAndGradients, softmax } from './math';
import * as readline from 'readline';
import {TransformerModel} from "./transformerModel";

// ==========================================
// CONFIGURATION
// ==========================================
const D_MODEL = 16;
const LEARNING_RATE = 0.05;
const EPOCHS = 200;

const TRAINING_DATA = "dog eats meat cat drinks milk cow eats grass dog drinks water cat eats fish";
const DATASET = [
    { input: "dog eats", target: "meat" },
    { input: "cat eats", target: "fish" },
    { input: "dog drinks", target: "water" },
    { input: "cat drinks", target: "milk" },
    { input: "cow eats", target: "grass" },
    { input: "cow drinks", target: "water" }
];

// ==========================================
// HELPER FUNCTIONS
// ==========================================

function printEmbeddings(embeddings: EmbeddingLayer, tokenizer: SimpleTokenizer, title: string): void {
    console.log(`\n=== ${title} ===`);
    for (let i = 0; i < tokenizer.getVocabSize(); i++) {
        const word = tokenizer.decode([i]);
        const vector = (embeddings as any).weights[i].map((v: number) => v.toFixed(3));
        console.log(`${word.padEnd(8)} : [${vector.join(", ")}]`);
    }
}

function calculateDistance(embeddings: EmbeddingLayer, tokenizer: SimpleTokenizer, word1: string, word2: string): number {
    const v1 = embeddings.forward(tokenizer.encode(word1)[0]!);
    const v2 = embeddings.forward(tokenizer.encode(word2)[0]!);
    return Math.sqrt(v1.reduce((sum, val, i) => sum + Math.pow(val - v2[i]!, 2), 0));
}

function forwardPass(
    indices: number[],
    embeddings: EmbeddingLayer,
    attention: AttentionLayer,
    ffn: NeuralLayer,
    lmHead: LMHead,
    dModel: number
): { logits: number[], ffnOutput: number[], lastWordVector: number[] } {
    // Build hidden state with embeddings and positional encoding
    let hiddenState: number[][] = [];
    for (let i = 0; i < indices.length; i++) {
        const rawVector = embeddings.forward(indices[i]!);
        const posVector = addPositionalEncoding(rawVector, i, dModel);
        hiddenState.push(posVector);
    }

    // Apply attention
    const attentionOutput = attention.forward(hiddenState);
    const lastWordVector = attentionOutput[attentionOutput.length - 1]!;

    // Pass through FFN and LM Head
    const ffnOutput = ffn.forward(lastWordVector);
    const logits = lmHead.forward(ffnOutput);

    return { logits, ffnOutput, lastWordVector };
}

function backwardPass(
    gradLogits: number[],
    ffnOutput: number[],
    lastWordVector: number[],
    lmHead: LMHead,
    ffn: NeuralLayer
): { gradW_LM: number[][], gradW_FFN: number[][], gradInput_FFN: number[] } {
    // Gradients for LM Head
    const gradW_LM = backwardLinear(ffnOutput, gradLogits);
    const gradInput_LM = backwardInput(gradLogits, lmHead.weights);

    // Gradients for FFN (with ReLU derivative)
    const gradFFN_Output = gradInput_LM.map((g, idx) => ffnOutput[idx]! > 0 ? g : g * 0.01);
    const gradW_FFN = backwardLinear(lastWordVector, gradFFN_Output);
    const gradInput_FFN = backwardInput(gradFFN_Output, (ffn as any).weights);

    return { gradW_LM, gradW_FFN, gradInput_FFN };
}

function updateWeights(
    embeddings: EmbeddingLayer,
    lmHead: LMHead,
    ffn: NeuralLayer,
    gradW_LM: number[][],
    gradW_FFN: number[][],
    gradInput_FFN: number[],
    wordIndex: number,
    learningRate: number,
    dModel: number
): void {
    // Update embeddings
    for (let d = 0; d < dModel; d++) {
        (embeddings as any).weights[wordIndex][d] -= learningRate * gradInput_FFN[d]!;
    }

    // Update LM Head weights
    for (let i = 0; i < lmHead.weights.length; i++) {
        for (let j = 0; j < lmHead.weights[i]!.length; j++) {
            lmHead.weights[i]![j] -= learningRate * gradW_LM[i]![j]!;
        }
    }

    // Update FFN weights
    const ffnWeights = (ffn as any).weights;
    for (let i = 0; i < ffnWeights.length; i++) {
        for (let j = 0; j < ffnWeights[i].length; j++) {
            ffnWeights[i][j] -= learningRate * gradW_FFN[i]![j]!;
        }
    }
}

function logTrainingProgress(
    epoch: number,
    loss: number,
    logits: number[],
    targetIndex: number,
    gradInput_FFN: number[],
    embeddings: EmbeddingLayer,
    tokenizer: SimpleTokenizer,
    initialDist: number
): void {
    if (epoch === 1 || epoch % 20 === 0) {
        const probs = softmax([logits])[0]!;
        console.log(`Epoch ${epoch.toString().padStart(3, '0')} | Loss: ${loss.toFixed(4)} | Probability: ${(probs[targetIndex]! * 100).toFixed(1)}%`);
    }

    if (epoch % 50 === 0) {
        const sumGrad = gradInput_FFN.reduce((a, b) => a + Math.abs(b), 0);
        console.log(`Epoch ${epoch} | Gradient energy: ${sumGrad.toFixed(8)}`);

        if (sumGrad === 0) {
            console.warn("⚠️ GRADIENT DIED! Error is not propagating.");
        }

        const dist = calculateDistance(embeddings, tokenizer, "cat", "dog");
        console.log(`Epoch ${epoch} | Distance "cat"-"dog": ${dist.toFixed(4)} | Change: ${(dist - initialDist).toFixed(6)}`);
    }
}

function predict(
    phrase: string,
    tokenizer: SimpleTokenizer,
    embeddings: EmbeddingLayer,
    attention: AttentionLayer,
    ffn: NeuralLayer,
    lmHead: LMHead,
    dModel: number
): void {
    console.log(`\n=== 🧪 MODEL TEST: ${phrase} ===`);
    const testIndices = tokenizer.encode(phrase);

    const { logits } = forwardPass(testIndices, embeddings, attention, ffn, lmHead, dModel);

    // Convert logits to probabilities (Softmax)
    const probs = softmax([logits])[0]!;

    // Print results
    probs.forEach((p, i) => {
        console.log(`Word "${tokenizer.decode([i])}": ${(p * 100).toFixed(1)}%`);
    });
}

// ==========================================
// MAIN TRAINING FUNCTION
// ==========================================

interface TrainedModel {
    tokenizer: SimpleTokenizer;
    embeddings: EmbeddingLayer;
    attention: AttentionLayer;
    ffn: NeuralLayer;
    lmHead: LMHead;
}

function train(): TrainedModel {
    // Initialize tokenizer and model components
    const tokenizer = new SimpleTokenizer();
    tokenizer.train(TRAINING_DATA);

    const embeddings = new EmbeddingLayer(tokenizer.getVocabSize(), D_MODEL);
    const attention = new AttentionLayer(D_MODEL);
    const ffn = new NeuralLayer(D_MODEL, D_MODEL);
    const lmHead = new LMHead(D_MODEL, tokenizer.getVocabSize());

    // Print initial state
    printEmbeddings(embeddings, tokenizer, "📋 Vectors BEFORE training");
    const initialDist = calculateDistance(embeddings, tokenizer, "cat", "meat");
    console.log(`Initial distance: ${initialDist.toFixed(6)}`);

    // Training loop
    for (let epoch = 1; epoch <= EPOCHS; epoch++) {
        for (const example of DATASET) {
            const indices = tokenizer.encode(example.input);
            const targetIndex = tokenizer.encode(example.target)[0]!;

            // Forward pass
            const { logits, ffnOutput, lastWordVector } = forwardPass(
                indices, embeddings, attention, ffn, lmHead, D_MODEL
            );

            // Calculate loss
            const { loss, gradients: gradLogits } = computeLossAndGradients(logits, targetIndex);

            // Backward pass
            const { gradW_LM, gradW_FFN, gradInput_FFN } = backwardPass(
                gradLogits, ffnOutput, lastWordVector, lmHead, ffn
            );

            // Update weights
            const wordIndex = indices[indices.length - 1]!;
            updateWeights(
                embeddings, lmHead, ffn, gradW_LM, gradW_FFN, gradInput_FFN,
                wordIndex, LEARNING_RATE, D_MODEL
            );

            // Log progress
            logTrainingProgress(
                epoch, loss, logits, targetIndex, gradInput_FFN,
                embeddings, tokenizer, initialDist
            );
        }
    }

    console.log("\n✅ Training completed. Weight matrices configured.");

    // Print final state
    printEmbeddings(embeddings, tokenizer, "Final Word Vectors (Embeddings)");

    console.log("\n=== LM Head weights for word 'meat' ===");
    const meatIndex = tokenizer.encode("meat")[0];
    const meatWeights = (lmHead as any).weights.map((row: number[]) => row[meatIndex]!.toFixed(3));
    console.log(`[${meatWeights.join(", ")}]`);

    return { tokenizer, embeddings, attention, ffn, lmHead };
}

// ==========================================
// INTERACTIVE TEST LOOP
// ==========================================

async function interactiveTestLoop(model: TrainedModel): Promise<void> {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const question = (prompt: string): Promise<string> => {
        return new Promise((resolve) => {
            rl.question(prompt, (answer: string) => {
                resolve(answer);
            });
        });
    };

    console.log("\n" + "=".repeat(50));
    console.log("🤖 Interactive Model Testing");
    console.log("=".repeat(50));
    console.log("Enter a phrase to test (e.g., 'dog eats', 'cat drinks')");
    console.log("Type 'exit' or 'quit' to stop\n");

    while (true) {
        const input = await question("Enter phrase: ");
        const trimmedInput = input.trim();

        if (trimmedInput.toLowerCase() === 'exit' || trimmedInput.toLowerCase() === 'quit') {
            console.log("\n👋 Goodbye!");
            rl.close();
            break;
        }

        if (trimmedInput === '') {
            console.log("⚠️  Please enter a phrase\n");
            continue;
        }

        predict(trimmedInput, model.tokenizer, model.embeddings, model.attention, model.ffn, model.lmHead, D_MODEL);
        console.log();
    }
}

// ==========================================
// ENTRY POINT
// ==========================================

async function main_2(): Promise<void> {
    const myLLM = new TransformerModel(1, D_MODEL);
    const model = train();
    await interactiveTestLoop(model);
}

main_2().catch(console.error);
