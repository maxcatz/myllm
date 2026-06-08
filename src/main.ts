import { SimpleTokenizer } from './tokenizer';
import { EmbeddingLayer, addPositionalEncoding, AttentionLayer, NeuralLayer, LMHead, backwardLinear, backwardInput } from './layers';
import { computeLossAndGradients } from './math';

const D_MODEL = 40;
const LEARNING_RATE = 0.05;
const EPOCHS = 200;

// 1. Initialization
const textData = "dog eats meat cat drinks milk cow eats grass dog drinks water";
const tokenizer = new SimpleTokenizer();
tokenizer.train(textData);

const embeddings = new EmbeddingLayer(tokenizer.getVocabSize(), D_MODEL);
const attention = new AttentionLayer(D_MODEL);
const ffn = new NeuralLayer(D_MODEL, D_MODEL);
const lmHead = new LMHead(D_MODEL, tokenizer.getVocabSize());

// Our task: input "dog eats" and predict "meat"
// const inputPhrase = "dog eats";
// const targetWord = "meat";
// const indices = tokenizer.encode(inputPhrase);
// const targetIndex = tokenizer.encode(targetWord)[0];

// console.log(`\n🚀 Starting training! Task: predict "${targetWord}" (index ${targetIndex}) after "${inputPhrase}"\n`);

console.log("\n=== 📋 Vectors BEFORE training ===");
for (let i = 0; i < tokenizer.getVocabSize(); i++) {
    const word = tokenizer.decode([i]);
    // Use `any` to peek into private weights
    const vector = (embeddings as any).weights[i].map((v: number) => v.toFixed(3));
    console.log(`${word.padEnd(8)} : [${vector.join(", ")}]`);
}

// 1.5. Fix initial distance for comparison
const getDist = () => {
    const v1 = embeddings.forward(tokenizer.encode("cat")[0]!);
    const v2 = embeddings.forward(tokenizer.encode("dog")[0]!);
    return Math.sqrt(v1.reduce((sum, val, i) => sum + Math.pow(val - v2[i]!, 2), 0));
};
const initialDist = getDist();
console.log(`Initial distance: ${initialDist.toFixed(6)}`);
const dataset = [
    { input: "dog eats", target: "meat" },
    { input: "cat eats", target: "meat" },
    { input: "dog drinks", target: "water" },
    { input: "cat drinks", target: "milk" },
    { input: "cow eats", target: "grass" }
];
// 2. Training loop (Gradient Descent)
for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    for (const example of dataset) {
        const indices = tokenizer.encode(example.input);
        const targetIndex = tokenizer.encode(example.target)[0]!;
    // ==========================================
    // STEP 1: FORWARD PASS
    // ==========================================
    let hiddenState: number[][] = [];
    for (let i = 0; i < indices.length; i++) {
        const rawVector = embeddings.forward(indices[i]!);
        const posVector = addPositionalEncoding(rawVector, i, D_MODEL);
        hiddenState.push(posVector);
    }

    const attentionOutput = attention.forward(hiddenState);

    // Take the vector of the LAST word ("eats") to predict the next one
    const lastWordVector = attentionOutput[attentionOutput.length - 1]!;

    // Pass through FFN and LM Head
    const ffnOutput = ffn.forward(lastWordVector);
    const logits = lmHead.forward(ffnOutput);


    // ==========================================
    // STEP 2: ERROR (LOSS)
    // ==========================================
    const { loss, gradients: gradLogits } = computeLossAndGradients(logits, targetIndex);

    // Log progress
    if (epoch === 1 || epoch % 20 === 0) {
        // Calculate probability of correct answer (just for display)
        const maxLogit = Math.max(...logits);
        const exps = logits.map(x => Math.exp(x - maxLogit));
        const probs = exps.map(x => x / exps.reduce((a, b) => a + b, 0));

        console.log(`Epoch ${epoch.toString().padStart(3, '0')} | Error (Loss): ${loss.toFixed(4)} | Probability "${'dog'}": ${(probs[targetIndex]! * 100).toFixed(1)}%`);
    }

    // ==========================================
    // STEP 3: BACKPROPAGATION (BACKWARD)
    // ==========================================

    // 3.1. Gradients for LM Head
    const gradW_LM = backwardLinear(ffnOutput, gradLogits);
    const gradInput_LM = backwardInput(gradLogits, lmHead.weights); // Pass error further down

    // 3.2. Gradients for FFN (taking ReLU into account!)
    // ReLU derivative: if output was 0, gradient doesn't pass (multiply by 0). Otherwise multiply by 1.
    const gradFFN_Output = gradInput_LM.map((g, idx) => ffnOutput[idx]! > 0 ? g : g*0.01);
    const gradW_FFN = backwardLinear(lastWordVector, gradFFN_Output);

    // IMPORTANT: Pass error further down to Embeddings
    const gradInput_FFN = backwardInput(gradFFN_Output, (ffn as any).weights);


    if (epoch % 50 === 0) {
        const sumGrad = gradInput_FFN.reduce((a, b) => a + Math.abs(b), 0);
        console.log(`Epoch ${epoch} | Sum of gradients (energy): ${sumGrad.toFixed(8)}`);

        if (sumGrad === 0) {
            console.warn("⚠️ GRADIENT DIED! Error is not propagating.");
        }
    }
    // ==========================================
    // STEP 4: OPTIMIZATION STEP (WEIGHT UPDATE)
    // ==========================================

    // 4.1. UPDATE EMBEDDINGS (Your "Gravity")
    // Take the index of the last word in the phrase that we're training
    const wordIndex = indices[indices.length - 1]!;
    for (let d = 0; d < D_MODEL; d++) {
        (embeddings as any).weights[wordIndex][d] -= LEARNING_RATE * gradInput_FFN[d]!;
    }

    // Update LM Head weights: Weight = Weight - (LR * Gradient)
    for (let i = 0; i < lmHead.weights.length; i++) {
        for (let j = 0; j < lmHead.weights[i]!.length; j++) {
            lmHead.weights[i]![j] -= LEARNING_RATE * gradW_LM[i]![j]!;
        }
    }

    // Update FFN weights
    // We'll need to remove the private modifier from ffn.weights in layers.ts,
    // or add an updateWeights method. For speed, we use TypeScript hack (any):
    const ffnWeights = (ffn as any).weights;
    for (let i = 0; i < ffnWeights.length; i++) {
        for (let j = 0; j < ffnWeights[i].length; j++) {
            ffnWeights[i][j] -= LEARNING_RATE * gradW_FFN[i]![j]!;
        }
    }

    // Inside training loop, after weight update:
    if (epoch % 50 === 0) {
        const vecCat = embeddings.forward(tokenizer.encode("cat")[0]!);
        const vecDog = embeddings.forward(tokenizer.encode("dog")[0]!);

        // Distance between "cat" and "dog"
        const dist = Math.sqrt(vecCat.reduce((sum, val, i) => sum + Math.pow(val - vecDog[i]!, 2), 0));

        console.log(`Epoch ${epoch} | Distance "cat"-"dog": ${dist.toFixed(4)} | Change: ${(dist - initialDist).toFixed(6)}`);
    }
    }
}

console.log("\n✅ Training completed. Weight matrices configured.");

// ==========================================
// STEP 5: WEIGHT INSPECTION (WHAT DID THE MODEL LEARN?)
// ==========================================
console.log("\n=== Final Word Vectors (Embeddings) ===");
for (let i = 0; i < tokenizer.getVocabSize(); i++) {
    const word = tokenizer.decode([i]);
    // Use hack with (any), since weights are private
    const vector = (embeddings as any).weights[i].map((v: number) => v.toFixed(3));
    console.log(`${word.padEnd(8)} : [${vector.join(", ")}]`);
}

console.log("\n=== LM Head weights for word 'meat' ===");
// See how the final layer responds to features
const meatIndex = tokenizer.encode("meat")[0];
const meatWeights = (lmHead as any).weights.map((row: number[]) => row[meatIndex]!.toFixed(3));
console.log(`[${meatWeights.join(", ")}]`);



// 1. Encode phrase
const testPhrase = process.argv[2] || "dog drinks";
console.log(`=== 🧪 MODEL TEST: ${testPhrase} ===`);
const testIndices = tokenizer.encode(testPhrase);

// 2. Forward pass (only forward propagation)
let hiddenStateTest: number[][] = [];
for (let i = 0; i < testIndices.length; i++) {
    const rawVector = embeddings.forward(testIndices[i]!); // Uses trained embeddings
    const posVector = addPositionalEncoding(rawVector, i, D_MODEL);
    hiddenStateTest.push(posVector);
}

const attOut = attention.forward(hiddenStateTest);
const ffnOut = ffn.forward(attOut[attOut.length - 1]!);
const finalLogits = lmHead.forward(ffnOut);

// 3. Convert logits to probabilities (Softmax)
const maxL = Math.max(...finalLogits);
const exps = finalLogits.map(x => Math.exp(x - maxL));
const sumE = exps.reduce((a, b) => a + b, 0);
const probs = exps.map(x => x / sumE);

// 4. Print result
probs.forEach((p, i) => {
    console.log(`Word "${tokenizer.decode([i])}": ${(p * 100).toFixed(1)}%`);
});
