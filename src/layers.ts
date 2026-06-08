import {multiply, scale, seededRandom, softmax, transpose} from './math';

export class EmbeddingLayer {
    private weights: number[][];

    constructor(vocabSize: number, embeddingDim: number) {
        this.weights = Array.from({ length: vocabSize }, () =>
            Array.from({ length: embeddingDim }, () => seededRandom() * 2 - 1)
        );
    }

    forward(tokenIndex: number): number[] {
        return [...this.weights[tokenIndex]]; // Return a copy of the vector
    }
}

export function addPositionalEncoding(vector: number[], pos: number, dModel: number): number[] {
    return vector.map((val, i) => {
        const divTerm = Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
        return i % 2 === 0 ? val + Math.sin(pos / divTerm) : val + Math.cos(pos / divTerm);
    });
}

export class AttentionLayer {
    private Wq: number[][];
    private Wk: number[][];
    private Wv: number[][];
    private dModel: number;

    constructor(dModel: number) {
        this.dModel = dModel;
        // Initialize weight matrices of size D x D
        this.Wq = Array.from({ length: dModel }, () => Array.from({ length: dModel }, () => seededRandom() - 0.5));
        this.Wk = Array.from({ length: dModel }, () => Array.from({ length: dModel }, () => seededRandom() - 0.5));
        this.Wv = Array.from({ length: dModel }, () => Array.from({ length: dModel }, () => seededRandom() - 0.5));
    }

    forward(input: number[][]): number[][] {
        const Q = multiply(input, this.Wq);
        const K = multiply(input, this.Wk);
        const V = multiply(input, this.Wv);

        const scores = multiply(Q, transpose(K));
        const scaledScores = scale(scores, 1 / Math.sqrt(this.dModel));
        const weights = softmax(scaledScores);

        return multiply(weights, V);
    }

    // Inside AttentionLayer class:
    backward(gradOutput: number[][], input: number[][]): { gradWq: number[][], gradWk: number[][], gradWv: number[][], gradInput: number[][] } {
        // 1. Calculate gradients for weights Wq, Wk, Wv
        // This requires matrix transposition and gradient multiplication
        // For simplicity, we'll do a "simplified" gradient pass
        // (in real frameworks, tensor optimization is used here)
        const gradWv = multiply(transpose(gradOutput), input); // Simplified

        // ... here will be multidimensional algebra for Q and K ...

        // Return gradients so they can be subtracted from weights
        return {
            gradWq: Array.from({length: this.dModel}, () => Array(this.dModel).fill(0)), // Placeholder for example
            gradWk: Array.from({length: this.dModel}, () => Array(this.dModel).fill(0)),
            gradWv,
            gradInput: gradOutput // Simplest pass-through
        };
    }
}

export class NeuralLayer {
    private weights: number[][];
    private bias: number[];

    constructor(inSize: number, outSize: number) {
        // Use Array.from for the inner array too, so random is called each time
        this.weights = Array.from({ length: inSize }, () =>
          Array.from({ length: outSize }, () => seededRandom() - 0.5)
        );
        this.bias = new Array(outSize).fill(0);
    }

    forward(input: number[]): number[] {
        const output = new Array(this.weights[0].length).fill(0);
        for (let j = 0; j < output.length; j++) {
            for (let i = 0; i < input.length; i++) {
                output[j] += input[i] * this.weights[i][j];
            }
            output[j] += this.bias[j];
            output[j] = output[j] > 0 ? output[j] : output[j] * 0.01; // ReLU activation
        }
        return output;
    }
}

// Backpropagation functions (Backprop)
export function backwardLinear(input: number[], gradOutput: number[]): number[][] {
    const gradWeights: number[][] = [];
    for (let i = 0; i < input.length; i++) {
        gradWeights[i] = [];
        for (let j = 0; j < gradOutput.length; j++) {
            gradWeights[i][j] = input[i] * gradOutput[j];
        }
    }
    return gradWeights;
}

export function backwardInput(gradOutput: number[], weights: number[][]): number[] {
    const inputSize = weights.length;
    const outputSize = weights[0].length;
    const gradInput: number[] = new Array(inputSize).fill(0);
    for (let i = 0; i < inputSize; i++) {
        for (let j = 0; j < outputSize; j++) {
            gradInput[i] += gradOutput[j] * weights[i][j];
        }
    }
    return gradInput;
}

export class LMHead {
    weights: number[][]; // Matrix [D_MODEL][VocabSize]

    constructor(dModel: number, vocabSize: number) {
        this.weights = Array.from({ length: dModel }, () =>
          Array.from({ length: vocabSize }, () => seededRandom() - 0.5)
        );
    }

    forward(input: number[]): number[] {
        const vocabSize = this.weights[0].length;
        const output = new Array(vocabSize).fill(0);

        for (let j = 0; j < vocabSize; j++) {
            for (let i = 0; i < input.length; i++) {
                output[j] += input[i] * this.weights[i][j];
            }
        }
        return output; // Returns 6 numbers (scores for each word)
    }
}
