import { multiply, transpose } from "./math";
import { seededRandom } from "./math";

export class LMHead {
    public weights: number[][]; // Matrix [D_MODEL][VocabSize]
    private gradW: number[][];

    public isTraining: boolean = true;
    private lastInput: number[][] = []; // Now caches 2D matrix

    constructor(dModel: number, vocabSize: number) {
        // Initialization of weights
        this.weights = Array.from({ length: dModel }, () =>
          Array.from({ length: vocabSize }, () => seededRandom() - 0.5)
        );

        this.gradW = Array.from({ length: dModel }, () =>
          new Array(vocabSize).fill(0)
        );
    }

    public setTrainingMode(mode: boolean): void {
        this.isTraining = mode;
        if (!mode) {
            this.lastInput = [];
        }
    }

    // Now accepts [SeqLen][D_MODEL] and returns [SeqLen][VocabSize]
    public forward(inputMatrix: number[][]): number[][] {
        if (this.isTraining) {
            this.lastInput = inputMatrix;
        }

        // Batched matrix multiplication: Input * Weights
        // Resulting matrix contains vocabulary scores (logits) for each word in the sequence
        return multiply(inputMatrix, this.weights);
    }

    public backward(gradOutput: number[][]): number[][] {
        const dModel = this.weights.length;
        const vocabSize = this.weights[0].length;

        // 1. Calculate gradients for weights (gradW) using batched math
        // Formula: gradW = Input^T * Grad
        const dW = multiply(transpose(this.lastInput), gradOutput);

        // Accumulating gradients for batch processing
        for (let i = 0; i < dModel; i++) {
            for (let j = 0; j < vocabSize; j++) {
                this.gradW[i][j] += dW[i][j];
            }
        }

        // 2. Calculate gradient for the previous layer (gradInput)
        // Formula: gradInput = Grad * W^T
        const gradInput = multiply(gradOutput, transpose(this.weights));

        // Return the error ready for the FFN layer
        return gradInput;
    }

    public updateWeights(learningRate: number): void {
        const dModel = this.weights.length;
        const vocabSize = this.weights[0].length;

        // Optimizer step (Gradient Descent)
        for (let i = 0; i < dModel; i++) {
            for (let j = 0; j < vocabSize; j++) {
                this.weights[i][j] -= learningRate * this.gradW[i][j];
                this.gradW[i][j] = 0; // Reset after step
            }
        }
    }
}
