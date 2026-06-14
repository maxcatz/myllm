import { seededRandom } from "./math";

export class LMHead {
    public weights: number[][]; // Matrix [D_MODEL][VocabSize]

    // Private gradient storage to prevent external mutation
    private gradW: number[][];

    // Layer state (switch for inference/training mode)
    public isTraining: boolean = true;
    private lastInput: number[] = [];

    constructor(dModel: number, vocabSize: number) {
        // Initialization of weights
        this.weights = Array.from({ length: dModel }, () =>
            Array.from({ length: vocabSize }, () => seededRandom() - 0.5)
        );

        // Pre-allocate memory for gradients (same size as weights)
        this.gradW = Array.from({ length: dModel }, () =>
            new Array(vocabSize).fill(0)
        );
    }

    public setTrainingMode(mode: boolean): void {
        this.isTraining = mode;
        if (!mode) {
            // Free memory when switching to inference mode
            this.lastInput = [];
        }
    }

    public forward(input: number[]): number[] {
        if (this.isTraining) {
            // Use spread operator [...input] to clone the array and
            // prevent accidental mutation from outside the class
            this.lastInput = [...input];
        }

        const vocabSize = this.weights[0].length;
        const output = new Array(vocabSize).fill(0);

        for (let j = 0; j < vocabSize; j++) {
            let sum = 0;
            const len = input.length;
            for (let i = 0; i < len; i++) {
                sum += input[i] * this.weights[i][j];
            }
            output[j] = sum;
        }
        return output;
    }

    public backward(gradOutput: number[]): number[] {
        const dModel = this.weights.length;
        const vocabSize = this.weights[0].length;

        // 1. Calculate gradients for weights (gradW) using accumulation
        // Formula: gradW += Input^T * Grad
        for (let i = 0; i < dModel; i++) {
            for (let j = 0; j < vocabSize; j++) {
                // Accumulating gradients for batch processing
                this.gradW[i][j] += this.lastInput[i] * gradOutput[j];
            }
        }

        // 2. Calculate gradient for the previous layer (gradInput)
        // Formula: gradInput = Grad * W^T
        const gradInput = new Array(dModel).fill(0);
        for (let i = 0; i < dModel; i++) {
            let sum = 0;
            for (let j = 0; j < vocabSize; j++) {
                sum += gradOutput[j] * this.weights[i][j];
            }
            gradInput[i] = sum;
        }

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

                // CRITICAL: Reset the gradient after the step
                // so errors don't carry over to the next batch
                this.gradW[i][j] = 0;
            }
        }
    }
}