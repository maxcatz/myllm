import { seededRandom } from "./math";

export class FFNLayer {
    public weights: number[][];
    public bias: number[];

    // Private gradient storages to prevent external mutation
    private gradW: number[][];
    private gradB: number[];

    // State for backpropagation
    public isTraining: boolean = true;
    private lastInput: number[] = [];
    private lastOutput: number[] = []; // Needed for the activation derivative

    constructor(inSize: number, outSize: number) {
        // Xavier/Glorot initialization for better convergence
        const limit = Math.sqrt(6 / (inSize + outSize));
        this.weights = Array.from({ length: inSize }, () =>
            Array.from({ length: outSize }, () => (seededRandom() * 2 - 1) * limit)
        );
        this.bias = new Array(outSize).fill(0);

        // Pre-allocate memory for gradients to avoid GC spikes during training
        this.gradW = Array.from({ length: inSize }, () => new Array(outSize).fill(0));
        this.gradB = new Array(outSize).fill(0);
    }

    public setTrainingMode(mode: boolean): void {
        this.isTraining = mode;
        if (!mode) {
            // Free memory when switching to inference mode
            this.lastInput = [];
            this.lastOutput = [];
        }
    }

    public forward(input: number[]): number[] {
        if (this.isTraining) {
            // Clone to avoid reference mutation from outside
            this.lastInput = [...input];
        }

        const inSize = this.weights.length;
        const outSize = this.weights[0].length;
        const output = new Array(outSize).fill(0);

        for (let j = 0; j < outSize; j++) {
            let sum = this.bias[j];
            for (let i = 0; i < inSize; i++) {
                sum += input[i] * this.weights[i][j];
            }

            // Leaky ReLU activation
            sum = sum > 0 ? sum : sum * 0.01;
            output[j] = sum;
        }

        if (this.isTraining) {
            // Cache the activated output for the backward pass
            this.lastOutput = [...output];
        }

        return output;
    }

    public backward(gradOutput: number[]): number[] {
        const inSize = this.weights.length;
        const outSize = this.weights[0].length;

        // 1. Backpropagate through Leaky ReLU activation (Chain Rule)
        const processedGrad = new Array(outSize).fill(0);
        for (let j = 0; j < outSize; j++) {
            // If forward output was > 0, gradient passes fully. Otherwise, multiply by 0.01
            processedGrad[j] = this.lastOutput[j] > 0 ? gradOutput[j] : gradOutput[j] * 0.01;
        }

        // 2. Calculate gradients for weights (gradW) and biases (gradB)
        for (let i = 0; i < inSize; i++) {
            for (let j = 0; j < outSize; j++) {
                // Using '+=' allows gradient accumulation if needed in the future
                this.gradW[i][j] += this.lastInput[i] * processedGrad[j];
            }
        }

        for (let j = 0; j < outSize; j++) {
            this.gradB[j] += processedGrad[j];
        }

        // 3. Calculate gradient for the previous layer (gradInput)
        const gradInput = new Array(inSize).fill(0);
        for (let i = 0; i < inSize; i++) {
            let sum = 0;
            for (let j = 0; j < outSize; j++) {
                sum += processedGrad[j] * this.weights[i][j];
            }
            gradInput[i] = sum;
        }

        // Return the error ready for the Attention layer
        return gradInput;
    }

    public updateWeights(learningRate: number): void {
        const inSize = this.weights.length;
        const outSize = this.weights[0].length;

        // Apply gradients and reset them immediately
        for (let i = 0; i < inSize; i++) {
            for (let j = 0; j < outSize; j++) {
                this.weights[i][j] -= learningRate * this.gradW[i][j];
                this.gradW[i][j] = 0; // Reset for the next epoch
            }
        }

        for (let j = 0; j < outSize; j++) {
            this.bias[j] -= learningRate * this.gradB[j];
            this.gradB[j] = 0; // Reset
        }
    }
}