import { multiply, transpose } from "./math";
import { seededRandom } from "./math";

export class FFNLayer {
    public weights: number[][]; // Matrix [inSize][outSize]
    public bias: number[];      // Vector [outSize]

    private gradW: number[][];
    private gradB: number[];

    public isTraining: boolean = true;

    // Cache for 2D matrices
    private lastInput: number[][] = [];
    private lastOutput: number[][] = [];

    constructor(inSize: number, outSize: number) {
        // Xavier/Glorot initialization
        const limit = Math.sqrt(6 / (inSize + outSize));
        this.weights = Array.from({ length: inSize }, () =>
          Array.from({ length: outSize }, () => (seededRandom() * 2 - 1) * limit)
        );
        this.bias = new Array(outSize).fill(0);

        this.gradW = Array.from({ length: inSize }, () => new Array(outSize).fill(0));
        this.gradB = new Array(outSize).fill(0);
    }

    public setTrainingMode(mode: boolean): void {
        this.isTraining = mode;
        if (!mode) {
            this.lastInput = [];
            this.lastOutput = [];
        }
    }

    public forward(inputMatrix: number[][]): number[][] {
        if (this.isTraining) {
            this.lastInput = inputMatrix;
        }

        const seqLen = inputMatrix.length;
        const outSize = this.weights[0].length;

        // 1. The magic of Transformer: multiply the whole sequence at once (Batched MatMul)
        const xw = multiply(inputMatrix, this.weights);
        const outputMatrix: number[][] = [];

        // 2. Apply bias and activation function (Leaky ReLU) position-wise
        for (let t = 0; t < seqLen; t++) {
            outputMatrix[t] = new Array(outSize).fill(0);
            for (let j = 0; j < outSize; j++) {
                let sum = xw[t][j] + this.bias[j];
                outputMatrix[t][j] = sum > 0 ? sum : sum * 0.01; // Leaky ReLU
            }
        }

        if (this.isTraining) {
            this.lastOutput = outputMatrix;
        }

        return outputMatrix;
    }

    public backward(gradOutput: number[][]): number[][] {
        const seqLen = gradOutput.length;
        const inSize = this.weights.length;
        const outSize = this.weights[0].length;

        // 1. Derivative of Leaky ReLU (position-wise)
        const processedGrad: number[][] = [];
        for (let t = 0; t < seqLen; t++) {
            processedGrad[t] = new Array(outSize).fill(0);
            for (let j = 0; j < outSize; j++) {
                processedGrad[t][j] = this.lastOutput[t][j] > 0 ? gradOutput[t][j] : gradOutput[t][j] * 0.01;

                // Accumulate gradient for bias (sum over all sequence positions)
                this.gradB[j] += processedGrad[t][j];
            }
        }

        // 2. Gradients for weights (Matrix Math)
        // dW = Input^T * Grad
        const dW = multiply(transpose(this.lastInput), processedGrad);
        for (let i = 0; i < inSize; i++) {
            for (let j = 0; j < outSize; j++) {
                this.gradW[i][j] += dW[i][j];
            }
        }

        // 3. Gradient to pass down to AttentionLayer
        // dInput = Grad * W^T
        const gradInput = multiply(processedGrad, transpose(this.weights));

        return gradInput;
    }

    public updateWeights(learningRate: number): void {
        const inSize = this.weights.length;
        const outSize = this.weights[0].length;

        // Update weights and biases
        for (let i = 0; i < inSize; i++) {
            for (let j = 0; j < outSize; j++) {
                this.weights[i][j] -= learningRate * this.gradW[i][j];
                this.gradW[i][j] = 0; // Reset
            }
        }
        for (let j = 0; j < outSize; j++) {
            this.bias[j] -= learningRate * this.gradB[j];
            this.gradB[j] = 0; // Reset
        }
    }
}
