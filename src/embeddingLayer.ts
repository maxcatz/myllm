import { seededRandom } from "./math";

export class EmbeddingLayer {
  public weights: number[][]; // Matrix [VocabSize][D_MODEL]
  private gradW: number[][];
  private dModel: number;

  public isTraining: boolean = true;
  private lastIndices: number[] = [];
  private updatedIndices: Set<number> = new Set();

  constructor(vocabSize: number, dModel: number) {
    this.dModel = dModel;
    this.weights = Array.from({ length: vocabSize }, () =>
      Array.from({ length: dModel }, () => (seededRandom() * 2 - 1) * 0.1)
    );
    this.gradW = Array.from({ length: vocabSize }, () =>
      new Array(dModel).fill(0)
    );
  }

  public setTrainingMode(mode: boolean): void {
    this.isTraining = mode;
    if (!mode) {
      this.lastIndices = [];
      this.updatedIndices.clear();
    }
  }

  public forward(indices: number[]): number[][] {
    if (this.isTraining) {
      this.lastIndices = [...indices];
    }

    const output: number[][] = [];

    for (let pos = 0; pos < indices.length; pos++) {
      const wordIndex = indices[pos];

      // 1. Extract the raw embedding vector (cloned to prevent mutation of weights)
      const vector = [...this.weights[wordIndex]];

      // 2. Inject spatial awareness (Positional Encoding)
      this.applyPositionalEncoding(vector, pos);

      output.push(vector);
    }

    return output;
  }

  public backward(gradOutput: number[][]): void {
    for (let t = 0; t < this.lastIndices.length; t++) {
      const wordIndex = this.lastIndices[t];
      this.updatedIndices.add(wordIndex);

      for (let d = 0; d < this.dModel; d++) {
        this.gradW[wordIndex][d] += gradOutput[t][d];
      }
    }
  }

  public updateWeights(learningRate: number): void {
    for (const wordIndex of this.updatedIndices) {
      for (let d = 0; d < this.dModel; d++) {
        this.weights[wordIndex][d] -= learningRate * this.gradW[wordIndex][d];
        this.gradW[wordIndex][d] = 0;
      }
    }
    this.updatedIndices.clear();
  }

  /**
   * Injects positional context into the word vector using sine and cosine functions.
   * Mutates the passed vector in-place for performance.
   */
  private applyPositionalEncoding(vector: number[], pos: number): void {
    for (let i = 0; i < this.dModel; i++) {
      const divTerm = Math.pow(10000, (2 * Math.floor(i / 2)) / this.dModel);

      if (i % 2 === 0) {
        vector[i] += Math.sin(pos / divTerm); // Even indices
      } else {
        vector[i] += Math.cos(pos / divTerm); // Odd indices
      }
    }
  }
}
