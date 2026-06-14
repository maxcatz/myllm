import {addMatrices, dotProduct, multiply, scale, seededRandom, softmax, transpose} from './math';

export class AttentionLayer {
  // Weight matrices [D_MODEL][D_MODEL]
  public wQ: number[][];
  public wK: number[][];
  public wV: number[][];

  // Private gradient accumulators
  private gradW_Q: number[][];
  private gradW_K: number[][];
  private gradW_V: number[][];

  private dModel: number;

  // State management
  public isTraining: boolean = true;

  // Cache for backpropagation (Crucial for Attention calculus)
  private lastInput: number[][] = [];  // Matrix X
  private lastQ: number[][] = [];      // Queries
  private lastK: number[][] = [];      // Keys
  private lastV: number[][] = [];      // Values
  private lastAttnWeights: number[][] = []; // Softmax probabilities

  constructor(dModel: number) {
    this.dModel = dModel;

    // Xavier/Glorot initialization for stable gradients
    const limit = Math.sqrt(6 / (dModel + dModel));
    const initMatrix = () => Array.from({ length: dModel }, () =>
      Array.from({ length: dModel }, () => (seededRandom() * 2 - 1) * limit)
    );

    this.wQ = initMatrix();
    this.wK = initMatrix();
    this.wV = initMatrix();

    const initGrad = () => Array.from({ length: dModel }, () => new Array(dModel).fill(0));
    this.gradW_Q = initGrad();
    this.gradW_K = initGrad();
    this.gradW_V = initGrad();
  }

  public setTrainingMode(mode: boolean): void {
    this.isTraining = mode;
    if (!mode) {
      // Aggressive memory cleanup for inference mode
      this.lastInput = [];
      this.lastQ = [];
      this.lastK = [];
      this.lastV = [];
      this.lastAttnWeights = [];
    }
  }

  public forward(input: number[][]): number[][] {
    // Cache input if training
    if (this.isTraining) {
      // Assuming input is created fresh in previous layer,
      // deep copy might be needed if external mutation is a risk.
      // For now, storing reference is generally safe if upstream layers return new arrays.
      this.lastInput = input;
    }

    // 1. Calculate Queries, Keys, Values
    // Q = X * Wq, K = X * Wk, V = X * Wv
    const Q = multiply(input, this.wQ);
    const K = multiply(input, this.wK);
    const V = multiply(input, this.wV);

    // 2. Calculate Attention Scores
    // Scores = (Q * K^T) / sqrt(dModel)
    const scores = multiply(Q, transpose(K));
    const scaledScores = scale(scores, 1 / Math.sqrt(this.dModel));

    // Apply Causal Mask
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < input.length; j++) {
        if (j > i) {
          scaledScores[i][j] = -1e9; // Mask future tokens
        }
      }
      // 3. Debug check (must be inside the loop where 'i' is defined)
      const rowSum = scaledScores[i].reduce((sum, val) => sum + Math.exp(val), 0);
      if (rowSum === 0 || isNaN(rowSum)) {
        console.error(`DANGER: Softmax breakdown at row ${i}. Check scores initialization.`);
      }
    }


    // 3. Apply Softmax to get probabilities
    const attnWeights = softmax(scaledScores);

    // 4. Calculate final contextualized vectors
    // Output = AttentionWeights * V
    const output = multiply(attnWeights, V);

    // Cache all intermediate mathematical steps for the backward pass
    if (this.isTraining) {
      // Since your math functions create new arrays, we can just store them directly
      this.lastQ = Q;
      this.lastK = K;
      this.lastV = V;
      this.lastAttnWeights = attnWeights;
    }

    return output;
  }

  public updateWeights(learningRate: number): void {
    // Apply accumulated gradients to Q, K, V matrices
    for (let i = 0; i < this.dModel; i++) {
      for (let j = 0; j < this.dModel; j++) {
        this.wQ[i][j] -= learningRate * this.gradW_Q[i][j];
        this.wK[i][j] -= learningRate * this.gradW_K[i][j];
        this.wV[i][j] -= learningRate * this.gradW_V[i][j];

        // Reset gradients for the next batch
        this.gradW_Q[i][j] = 0;
        this.gradW_K[i][j] = 0;
        this.gradW_V[i][j] = 0;
      }
    }
  }

  public backward(gradOutput: number[][]): number[][] {
    const seqLen = this.lastInput.length;
    const dModel = this.dModel;

    // ==========================================
    // STEP 1: Output = AttentionWeights * V
    // ==========================================
    // gradV = A^T * gradOutput
    const gradV = multiply(transpose(this.lastAttnWeights), gradOutput);

    // gradA = gradOutput * V^T
    const gradA = multiply(gradOutput, transpose(this.lastV));

    // ==========================================
    // STEP 2: Softmax Backward
    // ==========================================
    const gradScores: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      gradScores[i] = new Array(seqLen).fill(0);

      const sumA_gradA = dotProduct(this.lastAttnWeights[i], gradA[i]);

      for (let j = 0; j < seqLen; j++) {
        gradScores[i][j] = this.lastAttnWeights[i][j] * (gradA[i][j] - sumA_gradA);
      }
    }

    // ==========================================
    // STEP 3: Scale Backward (1 / sqrt(dModel))
    // ==========================================
    const scaleFactor = 1 / Math.sqrt(dModel);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        gradScores[i][j] *= scaleFactor;
      }
    }

    // ==========================================
    // STEP 4: Scores = Q * K^T
    // ==========================================
    // gradQ = gradScores * K
    const gradQ = multiply(gradScores, this.lastK);

    // gradK = gradScores^T * Q
    const gradK = multiply(transpose(gradScores), this.lastQ);

    // ==========================================
    // STEP 5: Gradients for Weights (W_Q, W_K, W_V)
    // ==========================================
    // dW = Input^T * grad
    const dW_Q = multiply(transpose(this.lastInput), gradQ);
    const dW_K = multiply(transpose(this.lastInput), gradK);
    const dW_V = multiply(transpose(this.lastInput), gradV);

    // Accumulate gradients
    for (let i = 0; i < dModel; i++) {
      for (let j = 0; j < dModel; j++) {
        this.gradW_Q[i][j] += dW_Q[i][j];
        this.gradW_K[i][j] += dW_K[i][j];
        this.gradW_V[i][j] += dW_V[i][j];
      }
    }

    // ==========================================
    // STEP 6: Gradient for the Input (gradInput)
    // ==========================================
    // gradInput = (gradQ * W_Q^T) + (gradK * W_K^T) + (gradV * W_V^T)
    const gIn_Q = multiply(gradQ, transpose(this.wQ));
    const gIn_K = multiply(gradK, transpose(this.wK));
    const gIn_V = multiply(gradV, transpose(this.wV));

    const gradInput = addMatrices(gIn_Q, gIn_K, gIn_V);

    return gradInput;
  }
}
