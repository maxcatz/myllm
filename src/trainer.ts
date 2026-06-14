import { TransformerModel } from "./transformerModel";

export class Trainer {
  constructor(private model: TransformerModel, private learningRate: number) {}

  // Main training loop
  public train(data: { input: number[], target: number[] }[], epochs: number): void {
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;

      for (const { input, target } of data) {
        // 1. Forward pass: Get logits [SeqLen][VocabSize]
        const logits = this.model.forward(input);

        // 2. Calculate Loss (Cross-Entropy) & initial Gradients
        const { loss, gradOutput } = this.calculateLossAndGradient(logits, target);
        totalLoss += loss;

        // 3. Backward pass: Propagate error through the network
        // We propagate the gradient back through all layers
        let grad = this.model.lmHead.backward(gradOutput);
        grad = this.model.ffn.backward(grad);
        grad = this.model.attention.backward(grad);
        this.model.embeddings.backward(grad);

        // 4. Update all weights
        this.model.lmHead.updateWeights(this.learningRate);
        this.model.ffn.updateWeights(this.learningRate);
        this.model.attention.updateWeights(this.learningRate);
        this.model.embeddings.updateWeights(this.learningRate);
      }

      if (epoch % 10 === 0) {
        console.log(`Epoch ${epoch}, Loss: ${totalLoss.toFixed(4)}`);
      }
    }
  }

  // Calculates Cross-Entropy Loss and the gradient for the output layer
  private calculateLossAndGradient(logits: number[][], target: number[]): { loss: number, gradOutput: number[][] } {
    const seqLen = logits.length;
    const vocabSize = logits[0].length;
    const gradOutput: number[][] = [];
    let loss = 0;

    for (let t = 0; t < seqLen; t++) {
      // Apply Softmax to get probabilities
      const probs = this.softmax(logits[t]);
      const targetId = target[t];

      // Loss = -log(prob[target])
      loss += -Math.log(probs[targetId] + 1e-9);

      // Gradient for Cross-Entropy + Softmax is simply (probs - target)
      gradOutput[t] = [...probs];
      gradOutput[t][targetId] -= 1;
    }

    return { loss, gradOutput };
  }

  private softmax(arr: number[]): number[] {
    const maxVal = Math.max(...arr); // For numerical stability
    const exp = arr.map(x => Math.exp(x - maxVal));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
  }
}
