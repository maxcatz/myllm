import {EmbeddingLayer} from "./embeddingLayer";
import {FFNLayer} from "./ffnLayer";
import {LMHead} from "./lmHead";
import {AttentionLayer} from "./attentionLayer";

export class TransformerModel {
  public embeddings: EmbeddingLayer;
  public attention: AttentionLayer;
  public ffn: FFNLayer;
  public lmHead: LMHead;


  private layers: any[];

  constructor(vocabSize: number, dModel: number) {
    this.embeddings = new EmbeddingLayer(vocabSize, dModel);
    // Transformer block
    this.attention = new AttentionLayer(dModel);
    this.ffn = new FFNLayer(dModel, dModel);
    // End of transformer block
    this.lmHead = new LMHead(dModel, vocabSize);

    this.layers = [this.embeddings, this.attention, this.ffn, this.lmHead];
  }

  public setTrainingMode(isTraining: boolean): void {
    for (const layer of this.layers) {
      if (typeof layer.setTrainingMode === 'function') {
        layer.setTrainingMode(isTraining);
      }
    }
  }

  public forward(indices: number[]): number[][] {
    const embOut = this.embeddings.forward(indices);
    const attnOut = this.attention.forward(embOut);
    const ffnOut = this.ffn.forward(attnOut);
    return this.lmHead.forward(ffnOut);
  }
}
