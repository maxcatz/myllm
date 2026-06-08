export class SimpleTokenizer {
    private vocab: Map<string, number> = new Map();
    private reverseVocab: Map<number, string> = new Map();
    private nextIndex: number = 0;

    train(text: string): void {
        const words = text.toLowerCase().split(/\s+/);
        for (const word of words) {
            if (!this.vocab.has(word)) {
                this.vocab.set(word, this.nextIndex);
                this.reverseVocab.set(this.nextIndex, word);
                this.nextIndex++;
            }
        }
    }

    encode(text: string): number[] {
        return text.toLowerCase().split(/\s+/).map(word => this.vocab.get(word) ?? 0);
    }

    decode(indices: number[]): string {
        return indices.map(idx => this.reverseVocab.get(idx) ?? "<unk>").join(" ");
    }
    
    getVocabSize(): number {
        return this.vocab.size;
    }
}