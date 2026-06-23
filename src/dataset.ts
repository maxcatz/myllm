// ==========================================
// TOY DATASET: Animals (Eat, Drink, Play)
// ==========================================

export class TextDataset {
  public vocab: string[] = [];
  public wordToId: Record<string, number> = {};
  public idToWord: Record<number, string> = {};
  public trainingData: { input: number[], target: number[] }[] = [];

  constructor(private rawSentences: string[]) {
    this.buildVocabulary();
    this.prepareData();
  }

  // Getter for convenient access to vocabulary size
  public get vocabSize(): number {
    return this.vocab.length;
  }

  // Private tokenization method
  private tokenize(sentence: string): string[] {
    return sentence.toLowerCase().split(/\s+/).filter(w => w);
  }

  // 1. Build vocabulary
  private buildVocabulary(): void {
    const uniqueWords = new Set<string>();

    // Collect unique words
    for (const sentence of this.rawSentences) {
      const words = this.tokenize(sentence);
      for (const word of words) {
        uniqueWords.add(word);
      }
    }

    // Initialize vocabulary with <eos> and sort remaining words
    this.vocab = ["<eos>", ...Array.from(uniqueWords).sort()];

    // Fill quick access dictionaries
    this.vocab.forEach((word, index) => {
      this.wordToId[word] = index;
      this.idToWord[index] = word;
    });
  }

  // 2. Prepare training data
  private prepareData(): void {
    for (const sentence of this.rawSentences) {
      const words = this.tokenize(sentence);
      const indices = words.map(word => this.wordToId[word]);

      // Input: exact copy of sentence indices
      const input = [...indices];

      // Target: shift left by 1, with <eos> at the end
      const target = [...indices.slice(1), this.wordToId["<eos>"]];

      this.trainingData.push({ input, target });
    }
  }

  // 3. Public method for encoding user input (Inference)
  public encode(sentence: string): number[] {
    const words = this.tokenize(sentence);
    return words.map(word => {
      const id = this.wordToId[word];
      return id !== undefined ? id : -1; // -1 for unknown words
    });
  }
}
