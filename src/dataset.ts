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

  // Геттер для удобного получения размера словаря
  public get vocabSize(): number {
    return this.vocab.length;
  }

  // Приватный метод токенизации
  private tokenize(sentence: string): string[] {
    return sentence.toLowerCase().split(/\s+/).filter(w => w);
  }

  // 1. Построение словаря
  private buildVocabulary(): void {
    const uniqueWords = new Set<string>();

    // Собираем уникальные слова
    for (const sentence of this.rawSentences) {
      const words = this.tokenize(sentence);
      for (const word of words) {
        uniqueWords.add(word);
      }
    }

    // Инициализируем словарь с <eos> и сортируем остальные слова
    this.vocab = ["<eos>", ...Array.from(uniqueWords).sort()];

    // Заполняем словари быстрого доступа
    this.vocab.forEach((word, index) => {
      this.wordToId[word] = index;
      this.idToWord[index] = word;
    });
  }

  // 2. Подготовка данных для обучения
  private prepareData(): void {
    for (const sentence of this.rawSentences) {
      const words = this.tokenize(sentence);
      const indices = words.map(word => this.wordToId[word]);

      // Input: точная копия индексов предложения
      const input = [...indices];

      // Target: сдвиг на 1 влево, в конце <eos>
      const target = [...indices.slice(1), this.wordToId["<eos>"]];

      this.trainingData.push({ input, target });
    }
  }

  // 3. Публичный метод для кодирования пользовательского ввода (Inference)
  public encode(sentence: string): number[] {
    const words = this.tokenize(sentence);
    return words.map(word => {
      const id = this.wordToId[word];
      return id !== undefined ? id : -1; // -1 для неизвестных слов
    });
  }
}

// ==========================================
// ИСПОЛЬЗОВАНИЕ
// ==========================================



// Теперь можно использовать:
// dataset.vocabSize
// dataset.trainingData
// dataset.encode("bear eats honey")
// dataset.idToWord[3]
