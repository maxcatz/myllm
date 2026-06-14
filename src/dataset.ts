// ==========================================
// TOY DATASET: Animals (Eat, Drink, Play)
// ==========================================

// 1. Define the raw training sentences as strings
const rawSentences = [
  "dog eats meat",
  "cat eats fish",
  "bear eats honey",
  "dog drinks water",
  "cat drinks milk",
  "bear drinks water",
  "dog plays ball",
  "cat plays ball",
  "bear plays ball",
  "human drinks vine",
  "dog sleeps well"
];

// Helper function to tokenize a sentence
function tokenize(sentence: string): string[] {
  return sentence.toLowerCase().split(/\s+/).filter(w => w);
}

// 2. Build vocabulary dynamically from rawSentences
function buildVocabulary(sentences: string[]): string[] {
  const uniqueWords = new Set<string>();
  
  // Collect all unique words from sentences
  for (const sentence of sentences) {
    const words = tokenize(sentence);
    for (const word of words) {
      uniqueWords.add(word);
    }
  }
  
  // Start with special tokens, then add sorted unique words
  return ["<eos>", ...Array.from(uniqueWords).sort()];
}

// Build the vocabulary
export const VOCAB = buildVocabulary(rawSentences);
export const VOCAB_SIZE = VOCAB.length;

// 3. Create fast lookup dictionaries
export const wordToId: Record<string, number> = {};
export const idToWord: Record<number, string> = {};

VOCAB.forEach((word, index) => {
  wordToId[word] = index;
  idToWord[index] = word;
});

// 4. Prepare the data for the Trainer
// The Trainer needs Input (X) and Target (Y) arrays of indices.
export const trainingData: { input: number[], target: number[] }[] = [];

for (const sentence of rawSentences) {
  // Tokenize the sentence into words
  const words = tokenize(sentence);
  
  // Convert words to their numeric IDs
  const indices = words.map(word => wordToId[word]);

  // Example for "dog eats meat" (1, 4, 7):
  // Input is the exact sentence: [1, 4, 7]
  const input = [...indices];

  // Target is shifted by 1 to the left, ending with <eos> (0):
  // Target: [4, 7, 0] ("eats", "meat", "<eos>")
  const target = [...indices.slice(1), wordToId["<eos>"]];

  trainingData.push({ input, target });
}
