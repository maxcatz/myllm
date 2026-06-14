// ==========================================
// TOY DATASET: Animals (Eat, Drink, Play)
// ==========================================

// 1. Define the exact vocabulary
export const VOCAB = [
  "<eos>",  // 0: End of Sequence (Critical for stopping generation)
  "dog",    // 1
  "cat",    // 2
  "bear",   // 3
  "eats",   // 4
  "drinks", // 5
  "plays",  // 6
  "meat",   // 7
  "fish",   // 8
  "honey",  // 9
  "water",  // 10
  "milk",   // 11
  "ball",    // 12
  "human",  // 13
  "vine",   // 14
  "sleeps", //15
  "well", //16

];

export const VOCAB_SIZE = VOCAB.length;

// 2. Create fast lookup dictionaries
export const wordToId: Record<string, number> = {};
export const idToWord: Record<number, string> = {};



VOCAB.forEach((word, index) => {
  wordToId[word] = index;
  idToWord[index] = word;
});

// 3. Define the raw training sentences
const rawSentences = [
  ["dog", "eats", "meat"],
  ["cat", "eats", "fish"],
  ["bear", "eats", "honey"],
  ["dog", "drinks", "water"],
  ["cat", "drinks", "milk"],
  ["bear", "drinks", "water"],
  ["dog", "plays", "ball"],
  ["cat", "plays", "ball"],
  ["bear", "plays", "ball"],
  ["human", "drinks", "vine"],
  ["dog", "sleeps", "well"]
];


// 4. Prepare the data for the Trainer
// The Trainer needs Input (X) and Target (Y) arrays of indices.
export const trainingData: { input: number[], target: number[] }[] = [];

for (const sentence of rawSentences) {
  // Convert words to their numeric IDs
  const indices = sentence.map(word => wordToId[word]);

  // Example for "dog eats meat" (1, 4, 7):
  // Input is the exact sentence: [1, 4, 7]
  const input = [...indices];

  // Target is shifted by 1 to the left, ending with <eos> (0):
  // Target: [4, 7, 0] ("eats", "meat", "<eos>")
  const target = [...indices.slice(1), wordToId["<eos>"]];

  trainingData.push({ input, target });
}
