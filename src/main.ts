import * as readline from 'readline';
import { TransformerModel } from "./transformerModel";
import { Trainer } from "./trainer";
import { TextDataset} from "./dataset";
import {softmax} from "./math";

// Initialize system
const D_MODEL = 16;
const LEARNING_RATE = 0.005;
const EPOCHS = 5000;
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

// Создаем экземпляр датасета
const dataset = new TextDataset(rawSentences);

const model = new TransformerModel(dataset.vocabSize, D_MODEL);
const trainer = new Trainer(model, LEARNING_RATE);

async function runInteractiveMode() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  console.log("\n====================================");
  console.log("🤖 Interactive Model Testing Ready");
  console.log("Enter phrases like: 'dog eats'");
  console.log("Type 'exit' to quit.");
  console.log("====================================\n");

  // Helper: translate words to IDs
  const encode = (phrase: string) => phrase.split(' ').map(w => dataset.wordToId[w] ?? -1);

  const ask = () => {
    rl.question('Phrase: ', (input) => {
      if (input.toLowerCase() === 'exit') {
        rl.close();
        return;
      }

      const indices = encode(input);
      if (indices.includes(-1)) {
        console.log("⚠️ Unknown words. Try: dog, cat, bear, eats, drinks, plays, meat, fish, honey, water, milk, ball");
      } else {
        // Set model to inference mode
        model.setTrainingMode(false);
        const logits = model.forward(indices);

        const proc = softmax(logits);
        indices.forEach((inputIdx, step) => {
          const inputWord = dataset.idToWord[inputIdx];

          // Получаем топ-5 для текущего шага
          const top5 = proc[step]
            .map((prob, i) => ({ word: dataset.idToWord[i], prob }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 5)
            .map(p => `${p.word} (${(p.prob * 100).toFixed(1)}%)`)
            .join(', ');

          console.log(`${inputWord} -> ${top5}`);
        });
        // Get prediction for the last token
        const lastTokenLogits = logits[logits.length - 1];
        const bestIdx = lastTokenLogits.indexOf(Math.max(...lastTokenLogits));

        console.log(`Prediction: ${dataset.idToWord[bestIdx]}`);
      }
      ask();
    });
  };

  ask();
}

// Execution flow
async function main() {
  console.log("Starting training...");
  trainer.train(dataset.trainingData, EPOCHS);
  console.log("Training complete!\n");

  await runInteractiveMode();
}

main().catch(console.error);
