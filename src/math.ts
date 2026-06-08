export function multiply(A: number[][], B: number[][]): number[][] {
    const rowsA = A.length, colsA = A[0].length;
    const rowsB = B.length, colsB = B[0].length;
    let result: number[][] = Array.from({ length: rowsA }, () => Array(colsB).fill(0));

    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
            for (let k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

export function scale(matrix: number[][], factor: number): number[][] {
    return matrix.map(row => row.map(val => val * factor));
}

export function transpose(matrix: number[][]): number[][] {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

export function softmax(matrix: number[][]): number[][] {
    return matrix.map(row => {
        const exps = row.map(x => Math.exp(x));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sumExps);
    });
}


/**
 * Calculates error and initial gradient for training
 * @param logits Raw scores from LM Head (array of 6 numbers)
 * @param targetIndex Index of the correct word in the vocabulary
 */
export function computeLossAndGradients(logits: number[], targetIndex: number): { loss: number, gradients: number[] } {
    // 1. Convert logits to probabilities (Softmax)
    const maxLogit = Math.max(...logits); // For computational stability (so exp doesn't explode)
    const exps = logits.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(x => x / sumExps);

    // 2. Calculate error magnitude: Cross-Entropy Loss
    // Take the probability the model gave to the CORRECT word, and take -log of it.
    // If probability is 1.0, logarithm equals 0 (no error). The closer to 0, the greater the error.
    const loss = -Math.log(probs[targetIndex] + 1e-9); // 1e-9 saves from log(0)

    // 3. Magic of initial gradient: grad = P - Y
    const gradients = [...probs];
    // For correct answer Y = 1, for all others Y = 0.
    gradients[targetIndex] -= 1.0;

    return { loss, gradients };
}


/**
 * Calculates Softmax gradient.
 * gradOutput - what came "from above" (error gradient with respect to probabilities)
 * probs - probabilities after softmax
 */
export function backwardSoftmax(gradOutput: number[], probs: number[]): number[] {
    const n = probs.length;
    const gradInput = new Array(n).fill(0);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            // Math: grad = S_i * (delta_ij - S_j)
            const delta = i === j ? 1 : 0;
            gradInput[i] += gradOutput[j]! * (probs[i]! * (delta - probs[j]!));
        }
    }
    return gradInput;
}

// Simple pseudo-random number generator (LCG)
let currentSeed = 42; // Magic number. If you change it to 43, results will change!

export function seededRandom() {
    currentSeed = (currentSeed * 16807) % 2147483647;
    return (currentSeed - 1) / 2147483646;
}
