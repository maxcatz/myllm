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
        const maxVal = Math.max(...row);
        const exps = row.map(x => Math.exp(x - maxVal));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sumExps);
    });
}

// Simple pseudo-random number generator (LCG)
let currentSeed = 42; // Magic number. If you change it to 43, results will change!

export function seededRandom() {
    currentSeed = (currentSeed * 16807) % 2147483647;
    return (currentSeed - 1) / 2147483646;
}


export function addMatrices(...matrices: number[][][]): number[][] {
    if (matrices.length === 0) return [];

    const rows = matrices[0].length;
    const cols = matrices[0][0].length;

    // Создаем пустую матрицу-результат
    const result = Array.from({ length: rows }, () => new Array(cols).fill(0));

    // Проходим по всем переданным матрицам и плюсуем их значения
    for (const matrix of matrices) {
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[i][j] += matrix[i][j];
            }
        }
    }
    return result;
}

export function dotProduct(vec1: number[], vec2: number[]): number {
    let sum = 0;
    const len = vec1.length;
    for (let i = 0; i < len; i++) {
        sum += vec1[i] * vec2[i];
    }

    return sum;
}
