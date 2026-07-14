// 1. Original data (100 houses)
const dataX = [
  [35, 1],
  [38, 1],
  [40, 1],
  [42, 1],
  [45, 1],
  [48, 1],
  [50, 1],
  [52, 1],
  [55, 1],
  [58, 1],
  [45, 2],
  [48, 2],
  [50, 2],
  [52, 2],
  [55, 2],
  [58, 2],
  [60, 2],
  [62, 2],
  [65, 2],
  [68, 2],
  [70, 2],
  [72, 2],
  [75, 2],
  [78, 2],
  [80, 2],
  [82, 2],
  [85, 2],
  [88, 2],
  [90, 2],
  [95, 2],
  [65, 3],
  [68, 3],
  [70, 3],
  [72, 3],
  [75, 3],
  [78, 3],
  [80, 3],
  [82, 3],
  [85, 3],
  [88, 3],
  [90, 3],
  [92, 3],
  [95, 3],
  [98, 3],
  [100, 3],
  [105, 3],
  [110, 3],
  [115, 3],
  [120, 3],
  [125, 3],
  [130, 3],
  [135, 3],
  [140, 3],
  [145, 3],
  [150, 3],
  [155, 3],
  [160, 3],
  [165, 3],
  [170, 3],
  [185, 3],
  [100, 4],
  [105, 4],
  [110, 4],
  [115, 4],
  [120, 4],
  [125, 4],
  [130, 4],
  [135, 4],
  [140, 4],
  [145, 4],
  [150, 4],
  [155, 4],
  [160, 4],
  [165, 4],
  [170, 4],
  [175, 4],
  [180, 4],
  [185, 4],
  [190, 4],
  [195, 4],
  [200, 4],
  [210, 4],
  [220, 4],
  [230, 4],
  [240, 4],
  [250, 4],
  [260, 4],
  [270, 4],
  [280, 4],
  [290, 4],
  [220, 5],
  [235, 5],
  [250, 5],
  [265, 5],
  [280, 5],
  [300, 5],
  [320, 5],
  [340, 5],
  [360, 5],
  [400, 5],
];

const y = [
  165, 172, 180, 185, 198, 205, 212, 220, 231, 240, 215, 224, 233, 241, 252,
  261, 270, 277, 289, 300, 308, 314, 326, 337, 345, 352, 364, 375, 383, 402,
  299, 310, 318, 325, 337, 348, 356, 363, 375, 386, 394, 401, 413, 424, 432,
  451, 470, 489, 508, 527, 546, 565, 584, 603, 622, 641, 660, 679, 698, 755,
  449, 468, 487, 506, 525, 544, 563, 582, 601, 620, 639, 658, 677, 696, 715,
  734, 753, 772, 791, 810, 831, 869, 907, 945, 983, 1021, 1059, 1097, 1135,
  1173, 940, 997, 1054, 1111, 1168, 1244, 1320, 1396, 1472, 1624,
];

// ========================================================
// BIAS TRICK: Insert column of '1' into matrix X
// Now each row becomes: [Square Footage, Rooms, 1]
// ========================================================
const X = dataX.map((row) => [...row, 1]);

console.log("--- Executing Normal Equation with Bias (Vanilla JS) ---\n");

// Linear Algebra Helper Functions for Dynamic Matrices
function transpose(matrix) {
  return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
}

function multiply(A, B) {
  const isColumn = !Array.isArray(B[0]);
  const matrixB = isColumn ? B.map((val) => [val]) : B;

  const rowsA = A.length,
    colsA = A[0].length;
  const colsB = matrixB[0].length;
  const result = Array(rowsA)
    .fill(0)
    .map(() => Array(colsB).fill(0));

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += A[i][k] * matrixB[k][j];
      }
      result[i][j] = sum;
    }
  }
  return isColumn ? result.map((row) => row[0]) : result;
}

// Inversion using Gauss-Jordan Elimination (necessary for dynamic 3x3 matrices)
function invertMatrix(matrix) {
  const n = matrix.length;
  const identity = Array(n)
    .fill(0)
    .map((_, i) =>
      Array(n)
        .fill(0)
        .map((_, j) => (i === j ? 1 : 0)),
    );
  const copy = matrix.map((row) => [...row]);

  for (let i = 0; i < n; i++) {
    let pivot = copy[i][i];
    if (pivot === 0) return null; // Matrix not invertible

    for (let j = 0; j < n; j++) {
      copy[i][j] /= pivot;
      identity[i][j] /= pivot;
    }

    for (let k = 0; k < n; k++) {
      if (k !== i) {
        let factor = copy[k][i];
        for (let j = 0; j < n; j++) {
          copy[k][j] -= factor * copy[i][j];
          identity[k][j] -= factor * identity[i][j];
        }
      }
    }
  }
  return identity;
}

// ==========================================
// APPLYING THE FORMULA: w = (X^T * X)^-1 * X^T * y
// ==========================================

// STEP 2: X^T
const XT = transpose(X);

// STEP 3: X^T * X
const XTX = multiply(XT, X);

// STEP 4: (X^T * X)^-1
const inverse = invertMatrix(XTX);

// STEP 5: X^T * y
const XTy = multiply(XT, y);

// STEP 6: Final weight results
const weights = multiply(inverse, XTy);

const w1 = weights[0]; // Weight of Square Footage
const w2 = weights[1]; // Weight of Rooms
const b = weights[2]; // The BIAS (Intercept) calculated automatically!

console.log("TRAINING RESULT (100 Houses):");
console.log(`w1 (Square Foot): R$ ${w1.toFixed(2)} thousand`);
console.log(`w2 (Per Room):    R$ ${w2.toFixed(2)} thousand`);
console.log(`b  (Bias/Fixed):  R$ ${b.toFixed(2)} thousand\n`);

console.log(
  `New Predictive Equation: Price = (${w1.toFixed(2)} * Square Footage) + (${w2.toFixed(2)} * Rooms) + (${b.toFixed(2)})`,
);
console.log("-------------------------------------------\n");

// --- TESTING WITH A PREDICTION ---
const newFootage = 50;
const newRooms = 2;

// UPDATED FORMULA: f(x) = w1*x1 + w2*x2 + b
const estimatedPrice = w1 * newFootage + w2 * newRooms + b;

console.log(
  `Real prediction for a house with ${newFootage}m² and ${newRooms} rooms:`,
);
console.log(
  `Estimated price: R$ ${(estimatedPrice * 1000).toLocaleString("pt-BR")}`,
);

function calculateWeights(dataX, y) {
  const X = dataX.map((row) => [...row, 1]);
  const XT = transpose(X);
  const XTX = multiply(XT, X);
  const inverse = invertMatrix(XTX);
  const XTy = multiply(XT, y);
  return multiply(inverse, XTy);
}

function predict() {
  const squareFootage = parseFloat(
    document.getElementById("inputMetros").value,
  );
  const rooms = parseFloat(document.getElementById("inputQuartos").value);

  if (isNaN(squareFootage) || isNaN(rooms)) {
    document.getElementById("resultado").innerText =
      "Please enter valid values.";
    return;
  }

  // Calculates the price using the global variables from your script
  const estimatedPrice = w1 * squareFootage + w2 * rooms + b;

  document.getElementById("resultado").innerHTML =
    `<strong>Estimated Price:</strong><br>R$ ${(estimatedPrice * 1000).toLocaleString("pt-BR", { minimumFractionDigits: 2 })}`;
}
window.onload = () => {
  document.getElementById("paramsDisplay").innerHTML =
    `<strong>Model Parameters (Calculated):</strong><br>` +
    `w1 (Square Footage): ${w1.toFixed(2)}<br>` +
    `w2 (Rooms): ${w2.toFixed(2)}<br>` +
    `b (Bias/Intercept): ${b.toFixed(2)}`;
};
