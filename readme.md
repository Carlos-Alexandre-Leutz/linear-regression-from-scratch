# Linear Regression from Scratch (JS)

This project implements the **Multiple Linear Regression** algorithm using only native JavaScript (*Vanilla JS*), without the aid of external scientific computing or linear algebra libraries.

The main goal is to demonstrate a deep understanding of the mathematical foundations behind machine learning, specifically through the implementation of the **Normal Equation**.

---

## 🚀 Live Demo
You can test the algorithm and interact with the model in real-time here:
**[https://SEU_USUARIO.github.io/NOME-DO-REPOSITORIO/](https://SEU_USUARIO.github.io/NOME-DO-REPOSITORIO/)**

---

## 🛠️ About the Project

Instead of using a "black box" (like `scikit-learn` or `TensorFlow`), this code manually performs:
* **Matrix Manipulation:** Data structuring for matrix calculations.
* **Normal Equation:** Implementation of the formula $\theta = (X^T X)^{-1} X^T y$.
* **Linear Algebra:** Implementation of matrix transposition, multiplication, and inversion (via Gauss-Jordan elimination).
* **Bias Trick:** Insertion of a column of ones to automatically calculate the intercept ($b$).



## 🛠️ Technologies Used
* **JavaScript (ES6+)**
* **Applied Mathematics:** Linear algebra applied to regression problems.

## 📊 The Model
The model was trained with a simulated dataset of 100 properties. The algorithm learned the relationship between two independent variables ($x_1$ = Square Footage, $x_2$ = Rooms) to predict a dependent variable ($y$ = Price).

The final predictive equation follows the pattern:
$$Price = (w_1 \cdot SquareFootage) + (w_2 \cdot Rooms) + b$$

## 🧠 Key Takeaways
This project helped consolidate essential skills for any data engineer or AI-focused developer:
* **Mathematical Insight:** Understanding how linear algebra solves prediction problems.
* **Algorithm Implementation:** Mastering the computational logic behind matrix operations without relying on external abstractions.

---
*Developed for technical demonstration and portfolio purposes.*
