// =============================================================================
// Neural Network from Scratch in Rust — XOR Learning
// =============================================================================
//
// We train a feedforward neural network to learn the XOR function:
//
//   (0,0) → 0    (0,1) → 1    (1,0) → 1    (1,1) → 0
//
// XOR is the canonical example because it is not linearly separable — no single
// hyperplane in R² can partition the four points into {0} and {1}. By the
// universal approximation theorem, a network with one hidden layer of
// sufficient width can represent any continuous function on a compact set,
// so a single hidden layer suffices here.
//
// Architecture:
//
//   x ∈ R²  →  Hidden layer (4 neurons, sigmoid)  →  Output (1 neuron, sigmoid)
//
// We use 4 hidden neurons (more than the minimum 2) for reliable convergence —
// with only 2 hidden neurons, gradient descent frequently lands in saddle
// points or poor local minima depending on initialization.
//
// =============================================================================

use std::fmt;

// =============================================================================
// SIMPLE PRNG (xorshift64)
// =============================================================================
//
// Neural network training requires random weight initialization to break
// symmetry between neurons. We use a minimal xorshift64 PRNG rather than
// pulling in a crate, since this is a teaching example.
//
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    // Returns a uniform random f64 in [min, max).
    fn uniform(&mut self, min: f64, max: f64) -> f64 {
        // xorshift64 — period 2^64 - 1, sufficient for our purposes.
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        let t = (self.state as f64) / (u64::MAX as f64); // t ∈ [0, 1)
        min + t * (max - min)
    }
}

// =============================================================================
// ACTIVATION FUNCTION: Sigmoid  σ(x) = 1 / (1 + e^{-x})
// =============================================================================
//
// The sigmoid function is a smooth, monotonically increasing map σ: R → (0,1).
//
// Key properties:
//   - σ(0) = 0.5
//   - lim_{x→+∞} σ(x) = 1,   lim_{x→-∞} σ(x) = 0
//   - σ'(x) = σ(x)(1 - σ(x))    (elegant self-referential derivative)
//   - Maximum gradient σ'(0) = 0.25, which causes the "vanishing gradient"
//     problem in deep networks. For our shallow 1-hidden-layer net, this is fine.
//
// We use sigmoid here for clarity. Modern deep networks prefer ReLU(x) = max(0,x)
// because its gradient is either 0 or 1 — no vanishing.
//
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// If y = σ(x), then σ'(x) = y(1 - y).
// We exploit this in backprop: we already computed y in the forward pass,
// so the derivative costs just one multiply and one subtract.
fn sigmoid_derivative(y: f64) -> f64 {
    y * (1.0 - y)
}

// =============================================================================
// NETWORK STRUCTURE
// =============================================================================
//
// A feedforward network computes a function f: R^n → R^m parameterized by
// weights W and biases b. For our two-layer network:
//
//   h = σ(W₁x + b₁)        ← hidden layer:  W₁ ∈ R^{4×2}, b₁ ∈ R^4
//   ŷ = σ(w₂ᵀh + b₂)       ← output layer:  w₂ ∈ R^4,    b₂ ∈ R
//
// The full parameter vector θ = (W₁, b₁, w₂, b₂) lives in R^{17}
// (4×2 + 4 + 4 + 1 = 17 parameters).
//
const HIDDEN_SIZE: usize = 4;
const INPUT_SIZE: usize = 2;

struct NeuralNetwork {
    // W₁ ∈ R^{HIDDEN×INPUT} — each row is the weight vector for one hidden neuron.
    weights_ih: [[f64; INPUT_SIZE]; HIDDEN_SIZE],

    // b₁ ∈ R^{HIDDEN}
    biases_h: [f64; HIDDEN_SIZE],

    // w₂ ∈ R^{HIDDEN} — weight vector from hidden layer to the single output.
    weights_ho: [f64; HIDDEN_SIZE],

    // b₂ ∈ R
    bias_o: f64,
}

impl NeuralNetwork {
    // Xavier/Glorot initialization: sample from Uniform(-√(6/(fan_in+fan_out)), +√(...))
    // This keeps the variance of activations roughly constant across layers,
    // preventing signals from exploding or vanishing before training even begins.
    fn new(rng: &mut Rng) -> Self {
        let limit_ih = (6.0 / (INPUT_SIZE + HIDDEN_SIZE) as f64).sqrt();
        let limit_ho = (6.0 / (HIDDEN_SIZE + 1) as f64).sqrt();

        let mut weights_ih = [[0.0; INPUT_SIZE]; HIDDEN_SIZE];
        let mut biases_h = [0.0; HIDDEN_SIZE];
        let mut weights_ho = [0.0; HIDDEN_SIZE];

        for h in 0..HIDDEN_SIZE {
            for i in 0..INPUT_SIZE {
                weights_ih[h][i] = rng.uniform(-limit_ih, limit_ih);
            }
            biases_h[h] = rng.uniform(-limit_ih, limit_ih);
            weights_ho[h] = rng.uniform(-limit_ho, limit_ho);
        }

        NeuralNetwork {
            weights_ih,
            biases_h,
            weights_ho,
            bias_o: rng.uniform(-limit_ho, limit_ho),
        }
    }

    // =========================================================================
    // FORWARD PASS
    // =========================================================================
    //
    // Evaluates the network: x ↦ ŷ
    //
    // For each hidden neuron j:
    //   zⱼ = Σᵢ W₁[j][i] · xᵢ + b₁[j]       (affine transformation)
    //   hⱼ = σ(zⱼ)                              (nonlinear activation)
    //
    // For the output:
    //   z_out = Σⱼ w₂[j] · hⱼ + b₂
    //   ŷ     = σ(z_out)
    //
    // We cache h (hidden activations) because backprop needs them.
    //
    fn forward(&self, x: [f64; INPUT_SIZE]) -> ForwardResult {
        let mut hidden = [0.0; HIDDEN_SIZE];

        for j in 0..HIDDEN_SIZE {
            let mut z = self.biases_h[j];
            for i in 0..INPUT_SIZE {
                z += self.weights_ih[j][i] * x[i];
            }
            hidden[j] = sigmoid(z);
        }

        let mut z_out = self.bias_o;
        for j in 0..HIDDEN_SIZE {
            z_out += self.weights_ho[j] * hidden[j];
        }
        let output = sigmoid(z_out);

        ForwardResult { hidden, output }
    }

    // =========================================================================
    // BACKPROPAGATION + GRADIENT DESCENT
    // =========================================================================
    //
    // Loss function: L(θ) = ½(y - ŷ)²  (mean squared error, factor of ½ for
    // clean derivatives).
    //
    // We want ∂L/∂θ for every parameter θ, then update: θ ← θ - η · ∂L/∂θ
    // where η is the learning rate.
    //
    // --- Derivation of gradients (chain rule) ---
    //
    // Let e = y - ŷ  (the signed error).
    //
    // OUTPUT LAYER:
    //   ∂L/∂ŷ  = -(y - ŷ) = -e
    //   ∂ŷ/∂z_out = σ'(z_out) = ŷ(1 - ŷ)
    //
    //   Define δ_out = -∂L/∂z_out = e · ŷ(1 - ŷ)
    //   (We flip the sign so the update becomes θ += η·δ·input, i.e., we
    //    ascend the negative-loss surface = descend the loss surface.)
    //
    //   ∂L/∂w₂[j] = -δ_out · hⱼ       →  Δw₂[j] = η · δ_out · hⱼ
    //   ∂L/∂b₂    = -δ_out             →  Δb₂    = η · δ_out
    //
    // HIDDEN LAYER (for neuron j):
    //   The error signal propagated back through the output weight:
    //     ε_j = δ_out · w₂[j]
    //
    //   δ_h[j] = ε_j · σ'(zⱼ) = ε_j · hⱼ(1 - hⱼ)
    //
    //   ∂L/∂W₁[j][i] = -δ_h[j] · xᵢ  →  ΔW₁[j][i] = η · δ_h[j] · xᵢ
    //   ∂L/∂b₁[j]    = -δ_h[j]        →  Δb₁[j]     = η · δ_h[j]
    //
    // This is the complete gradient for all 17 parameters, computed in O(n)
    // where n = number of parameters. The same pattern generalizes to any
    // number of layers — you just keep propagating δ backwards.
    //
    fn train(&mut self, x: [f64; INPUT_SIZE], y: f64, lr: f64) -> f64 {
        // --- Forward pass ---
        let fwd = self.forward(x);
        let predicted = fwd.output;
        let hidden = fwd.hidden;

        let error = y - predicted;

        // --- Output layer delta ---
        let delta_out = error * sigmoid_derivative(predicted);

        // --- Hidden layer deltas ---
        let mut delta_h = [0.0; HIDDEN_SIZE];
        for j in 0..HIDDEN_SIZE {
            let backprop_error = delta_out * self.weights_ho[j];
            delta_h[j] = backprop_error * sigmoid_derivative(hidden[j]);
        }

        // --- Gradient descent updates ---

        // w₂ ← w₂ + η · δ_out · h
        for j in 0..HIDDEN_SIZE {
            self.weights_ho[j] += lr * delta_out * hidden[j];
        }
        self.bias_o += lr * delta_out;

        // W₁ ← W₁ + η · δ_h · xᵀ
        for j in 0..HIDDEN_SIZE {
            for i in 0..INPUT_SIZE {
                self.weights_ih[j][i] += lr * delta_h[j] * x[i];
            }
            self.biases_h[j] += lr * delta_h[j];
        }

        // Return ½(y - ŷ)² for loss tracking.
        0.5 * error * error
    }
}

struct ForwardResult {
    hidden: [f64; HIDDEN_SIZE],
    output: f64,
}

impl fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  W₁ (input→hidden):")?;
        for (j, row) in self.weights_ih.iter().enumerate() {
            writeln!(f, "    neuron {j}: [{:.4}, {:.4}]", row[0], row[1])?;
        }
        writeln!(f, "  b₁: [{:.4}, {:.4}, {:.4}, {:.4}]",
            self.biases_h[0], self.biases_h[1], self.biases_h[2], self.biases_h[3])?;
        writeln!(f, "  w₂ (hidden→output): [{:.4}, {:.4}, {:.4}, {:.4}]",
            self.weights_ho[0], self.weights_ho[1], self.weights_ho[2], self.weights_ho[3])?;
        writeln!(f, "  b₂: {:.4}", self.bias_o)
    }
}

// =============================================================================
// MAIN
// =============================================================================
fn main() {
    let mut rng = Rng::new(42);
    let mut nn = NeuralNetwork::new(&mut rng);

    // XOR truth table
    let data: [([f64; 2], f64); 4] = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    // η = 2.0 is aggressive but effective for this toy problem.
    // In production you'd use adaptive methods like Adam (which maintains
    // per-parameter learning rates using 1st and 2nd moment estimates).
    let lr = 2.0;
    let epochs = 10_000;

    println!("=== Neural Network from Scratch: Learning XOR ===\n");
    println!("Architecture: 2 → 4 (sigmoid) → 1 (sigmoid)");
    println!("Parameters:   17  (4×2 + 4 + 4 + 1)");
    println!("Loss:         L(θ) = ½ Σ (yᵢ - ŷᵢ)²");
    println!("Optimizer:    SGD, η = {lr}");
    println!("Init:         Xavier/Glorot uniform\n");

    println!("Network before training:");
    println!("{nn}");

    // =========================================================================
    // TRAINING LOOP
    // =========================================================================
    //
    // Each epoch presents all 4 training examples (online/stochastic SGD).
    //
    // In mini-batch SGD (used in practice), you'd accumulate gradients over a
    // batch before updating. Here with only 4 examples, we update after each
    // one — the stochastic noise actually helps escape shallow local minima.
    //
    println!("Training...\n");
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for &(x, y) in &data {
            total_loss += nn.train(x, y, lr);
        }

        if epoch % 2000 == 0 || epoch == epochs - 1 {
            println!("  epoch {:>5} | loss = {:.8}", epoch, total_loss);
        }
    }

    // =========================================================================
    // EVALUATION
    // =========================================================================
    println!("\n--- Trained network ---\n");
    println!("{nn}");

    println!("  x₁  x₂  │ y (true)  ŷ (predicted)  round(ŷ)");
    println!(" ──────────┼──────────────────────────────────── ");
    for &(x, y) in &data {
        let pred = nn.forward(x).output;
        println!(
            "   {}   {}  │    {}       {:.6}         {}",
            x[0] as u8,
            x[1] as u8,
            y as u8,
            pred,
            if pred > 0.5 { 1 } else { 0 }
        );
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n=== The algorithm in one paragraph ===\n");
    println!("We define a parameterized function f_θ(x) (the network), a loss");
    println!("L(θ) = ½Σ(y - f_θ(x))² measuring prediction error, and compute");
    println!("∇_θ L via the chain rule (backpropagation). Then we iterate");
    println!("θ ← θ - η∇_θ L (gradient descent) until L is small. That's it.");
    println!("Everything in modern ML — transformers, diffusion models, RL —");
    println!("is variations on this theme with different architectures and");
    println!("loss functions.");
}
