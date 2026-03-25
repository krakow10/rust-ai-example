// =============================================================================
// Same XOR Network, but using the Burn framework
// =============================================================================
//
// Compare this to main.rs (the from-scratch version). Everything we manually
// coded — forward pass, backprop, gradient descent — Burn handles for us.
//
// The Python/PyTorch equivalent would be:
//
//   class XorNet(nn.Module):
//       def __init__(self):
//           super().__init__()
//           self.linear1 = nn.Linear(2, 4)
//           self.linear2 = nn.Linear(4, 1)
//           self.sigmoid = nn.Sigmoid()
//
//       def forward(self, x):
//           x = self.sigmoid(self.linear1(x))
//           return self.sigmoid(self.linear2(x))
//
// You'll see the Burn version is structurally almost identical.
//
// =============================================================================

use burn::prelude::*;                          // Tensor, Backend, Module, Config
use burn::backend::{Autodiff, Cuda};        // CPU backend + automatic differentiation
use burn::module::AutodiffModule;              // Provides .valid() for inference mode
use burn::nn::{Linear, LinearConfig, Relu, Sigmoid}; // Layer types
use burn::nn::loss::{MseLoss, Reduction};      // Loss function
use burn::optim::{Optimizer, SgdConfig, GradientsParams}; // Optimizer trait + SGD config

// =============================================================================
// BACKEND SELECTION
// =============================================================================
//
// Burn separates the "what" (your model) from the "where" (the backend).
// The same model code can run on CPU, GPU (CUDA/Vulkan/Metal), or even WASM.
//
// NdArray  = pure-Rust CPU backend (no external dependencies, great for learning)
// Autodiff = wrapper that adds automatic differentiation (computes gradients for us)
//
// In Python terms: NdArray ≈ numpy, Autodiff ≈ torch.autograd
//
type TrainingBackend = Autodiff<Cuda<f32>>;
// When we want to run inference without tracking gradients:
type InferenceBackend = Cuda<f32>;

// =============================================================================
// MODEL DEFINITION
// =============================================================================
//
// #[derive(Module)] is Burn's version of PyTorch's nn.Module.
// It auto-generates code to:
//   - Collect all parameters (weights/biases) for the optimizer
//   - Move the model between devices
//   - Switch between training and inference modes
//
// The generic <B: Backend> means this model works with ANY backend.
// This is a Rust pattern called "monomorphization" — the compiler generates
// specialized code for each backend you use, with zero runtime overhead.
//
#[derive(Module, Debug)]
struct XorNet<B: Backend> {
    // Linear layer: y = Wx + b
    // linear1: R² → R⁴  (2 inputs, 4 hidden neurons — same as our from-scratch version)
    linear1: Linear<B>,
    // linear2: R⁴ → R¹  (4 hidden neurons, 1 output)
    linear2: Linear<B>,
    // Sigmoid activation (stateless — no parameters)
    activation: Relu,
    sigmoid: Sigmoid,
}

// In Burn, model construction uses the Config pattern.
// This separates hyperparameters (hidden_size) from runtime state (weights).
//
// Python equivalent:
//   config = {"hidden_size": 4}
//   model = XorNet(**config)
//
#[derive(Config, Debug)]
struct XorNetConfig {
    #[config(default = 4)]
    hidden_size: usize,
}

impl XorNetConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> XorNet<B> {
        XorNet {
            // LinearConfig::new(in_features, out_features) — just like nn.Linear(2, 4)
            // .init(device) allocates the weight tensors on the specified device.
            // Weights are initialized with Glorot uniform by default (same as our
            // hand-rolled Xavier init in main.rs).
            linear1: LinearConfig::new(2, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, 1).init(device),
            activation: Relu::new(),
            sigmoid: Sigmoid::new(),
        }
    }
}

impl<B: Backend> XorNet<B> {
    // =========================================================================
    // FORWARD PASS
    // =========================================================================
    //
    // Compare to from-scratch version:
    //
    //   FROM SCRATCH:                          BURN:
    //   z = Σ W[j][i] * x[i] + b[j]          x = self.linear1.forward(x)
    //   h = sigmoid(z)                         x = self.sigmoid.forward(x)
    //   z_out = Σ w₂[j] * h[j] + b₂          x = self.linear2.forward(x)
    //   ŷ = sigmoid(z_out)                     x = self.sigmoid.forward(x)
    //
    // The framework handles the matrix multiplication, bias addition, and
    // batching automatically. We just describe the data flow.
    //
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Tensor<B, 2> means a 2D tensor on backend B.
        // The "2" is a compile-time dimension count (like numpy ndarray's ndim).
        // Shape here: [batch_size, 2] → [batch_size, 4] → [batch_size, 1]

        let x = self.linear1.forward(x);   // Affine: [batch, 2] → [batch, 4]
        let x = self.activation.forward(x);    // Elementwise σ
        let x = self.linear2.forward(x);    // Affine: [batch, 4] → [batch, 1]
        self.sigmoid.forward(x)        // Final σ → outputs in (0, 1)
    }
}

// =============================================================================
// MAIN — Training Loop
// =============================================================================
fn main() {
    let device = Default::default(); // CPU device for NdArray backend

    // Initialize model with random weights.
    let mut model: XorNet<TrainingBackend> = XorNetConfig::new().init(&device);

    // SGD optimizer — same algorithm we hand-coded in main.rs.
    //
    // In the from-scratch version we did:
    //   weight += lr * delta * input
    //
    // Burn's SGD does the same thing but handles all parameters automatically.
    let mut optim = SgdConfig::new().init();

    let lr = 0.5;       // Lower than from-scratch (0.5 vs 2.0) because Burn's
                         // MSE averages over the batch, changing the gradient scale.
    let epochs = 10_000;

    // =========================================================================
    // PREPARE DATA
    // =========================================================================
    //
    // In PyTorch you'd write:
    //   inputs = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    //   targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    //
    // Burn is nearly identical — from_floats creates a tensor from a 2D array.
    // Shape: [4, 2] — 4 samples, 2 features each.
    //
    let inputs = Tensor::<TrainingBackend, 2>::from_floats(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        &device,
    );
    // Shape: [4, 1] — 4 samples, 1 target each.
    let targets = Tensor::<TrainingBackend, 2>::from_floats(
        [[0.0], [1.0], [1.0], [0.0]],
        &device,
    );

    println!("=== XOR with Burn Framework ===\n");
    println!("Architecture: 2 → 4 (sigmoid) → 1 (sigmoid)");
    println!("Optimizer:    SGD, η = {lr}");
    println!("Backend:      NdArray (pure Rust CPU)\n");
    println!("Training...\n");

    // =========================================================================
    // THE TRAINING LOOP
    // =========================================================================
    //
    // This is the standard ML training loop, identical in structure to PyTorch:
    //
    //   PYTORCH:                                BURN:
    //   pred = model(inputs)                    pred = model.forward(inputs)
    //   loss = F.mse_loss(pred, targets)        loss = MseLoss::new().forward(...)
    //   loss.backward()                         grads = loss.backward()
    //   optimizer.step()                        model = optim.step(lr, model, grads)
    //   optimizer.zero_grad()                   (automatic — no zero_grad needed!)
    //
    // Key difference from from-scratch:
    //   - We NEVER compute derivatives manually
    //   - loss.backward() uses automatic differentiation (autograd) to compute
    //     ∂L/∂θ for ALL parameters in one call — the same chain-rule math we
    //     did by hand, but the framework builds and traverses the computation
    //     graph for us.
    //
    for epoch in 0..epochs {
        // Forward pass — compute predictions for ALL 4 inputs at once (batched).
        // In the from-scratch version we looped over examples one at a time.
        let predictions = model.forward(inputs.clone());

        // Compute MSE loss: L = (1/n) Σ (yᵢ - ŷᵢ)²
        let loss = MseLoss::new().forward(
            predictions,
            targets.clone(),
            Reduction::Mean,
        );

        if epoch % 2000 == 0 || epoch == epochs - 1 {
            // .clone().into_scalar() extracts the f32 value from a 1-element tensor.
            println!("  epoch {:>5} | loss = {:.8}", epoch, loss.clone().into_scalar());
        }

        // Backward pass — this is where the magic happens.
        // Burn traced every operation in the forward pass (building a computation
        // graph), and now walks it backwards to compute all gradients via the
        // chain rule. This is EXACTLY what we did by hand in main.rs, but
        // automated for any architecture, no matter how complex.
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        // Gradient descent step: θ ← θ - η∇L for all parameters.
        // Note: Burn uses ownership transfer (model is consumed and returned).
        // This is Rust's way of ensuring you can't accidentally use stale weights.
        model = optim.step(lr, model, grads);
    }

    // =========================================================================
    // EVALUATION
    // =========================================================================
    //
    // .valid() strips the Autodiff wrapper — we don't need gradient tracking
    // for inference. This is like model.eval() in PyTorch.
    //
    let model_valid: XorNet<InferenceBackend> = model.valid();

    let test_inputs = Tensor::<InferenceBackend, 2>::from_floats(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        &device,
    );
    let expected = [0.0_f32, 1.0, 1.0, 0.0];

    let predictions = model_valid.forward(test_inputs);

    let pairs = [[0, 0], [0, 1], [1, 0], [1, 1]];

    println!("\n--- Results ---\n");
    println!("  x₁  x₂  │ expected  predicted  correct?");
    println!(" ──────────┼──────────────────────────────");
    for (i, pair) in pairs.iter().enumerate() {
        let pred: f32 = predictions.clone().slice([i..i + 1, 0..1]).into_scalar();
        let rounded = if pred > 0.5 { 1.0 } else { 0.0 };
        let check = if rounded == expected[i] { "yes" } else { "NO" };
        println!(
            "   {}   {}  │   {}       {:.5}     {}",
            pair[0], pair[1], expected[i], pred, check
        );
    }

    // =========================================================================
    // COMPARISON SUMMARY
    // =========================================================================
    println!("\n=== From-scratch vs Burn ===\n");
    println!("  WHAT WE WROTE BY HAND:          WHAT BURN DID FOR US:");
    println!("  ─────────────────────           ─────────────────────");
    println!("  sigmoid()                       Sigmoid::new()");
    println!("  sigmoid_derivative()            (autograd handles it)");
    println!("  forward() with loops            model.forward(x) via Linear");
    println!("  backprop chain rule             loss.backward()");
    println!("  weight += lr * delta * input    optim.step(lr, model, grads)");
    println!("  ~200 lines                      ~50 lines of model code");
    println!();
    println!("The math is identical. The framework just automates the tedious parts.");
}
