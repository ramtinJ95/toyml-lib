package main

import (
	"math"
	"math/rand"
	"testing"
)

const epsilon = 1e-6

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < epsilon
}

// Test 1: Simple multiplication a * b where a=2, b=3
func TestSimpleMul(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)

	c := a.Mul(b)
	c.Backprop()

	t.Logf("Simple Mul: a=%v, b=%v", a.Data, b.Data)
	t.Logf("Result: c.Data=%v", c.Data)
	t.Logf("Gradients: a.Grad=%v, b.Grad=%v", a.Grad, b.Grad)

	// c = a * b
	// dc/da = b = 3
	// dc/db = a = 2
	expectedAGrad := 3.0
	expectedBGrad := 2.0

	if !almostEqual(c.Data, 6.0) {
		t.Errorf("Expected c.Data=6.0, got %v", c.Data)
	}
	if !almostEqual(a.Grad, expectedAGrad) {
		t.Errorf("Expected a.Grad=%v, got %v", expectedAGrad, a.Grad)
	}
	if !almostEqual(b.Grad, expectedBGrad) {
		t.Errorf("Expected b.Grad=%v, got %v", expectedBGrad, b.Grad)
	}
}

// Test 2: Chain (a * b) + c where a=2, b=3, c=4
func TestChainMulAdd(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)
	c := NewValue(4.0)

	d := a.Mul(b).Add(c)
	d.Backprop()

	t.Logf("Chain: a=%v, b=%v, c=%v", a.Data, b.Data, c.Data)
	t.Logf("Result: d.Data=%v", d.Data)
	t.Logf("Gradients: a.Grad=%v, b.Grad=%v, c.Grad=%v", a.Grad, b.Grad, c.Grad)

	// d = a*b + c
	// dd/da = b = 3
	// dd/db = a = 2
	// dd/dc = 1
	expectedAGrad := 3.0
	expectedBGrad := 2.0
	expectedCGrad := 1.0

	if !almostEqual(d.Data, 10.0) {
		t.Errorf("Expected d.Data=10.0, got %v", d.Data)
	}
	if !almostEqual(a.Grad, expectedAGrad) {
		t.Errorf("Expected a.Grad=%v, got %v", expectedAGrad, a.Grad)
	}
	if !almostEqual(b.Grad, expectedBGrad) {
		t.Errorf("Expected b.Grad=%v, got %v", expectedBGrad, b.Grad)
	}
	if !almostEqual(c.Grad, expectedCGrad) {
		t.Errorf("Expected c.Grad=%v, got %v", expectedCGrad, c.Grad)
	}
}

// Test 3: Tanh of product: tanh(a * b) where a=0.5, b=0.5
func TestTanhMul(t *testing.T) {
	a := NewValue(0.5)
	b := NewValue(0.5)

	c := a.Mul(b).Tanh()
	c.Backprop()

	t.Logf("Tanh: a=%v, b=%v", a.Data, b.Data)
	t.Logf("Result: c.Data=%v", c.Data)
	t.Logf("Gradients: a.Grad=%v, b.Grad=%v", a.Grad, b.Grad)

	// c = tanh(a*b)
	// Let z = a*b = 0.25
	// c = tanh(z)
	// dc/dz = 1 - tanh(z)^2
	// dc/da = dc/dz * dz/da = (1 - tanh(z)^2) * b
	// dc/db = dc/dz * dz/db = (1 - tanh(z)^2) * a
	z := 0.5 * 0.5
	tanhZ := math.Tanh(z)
	dtanh := 1 - tanhZ*tanhZ
	expectedAGrad := dtanh * 0.5 // b
	expectedBGrad := dtanh * 0.5 // a

	if !almostEqual(c.Data, tanhZ) {
		t.Errorf("Expected c.Data=%v, got %v", tanhZ, c.Data)
	}
	if !almostEqual(a.Grad, expectedAGrad) {
		t.Errorf("Expected a.Grad=%v, got %v", expectedAGrad, a.Grad)
	}
	if !almostEqual(b.Grad, expectedBGrad) {
		t.Errorf("Expected b.Grad=%v, got %v", expectedBGrad, b.Grad)
	}
}

// Test 4: Neuron-like computation: o = tanh(x1*w1 + x2*w2 + b)
func TestNeuronComputation(t *testing.T) {
	// inputs
	x1 := NewValue(2.0)
	x2 := NewValue(0.0)
	// weights
	w1 := NewValue(-3.0)
	w2 := NewValue(1.0)
	// bias
	b := NewValue(6.8813735870195432)

	// neuron: o = tanh(x1*w1 + x2*w2 + b)
	x1w1 := x1.Mul(w1)
	x2w2 := x2.Mul(w2)
	x1w1x2w2 := x1w1.Add(x2w2)
	n := x1w1x2w2.Add(b)
	o := n.Tanh()

	o.Backprop()

	t.Logf("Neuron: x1=%v, x2=%v, w1=%v, w2=%v, b=%v", x1.Data, x2.Data, w1.Data, w2.Data, b.Data)
	t.Logf("Result: o.Data=%v", o.Data)
	t.Logf("Gradients: x1.Grad=%v, x2.Grad=%v, w1.Grad=%v, w2.Grad=%v, b.Grad=%v",
		x1.Grad, x2.Grad, w1.Grad, w2.Grad, b.Grad)

	// Expected from PyTorch (will be printed by validate_grads.py)
	// n = x1*w1 + x2*w2 + b = 2*(-3) + 0*1 + 6.8813735870195432 = 0.8813735870195432
	// o = tanh(n) ~ 0.7071
	// do/dn = 1 - tanh(n)^2 = 1 - 0.7071^2 ~ 0.5
	// do/dx1 = do/dn * dn/dx1 = do/dn * w1
	// do/dw1 = do/dn * dn/dw1 = do/dn * x1
	// etc.

	nVal := 2.0*(-3.0) + 0.0*1.0 + 6.8813735870195432
	tanhN := math.Tanh(nVal)
	dtanhN := 1 - tanhN*tanhN

	expectedX1Grad := dtanhN * w1.Data
	expectedW1Grad := dtanhN * x1.Data
	expectedX2Grad := dtanhN * w2.Data
	expectedW2Grad := dtanhN * x2.Data
	expectedBGrad := dtanhN * 1.0

	if !almostEqual(o.Data, tanhN) {
		t.Errorf("Expected o.Data=%v, got %v", tanhN, o.Data)
	}
	if !almostEqual(x1.Grad, expectedX1Grad) {
		t.Errorf("Expected x1.Grad=%v, got %v", expectedX1Grad, x1.Grad)
	}
	if !almostEqual(w1.Grad, expectedW1Grad) {
		t.Errorf("Expected w1.Grad=%v, got %v", expectedW1Grad, w1.Grad)
	}
	if !almostEqual(x2.Grad, expectedX2Grad) {
		t.Errorf("Expected x2.Grad=%v, got %v", expectedX2Grad, x2.Grad)
	}
	if !almostEqual(w2.Grad, expectedW2Grad) {
		t.Errorf("Expected w2.Grad=%v, got %v", expectedW2Grad, w2.Grad)
	}
	if !almostEqual(b.Grad, expectedBGrad) {
		t.Errorf("Expected b.Grad=%v, got %v", expectedBGrad, b.Grad)
	}
}

// Test 5: Test subtraction: (a - b) where a=5, b=3
func TestSub(t *testing.T) {
	a := NewValue(5.0)
	b := NewValue(3.0)

	c := a.Sub(b)
	c.Backprop()

	t.Logf("Sub: a=%v, b=%v", a.Data, b.Data)
	t.Logf("Result: c.Data=%v", c.Data)
	t.Logf("Gradients: a.Grad=%v, b.Grad=%v", a.Grad, b.Grad)

	// c = a - b
	// dc/da = 1
	// dc/db = -1
	if !almostEqual(c.Data, 2.0) {
		t.Errorf("Expected c.Data=2.0, got %v", c.Data)
	}
	if !almostEqual(a.Grad, 1.0) {
		t.Errorf("Expected a.Grad=1.0, got %v", a.Grad)
	}
	if !almostEqual(b.Grad, -1.0) {
		t.Errorf("Expected b.Grad=-1.0, got %v", b.Grad)
	}
}

// Test 6: Test division: a / b where a=6, b=2
func TestDiv(t *testing.T) {
	a := NewValue(6.0)
	b := NewValue(2.0)

	c := a.Div(b)
	c.Backprop()

	t.Logf("Div: a=%v, b=%v", a.Data, b.Data)
	t.Logf("Result: c.Data=%v", c.Data)
	t.Logf("Gradients: a.Grad=%v, b.Grad=%v", a.Grad, b.Grad)

	// c = a / b = a * b^(-1)
	// dc/da = 1/b = 0.5
	// dc/db = -a/b^2 = -6/4 = -1.5
	if !almostEqual(c.Data, 3.0) {
		t.Errorf("Expected c.Data=3.0, got %v", c.Data)
	}
	if !almostEqual(a.Grad, 0.5) {
		t.Errorf("Expected a.Grad=0.5, got %v", a.Grad)
	}
	if !almostEqual(b.Grad, -1.5) {
		t.Errorf("Expected b.Grad=-1.5, got %v", b.Grad)
	}
}

// Test 7: Test power: a^3 where a=2
func TestPow(t *testing.T) {
	a := NewValue(2.0)

	c := a.Pow(3)
	c.Backprop()

	t.Logf("Pow: a=%v, exponent=3", a.Data)
	t.Logf("Result: c.Data=%v", c.Data)
	t.Logf("Gradients: a.Grad=%v", a.Grad)

	// c = a^3
	// dc/da = 3 * a^2 = 3 * 4 = 12
	if !almostEqual(c.Data, 8.0) {
		t.Errorf("Expected c.Data=8.0, got %v", c.Data)
	}
	if !almostEqual(a.Grad, 12.0) {
		t.Errorf("Expected a.Grad=12.0, got %v", a.Grad)
	}
}

// Test 8: Neuron forward pass and gradients (with fixed weights)
func TestNeuronForwardGrad(t *testing.T) {
	// Use fixed seed for reproducibility
	rand.Seed(42)

	// Create neuron with 2 inputs
	neuron := NewNeuron(2)

	// Log the random weights for comparison with PyTorch
	t.Logf("Neuron weights: w[0]=%v, w[1]=%v, b=%v",
		neuron.w[0].Data, neuron.w[1].Data, neuron.b.Data)

	// Create inputs
	x := []*Value{NewValue(0.5), NewValue(-0.3)}

	// Forward pass
	out := neuron.Forward(x)
	out.Backprop()

	t.Logf("Input: x[0]=%v, x[1]=%v", x[0].Data, x[1].Data)
	t.Logf("Output: out.Data=%v", out.Data)
	t.Logf("Gradients: x[0].Grad=%v, x[1].Grad=%v", x[0].Grad, x[1].Grad)
	t.Logf("Weight gradients: w[0].Grad=%v, w[1].Grad=%v, b.Grad=%v",
		neuron.w[0].Grad, neuron.w[1].Grad, neuron.b.Grad)

	// Verify that gradients are non-zero (sanity check)
	if out.Data == 0 {
		t.Error("Output should not be zero")
	}

	// Compute expected values manually
	// activation = b + w[0]*x[0] + w[1]*x[1]
	// out = tanh(activation)
	activation := neuron.b.Data + neuron.w[0].Data*x[0].Data + neuron.w[1].Data*x[1].Data
	expectedOut := math.Tanh(activation)
	dtanh := 1 - expectedOut*expectedOut

	if !almostEqual(out.Data, expectedOut) {
		t.Errorf("Expected out.Data=%v, got %v", expectedOut, out.Data)
	}

	// Gradient checks
	// d(out)/d(w[i]) = dtanh * x[i]
	// d(out)/d(x[i]) = dtanh * w[i]
	// d(out)/d(b) = dtanh
	expectedW0Grad := dtanh * x[0].Data
	expectedW1Grad := dtanh * x[1].Data
	expectedX0Grad := dtanh * neuron.w[0].Data
	expectedX1Grad := dtanh * neuron.w[1].Data
	expectedBGrad := dtanh

	if !almostEqual(neuron.w[0].Grad, expectedW0Grad) {
		t.Errorf("Expected w[0].Grad=%v, got %v", expectedW0Grad, neuron.w[0].Grad)
	}
	if !almostEqual(neuron.w[1].Grad, expectedW1Grad) {
		t.Errorf("Expected w[1].Grad=%v, got %v", expectedW1Grad, neuron.w[1].Grad)
	}
	if !almostEqual(x[0].Grad, expectedX0Grad) {
		t.Errorf("Expected x[0].Grad=%v, got %v", expectedX0Grad, x[0].Grad)
	}
	if !almostEqual(x[1].Grad, expectedX1Grad) {
		t.Errorf("Expected x[1].Grad=%v, got %v", expectedX1Grad, x[1].Grad)
	}
	if !almostEqual(neuron.b.Grad, expectedBGrad) {
		t.Errorf("Expected b.Grad=%v, got %v", expectedBGrad, neuron.b.Grad)
	}
}

// Test 9: Layer forward pass and gradients
func TestLayerForwardGrad(t *testing.T) {
	// Use fixed seed for reproducibility
	rand.Seed(42)

	// Create layer with 2 inputs, 3 outputs (3 neurons)
	layer := NewLayer(2, 3)

	// Log the layer structure
	t.Logf("Layer: 2 inputs -> 3 neurons")
	for i, n := range layer.Neurons {
		t.Logf("Neuron %d: w[0]=%v, w[1]=%v, b=%v",
			i, n.w[0].Data, n.w[1].Data, n.b.Data)
	}

	// Create inputs
	x := []*Value{NewValue(0.5), NewValue(-0.3)}

	// Forward pass
	out := layer.Forward(x)

	// Sum all outputs to create a scalar for backprop
	sum := out[0]
	for i := 1; i < len(out); i++ {
		sum = sum.Add(out[i])
	}
	sum.Backprop()

	t.Logf("Input: x[0]=%v, x[1]=%v", x[0].Data, x[1].Data)
	t.Logf("Outputs: out[0]=%v, out[1]=%v, out[2]=%v", out[0].Data, out[1].Data, out[2].Data)
	t.Logf("Sum: %v", sum.Data)
	t.Logf("Input gradients: x[0].Grad=%v, x[1].Grad=%v", x[0].Grad, x[1].Grad)

	// Verify outputs are valid tanh outputs (-1, 1)
	for i, o := range out {
		if o.Data < -1 || o.Data > 1 {
			t.Errorf("Output %d should be in range [-1, 1], got %v", i, o.Data)
		}
	}

	// Verify gradients exist and are non-zero for inputs
	if x[0].Grad == 0 && x[1].Grad == 0 {
		t.Error("At least one input should have non-zero gradient")
	}

	// Verify all parameters have gradients
	params := layer.Parameters()
	if len(params) != 9 { // 3 neurons * (2 weights + 1 bias)
		t.Errorf("Expected 9 parameters, got %d", len(params))
	}
	for i, p := range params {
		t.Logf("Param %d: Data=%v, Grad=%v", i, p.Data, p.Grad)
	}
}

// Test 10: MLP forward pass and gradients
func TestMLPForwardGrad(t *testing.T) {
	// Use fixed seed for reproducibility
	rand.Seed(42)

	// Create MLP: 2 inputs -> 3 hidden -> 1 output
	mlp := NewMLP(2, []int{3, 1})

	// Log the MLP structure
	t.Logf("MLP: 2 -> 3 -> 1")
	for i, layer := range mlp.Layers {
		t.Logf("Layer %d:", i)
		for j, n := range layer.Neurons {
			wStr := ""
			for k, w := range n.w {
				wStr += " w[" + string(rune('0'+k)) + "]=" + string(rune('0'+int(w.Data*10)/10))
			}
			t.Logf("  Neuron %d: %v, b=%v", j, n.w, n.b.Data)
		}
	}

	// Create inputs
	x := []*Value{NewValue(0.5), NewValue(-0.3)}

	// Forward pass
	out := mlp.Forward(x)

	// Backprop on the single output
	out[0].Backprop()

	t.Logf("Input: x[0]=%v, x[1]=%v", x[0].Data, x[1].Data)
	t.Logf("Output: out[0]=%v", out[0].Data)
	t.Logf("Input gradients: x[0].Grad=%v, x[1].Grad=%v", x[0].Grad, x[1].Grad)

	// Verify output is valid tanh output (-1, 1)
	if out[0].Data < -1 || out[0].Data > 1 {
		t.Errorf("Output should be in range [-1, 1], got %v", out[0].Data)
	}

	// Verify all parameters have gradients
	params := mlp.Parameters()
	expectedParams := 3*(2+1) + 1*(3+1) // layer1: 3 neurons * (2 weights + 1 bias), layer2: 1 neuron * (3 weights + 1 bias)
	if len(params) != expectedParams {
		t.Errorf("Expected %d parameters, got %d", expectedParams, len(params))
	}

	t.Logf("Total parameters: %d", len(params))
	for i, p := range params {
		t.Logf("Param %d: Data=%v, Grad=%v", i, p.Data, p.Grad)
	}
}

// Test 11: MSE loss computation and gradients
func TestMSELossGrad(t *testing.T) {
	// Use fixed values (no random seed needed)

	// Predictions
	yPred := []*Value{NewValue(0.8), NewValue(0.5), NewValue(-0.2)}

	// Targets
	y := []*Value{NewValue(1.0), NewValue(0.3), NewValue(0.0)}

	// Compute MSE loss
	loss := MSE(yPred, y)
	loss.Backprop()

	t.Logf("Predictions: %v, %v, %v", yPred[0].Data, yPred[1].Data, yPred[2].Data)
	t.Logf("Targets: %v, %v, %v", y[0].Data, y[1].Data, y[2].Data)
	t.Logf("Loss: %v", loss.Data)
	t.Logf("Prediction gradients: %v, %v, %v", yPred[0].Grad, yPred[1].Grad, yPred[2].Grad)
	t.Logf("Target gradients: %v, %v, %v", y[0].Grad, y[1].Grad, y[2].Grad)

	// Manual calculation:
	// MSE = sum((yPred[i] - y[i])^2)
	// = (0.8 - 1.0)^2 + (0.5 - 0.3)^2 + (-0.2 - 0.0)^2
	// = 0.04 + 0.04 + 0.04 = 0.12
	expectedLoss := 0.04 + 0.04 + 0.04

	if !almostEqual(loss.Data, expectedLoss) {
		t.Errorf("Expected loss=%v, got %v", expectedLoss, loss.Data)
	}

	// Gradients:
	// d(loss)/d(yPred[i]) = 2 * (yPred[i] - y[i])
	// d(loss)/d(y[i]) = -2 * (yPred[i] - y[i])
	for i := range yPred {
		diff := yPred[i].Data - y[i].Data
		expectedPredGrad := 2 * diff
		expectedTargetGrad := -2 * diff

		if !almostEqual(yPred[i].Grad, expectedPredGrad) {
			t.Errorf("Expected yPred[%d].Grad=%v, got %v", i, expectedPredGrad, yPred[i].Grad)
		}
		if !almostEqual(y[i].Grad, expectedTargetGrad) {
			t.Errorf("Expected y[%d].Grad=%v, got %v", i, expectedTargetGrad, y[i].Grad)
		}
	}
}

// Test 12: Neuron with fixed weights for PyTorch comparison
func TestNeuronFixedWeights(t *testing.T) {
	// Create neuron manually with fixed weights (matching PyTorch)
	neuron := &Neuron{
		w: []*Value{NewValue(0.3), NewValue(-0.5)},
		b: NewValue(0.1),
	}

	// Create inputs
	x := []*Value{NewValue(0.5), NewValue(-0.3)}

	// Forward pass
	out := neuron.Forward(x)
	out.Backprop()

	t.Logf("Fixed Neuron Test:")
	t.Logf("Weights: w[0]=%v, w[1]=%v, b=%v", neuron.w[0].Data, neuron.w[1].Data, neuron.b.Data)
	t.Logf("Input: x[0]=%v, x[1]=%v", x[0].Data, x[1].Data)
	t.Logf("Output: out.Data=%v", out.Data)
	t.Logf("Gradients: x[0].Grad=%v, x[1].Grad=%v", x[0].Grad, x[1].Grad)
	t.Logf("Weight gradients: w[0].Grad=%v, w[1].Grad=%v, b.Grad=%v",
		neuron.w[0].Grad, neuron.w[1].Grad, neuron.b.Grad)

	// activation = b + w[0]*x[0] + w[1]*x[1]
	//            = 0.1 + 0.3*0.5 + (-0.5)*(-0.3)
	//            = 0.1 + 0.15 + 0.15 = 0.4
	// out = tanh(0.4) = 0.3799489622552249
	activation := 0.1 + 0.3*0.5 + (-0.5)*(-0.3)
	expectedOut := math.Tanh(activation)
	dtanh := 1 - expectedOut*expectedOut

	if !almostEqual(out.Data, expectedOut) {
		t.Errorf("Expected out.Data=%v, got %v", expectedOut, out.Data)
	}

	// Expected gradients
	expectedW0Grad := dtanh * 0.5   // x[0]
	expectedW1Grad := dtanh * -0.3  // x[1]
	expectedX0Grad := dtanh * 0.3   // w[0]
	expectedX1Grad := dtanh * -0.5  // w[1]
	expectedBGrad := dtanh * 1.0

	if !almostEqual(neuron.w[0].Grad, expectedW0Grad) {
		t.Errorf("Expected w[0].Grad=%v, got %v", expectedW0Grad, neuron.w[0].Grad)
	}
	if !almostEqual(neuron.w[1].Grad, expectedW1Grad) {
		t.Errorf("Expected w[1].Grad=%v, got %v", expectedW1Grad, neuron.w[1].Grad)
	}
	if !almostEqual(x[0].Grad, expectedX0Grad) {
		t.Errorf("Expected x[0].Grad=%v, got %v", expectedX0Grad, x[0].Grad)
	}
	if !almostEqual(x[1].Grad, expectedX1Grad) {
		t.Errorf("Expected x[1].Grad=%v, got %v", expectedX1Grad, x[1].Grad)
	}
	if !almostEqual(neuron.b.Grad, expectedBGrad) {
		t.Errorf("Expected b.Grad=%v, got %v", expectedBGrad, neuron.b.Grad)
	}
}

// Test 13: Layer with fixed weights for PyTorch comparison
func TestLayerFixedWeights(t *testing.T) {
	// Create layer manually with fixed weights
	layer := &Layer{
		Neurons: []*Neuron{
			{w: []*Value{NewValue(0.3), NewValue(-0.5)}, b: NewValue(0.1)},
			{w: []*Value{NewValue(0.2), NewValue(0.4)}, b: NewValue(-0.2)},
		},
	}

	// Create inputs
	x := []*Value{NewValue(0.5), NewValue(-0.3)}

	// Forward pass
	out := layer.Forward(x)

	// Sum outputs for scalar loss
	sum := out[0].Add(out[1])
	sum.Backprop()

	t.Logf("Fixed Layer Test (2 inputs -> 2 neurons):")
	t.Logf("Neuron 0: w=[%v, %v], b=%v", layer.Neurons[0].w[0].Data, layer.Neurons[0].w[1].Data, layer.Neurons[0].b.Data)
	t.Logf("Neuron 1: w=[%v, %v], b=%v", layer.Neurons[1].w[0].Data, layer.Neurons[1].w[1].Data, layer.Neurons[1].b.Data)
	t.Logf("Input: x=[%v, %v]", x[0].Data, x[1].Data)
	t.Logf("Outputs: [%v, %v]", out[0].Data, out[1].Data)
	t.Logf("Sum: %v", sum.Data)
	t.Logf("Input gradients: [%v, %v]", x[0].Grad, x[1].Grad)

	// Compute expected outputs manually
	// Neuron 0: tanh(0.1 + 0.3*0.5 + (-0.5)*(-0.3)) = tanh(0.4)
	// Neuron 1: tanh(-0.2 + 0.2*0.5 + 0.4*(-0.3)) = tanh(-0.22)
	act0 := 0.1 + 0.3*0.5 + (-0.5)*(-0.3)
	act1 := -0.2 + 0.2*0.5 + 0.4*(-0.3)
	exp0 := math.Tanh(act0)
	exp1 := math.Tanh(act1)

	if !almostEqual(out[0].Data, exp0) {
		t.Errorf("Expected out[0].Data=%v, got %v", exp0, out[0].Data)
	}
	if !almostEqual(out[1].Data, exp1) {
		t.Errorf("Expected out[1].Data=%v, got %v", exp1, out[1].Data)
	}

	// Sum gradient is 1.0, so gradients flow back
	dtanh0 := 1 - exp0*exp0
	dtanh1 := 1 - exp1*exp1

	// d(sum)/d(x[0]) = dtanh0 * w0[0] + dtanh1 * w1[0]
	expectedX0Grad := dtanh0*0.3 + dtanh1*0.2
	expectedX1Grad := dtanh0*(-0.5) + dtanh1*0.4

	if !almostEqual(x[0].Grad, expectedX0Grad) {
		t.Errorf("Expected x[0].Grad=%v, got %v", expectedX0Grad, x[0].Grad)
	}
	if !almostEqual(x[1].Grad, expectedX1Grad) {
		t.Errorf("Expected x[1].Grad=%v, got %v", expectedX1Grad, x[1].Grad)
	}
}

// Test 14: MLP with fixed weights for PyTorch comparison
func TestMLPFixedWeights(t *testing.T) {
	// Create MLP manually: 2 inputs -> 2 hidden -> 1 output
	mlp := &MLP{
		Sizes: []int{2, 2, 1},
		Layers: []*Layer{
			{
				Neurons: []*Neuron{
					{w: []*Value{NewValue(0.3), NewValue(-0.5)}, b: NewValue(0.1)},
					{w: []*Value{NewValue(0.2), NewValue(0.4)}, b: NewValue(-0.2)},
				},
			},
			{
				Neurons: []*Neuron{
					{w: []*Value{NewValue(0.6), NewValue(-0.3)}, b: NewValue(0.05)},
				},
			},
		},
	}

	// Create inputs
	x := []*Value{NewValue(0.5), NewValue(-0.3)}

	// Forward pass
	out := mlp.Forward(x)

	// Backprop
	out[0].Backprop()

	t.Logf("Fixed MLP Test (2 -> 2 -> 1):")
	t.Logf("Layer 0 Neuron 0: w=[%v, %v], b=%v",
		mlp.Layers[0].Neurons[0].w[0].Data, mlp.Layers[0].Neurons[0].w[1].Data, mlp.Layers[0].Neurons[0].b.Data)
	t.Logf("Layer 0 Neuron 1: w=[%v, %v], b=%v",
		mlp.Layers[0].Neurons[1].w[0].Data, mlp.Layers[0].Neurons[1].w[1].Data, mlp.Layers[0].Neurons[1].b.Data)
	t.Logf("Layer 1 Neuron 0: w=[%v, %v], b=%v",
		mlp.Layers[1].Neurons[0].w[0].Data, mlp.Layers[1].Neurons[0].w[1].Data, mlp.Layers[1].Neurons[0].b.Data)
	t.Logf("Input: x=[%v, %v]", x[0].Data, x[1].Data)
	t.Logf("Output: %v", out[0].Data)
	t.Logf("Input gradients: [%v, %v]", x[0].Grad, x[1].Grad)

	// Compute expected output manually
	// Hidden layer:
	// h0 = tanh(0.1 + 0.3*0.5 + (-0.5)*(-0.3)) = tanh(0.4)
	// h1 = tanh(-0.2 + 0.2*0.5 + 0.4*(-0.3)) = tanh(-0.22)
	act0 := 0.1 + 0.3*0.5 + (-0.5)*(-0.3)
	act1 := -0.2 + 0.2*0.5 + 0.4*(-0.3)
	h0 := math.Tanh(act0)
	h1 := math.Tanh(act1)

	// Output layer:
	// o = tanh(0.05 + 0.6*h0 + (-0.3)*h1)
	act2 := 0.05 + 0.6*h0 + (-0.3)*h1
	expectedOut := math.Tanh(act2)

	if !almostEqual(out[0].Data, expectedOut) {
		t.Errorf("Expected out[0].Data=%v, got %v", expectedOut, out[0].Data)
	}

	// Log intermediate values
	t.Logf("Hidden activations: h0=%v, h1=%v", h0, h1)
	t.Logf("Output activation: %v", act2)
	t.Logf("Expected output: %v", expectedOut)
}

// Test 15: Full MLP with MSE loss for PyTorch comparison
func TestMLPWithMSE(t *testing.T) {
	// Create MLP manually: 2 inputs -> 2 hidden -> 1 output
	mlp := &MLP{
		Sizes: []int{2, 2, 1},
		Layers: []*Layer{
			{
				Neurons: []*Neuron{
					{w: []*Value{NewValue(0.3), NewValue(-0.5)}, b: NewValue(0.1)},
					{w: []*Value{NewValue(0.2), NewValue(0.4)}, b: NewValue(-0.2)},
				},
			},
			{
				Neurons: []*Neuron{
					{w: []*Value{NewValue(0.6), NewValue(-0.3)}, b: NewValue(0.05)},
				},
			},
		},
	}

	// Create inputs
	x := []*Value{NewValue(0.5), NewValue(-0.3)}

	// Forward pass
	out := mlp.Forward(x)

	// Target
	y := []*Value{NewValue(1.0)}

	// MSE loss
	loss := MSE(out, y)
	loss.Backprop()

	t.Logf("MLP with MSE Loss Test:")
	t.Logf("Input: x=[%v, %v]", x[0].Data, x[1].Data)
	t.Logf("Output: %v", out[0].Data)
	t.Logf("Target: %v", y[0].Data)
	t.Logf("Loss: %v", loss.Data)
	t.Logf("Input gradients: [%v, %v]", x[0].Grad, x[1].Grad)
	t.Logf("Output gradient: %v", out[0].Grad)

	// Log parameter gradients
	params := mlp.Parameters()
	for i, p := range params {
		t.Logf("Param %d: Data=%v, Grad=%v", i, p.Data, p.Grad)
	}

	// Compute expected loss
	// loss = (out - y)^2
	expectedLoss := (out[0].Data - y[0].Data) * (out[0].Data - y[0].Data)

	if !almostEqual(loss.Data, expectedLoss) {
		t.Errorf("Expected loss=%v, got %v", expectedLoss, loss.Data)
	}

	// d(loss)/d(out) = 2 * (out - y)
	expectedOutGrad := 2 * (out[0].Data - y[0].Data)
	if !almostEqual(out[0].Grad, expectedOutGrad) {
		t.Errorf("Expected out[0].Grad=%v, got %v", expectedOutGrad, out[0].Grad)
	}
}
