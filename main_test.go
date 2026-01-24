package main

import (
	"math"
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
