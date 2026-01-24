package main

import "math/rand"

type Neuron struct {
	w []*Value
	b *Value
}

func NewNeuron(nin int) *Neuron {
	w := make([]*Value, nin)
	for i := range w {
		w[i] = NewValue(2*rand.Float64() - 1)
	}
	b := NewValue(2*rand.Float64() - 1)
	return &Neuron{
		w: w,
		b: b,
	}
}

func (n *Neuron) Forward(x []*Value) *Value {
	if len(n.w) != len(x) {
		panic("Weight and data diff dimensions")
	}

	act := n.b
	for i := range x {
		act = act.Add(n.w[i].Mul(x[i]))
	}

	return act.Tanh()
}

func (n *Neuron) Parameters() []*Value {
	return append(n.w, n.b)
}

// Layer

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(nin, nout int) *Layer {
	neurons := make([]*Neuron, nout)
	for i := range nout {
		neurons[i] = NewNeuron(nin)
	}

	return &Layer{Neurons: neurons}
}

func (l *Layer) Forward(x []*Value) []*Value {
	out := make([]*Value, len(l.Neurons))
	for i, n := range l.Neurons {
		out[i] = n.Forward(x)
	}

	return out
}

func (l *Layer) Parameters() []*Value {
	res := []*Value{}
	for _, n := range l.Neurons {
		res = append(res, n.Parameters()...)
	}
	return res
}

// MLP

type MLP struct {
	Sizes  []int
	Layers []*Layer
}

func NewMLP(nin int, nouts []int) *MLP {
	sizes := append([]int{nin}, nouts...)
	layers := make([]*Layer, len(nouts))

	for i := range nouts {
		layers[i] = NewLayer(sizes[i], sizes[i+1])
	}

	return &MLP{Sizes: sizes, Layers: layers}
}

func (mlp *MLP) Forward(x []*Value) []*Value {
	for _, l := range mlp.Layers {
		x = l.Forward(x)
	}

	return x
}

func (mlp *MLP) Parameters() []*Value {
	res := []*Value{}
	for _, l := range mlp.Layers {
		res = append(res, l.Parameters()...)
	}
	return res
}

// Loss

func MSE(yPred, y []*Value) *Value {
	if len(y) != len(yPred) {
		panic("y and yPred different sizes")
	}
	sum := NewValue(0)
	for i := range y {
		sum = sum.Add(yPred[i].Sub(y[i]).Pow(2))
	}

	return sum
}
